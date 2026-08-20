"""Run the corrected original MEL event stage on two supplied segmentations.

This entry point deliberately skips CZI loading, deconvolution, thresholding,
and the legacy review GUI. It keeps MEL's association, event classification,
duplicate filtering, and event-location stages with documented defect fixes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.ndimage import center_of_mass, label as connected_components

# Keep the maintained runner usable on headless Linux/CI machines.  Users can
# still override this explicitly before launching the process.
os.environ.setdefault("MPLBACKEND", "Agg")

import MEL
import Morphology


EVENT_NAMES = ("fusion", "fission", "depolarisation")
IMPLEMENTATION_ID = "corrected-original-mel-segmentation-stage/1.0"


def _prepare_segmentation(
    segmentation: np.ndarray,
    *,
    min_volume: int,
    input_kind: str,
) -> tuple[np.ndarray, np.ndarray, int, dict[int, int], str]:
    """Return a binary stack and contiguous positive labels.

    Boolean, 0/1, and conventional 0/255 inputs are treated as binary masks and
    connected components are labelled.  Other positive integer values are
    treated as supplied object IDs.  MEL uses contiguous internal IDs, so the
    returned mapping preserves the relationship to supplied labels.
    """

    array = np.asarray(segmentation)
    if array.ndim != 3:
        raise ValueError(
            "each segmentation must be a 3-D array in (z, y, x) order; "
            f"received shape {array.shape}"
        )
    if min_volume < 1:
        raise ValueError("min_volume must be at least 1")
    if input_kind not in {"auto", "binary", "labels"}:
        raise ValueError("input_kind must be 'auto', 'binary', or 'labels'")
    is_boolean = np.issubdtype(array.dtype, np.bool_)
    if not is_boolean and not np.issubdtype(array.dtype, np.number):
        raise TypeError("segmentations must contain numeric labels")
    if not np.all(np.isfinite(array)):
        raise ValueError("segmentations may not contain NaN or infinity")
    if np.any(array < 0) or not np.all(array == np.floor(array)):
        raise ValueError("segmentation labels must be non-negative integers")

    unique_values = np.unique(array)
    auto_binary = is_boolean or np.all(np.isin(unique_values, (0, 1)))
    auto_binary = auto_binary or np.all(np.isin(unique_values, (0, 255)))
    is_binary_encoding = input_kind == "binary" or (
        input_kind == "auto" and auto_binary
    )
    if is_binary_encoding:
        initial_labels, initial_count = connected_components(array.astype(bool))
        source_ids = range(1, initial_count + 1)
        resolved_input_kind = "binary"
    else:
        initial_labels = array.astype(np.int64, copy=False)
        source_ids = np.unique(initial_labels[initial_labels > 0])
        resolved_input_kind = "labels"

    labels = np.zeros(array.shape, dtype=np.int32)
    input_label_map: dict[int, int] = {}
    next_id = 1
    for source_id in source_ids:
        object_mask = initial_labels == source_id
        if int(np.count_nonzero(object_mask)) >= min_volume:
            labels[object_mask] = next_id
            input_label_map[next_id] = int(source_id)
            next_id += 1

    binary = (labels > 0).astype(np.uint8)
    return binary, labels, next_id - 1, input_label_map, resolved_input_kind


def _locations_to_lists(
    locations: list[list[np.ndarray]],
) -> dict[str, list[list[float]]]:
    return {
        name: [np.asarray(location, dtype=float).tolist() for location in values]
        for name, values in zip(EVENT_NAMES, locations)
    }


def _labels_to_lists(
    labels: list[list[Any]],
    input_label_maps: tuple[dict[int, int], dict[int, int], dict[int, int]],
) -> dict[str, list[Any]]:
    converted: dict[str, list[Any]] = {}
    for name, values, input_label_map in zip(EVENT_NAMES, labels, input_label_maps):
        rows = []
        for value in values:
            if isinstance(value, tuple):
                rows.append([input_label_map[int(item)] for item in value])
            else:
                rows.append(input_label_map[int(value)])
        converted[name] = rows
    return converted


def _serialise_label_map(input_label_map: dict[int, int]) -> list[dict[str, int]]:
    return [
        {"internal_label": internal, "input_label": source}
        for internal, source in sorted(input_label_map.items())
    ]


def analyse_segmentations(
    frame_1_segmentation: np.ndarray,
    frame_2_segmentation: np.ndarray,
    *,
    duplicate_distance: float = 10,
    distance_threshold: float = 20,
    overlap_threshold: float = 0.5,
    min_volume: int = 40,
    input_kind: str = "auto",
) -> dict[str, Any]:
    """Apply MEL's event stage to two delivered segmentations.

    Coordinates in the result use ``(z, y, x)`` index order.  Distances are in
    index-space voxels, matching the original implementation.
    """

    frame_1 = np.asarray(frame_1_segmentation)
    frame_2 = np.asarray(frame_2_segmentation)
    if frame_1.shape != frame_2.shape:
        raise ValueError(
            "the two segmentations must use the same (z, y, x) grid; "
            f"received {frame_1.shape} and {frame_2.shape}"
        )
    if not all(
        math.isfinite(value) for value in (duplicate_distance, distance_threshold)
    ):
        raise ValueError("distance parameters must be finite")
    if duplicate_distance < 0 or distance_threshold < 0:
        raise ValueError("distance parameters must be non-negative")
    if not math.isfinite(overlap_threshold) or not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between 0 and 1")

    binary_1, labels_1, count_1, input_label_map_1, resolved_kind_1 = (
        _prepare_segmentation(
            frame_1,
            min_volume=min_volume,
            input_kind=input_kind,
        )
    )
    binary_2, labels_2, count_2, input_label_map_2, resolved_kind_2 = (
        _prepare_segmentation(
            frame_2,
            min_volume=min_volume,
            input_kind=input_kind,
        )
    )

    label_stack_1 = Morphology.stack3DTo4D(labels_1, count_1)
    label_stack_2 = Morphology.stack3DTo4D(labels_2, count_2)
    filtered_labels_1 = MEL.gaussianFilter(label_stack_1)
    filtered_labels_2 = MEL.gaussianFilter(label_stack_2)

    centres_1 = list(center_of_mass(binary_1, labels_1, range(1, count_1 + 1)))
    centres_1.insert(0, (0, 0, 0))
    centres_1_array = np.asarray(centres_1, dtype=float)

    overlap_1_to_2 = MEL.compareOverlapV2(filtered_labels_1, filtered_labels_2)
    overlap_2_to_1 = overlap_1_to_2.T
    associated_1, associated_2 = MEL.getFragFusePairs(overlap_1_to_2, overlap_2_to_1)
    within_1, within_2 = MEL.backAndForthLabelMatching(associated_1, associated_2)

    canny_1 = MEL.labelToCanny(label_stack_1)
    canny_2 = MEL.labelToCanny(label_stack_2)
    halfway_1, distances_1, vectors_1 = MEL.getAllHalfWayPoints(canny_1, within_1)
    halfway_2, distances_2, vectors_2 = MEL.getAllHalfWayPoints(canny_2, within_2)
    statuses_1, statuses_2 = MEL.linkStatusPerLabel(
        binary_1,
        binary_2,
        overlap_1_to_2,
        overlap_2_to_1,
        associated_1,
        associated_2,
        within_1,
        within_2,
        distances_1,
        vectors_1,
        distances_2,
        vectors_2,
        distanceThreshold=distance_threshold,
        bigPercThresh=overlap_threshold,
    )

    (
        event_image,
        counts,
        locations,
        label_pairs,
        duplicate_counts,
        duplicate_locations,
        duplicate_label_pairs,
    ) = MEL.generateGradientImage(
        binary_1,
        centres_1_array,
        statuses_1,
        statuses_2,
        within_1,
        within_2,
        halfway_1,
        halfway_2,
        duplicateDistance=duplicate_distance,
    )

    return {
        "implementation": IMPLEMENTATION_ID,
        "counts": dict(zip(EVENT_NAMES, (int(value) for value in counts))),
        "locations": _locations_to_lists(locations),
        "label_pairs": _labels_to_lists(
            label_pairs,
            (input_label_map_1, input_label_map_2, input_label_map_1),
        ),
        "duplicates": {
            "fusion": int(duplicate_counts[0]),
            "fission": int(duplicate_counts[1]),
        },
        "duplicate_locations": {
            "fusion": [
                np.asarray(location, dtype=float).tolist()
                for location in duplicate_locations[0]
            ],
            "fission": [
                np.asarray(location, dtype=float).tolist()
                for location in duplicate_locations[1]
            ],
        },
        "duplicate_label_pairs": {
            "fusion": [
                [input_label_map_1[int(item)] for item in pair]
                for pair in duplicate_label_pairs[0]
            ],
            "fission": [
                [input_label_map_2[int(item)] for item in pair]
                for pair in duplicate_label_pairs[1]
            ],
        },
        "coordinate_order": ["z", "y", "x"],
        "distance_unit": "index-space voxel",
        "input_shape": list(frame_1.shape),
        "input_object_counts": {"frame_1": count_1, "frame_2": count_2},
        "resolved_input_kinds": {
            "frame_1": resolved_kind_1,
            "frame_2": resolved_kind_2,
        },
        "input_label_maps": {
            "frame_1": _serialise_label_map(input_label_map_1),
            "frame_2": _serialise_label_map(input_label_map_2),
        },
        "parameters": {
            "duplicate_distance": duplicate_distance,
            "distance_threshold": distance_threshold,
            "overlap_threshold": overlap_threshold,
            "min_volume": min_volume,
            "input_kind": input_kind,
        },
        "event_image": event_image,
    }


def _json_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "event_image"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run corrected original MEL from the segmentation stage on two 3-D TIFF "
            "label stacks."
        )
    )
    parser.add_argument("frame_1", type=Path, help="frame 1 label/binary TIFF")
    parser.add_argument("frame_2", type=Path, help="frame 2 label/binary TIFF")
    parser.add_argument("--output", type=Path, required=True, help="output JSON path")
    parser.add_argument(
        "--event-image",
        type=Path,
        help="optional RGB event-overlay TIFF path",
    )
    parser.add_argument("--duplicate-distance", type=float, default=10)
    parser.add_argument("--distance-threshold", type=float, default=20)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--min-volume", type=int, default=40)
    parser.add_argument(
        "--input-kind",
        choices=("auto", "binary", "labels"),
        default="auto",
        help="interpret both inputs automatically, as masks, or as label images",
    )
    return parser


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    """Compare path names and, when both exist, their filesystem identity."""

    if first.resolve() == second.resolve():
        return True
    try:
        return first.samefile(second)
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    input_paths = (args.frame_1, args.frame_2)
    if any(_paths_refer_to_same_file(args.output, path) for path in input_paths):
        parser.error("--output must not overwrite either input TIFF")
    if args.event_image is not None and any(
        _paths_refer_to_same_file(args.event_image, path) for path in input_paths
    ):
        parser.error("--event-image must not overwrite either input TIFF")
    if args.event_image is not None and _paths_refer_to_same_file(
        args.output, args.event_image
    ):
        parser.error("--output and --event-image must be different files")

    frame_1 = tifffile.imread(args.frame_1)
    frame_2 = tifffile.imread(args.frame_2)
    result = analyse_segmentations(
        frame_1,
        frame_2,
        duplicate_distance=args.duplicate_distance,
        distance_threshold=args.distance_threshold,
        overlap_threshold=args.overlap_threshold,
        min_volume=args.min_volume,
        input_kind=args.input_kind,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_result(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.event_image is not None:
        args.event_image.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            args.event_image,
            np.clip(result["event_image"] * 255, 0, 255).astype(np.uint8),
        )

    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
