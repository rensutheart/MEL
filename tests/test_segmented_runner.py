import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _two_to_one_labels():
    frame_1 = np.zeros((5, 40, 40), dtype=np.uint16)
    frame_1[:, 12:19, 8:15] = 1
    frame_1[:, 12:19, 22:29] = 2

    frame_2 = np.zeros_like(frame_1)
    frame_2[:, 12:19, 8:29] = 1
    return frame_1, frame_2


def test_overlap_uses_numpy_and_matches_the_reference_calculation():
    import MEL

    frame_1 = np.zeros((3, 2, 3, 4), dtype=np.float32)
    frame_2 = np.zeros((2, 2, 3, 4), dtype=np.float32)
    frame_1[1, :, 0:2, 0:2] = 1
    frame_1[2, :, 1:3, 2:4] = 0.5
    frame_2[1, :, 1:3, 1:4] = 0.75

    expected = np.zeros((3, 2), dtype=np.float64)
    for z_index in range(frame_1.shape[1]):
        for label_1 in range(frame_1.shape[0]):
            for label_2 in range(frame_2.shape[0]):
                expected[label_1, label_2] += np.sum(
                    frame_1[label_1, z_index] * frame_2[label_2, z_index]
                )

    np.testing.assert_allclose(MEL.compareOverlapV2(frame_1, frame_2), expected)


def test_structure_check_does_not_overflow_coordinates_above_127():
    import MEL

    stack = np.zeros((2, 180, 180), dtype=np.uint8)
    stack[0, 140, 140] = 1
    vector = (np.array([0, 130, 130]), np.array([0, 151, 151]))

    assert MEL.checkStructuresInBetween(stack, vector, numPoints=21)


def test_structure_check_ignores_the_two_participant_endpoints():
    import MEL

    stack = np.zeros((1, 32, 32), dtype=np.uint8)
    stack[0, 10, 10] = 1
    stack[0, 20, 20] = 1
    vector = (np.array([0, 10, 10]), np.array([0, 20, 20]))

    assert not MEL.checkStructuresInBetween(stack, vector)
    stack[0, 15, 15] = 1
    assert MEL.checkStructuresInBetween(stack, vector)


def test_structure_check_preserves_the_original_interior_sampling_lattice():
    import MEL

    stack = np.zeros((1, 24, 24), dtype=np.uint8)
    stack[0, 10, 10] = 1
    vector = (np.array([0, 0, 0]), np.array([0, 20, 20]))

    # The historical four-point lattice sampled fractions i/4. Excluding its
    # participant endpoint must retain the interior samples at 5, 10, and 15.
    assert MEL.checkStructuresInBetween(stack, vector, numPoints=4)


def test_segmented_fusion_and_reverse_fission_are_detected():
    from segmentation_runner import analyse_segmentations

    frame_1, frame_2 = _two_to_one_labels()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fusion = analyse_segmentations(frame_1, frame_2, duplicate_distance=8)
        fission = analyse_segmentations(frame_2, frame_1, duplicate_distance=8)

    assert fusion["counts"] == {"fusion": 1, "fission": 0, "depolarisation": 0}
    assert fission["counts"] == {"fusion": 0, "fission": 1, "depolarisation": 0}
    assert fusion["locations"]["fusion"] == [[0.0, 15.0, 18.0]]
    assert fission["locations"]["fission"] == [[0.0, 15.0, 18.0]]
    assert {tuple(sorted(pair)) for pair in fusion["label_pairs"]["fusion"]} == {
        (1, 2)
    }
    assert {tuple(sorted(pair)) for pair in fission["label_pairs"]["fission"]} == {
        (1, 2)
    }
    assert fusion["duplicates"] == {"fusion": 0, "fission": 0}
    assert fission["duplicates"] == {"fusion": 0, "fission": 0}


@pytest.mark.parametrize(
    "dtype, foreground_value",
    [(np.bool_, True), (np.uint8, 255)],
)
def test_common_binary_mask_encodings_are_component_labelled(
    dtype, foreground_value
):
    from segmentation_runner import analyse_segmentations

    labels_1, labels_2 = _two_to_one_labels()
    frame_1 = np.where(labels_1 > 0, foreground_value, 0).astype(dtype)
    frame_2 = np.where(labels_2 > 0, foreground_value, 0).astype(dtype)

    result = analyse_segmentations(frame_1, frame_2, duplicate_distance=8)

    assert result["input_object_counts"] == {"frame_1": 2, "frame_2": 1}
    assert result["counts"] == {"fusion": 1, "fission": 0, "depolarisation": 0}


def test_explicit_labels_mode_preserves_a_single_label_255():
    from segmentation_runner import analyse_segmentations

    frame_1 = np.zeros((3, 16, 16), dtype=np.uint16)
    frame_1[:, :5, :5] = 255
    frame_2 = np.zeros_like(frame_1)

    result = analyse_segmentations(
        frame_1,
        frame_2,
        input_kind="labels",
        min_volume=1,
    )

    assert result["input_label_maps"]["frame_1"] == [
        {"internal_label": 1, "input_label": 255}
    ]
    assert result["label_pairs"]["depolarisation"] == [255]


def test_supplied_label_ids_are_preserved_in_event_output():
    from segmentation_runner import analyse_segmentations

    frame_1, frame_2 = _two_to_one_labels()
    frame_1 = np.where(frame_1 == 1, 10, np.where(frame_1 == 2, 20, 0))
    frame_2 = np.where(frame_2 == 1, 7, 0)

    result = analyse_segmentations(frame_1, frame_2, duplicate_distance=8)
    reverse = analyse_segmentations(frame_2, frame_1, duplicate_distance=8)

    assert {tuple(sorted(pair)) for pair in result["label_pairs"]["fusion"]} == {
        (10, 20)
    }
    assert result["input_label_maps"] == {
        "frame_1": [
            {"internal_label": 1, "input_label": 10},
            {"internal_label": 2, "input_label": 20},
        ],
        "frame_2": [{"internal_label": 1, "input_label": 7}],
    }
    assert {
        tuple(sorted(pair)) for pair in reverse["label_pairs"]["fission"]
    } == {(10, 20)}


def test_duplicate_event_pairs_also_use_supplied_label_ids():
    from segmentation_runner import analyse_segmentations

    frame_1 = np.zeros((5, 80, 50), dtype=np.uint16)
    frame_2 = np.zeros_like(frame_1)
    for row, first_id, second_id, merged_id in [
        (12, 10, 20, 7),
        (52, 30, 40, 9),
    ]:
        frame_1[:, row : row + 7, 5:12] = first_id
        frame_1[:, row : row + 7, 19:26] = second_id
        frame_2[:, row : row + 7, 5:26] = merged_id

    result = analyse_segmentations(
        frame_1,
        frame_2,
        duplicate_distance=1000,
        min_volume=1,
    )

    accepted = {tuple(sorted(pair)) for pair in result["label_pairs"]["fusion"]}
    duplicates = {
        tuple(sorted(pair)) for pair in result["duplicate_label_pairs"]["fusion"]
    }
    assert accepted | duplicates == {(10, 20), (30, 40)}
    assert result["duplicates"]["fusion"] == 1


def test_empty_frames_have_no_events():
    from segmentation_runner import analyse_segmentations

    empty = np.zeros((3, 24, 24), dtype=np.uint8)
    result = analyse_segmentations(empty, empty, min_volume=1)

    assert result["input_object_counts"] == {"frame_1": 0, "frame_2": 0}
    assert result["counts"] == {"fusion": 0, "fission": 0, "depolarisation": 0}
    assert not np.any(result["event_image"])


def test_near_border_disappearance_is_counted_without_coordinate_underflow():
    from segmentation_runner import analyse_segmentations

    frame_1 = np.zeros((5, 32, 32), dtype=np.uint8)
    frame_1[:, :5, :5] = 1
    frame_2 = np.zeros_like(frame_1)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = analyse_segmentations(frame_1, frame_2, min_volume=1)

    assert result["counts"] == {"fusion": 0, "fission": 0, "depolarisation": 1}
    assert result["locations"]["depolarisation"] == [[2.0, 2.0, 2.0]]


@pytest.mark.parametrize(
    "frame_1, frame_2, error, message",
    [
        (
            np.zeros((4, 4), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            ValueError,
            "3-D array",
        ),
        (
            np.zeros((2, 4, 4), dtype=np.uint8),
            np.zeros((3, 4, 4), dtype=np.uint8),
            ValueError,
            r"same \(z, y, x\) grid",
        ),
        (
            -np.ones((2, 4, 4), dtype=np.int8),
            np.zeros((2, 4, 4), dtype=np.uint8),
            ValueError,
            "non-negative integers",
        ),
    ],
)
def test_invalid_segmentations_fail_with_actionable_errors(
    frame_1, frame_2, error, message
):
    from segmentation_runner import analyse_segmentations

    with pytest.raises(error, match=message):
        analyse_segmentations(frame_1, frame_2, min_volume=1)


@pytest.mark.parametrize(
    "parameter, value",
    [
        ("duplicate_distance", np.nan),
        ("duplicate_distance", np.inf),
        ("distance_threshold", np.nan),
        ("distance_threshold", np.inf),
    ],
)
def test_distance_parameters_must_be_finite(parameter, value):
    from segmentation_runner import analyse_segmentations

    empty = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="finite"):
        analyse_segmentations(empty, empty, min_volume=1, **{parameter: value})


def test_input_kind_must_be_explicitly_supported():
    from segmentation_runner import analyse_segmentations

    empty = np.zeros((2, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="input_kind"):
        analyse_segmentations(empty, empty, input_kind="guess", min_volume=1)


def test_cli_writes_machine_readable_results(tmp_path):
    frame_1, frame_2 = _two_to_one_labels()
    frame_1_path = tmp_path / "frame-1.tif"
    frame_2_path = tmp_path / "frame-2.tif"
    output_path = tmp_path / "events.json"
    event_image_path = tmp_path / "events.tif"
    tifffile.imwrite(frame_1_path, frame_1)
    tifffile.imwrite(frame_2_path, frame_2)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "segmentation_runner.py"),
            str(frame_1_path),
            str(frame_2_path),
            "--output",
            str(output_path),
            "--event-image",
            str(event_image_path),
            "--duplicate-distance",
            "8",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["implementation"] == (
        "corrected-original-mel-segmentation-stage/1.0"
    )
    assert result["counts"]["fusion"] == 1
    assert result["coordinate_order"] == ["z", "y", "x"]
    assert result["input_shape"] == [5, 40, 40]
    assert tifffile.imread(event_image_path).shape == (5, 40, 40, 3)


def test_cli_refuses_to_overwrite_an_input(tmp_path):
    frame_1, frame_2 = _two_to_one_labels()
    frame_1_path = tmp_path / "frame-1.tif"
    frame_2_path = tmp_path / "frame-2.tif"
    tifffile.imwrite(frame_1_path, frame_1)
    tifffile.imwrite(frame_2_path, frame_2)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "segmentation_runner.py"),
            str(frame_1_path),
            str(frame_2_path),
            "--output",
            str(frame_1_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    np.testing.assert_array_equal(tifffile.imread(frame_1_path), frame_1)


def test_cli_refuses_a_hard_link_alias_of_an_input(tmp_path):
    frame_1, frame_2 = _two_to_one_labels()
    frame_1_path = tmp_path / "frame-1.tif"
    frame_2_path = tmp_path / "frame-2.tif"
    output_alias = tmp_path / "events.json"
    tifffile.imwrite(frame_1_path, frame_1)
    tifffile.imwrite(frame_2_path, frame_2)
    try:
        os.link(frame_1_path, output_alias)
    except OSError as error:
        pytest.skip(f"hard links are unavailable on this filesystem: {error}")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "segmentation_runner.py"),
            str(frame_1_path),
            str(frame_2_path),
            "--output",
            str(output_alias),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    np.testing.assert_array_equal(tifffile.imread(frame_1_path), frame_1)


def test_modern_rgb_rescale_and_panel_output(tmp_path):
    import ImageAnalysis

    rgb_stack = np.zeros((2, 4, 5, 3), dtype=np.float32)
    assert ImageAnalysis.rescaleStackXY_RGB(rgb_stack, 2).shape == (2, 8, 10, 3)

    frame_1 = np.zeros((2, 8, 8), dtype=np.float32)
    frame_2 = np.ones_like(frame_1)
    events = np.zeros((2, 8, 8, 3), dtype=np.float32)
    output_path = tmp_path / "panel.tif"
    panel = ImageAnalysis.saveCroppedImagePanel(
        frame_1,
        frame_2,
        events,
        0,
        8,
        0,
        8,
        outputPath=output_path,
    )

    assert panel.shape == (8, 28, 3)
    assert output_path.is_file()
