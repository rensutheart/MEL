# MEL — Mitochondrial Event Localiser

MEL localises and counts candidate mitochondrial fusion, fission, and
depolarisation events between two 3-D frames. This maintained route runs the
corrected original MEL event stage on segmentations supplied by the user.

> **Recommended maintained route:** start from two already-segmented 3-D TIFF
> stacks and run MEL from the segmentation stage. This skips the old CZI
> deconvolution, thresholding, and review GUI, which is the intended use when
> segmentation is supplied by another tool.

This repository is research software. The modern smoke tests use controlled
synthetic segmentations; they are not biological or clinical validation.

## Citation and use notice

If you use MEL, please cite:

> Theart RP, Kriel J, du Toit A, Loos B, Niesler TR (2020).
> *Mitochondrial event localiser (MEL) to quantitatively describe fission,
> fusion and depolarisation in the three-dimensional space.* PLOS ONE 15(12):
> e0229634. <https://doi.org/10.1371/journal.pone.0229634>

The existing project notice remains in force:

> THIS CODE MAY NOT BE USED FOR COMMERCIAL PURPOSES WITHOUT PERMISSION FROM
> THE AUTHOR.

The repository does not currently contain a conventional open-source licence
file. Contact the author before commercial use or redistribution.

## Quick start: supplied segmentations

### Requirements

- 64-bit Python 3.12 (tested with Python 3.12.10 on Windows x86-64)
- The headless code is intended to be portable to Linux and macOS, but those
  platforms have not yet been independently tested in this repository
- No GPU, TensorFlow, Flowdec, napari, or display server is required for this
  route

`requirements.txt` pins the direct runtime dependencies used by the current
smoke tests. `requirements-lock.txt` records the complete resolved runtime
environment from the tested Windows installation.

### Install on Windows PowerShell

```powershell
git clone https://github.com/rensutheart/MEL.git
cd MEL
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

For the closest reproduction of the tested Windows environment, install
`requirements-lock.txt` instead of `requirements.txt`.

### Install on Linux or macOS

```bash
git clone https://github.com/rensutheart/MEL.git
cd MEL
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

### Input format

Provide two TIFF files with the same 3-D shape in **`(z, y, x)`** order:

- background must be `0`;
- a Boolean, binary `0/1`, or conventional binary `0/255` stack is
  connected-component labelled automatically; or
- positive integer values may provide object labels directly;
- labels smaller than `--min-volume` are removed (default: 40 voxels).

Automatic mode interprets `{0, 255}` as a conventional binary mask. If `255`
is instead a genuine object ID, pass `--input-kind labels`. Use
`--input-kind binary` to explicitly connected-component label any nonzero mask.

The two files must already be registered to the same voxel grid. Reported
locations are also in `(z, y, x)` order and distances are index-space voxels;
the runner does not silently convert them to physical units.

### Run

Windows:

```powershell
.venv\Scripts\python segmentation_runner.py frame_1_labels.tif frame_2_labels.tif `
  --output events.json `
  --event-image events.tif
```

Linux/macOS:

```bash
.venv/bin/python segmentation_runner.py frame_1_labels.tif frame_2_labels.tif \
  --output events.json \
  --event-image events.tif
```

`events.json` contains:

- fusion, fission, and depolarisation counts;
- event locations in `(z, y, x)`;
- the participating input label IDs, plus the internal-to-input label map;
- candidates removed by duplicate filtering; and
- the exact parameter values and corrected-implementation identifier used.

`events.tif` is optional. It is an RGB stack showing MEL's event markers over
frame 1.

Useful options:

```text
--duplicate-distance 10    suppress event sites closer than this many voxels
--distance-threshold 20    maximum participant separation in voxels
--overlap-threshold 0.5    minimum relative overlap used by legacy MEL
--min-volume 40            remove smaller labelled objects
--input-kind auto          choose auto, binary, or labels for both inputs
```

For nearby but genuinely distinct events, lower `--duplicate-distance`. Record
the chosen value with your result; changing it can change the event count.

### Direct Python use

```python
import tifffile
from segmentation_runner import analyse_segmentations

frame_1 = tifffile.imread("frame_1_labels.tif")
frame_2 = tifffile.imread("frame_2_labels.tif")

result = analyse_segmentations(
    frame_1,
    frame_2,
    duplicate_distance=10,
    distance_threshold=20,
    overlap_threshold=0.5,
    min_volume=40,
    input_kind="auto",
)

print(result["counts"])
print(result["locations"])
event_image = result["event_image"]
```

## Verify the installation

Install the test dependency and run the smoke tests:

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

On Linux or macOS, replace `.venv\Scripts\python` with `.venv/bin/python`.

The tests cover a hand-calculated overlap, two-to-one fusion, reverse
one-to-two fission, disappearance/depolarisation at an image edge, empty
frames, Boolean and 0/255 masks, preservation of supplied label IDs, invalid
input, output-path safety, command-line JSON/TIFF output, and the coordinate
bugs corrected for current NumPy/SciPy/scikit-image.

## What was modernised

The runner retains the original MEL pairwise association, classification,
location, and duplicate-filtering stages, with the explicit defect corrections
below. It is identified in every result as
`corrected-original-mel-segmentation-stage/1.0`. It is not a claim of
bug-for-bug or byte-for-byte parity with the 2020 executable; affected edge
cases can intentionally produce different results.

The maintenance changes needed to run the corrected stage on current Python
are:

- replaced the TensorFlow-only overlap loop with the mathematically equivalent
  NumPy CPU calculation;
- isolated the maintained runner from legacy CZI, Flowdec, TIFF-metadata, mesh,
  and GUI dependencies, using lazy imports where shared modules need them;
- corrected the structure-between check, which previously counted a
  participant endpoint as an intervening structure and used overflowing
  8-bit coordinates;
- corrected unsigned event-marker coordinates that underflowed near image
  borders;
- made variable-length label associations explicit for NumPy 2;
- updated removed SciPy/scikit-image/NumPy/Pandas APIs; and
- added a real command-line entry point instead of requiring author-local paths
  inside `MEL_main.py`.

These are compatibility and defect corrections supported by small synthetic
regressions. They are not a reproduction of the paper's reported results or a
claim that the method's scientific performance has been newly validated.

### Known limitation

The original event algorithm materialises several full 3-D arrays per object.
Memory use therefore grows with both image size and object count. The maintained
route is suitable for the small/cropped two-frame comparisons covered here, but
it has not been redesigned or benchmarked as a large-volume scalable engine.

## Legacy raw-CZI workflow

`MEL_main.py` retains the archival shape of the 2020 end-to-end workflow: CZI
loading, Flowdec deconvolution, preprocessing/thresholding, MEL, and an
interactive Pyglet/Glooey review window. It also contains author-local,
dataset-specific settings near the top of the file (`filePath`, `writePath`,
`positionNum`, `startFileIndex`, and `startFrame`).

That lane depends on an end-of-life Python 3.8 / TensorFlow 2.3-era stack,
additional unpinned GUI/CZI dependencies, and the bundled `modified-flowdec`
tree. It is retained as historical source only: it is **not** installed by the
modern requirements, is not covered by the maintained tests, and is not an
exactly reproducible route from the instructions in this README. Moreover, it
imports the corrected shared `MEL.py`, so it must not be treated as literal
bug-for-bug 2020 behavior.

For inspection of the unmodified public snapshot, use Git commit
`2dc2f1c3a0f0cbb01292f7d732d737f6adb6da65`. A separate historically pinned
environment and dataset-specific documentation would still be required to run
that raw-CZI/GUI lane reproducibly.

## Troubleshooting

- **`ModuleNotFoundError` after installation:** run the command with the Python
  executable inside `.venv`, not another system Python.
- **“must be a 3-D array”:** export one `(z, y, x)` volume per frame rather than
  a time series or channel-first array.
- **No objects/events:** inspect the label values and try a smaller
  `--min-volume`; do not tune thresholds without recording the change.
- **Two nearby events became one:** lower `--duplicate-distance` cautiously.
- **Need CZI deconvolution or the old review window:** that is the separate
  legacy lane above, not a missing requirement in the segmentation route.
