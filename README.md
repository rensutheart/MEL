# MEL
Mitochondrial Event Localiser

THIS CODE MAY NOT BE USED FOR COMMERCIAL PURPOSES WITHOUT PERMISSION FROM THE AUTHOR.

If this method or tool is used, please cite the following [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0229634)

Theart RP, Kriel J, du Toit A, Loos B, Niesler TR (2020) Mitochondrial event localiser (MEL) to quantitativelydescribe fission, fusion and depolarisation in the three-dimensional space. PLoS ONE 15(12): e0229634. https://doi.org/10.1371/journal.pone.0229634

## Install the following libraries (indented lines should be installed automatically)
```
pip install tensorflow
	pip install numpy
pip install scikit-image
	pip install scipy
	pip install pillow
	pip install tifffile
	pip install matplotlib
pip install pandas
pip install trimesh
pip install czifile
pip install lxml
pip install ExifRead
pip install opencv-python
pip install pyglet
pip install glooey
```
To install TensorFlow, please follow this guide: https://www.tensorflow.org/install/gpu

Install flowdec manually since the original flowdec library causes issues: https://github.com/hammerlab/flowdec

## Usage
At the top of MEL_main.py the following variables are used to process batches of samples

* `filePath`: the path to where the input .czi samples are stored
* `writePath`: the path to where the output results are stored
* `positionNum`: if multiple positions exist in the same .czi file, then this select the desired position (default: 0)
* `startFileIndex`: if multiple files exist in the filePath, indicates which file must be processed (default: 0)
* `startFrame`: the index of the frame in the time-lapse sequence to start at

## Running
Run the code by typing `python MEL_main.py` in the command prompt or terminal.
