import numpy as np
from scipy import ndimage
from time import time
import pandas as pd
from skimage import measure
from skimage import io

import matplotlib.pyplot as plt

import trimesh

import TiffMetadata

import math

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from scipy.ndimage import zoom
from scipy import signal
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.stats import moment
from scipy.spatial import ConvexHull


def labelStack(binarisedImageStack, minVolume=40):
    (ar, countsArr) = np.unique(binarisedImageStack, return_counts=True)
    print(countsArr)
    if len(countsArr) > 2:
        print('You must provide a binarised stack')
        return

    # get all labels (similar to what I tried with watershed)
    labeled, numpatches = ndimage.label(binarisedImageStack)

    # since labels start from 1 use this range
    sizes = ndimage.sum(binarisedImageStack, labeled, range(1, numpatches + 1))

    # to ensure "black background" is excluded add 1, and labels only start from 1
    filteredIndexes = np.where(sizes >= minVolume)[0] + 1

    filteredBinaryIndexes = np.zeros(numpatches + 1, np.uint8)
    filteredBinaryIndexes[filteredIndexes] = 1
    filteredBinary = filteredBinaryIndexes[labeled]

    labeledStack, numLabels = ndimage.label(filteredBinary)
    print("Initial num labels: {}, num lables after filter: {}".format(numpatches, numLabels))

    return filteredBinary, labeledStack, numLabels


def stack3DTo4D(labeledStack, numLabels):
    print("\nstack3DTo4D")

    if(numLabels > 150):
        print("MEMORY WARNING: More than 150 labels, kernel algorithm could possibly run out of VRAM.")

    sliceArray = []
    label_startTime = time()

    sliceCount = 0
    print("Slice done: ", end='')
    for s in labeledStack:
        frame_label_matrix = np.zeros((numLabels + 1, s.shape[0], s.shape[1], 1), dtype=np.float32)

        for y in range(0, s.shape[0]):
            for x in range(0, s.shape[1]):
                if (s[y, x] != 0):  # not black
                    try:
                        frame_label_matrix[s[y, x], y, x, 0] = 1
                    except:
                        print("Not found: {} at x and y ({},{})".format(s[y, x], x, y))


        # split labels, and process segment per segment

        sliceArray.append(frame_label_matrix)

        print(" {} ".format(sliceCount), end='')
        sliceCount += 1

    output = np.swapaxes(np.array(sliceArray)[:, :, :, :, 0], 0, 1)
    print("\nLabel Time: ", time() - label_startTime)

    return output

def fullStackToMesh(stackLabels, scaleVector=None):
    if scaleVector == None:
        scaleVector = [1,1,1,1]

    properties = measure.regionprops(stackLabels)

    listOfCoords = properties[0].coords
    for index in range(1, len(properties)):
        listOfCoords = np.vstack((listOfCoords, properties[index].coords))

    return trimesh.voxel.base.ops.points_to_marching_cubes(listOfCoords).apply_transform(np.eye(4,4)*scaleVector)

def getMetadata(filename):
    return TiffMetadata.metadata(filename)

def exportMeshAsPng(mesh, path, rotation):
    if not path.lower().endswith('.png'):
        path = path + '.png'
    meshScene = mesh.scene()
    rotationMatrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    transformationMatrix = np.eye(4, 4)
    transformationMatrix[0:3, 0:3] = rotationMatrix
    meshScene.apply_transform(transformationMatrix)
    pngBytes = bytes(mesh.scene().save_image((500, 500)))
    file = open(path, "wb")
    file.write(pngBytes)
    file.close()


def exportMesh(mesh, path, type):
    mesh.export(path, type)
