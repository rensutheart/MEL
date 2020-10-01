import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from scipy.ndimage.measurements import center_of_mass
from scipy import ndimage
from scipy import signal
from skimage.filters import unsharp_mask
from skimage.filters import threshold_otsu, threshold_li, threshold_mean, gaussian
from skimage.transform import rescale
from skimage import data, io
from skimage import feature
from skimage.exposure import rescale_intensity
from skimage.morphology import white_tophat
from skimage.morphology import disk, square, ball

from scipy.spatial.distance import cdist

import tensorflow as tf

from time import time

import Morphology
import ImageAnalysis

import trimesh

import os
from os import listdir
from os.path import isfile, join

black = np.zeros(3)

x = []

# 0 unasissgned, 1 Nothing, 2 Fuse, 3 Fragment, 4 depolarize, 5 Frag-Fuse
class transType:
    UNASSIGNED = 0
    NOTHING = 1
    FUSE = 2
    FRAGMENT = 3
    DEPOLARIZE = 4
    UNCERTAIN = 5



def backAndForthLabelMatching(listAssociatedLabelsF1, listAssociatedLabelsF2):
    print("\nStart backAndForthLabelMatching")
    backAndForthStartTime = time()

    withinAssociatedLabelsF1 = []
    withinAssociatedLabelsF2 = []

    # Using the associated labels in F2, which which of those labels in F1 that one also refers to.
    # begin at 1 since 0 is background
    F1Label_index = 1
    for l_F1 in listAssociatedLabelsF1:
        tempList = set()

        if (l_F1.shape[0] > 0):
            for l_F2 in l_F1:  # for each label
                for F1_Label in listAssociatedLabelsF2[l_F2]:
                    if (F1Label_index != F1_Label and F1_Label != 0): # 0 is the background
                        tempList.add(F1_Label)

        withinAssociatedLabelsF1.append(np.array(list(tempList)))
        F1Label_index += 1

    # Using the associated labels in F1, which which of those labels in F2 that one also refers to.
    F2Label_index = 1
    for l_F2 in listAssociatedLabelsF2:
        tempList = set()

        if (l_F2.shape[0] > 0):
            for l_F1 in l_F2:  # for each label
                for F2_Label in listAssociatedLabelsF1[l_F1]:
                    if (F2Label_index != F2_Label and F2_Label != 0):
                        tempList.add(F2_Label)

        withinAssociatedLabelsF2.append(np.array(list(tempList)))
        F2Label_index += 1

    print("backAndForthLabelMatching time: ", time() - backAndForthStartTime)

    return np.array(withinAssociatedLabelsF1), np.array(withinAssociatedLabelsF2)


def labelToCanny(labeledStack):
    print("\nStart labelToCanny")
    cannyStartTime = time()
    cannyLabledStack = []
    zeros = np.zeros_like(labeledStack[0][0], dtype=np.uint8)

    print("Label done: ", end='')
    labelCount = 0
    for label in labeledStack:
        canny3D = []
        for s in label:
            if (np.sum(s) != 0):
                edges = feature.canny(s)

                canny3D.append(edges)
            else:
                canny3D.append(zeros)
        canny3D = np.array(canny3D, dtype=np.uint8)
        cannyLabledStack.append(canny3D)

        print(" {} ".format(labelCount), end='')
        labelCount += 1

    print("\nCanny time: ", time() - cannyStartTime)

    return np.array(cannyLabledStack, dtype=np.uint8)


def getHalfWayPointBetweenLabels(labeledStackF1, fromIndex, toIndex, plot=False):
    pointsFrom = np.array(np.where(labeledStackF1[fromIndex] > 0)).T
    pointsTo = np.array(np.where(labeledStackF1[toIndex] > 0)).T
    out = cdist(pointsFrom, pointsTo)

    t = np.min(out, axis=1)  # get the minimum distance for each from pixel and store in column vector
    p = np.argmin(t)  # get the from pixel with the minimum distance to the to pixel
    q = np.argmin(out[p])  # get the to pixel which is the minimum for that specific from pixel (should usually be 0 since it is sorted)

    # get coordinate points
    fromPoint = pointsFrom[p]
    toPoint = pointsTo[q]
    halfwayPoint = (fromPoint + toPoint) / 2
    distance = np.linalg.norm(fromPoint - toPoint)

    if (plot):
        fig = plt.figure()  # figsize=(20,15))
        plt.imshow(labeledStackF1[fromIndex][fromPoint[0]] + labeledStackF1[toIndex][toPoint[0]] * 0.5)
        plt.plot((fromPoint[2], toPoint[2]), (fromPoint[1], toPoint[1]))
        plt.plot(halfwayPoint[2], halfwayPoint[1], 'ro')
        plt.show()

        plt.close('all')

    return halfwayPoint, distance, (fromPoint, toPoint)


def getAllHalfWayPoints(labeledStackF1, withinAssociatedLabelsF1):
    print("\nStart getAllHalfWayPoints")
    halfwayTime = time()
    withinAssociatedLabelsF1_HWP = []
    withinAssociatedLabels_vector = []

    labelDistanceF1 = []
    F1_label_index = 0
    print("Processing label: ", end='')
    for associatedLabelF1 in withinAssociatedLabelsF1:
        print(" {}".format(F1_label_index), end='')
        tempList = []
        tempDistList = []
        tempListVector = []
        if (associatedLabelF1.shape[0] > 0):
            for label_F1_2 in associatedLabelF1:
                HWP, dist, vector = getHalfWayPointBetweenLabels(labeledStackF1, F1_label_index, label_F1_2, False)
                tempList.append(HWP)
                tempDistList.append(dist)
                tempListVector.append(vector)

        withinAssociatedLabelsF1_HWP.append(tempList)
        labelDistanceF1.append(tempDistList)
        withinAssociatedLabels_vector.append(tempListVector)

        F1_label_index += 1
    print("\nHalfway Time: ", time() - halfwayTime)

    return np.array(withinAssociatedLabelsF1_HWP), np.array(labelDistanceF1), np.array(withinAssociatedLabels_vector)

def generateKernel(kernelSize=9, size=10, s=0.5, showKernel=False):
    # zDivFactor=2
    kernelZ = 1  # kernelSize//zDivFactor

    x, y = np.mgrid[-1.0:1.0:2 / size, -1.0:1.0:2 / size]

    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([0.0, 0.0])

    sigma = np.array([s, s])
    covariance = np.diag(sigma ** 2)

    out = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    zOut = signal.gaussian(kernelZ, kernelZ * 0.2)
    zOut = zOut / zOut[kernelZ // 2]
    if showKernel:
        plt.figure()
        plt.plot(zOut)

    # Reshape back to a (30, 30) grid.
    out = out.reshape(x.shape)
    kernel = rescale(out, kernelSize / size, multichannel=False)
    normalizationNum = kernel[kernelSize // 2, kernelSize // 2]
    kernel = kernel / normalizationNum

    fullKernel = []

    for i in range(0, kernelZ):
        iKernel = kernel * zOut[i]
        fullKernel.append(iKernel)
        if showKernel:
            plt.imshow(iKernel, vmin=0, vmax=1)
            plt.show()

    return np.array(fullKernel)

def generateRGBkernels(kernelSize = 10):

    splat_kernel = generateKernel(kernelSize, size=kernelSize, s=0.4)
    kernel_zeros = np.zeros_like(splat_kernel)
    red_kernel = np.swapaxes(np.stack((splat_kernel, kernel_zeros, kernel_zeros))[:, 0, :, :], 0, 2)
    green_kernel = np.swapaxes(np.stack((kernel_zeros, splat_kernel, kernel_zeros))[:, 0, :, :], 0, 2)
    blue_kernel = np.swapaxes(np.stack((kernel_zeros, kernel_zeros, splat_kernel))[:, 0, :, :], 0, 2)

    # 3 layers (*0.5 since each one repeats twice, and overlays)
    red_kernel_3D = np.stack((red_kernel * 0.2, red_kernel * 0.5, red_kernel * 0.2))
    green_kernel_3D = np.stack((green_kernel * 0.2, green_kernel * 0.5, green_kernel * 0.2))
    blue_kernel_3D = np.stack((blue_kernel * 0.2, blue_kernel * 0.5, blue_kernel * 0.2))

    return (red_kernel_3D, green_kernel_3D, blue_kernel_3D)


def gaussianFilter(labeledStack, sigma3D=0.25, sigma2D=1.0):
    print("\ngaussianFilter")
    startTime = time()

    labeledStackKernel = np.zeros(labeledStack.shape, dtype=np.float32)

    print("Label done: ", end='')
    for labelIndex in range(0, labeledStack.shape[0]):
        '''
        pass3D = gaussian(labeledStack[labelIndex], sigma3D)  # do a "light" gaussian blur in x, y, z demension
#        pass3D = labeledStack[labelIndex]

        # do a 2D gaussian filter
        for sl in range(0,pass3D.shape[0]):
            pass3D[sl] = gaussian(pass3D[sl], sigma2D)


        labeledStackKernel[labelIndex] = np.copy(pass3D)
        '''

        labeledStackKernel[labelIndex] = gaussian(labeledStack[labelIndex], sigma3D)  # do a "light" gaussian blur in x, y, z demension 

        print(" {} ".format(labelIndex), end='')

    print("\ngaussianFilter Time: ", time() - startTime)

    return labeledStackKernel


def compareOverlapV2(labeledStackF1, labeledStackF2):
    print("\ncompareOverlapV2")
    startTime = time()

    volume_overlap_F1_to_F2 = np.zeros((labeledStackF1.shape[0], labeledStackF2.shape[0]))

    try:

        print("Slice done: ", end='')
        with tf.device('/gpu:0'):
            for sl in range(0, labeledStackF1.shape[1]):
                f1TensorSlice = tf.constant(labeledStackF1[:, sl, :, :])
                f2TensorSlice = tf.constant(labeledStackF2[:, sl, :, :])

                for f1LabIndex in range(0, labeledStackF1.shape[0]):
                    volume = tf.reduce_sum(tf.reduce_sum(tf.multiply(f1TensorSlice[f1LabIndex], f2TensorSlice), axis=2),
                                        axis=1).numpy()
                    volume_overlap_F1_to_F2[f1LabIndex] = np.add(volume, volume_overlap_F1_to_F2[
                        f1LabIndex])  # don't erase with each slice iteration

                print(" {} ".format(sl), end='')

        print("\ncompareOverlapV2 Time: ", time() - startTime)\
    
    except:
        print("Could not run compare. Possibly ran out of memory")
        return False

    return volume_overlap_F1_to_F2  # , np.swapaxes(volume_overlap_F1_to_F2,0,1)

def checkStructuresInBetween(filteredBinary, vector, numPoints=50):
    start = np.array(vector[0])
    end = np.array(vector[1])
    difference = end - start
    step = difference / numPoints;

    for i in range(0, numPoints):
        coord = np.around(start + step*i).astype(np.int8)
        if filteredBinary[coord[0], coord[1], coord[2]] == 1:
            return True

    return False # no structures found between


def linkStatusPerLabel(filteredBinaryF1, filteredBinaryF2, overlapF1_to_F2, overlapF2_to_F1, listAssociatedLabelsF1,
                       listAssociatedLabelsF2, withinAssociatedLabelsF1, withinAssociatedLabelsF2,
                       labelDistanceF1, vector_F1,  labelDistanceF2, vector_F2, distanceThreshold=20,
                       bigPercThresh=0.5, smallPercThresh=0.5):
    print("\nlinkStatusPerLabel")
    startTime = time()
    # chosenStatusF1 is matchined to labels in listAssociatedLabelsF1
    # withinChosenStatusF1 is matched to labels in withinAssociatedLabelsF1
    #    chosenStatusF1 = []  # -1 error state, 0 unasissgned, 1 Nothing, 2 Fuse, 3 Fragment, 4 depolarize, 5 Unrelated (small and small percentage overlap)
    withinChosenStatusF1_Fuse = []  # this cannot contain fragmentation locations... so just fusion
    withinChosenStatusF2_Fragment = []  # this cannot contain fusion locations... so just fragmentation

    # CALCULATE RELATIVE PERCENTAGES of overlapping labels
    totalPerc_F1 = []
    for F1_label in range(0, listAssociatedLabelsF1.shape[0]):
        labelVols_F1 = []
        for F2_label in listAssociatedLabelsF1[F1_label]:
            labelVols_F1.append(overlapF1_to_F2[F1_label][F2_label])


        totalPerc = np.array(labelVols_F1) / np.sum(labelVols_F1)
        totalPerc_F1.append(totalPerc)

    totalPerc_F2 = []
    for F2_label in range(0, listAssociatedLabelsF2.shape[0]):
        labelVols_F2 = []
        for F1_label in listAssociatedLabelsF2[F2_label]:
            labelVols_F2.append(overlapF2_to_F1[F2_label][F1_label])

        totalPerc = np.array(labelVols_F2) / np.sum(labelVols_F2)
        totalPerc_F2.append(totalPerc)

    # Determine what will happen for labels in F1
    withinChosenStatusF1_Fuse.append([])  # first zero background label
    for F1_i in range(1, withinAssociatedLabelsF1.shape[0]):  # start from 1, since background is label 0
        currentF1List = []
        if (listAssociatedLabelsF1[F1_i].shape[0] == 0):  # depolarise
            currentF1List.append(4)
        elif (withinAssociatedLabelsF1[F1_i].shape[0] != 0):  # first label (for background)
            for otherLabelF1_index in range(0, withinAssociatedLabelsF1[F1_i].shape[0]):  # associated label
                currentF1List.append(0)  # unassigned label... at the end there should be no zeros
                if (F1_i != withinAssociatedLabelsF1[F1_i][otherLabelF1_index]): # since can't fuse to self
                    otherLabelF1 = withinAssociatedLabelsF1[F1_i][otherLabelF1_index]
                    if (labelDistanceF1[F1_i][otherLabelF1_index] < distanceThreshold):  # some labels are not really associated due to large distance
                        # check if there is a structure inbetween... to remove any remaining accidental mataches. Crudely look at the HWP
                        if (checkStructuresInBetween(filteredBinaryF1, vector_F1[F1_i][otherLabelF1_index])):
                            print("{} to {} in F1 was detected as a false match... Structure in between".format(F1_i, otherLabelF1))
                            pass
                        else:
                            matchingLabelsInF2 = np.intersect1d(listAssociatedLabelsF1[F1_i],
                                                                listAssociatedLabelsF1[otherLabelF1])
                            if (matchingLabelsInF2.shape[0] == 0):  # unrelated
                                currentF1List[len(currentF1List) - 1] = 5

                            for match in matchingLabelsInF2:
                                match_index_1 = np.where(listAssociatedLabelsF1[F1_i] == match)[0][0]
                                match_index_2 = np.where(listAssociatedLabelsF1[otherLabelF1] == match)[0][0]

                                if (totalPerc_F1[F1_i][match_index_1] > bigPercThresh and
                                        totalPerc_F1[otherLabelF1][match_index_2] > bigPercThresh):  # fuse
                                    currentF1List[len(currentF1List) - 1] = 2
                                else:  # nothing, since either big-small combination (probably coincidental) or small small is nothing...
                                    currentF1List[len(currentF1List) - 1] = 1
                    else:  # unrelated
                        currentF1List[len(currentF1List) - 1] = 5
                else:
                    currentF1List[len(currentF1List) - 1] = 1 # nothing happens

        withinChosenStatusF1_Fuse.append(currentF1List)

        # Determine what will happen for labels in F2
    withinChosenStatusF2_Fragment.append([])  # first zero background label
    for F2_i in range(1, withinAssociatedLabelsF2.shape[0]):  # start from 1, since background is label 0
        currentF2List = []
        if (listAssociatedLabelsF2[F2_i].shape[0] == 0):  # deploraize
            currentF2List.append(4)
        elif (withinAssociatedLabelsF2[F2_i].shape[0] != 0):  # first label
            for otherLabelF2_index in range(0, withinAssociatedLabelsF2[F2_i].shape[0]):  # associated label
                currentF2List.append(0)  # unassigned label... at the end there should be no zeors
                if (F2_i != withinAssociatedLabelsF2[F2_i][otherLabelF2_index]):  # since can't fragment to self
                    otherLabelF2 = withinAssociatedLabelsF2[F2_i][otherLabelF2_index]
                    if (labelDistanceF2[F2_i][otherLabelF2_index] < distanceThreshold):  # some labels are not really associated due to large distance
                        # check if there is a structure inbetween... to remove any remaining accidental mataches. Crudely look at the HWP
                        if (checkStructuresInBetween(filteredBinaryF2, vector_F2[F2_i][otherLabelF2_index])):
                            print("{} to {} in F2 was detected as a false match... Structure in between".format(F2_i, otherLabelF2))
                            pass
                        else:
                            matchingLabelsInF1 = np.intersect1d(listAssociatedLabelsF2[F2_i],
                                                                listAssociatedLabelsF2[otherLabelF2])
                            if (matchingLabelsInF1.shape[0] == 0):  # unrelated
                                currentF2List[len(currentF2List) - 1] = 5

                            for match in matchingLabelsInF1:
                                match_index_1 = np.where(listAssociatedLabelsF2[F2_i] == match)[0][0]
                                match_index_2 = np.where(listAssociatedLabelsF2[otherLabelF2] == match)[0][0]
                                if (totalPerc_F2[F2_i][match_index_1] > bigPercThresh and
                                        totalPerc_F2[otherLabelF2][match_index_2] > bigPercThresh):  # fragment
                                    currentF2List[len(currentF2List) - 1] = 3
                                else:  # nothing, since either big-small combination (probably coincidental) or small samll is nothing...
                                    currentF2List[len(currentF2List) - 1] = 1
                    else:  # unrelated
                        currentF2List[len(currentF2List) - 1] = 5
                else:
                    currentF2List[len(currentF2List) - 1] = 1 # nothing happens

        withinChosenStatusF2_Fragment.append(currentF2List)

    print("linkStatusPerLabel Time: ", time() - startTime)

    return withinChosenStatusF1_Fuse, withinChosenStatusF2_Fragment


def getFragFusePairs(overlapF1_to_F2, overlapF2_to_F1):
    listAssociatedLabelsF1 = []
    # use as range starting from 1, since 0 is background
    for i in range(0, overlapF1_to_F2.shape[0]):
        listAssociatedLabelsF1.append(np.argwhere(overlapF1_to_F2[i] > 0)[:, 0])

    listAssociatedLabelsF1 = np.array(listAssociatedLabelsF1)

    listAssociatedLabelsF2 = []
    for i in range(0, overlapF2_to_F1.shape[0]):
        listAssociatedLabelsF2.append(np.argwhere(overlapF2_to_F1[i] > 0)[:, 0])

    listAssociatedLabelsF2 = np.array(listAssociatedLabelsF2)

    return listAssociatedLabelsF1, listAssociatedLabelsF2


def addKernelToOutputStack(kernel, outputStack, location, Frame1):
    kernelSize = kernel.shape[1]
    kernelDepth = kernel.shape[0]
    location = np.around(location).astype(np.uint16)

    # ensure I stay within the bounds of the image
    startX = location[1] - kernelSize // 2
    endX = location[1] + kernelSize // 2 + kernelSize % 2  # for odd numbers
    startY = location[2] - kernelSize // 2
    endY = location[2] + kernelSize // 2 + kernelSize % 2  # for odd numbers
    startZ = location[0] - kernelDepth // 2
    endZ = location[0] + kernelDepth // 2 + kernelDepth % 2  # for odd numbers

    kernelStartX = max(0, -startX)  # if negative then remove that part
    kernelEndX = min(0, (Frame1.shape[1]) - endX) + kernelSize
    kernelStartY = max(0, -startY)
    kernelEndY = min(0, (Frame1.shape[2]) - endY) + kernelSize
    kernelStartZ = max(0, -startZ)
    kernelEndZ = min(0, (Frame1.shape[0]) - endZ) + kernelDepth

    startX = max(startX, 0)
    endX = min(endX, Frame1.shape[1])
    startY = max(startY, 0)
    endY = min(endY, Frame1.shape[2])
    startZ = max(startZ, 0)
    endZ = min(endZ, Frame1.shape[0])

    try:
        outputStack[startZ:endZ, startX:endX, startY:endY] += kernel[kernelStartZ:kernelEndZ, kernelStartX:kernelEndX,
                                                              kernelStartY:kernelEndY]
    except:
        print("ERROR in addKernelToOutputStack. Sizes:")
        print("{} {}   {} {}   {} {} Kernel: {} {}   {} {}   {} {}".format(startX, endX, startY, endY, startZ, endZ,
                                                                           kernelStartX, kernelEndX, kernelStartY,
                                                                           kernelEndY, kernelStartZ, kernelEndZ))
        print(location)
        return outputStack, False

    return outputStack, True


def generateGradientImage(Frame1, CoM_F1, withinChosenStatusF1_Fuse, withinChosenStatusF2_Fragment,
                          withinAssociatedLabelsF1, withinAssociatedLabelsF2, withinAssociatedLabelsF1_HWP,
                          withinAssociatedLabelsF2_HWP, duplicateDistance=10):
    print("\ngenerateGradientImage")
    startTime = time()

    outputStack = np.zeros((Frame1.shape[0], Frame1.shape[1], Frame1.shape[2], 3), dtype=np.float64)
    originalImage = np.stack((Frame1,) * 3, axis=-1)
    if (np.max(Frame1) > 1):
        originalImage = originalImage / 255

    outputStack += originalImage

    depLocations = []
    fuseLocations = []
    fragLocations = []

    depLabels = []
    fuseLabelPairsF1 = []
    fragLabelPairsF2 = []

    duplicateFuseLocations = []
    duplicateFragLocations = []

    duplicateFuseLabelPairsF1 = []
    duplicateFragLabelPairsF2 = []

    numDepolarization = 0
    numFuse = 0
    numFragment = 0

    duplicateNumFuse = 0
    duplicateNumFragment = 0

    (red_kernel_3D, green_kernel_3D, blue_kernel_3D) = generateRGBkernels(20)

    # depolarize labels
    for label_index in range(1, len(withinChosenStatusF1_Fuse)):
        if (len(withinChosenStatusF1_Fuse[label_index]) == 1):
            if (withinChosenStatusF1_Fuse[label_index][0] == 4 and label_index not in depLabels):  # depolarize
                outputStack, success = addKernelToOutputStack(blue_kernel_3D, outputStack, CoM_F1[label_index], Frame1)

                if (not success):
                    print(
                        "Depolarize Error at {} to {}\n".format(label_index, withinAssociatedLabelsF1[label_index][0]))
                else:
                    numDepolarization += 1
                    depLabels.append(label_index)
                    depLocations.append(CoM_F1[label_index])

    # fuse labels
    for label_index in range(1, withinAssociatedLabelsF1_HWP.shape[0]):  # start from 1 since background is label 0
        for other_label_index in range(0, len(withinAssociatedLabelsF1_HWP[label_index])):
            pairTuple = (label_index, withinAssociatedLabelsF1[label_index][other_label_index])
            revPairTuple = (withinAssociatedLabelsF1[label_index][other_label_index], label_index)
            if (withinChosenStatusF1_Fuse[label_index][other_label_index] == 2 and #fusion
                     pairTuple not in fuseLabelPairsF1 and revPairTuple not in fuseLabelPairsF1): # does not already exist
                HWP = withinAssociatedLabelsF1_HWP[label_index][other_label_index]
                if isUniqueInDistance(HWP, fuseLocations, duplicateDistance):
                    outputStack, success = addKernelToOutputStack(green_kernel_3D, outputStack,
                                                                  withinAssociatedLabelsF1_HWP[label_index][other_label_index], Frame1)
                    if (not success):
                        print("Fuse Error at {} to {}\n".format(pairTuple[0], pairTuple[1]))

                    numFuse += 1
                    fuseLabelPairsF1.append(pairTuple)
                    fuseLocations.append(HWP)
                else:
                    duplicateNumFuse += 1
                    duplicateFuseLabelPairsF1.append(pairTuple)
                    duplicateFuseLocations.append(HWP)

    # fragment labels
    for label_index in range(1, withinAssociatedLabelsF2_HWP.shape[0]):  # start from 1 since background is label 0
        for other_label_index in range(0, len(withinAssociatedLabelsF2_HWP[label_index])):
            pairTuple = (label_index, withinAssociatedLabelsF2[label_index][other_label_index])
            revPairTuple = (withinAssociatedLabelsF2[label_index][other_label_index], label_index)
            if (withinChosenStatusF2_Fragment[label_index][other_label_index] == 3 and #fission
                     pairTuple not in fragLabelPairsF2 and revPairTuple not in fragLabelPairsF2): # does not already exist

                HWP = withinAssociatedLabelsF2_HWP[label_index][other_label_index]
                if isUniqueInDistance(HWP, fragLocations, duplicateDistance):
                    outputStack, success = addKernelToOutputStack(red_kernel_3D, outputStack,
                                                                  withinAssociatedLabelsF2_HWP[label_index][other_label_index], Frame1)
                    if (not success):
                        print("Fragment Error at {} to {}\n".format(pairTuple[0], pairTuple[1]))

                    numFragment += 1
                    fragLabelPairsF2.append(pairTuple)
                    fragLocations.append(HWP)
                else:
                    duplicateNumFragment += 1
                    duplicateFragLabelPairsF2.append(pairTuple)
                    duplicateFragLocations.append(HWP)

    outputStack = np.clip(outputStack, 0, 1)

    print("\ngenerateGradientImage Time: ", time() - startTime)

    return (outputStack,
            [numFuse, numFragment, numDepolarization], # outcomes
            [fuseLocations, fragLocations, depLocations], # locations
            [fuseLabelPairsF1, fragLabelPairsF2, depLabels], # labels
            [duplicateNumFuse, duplicateNumFragment],
            [duplicateFuseLocations, duplicateFragLocations],
            [duplicateFuseLabelPairsF1, duplicateFragLabelPairsF2]
            )

def generateGradientImageFromLocation(originalStack, locations):
    (red_kernel_3D, green_kernel_3D, blue_kernel_3D) = generateRGBkernels(20)
    kernels = [green_kernel_3D, red_kernel_3D, blue_kernel_3D]
    outputStack = np.stack((originalStack,originalStack,originalStack), axis=-1)

    countType = 0
    for locationType in locations:
        for location in locationType:
            outputStack, success = addKernelToOutputStack(kernels[countType], outputStack, location, originalStack)

            if not success:
                print('SOME ERROR OCCURED IN generateGradientImageFromLocation')

        countType += 1

    return outputStack


def outputImagePanel(outputPath, label, folder, Frame1, Frame2, outputStack, filteredBinaryF1, filteredBinaryF2):
    cm = plt.get_cmap('viridis')
    output = np.dstack((cm(Frame1)[:, :, :, 0:3], cm(Frame2)[:, :, :, 0:3], outputStack))
    output2 = np.dstack(
        (np.stack((filteredBinaryF1,) * 3, axis=-1), np.stack((filteredBinaryF2,) * 3, axis=-1), outputStack))
    output3 = np.hstack((output, output2))
    # plt.imshow(output3[0])

    frameOutputPath = "{}/panel/".format(outputPath, folder)
    if not os.path.exists(frameOutputPath):
        os.makedirs(frameOutputPath)

    io.imsave("{}{}.tif".format(frameOutputPath, label), (output3 * 255).astype(np.uint8))


def showMesh(filteredBinary, labels, locations, dupLabels, dupLocations, zScale, xyScale):
    sc = trimesh.Scene()
    fullMesh = Morphology.fullStackToMesh(filteredBinary, [zScale, xyScale, xyScale, 1])
    sc.add_geometry(fullMesh)

    for fuseFragDep in range(0, 3):
        # show main fuse/frag/dep
        for index in range(0, len(labels[fuseFragDep])):

            sphereColor = [fuseFragDep * 1.0, ((fuseFragDep + 1) % 2) * 1.0, 0.0, 0.75]
            if fuseFragDep == 2:  # depoloarisation
                sphereColor = [0.0, 0.0, 1.0, 0.75]

            sphere = trimesh.creation.icosphere(1, 2.5*xyScale, sphereColor)
            sphere.apply_translation(np.array([locations[fuseFragDep][index][0]*zScale, locations[fuseFragDep][index][1] * xyScale, locations[fuseFragDep][index][2] * xyScale]))
            sc.add_geometry(sphere)

        if fuseFragDep != 2:
            # show duplicate fuse/frag/dep
            for index in range(0, len(dupLabels[fuseFragDep])):
                sphereColor = [fuseFragDep * 1.0, ((fuseFragDep + 1) % 2) * 1.0, 1.0, 0.75]
                sphere = trimesh.creation.icosphere(1, 1*xyScale, sphereColor)
                sphere.apply_translation(np.array([dupLocations[fuseFragDep][index][0]*zScale, dupLocations[fuseFragDep][index][1] * xyScale, dupLocations[fuseFragDep][index][2] * xyScale]))
                sc.add_geometry(sphere)


    sc.show()

def printTable(table):
    for row in table:
        for col in row:
            print("{}, ".format(col), end="")
        print()


def isInUnique(HWP, unique):
    for u in unique:
        if np.array_equal(HWP, u):
            return True

    return False

def isUniqueInDistance(point, unique, distanceXY=5):
    for u in unique:
        # print(np.linalg.norm(point - u))
        if np.linalg.norm(point - u) < distanceXY:
            return False

    return True

def findCloseEvents(withinAssociatedLabels, withinAssociatedLabels_HWP, withinChosenStatus_Fuse, distanceXY=5):
    foundHWP = []
    tuples = []
    duplicateHWP = []
    duplicates = []

    for f1Label in range(1, len(withinAssociatedLabels_HWP)):
        for otherIndex in range(0, len(withinAssociatedLabels_HWP[f1Label])):
            if withinChosenStatus_Fuse[f1Label][otherIndex] == 2: #fusion
                HWP = withinAssociatedLabels_HWP[f1Label][otherIndex]
                pairTuple = (f1Label, withinAssociatedLabels[f1Label][otherIndex])
                revPair = (withinAssociatedLabels[f1Label][otherIndex], f1Label)
                if(revPair not in tuples and revPair not in duplicates and
                pairTuple not in tuples and pairTuple not in duplicates):


                    if isUniqueInDistance(HWP, foundHWP, distanceXY):
                        foundHWP.append(HWP)
                        tuples.append(pairTuple)
                    else:
                        #print('Duplicate found at {} for labels {}'.format(HWP, pairTuple))
                        duplicateHWP.append(HWP)
                        duplicates.append(pairTuple)


    return tuples, foundHWP, duplicates, duplicateHWP


def checkEvent(typeEvent, stack4D_F1, stack4D_F2, associatedFromF1_to_F2, associatedFromF2_to_F1,
               associatedLabelInSame, location, stackF1, stackF2, zScale, xyScale, windowSize=100, binarize=True):
    sc = trimesh.Scene()

    thisLocation = location.copy()
    highlighedLabels = np.zeros((stackF1.shape[0], windowSize, windowSize*3, 3))
    stackF1_3 = np.stack((stackF1,) * 3, axis=-1)
    stackF2_3 = np.stack((stackF2,) * 3, axis=-1)

    (red_kernel_3D, green_kernel_3D, blue_kernel_3D) = generateRGBkernels(10)
    #print(associatedLabelInSame)
    #print(location)
    if type(associatedLabelInSame) is int:
        l1 = associatedLabelInSame
        l2 = 0
    else:
        l1 = associatedLabelInSame[0]
        l2 = associatedLabelInSame[1]
    if typeEvent == transType.FUSE:
        label1 = np.int8(stack4D_F1[l1])
        labelMesh1 = Morphology.fullStackToMesh(label1, [zScale, 1, 1, 1])
        labelMesh1.visual.face_colors = [0.0, 0.5, 0.0, 0.5]
        sc.add_geometry(labelMesh1)

        label2 = np.int8(stack4D_F1[l2])
        labelMesh2 = Morphology.fullStackToMesh(label2, [zScale, 1, 1, 1])
        labelMesh2.visual.face_colors = [0.5, 0.0, 0.0, 0.5]
        sc.add_geometry(labelMesh2)

        structureList = np.intersect1d(associatedFromF1_to_F2[l1], associatedFromF1_to_F2[l2])
        #print(structureList)
        combined = np.zeros_like(stack4D_F2[0])
        for structure in structureList:
            combined = np.maximum(combined, stack4D_F2[structure])

        combined = np.int8(combined)
        otherFrameMesh = Morphology.fullStackToMesh(combined, [zScale, 1, 1 , 1])
        otherFrameMesh.visual.face_colors = [0.0, 0.0, 0.0, 0.4]
        sc.add_geometry(otherFrameMesh)

        sphere = trimesh.creation.icosphere(1, 1/zScale, [0.0, 1.0, 0.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation[0] *= zScale
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        zCoord = int(location[0])
        xCoord = int(location[1])
        yCoord = int(location[2])

        #overload the structures with a faint green
        structures = np.float64(stack4D_F1[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:,:,:,0] = structures[:,:,:,2] = 0
        structures[:,:,:,1] = structures[:,:,:,1]*np.max(stackF1_3)*0.6

        structures2 = np.float64(stack4D_F1[l2])
        structures2 = np.stack((structures2,) * 3, axis=-1)
        structures2[:,:,:,1] = structures2[:,:,:,2] = 0
        structures2[:,:,:,0] = structures2[:,:,:,0]*np.max(stackF1_3)*0.6

        labelOverlaid = np.maximum(stackF1_3*0.5, np.maximum(structures, structures2))

        labelOverlaid, success = addKernelToOutputStack(green_kernel_3D, labelOverlaid, location, stackF1_3)
        highlighedLabels[:,0:100, 0:100] = ImageAnalysis.padStackXY(labelOverlaid, windowSize // 2)[:,xCoord:xCoord+windowSize,yCoord:yCoord+windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY(stackF1_3, windowSize // 2)[:,xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY(stackF2_3, windowSize // 2)[:,xCoord:xCoord+windowSize,yCoord:yCoord+windowSize]
    elif typeEvent == transType.FRAGMENT:
        label1 = np.int8(stack4D_F2[l1])
        labelMesh1 = Morphology.fullStackToMesh(label1, [zScale, 1, 1, 1])
        labelMesh1.visual.face_colors = [0.5, 0.0, 0.0, 0.5]
        sc.add_geometry(labelMesh1)

        label2 = np.int8(stack4D_F2[l2])
        labelMesh2 = Morphology.fullStackToMesh(label2, [zScale, 1, 1, 1])
        labelMesh2.visual.face_colors = [0.0, 0.5, 0.0, 0.5]
        sc.add_geometry(labelMesh2)

        structureList = np.intersect1d(associatedFromF2_to_F1[l1], associatedFromF2_to_F1[l2])
        # print(structureList)
        combined = np.zeros_like(stack4D_F1[0])
        for structure in structureList:
            combined = np.maximum(combined, stack4D_F1[structure])

        combined = np.int8(combined)
        otherFrameMesh = Morphology.fullStackToMesh(combined, [zScale, 1, 1, 1])
        otherFrameMesh.visual.face_colors = [0.0, 0.0, 0.0, 0.4]
        sc.add_geometry(otherFrameMesh)

        sphere = trimesh.creation.icosphere(1, 1 / zScale, [1.0, 0.0, 0.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation[0] *= zScale
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        zCoord = int(location[0])
        xCoord = int(location[1])
        yCoord = int(location[2])

        # overload the structures with a faint red
        structures = np.float64(stack4D_F2[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:,:,:,1] = structures[:,:,:,2] = 0
        structures[:,:,:,0] = structures[:,:,:,0]*np.max(stackF2_3)*0.6

        structures2 = np.float64(stack4D_F2[l2])
        structures2 = np.stack((structures2,) * 3, axis=-1)
        structures2[:,:,:,0] = structures2[:,:,:,2] = 0
        structures2[:,:,:,1] = structures2[:,:,:,1]*np.max(stackF2_3)*0.6

        labelOverlaid = np.maximum(stackF2_3*0.5, np.maximum(structures, structures2))

        labelOverlaid, success = addKernelToOutputStack(red_kernel_3D, labelOverlaid, location, stackF2_3)
        highlighedLabels[:, 0:100, 0:100] = ImageAnalysis.padStackXY(labelOverlaid, windowSize // 2)[:,
                                            xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY(stackF2_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY(stackF1_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
    elif typeEvent == transType.DEPOLARIZE:
        label1 = np.int8(stack4D_F1[l1])
        labelMesh1 = Morphology.fullStackToMesh(label1, [zScale, 1, 1, 1])
        labelMesh1.visual.face_colors = [0.0, 0.0, 0.5, 0.5]
        sc.add_geometry(labelMesh1)

        structureList = associatedFromF1_to_F2[l1]
        if(structureList.shape[0] != 0):
            print("There are not supposed to be associated structures")

        sphere = trimesh.creation.icosphere(1, 1 / zScale, [0.0, 0.0, 1.0, 1.0])
        sphere.apply_scale([zScale, zScale, zScale])
        thisLocation[0] *= zScale
        sphere.apply_translation(thisLocation)
        sc.add_geometry(sphere)

        zCoord = int(location[0])
        xCoord = int(location[1])
        yCoord = int(location[2])

        # overload the structures with a faint blue
        structures = np.float64(stack4D_F1[l1])
        structures = np.stack((structures,) * 3, axis=-1)
        structures[:, :, :, 0] = structures[:, :, :, 1] = 0
        structures[:, :, :, 2] = structures[:, :, :, 2] * np.max(stackF1_3) * 0.6

        labelOverlaid = np.maximum(stackF1_3*0.5, structures)

        labelOverlaid, success = addKernelToOutputStack(blue_kernel_3D, labelOverlaid, location, stackF1_3)
        highlighedLabels[:, 0:100, 0:100] = ImageAnalysis.padStackXY(labelOverlaid, windowSize // 2)[:,
                                            xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 100:200] = ImageAnalysis.padStackXY(stackF1_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
        highlighedLabels[:, 0:100, 200:300] = ImageAnalysis.padStackXY(stackF2_3, windowSize // 2)[:,
                                              xCoord:xCoord + windowSize, yCoord:yCoord + windowSize]
    else:
        print("The type must be either FUSE, FRAGMENT, or DEPOLARISE")
        return


    from scipy.spatial.transform import Rotation as R
    rotation = [0, 90, 0]
    rotationMatrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    transformationMatrix = np.eye(4, 4)
    transformationMatrix[0:3, 0:3] = rotationMatrix
    sc.apply_transform(transformationMatrix)

    return highlighedLabels, sc


# how many time frames to skip between comparisons. 0 is the lowest (means consecutive frames)
def runMEL(frame1Stack, frame2Stack, frameText, duplicateDistance=10,
filteredBinaryF1=None, stackLabelsF1=None, numLabelsF1=None, labeledStackF1=None, filteredLabeledStackF1=None, cannyLabeledStackF1=None):
    progamStartTime = time()

    try:
        print("\nFRAME 1")
        #ImageAnalysis.plotStackHist(frame1Stack, remove_zero=True)
        #filteredBinaryF1, stackLabelsF1, numLabelsF1 = Morphology.labelStack(ImageAnalysis.binarizeStack(frame1Stack))

        if type(filteredBinaryF1) == type(None) or type(stackLabelsF1) == type(None) or type(numLabelsF1) == type(None) or type(labeledStackF1) == type(None) or type(filteredLabeledStackF1) == type(None):
            (low, high) = ImageAnalysis.determineHysteresisThresholds(frame1Stack)
            # frame1StackThreshold = frame1Stack * ImageAnalysis.hysteresisThresholdingStack(frame1Stack, low, high)
            # frame1StackThreshold = ImageAnalysis.unsharpMask(frame1StackThreshold)
            # #ImageAnalysis.chooseHysteresisParams(frame1StackThreshold)
            # frame1StackThreshold = ImageAnalysis.binarizeStack(frame1StackThreshold)
            frame1StackThreshold = ImageAnalysis.hysteresisThresholdingStack(frame1Stack, low, high)
            filteredBinaryF1, stackLabelsF1, numLabelsF1 = Morphology.labelStack(frame1StackThreshold)

            labeledStackF1 = Morphology.stack3DTo4D(stackLabelsF1, numLabelsF1)

            filteredLabeledStackF1 = gaussianFilter(labeledStackF1)
        else:
            print("DATA PROVIDED")

        CoM_F1 = center_of_mass(filteredBinaryF1, stackLabelsF1, range(1, numLabelsF1 + 1))
        CoM_F1.insert(0, (0, 0, 0))  # to offset everything by 1, to match other arrays (start from label 1)
        CoM_F1 = np.array(CoM_F1)

        print("\nFRAME 2")
        #filteredBinaryF2, stackLabelsF2, numLabelsF2 = Morphology.labelStack(ImageAnalysis.binarizeStack(frame2Stack))
        (low, high) = ImageAnalysis.determineHysteresisThresholds(frame2Stack)
        # frame2StackThreshold = frame2Stack * ImageAnalysis.hysteresisThresholdingStack(frame2Stack, low, high)
        # frame2StackThreshold = ImageAnalysis.unsharpMask(frame2StackThreshold)
        # frame2StackThreshold = ImageAnalysis.binarizeStack(frame2StackThreshold)
        frame2StackThreshold = ImageAnalysis.hysteresisThresholdingStack(frame2Stack, low, high)
        filteredBinaryF2, stackLabelsF2, numLabelsF2 = Morphology.labelStack(frame2StackThreshold)
        print("{} labels detected in Frame 2".format(numLabelsF2))
        labeledStackF2 = Morphology.stack3DTo4D(stackLabelsF2, numLabelsF2)
        filteredLabeledStackF2 = gaussianFilter(labeledStackF2)

        
        # overlapF1_to_F2 defined shape (#labels F1, #labels F2) NOTE - these matrices are not symmetrical
        #            overlapF1_to_F2 = compareOverlapV2(labeledStackF1, filteredLabeledStackF2)
        #            overlapF2_to_F1  = compareOverlapV2(labeledStackF2, filteredLabeledStackF1)

        # overlapF1_to_F2 defined shape (#labels F1, #labels F2) the matrix is symmetrical in the new design
        overlapF1_to_F2 = compareOverlapV2(filteredLabeledStackF1, filteredLabeledStackF2)
        if type(overlapF1_to_F2) == bool: # could not run compare
            return (False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)

        overlapF2_to_F1 = overlapF1_to_F2.T

        # list labels associated with current label
        listAssociatedLabelsF1, listAssociatedLabelsF2 = getFragFusePairs(overlapF1_to_F2, overlapF2_to_F1)
        withinAssociatedLabelsF1, withinAssociatedLabelsF2 = backAndForthLabelMatching(listAssociatedLabelsF1,
                                                                                    listAssociatedLabelsF2)

        # Without this step I run out of RAM, drastically reduce the number of pixels to compare
        if type(cannyLabeledStackF1) == type(None):
            cannyLabeledStackF1 = labelToCanny(labeledStackF1)
        else:
            print("Canny Data provdided")
        cannyLabeledStackF2 = labelToCanny(labeledStackF2)        

        withinAssociatedLabelsF1_HWP, labelDistanceF1, vector_F1 = getAllHalfWayPoints(cannyLabeledStackF1,
                                                                            withinAssociatedLabelsF1)
        withinAssociatedLabelsF2_HWP, labelDistanceF2, vector_F2 = getAllHalfWayPoints(cannyLabeledStackF2,
                                                                            withinAssociatedLabelsF2)

        # determine link status
        withinChosenStatusF1_Fuse, withinChosenStatusF2_Fragment = linkStatusPerLabel(filteredBinaryF1,
                                                                                    filteredBinaryF2,
                                                                                    overlapF1_to_F2,
                                                                                    overlapF2_to_F1,
                                                                                    listAssociatedLabelsF1,
                                                                                    listAssociatedLabelsF2,
                                                                                    withinAssociatedLabelsF1,
                                                                                    withinAssociatedLabelsF2,
                                                                                    #withinAssociatedLabelsF1_HWP,
                                                                                    labelDistanceF1,
                                                                                    vector_F1,
                                                                                    #withinAssociatedLabelsF2_HWP,
                                                                                    labelDistanceF2,
                                                                                    vector_F2,
                                                                                    bigPercThresh=0.5)

        (outputStack, outcomes, locations, labels, dupOutcomes, dupLocations, dupLabels) = generateGradientImage(frame1Stack, CoM_F1, withinChosenStatusF1_Fuse,
                                                                withinChosenStatusF2_Fragment,
                                                                withinAssociatedLabelsF1,
                                                                withinAssociatedLabelsF2,
                                                                withinAssociatedLabelsF1_HWP,
                                                                withinAssociatedLabelsF2_HWP,
                                                                duplicateDistance)

        checkEventsList = (labeledStackF1, labeledStackF2, listAssociatedLabelsF1, listAssociatedLabelsF2)
        
        totalStructures = len(labeledStackF1)

        print("Max (should be 1): ", np.max(labeledStackF1))
        averageStructureVolume = 0
        for structure in labeledStackF1:
            averageStructureVolume += np.sum(structure)
        
        averageStructureVolume /= totalStructures

        print("\nMEL run time: ", time() - progamStartTime)
    
    except Exception as e:
        print("SOME EXCEPTION OCCURED")
        print(e)
        return (False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False)

    return (outputStack, filteredBinaryF1, outcomes, locations, labels, dupOutcomes, dupLocations, dupLabels, checkEventsList, totalStructures, averageStructureVolume,
    filteredBinaryF2, stackLabelsF2, numLabelsF2, labeledStackF2, filteredLabeledStackF2, cannyLabeledStackF2)
