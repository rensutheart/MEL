from os import listdir
from os.path import isfile, join

from skimage import data, io
from skimage.filters import threshold_otsu, threshold_li, threshold_yen, threshold_isodata, threshold_mean, gaussian, apply_hysteresis_threshold,  unsharp_mask
from skimage.exposure import rescale_intensity
from skimage.transform import rescale

from skimage.morphology import white_tophat
from skimage.morphology import disk, square, ball

import pandas as pd

import TiffMetadata
from scipy import ndimage

from CZI_Processor import cziFile
import tifffile

import cv2

import numpy as np

from skimage.exposure import histogram

import matplotlib.pyplot as plt

import pylab


def plotRGBhist(img, bins=256, remove_zero=False, remove_max=False):
    if len(img.shape) != 3:
        print("The provided image must be (X, Y, C)")
        return

    startIndex = 0
    if remove_zero:
        startIndex = 1

    plt.figure()  # this is necessary to ensure no existing figure is overwritten
    r_counts, r_centers = histogram(img[:,:,0], nbins=bins)
    g_counts, g_centers = histogram(img[:, :, 1], nbins=bins)
    b_counts, b_centers = histogram(img[:, :, 2], nbins=bins)

    if remove_max:
        plt.plot(r_centers[startIndex:-1], r_counts[startIndex:-1], color='red')
        plt.plot(g_centers[startIndex:-1], g_counts[startIndex:-1], color='green')
        plt.plot(b_centers[startIndex:-1], b_counts[startIndex:-1], color='blue')
    else:
        plt.plot(r_centers[startIndex:], r_counts[startIndex:], color='red')
        plt.plot(g_centers[startIndex:], g_counts[startIndex:], color='green')
        plt.plot(b_centers[startIndex:], b_counts[startIndex:], color='blue')

    plt.show()

def plotGrayhist(img, bins=256, remove_zero=False, remove_max=False):
    startIndex = 0
    if remove_zero:
        startIndex = 1

    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.reshape(img, (img.shape[0], img.shape[1]))
    elif len(img.shape) != 2:
        print("The provided image must be (X, Y)")
        return

    plt.figure()  # this is necessary to ensure no existing figure is overwritten
    counts, centers = histogram(img, nbins=bins)

    if remove_max:
        plt.plot(centers[startIndex:-1], counts[startIndex:-1], color='black')
    else:
        plt.plot(centers[startIndex:], counts[startIndex:], color='black')

    plt.show()

def plotImageHist(img, bins=256, remove_zero=False, remove_max=False):
    imgShape = img.shape;

    if len(imgShape) == 2:  # gray
        plotGrayhist(img, bins, remove_zero, remove_max)
    elif len(imgShape) == 3:  # rgb
        plotRGBhist(img, bins, remove_zero, remove_max)
    else:
        print("img does not have correct shape. Expected (X, Y, C) or (X, Y)")
        return

def plotStackHist(stack, bins=256, remove_zero=False, remove_max=False):
    stackShape = stack.shape;

    if len(stackShape) == 3: # gray
        plotGrayhist(np.reshape(stack, (stackShape[1], stackShape[2] * stackShape[0])), bins, remove_zero, remove_max)
    elif len(stackShape) == 4: # rgb
        plotRGBhist(np.reshape(stack, (stackShape[1], stackShape[2] * stackShape[0], stackShape[3])), bins, remove_zero, remove_max)
    else:
        print("stack does not have correct shape. Expected (Sl, X, Y, C) or (Sl, X, Y)")
        return

def loadTimelapseTif(path, scaleFactor=1, pad=False):
    stackTimelapsePath = [f for f in listdir(path) if isfile(join(path, f)) and (f.endswith("tif") or f.endswith("tiff"))]
    stackTimelapsePath.sort()

    timelapse = [padStack3D(rescaleStackXY(io.imread(path + f), scaleFactor)) if pad else io.imread(path + f) for f in stackTimelapsePath]
    metadata = [TiffMetadata.metadata(path + f) for f in stackTimelapsePath]

    return timelapse, metadata

def loadTifStack(filename, scaleFactor=1, pad=False):
    stack = padStack3D(rescaleStackXY(io.imread(filename), scaleFactor)) if pad else io.imread(filename)
    metadata = TiffMetadata.metadata(filename)

    return stack, metadata

def loadGenericImage(filename):
    return io.imread(filename)

def saveGenericImage(filename, image):
    io.imsave(filename, image)

def saveTifStack(filename, imageArray):
    if(not str(filename).lower().endswith('tif') and not str(filename).lower().endswith('tiff')):
        filename += '.tif'
    io.imsave(filename, imageArray)

def loadCZIFile(filename):
    return cziFile(filename)

def padImageTo3D(image):
    if(image.shape[0] == 1):
        return padStack3D(image)
    else:
        return padStack3D(np.expand_dims(image, axis=0))

def padStack3D(stack):
    newStack = []
    # TODO: Check if I maybe need to add padding of 3 (As before) instead of only 1
    blankSlice = cv2.copyMakeBorder(np.zeros_like(stack[0]), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    newStack.append(blankSlice)
    for im in stack:
        newStack.append(cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    newStack.append(blankSlice)

    return np.stack(newStack)

def padStackXY(stack, paddingWidth=10):
    newStack = []
    pw = int(paddingWidth)

    for im in stack:
        newStack.append(cv2.copyMakeBorder(im, pw, pw, pw, pw, cv2.BORDER_CONSTANT, value=(0, 0, 0)))

    return np.stack(newStack)

def binarizeStack(stack, method='otsu'):
    if method.lower() == 'otsu':
        thresh = threshold_otsu(stack)
    elif method.lower() == 'li':
        thresh = threshold_li(stack)
    elif method.lower() == 'mean':
        thresh = threshold_mean(stack)
    elif method.lower() == 'yen':
        thresh = threshold_yen(stack)
    elif method.lower() == 'isodata':
        thresh = threshold_isodata(stack)
    else:
        print("Error method {} is not valid for binarizeStack. Defaulted to Otsu.".format(method))
        thresh = threshold_otsu(stack)

    return (stack > thresh)

def hysteresisThresholdingStack(stack, low=0.25, high=0.7):
    return apply_hysteresis_threshold(stack, low, high) > 0

def determineHysteresisThresholds(img, outputPath=None, bins=256, movingAverageFrame=20, cutOffSlope=2, highVal=0.95):
    counts, centers = histogram(img, nbins=bins)
    #remove 'black'
    counts = counts[1:]
    centers = centers[1:]

    df = pd.DataFrame(counts)
    movingAverage = df.rolling(movingAverageFrame, center=True).mean()

    startIntensity = 10
    useIntensityLow = startIntensity
    useIntensityHigh = 0

    for i in range(len(movingAverage[0])*3//4,startIntensity, -1):
        if movingAverage[0][i-10]/movingAverage[0][i+10] >= cutOffSlope:
              useIntensityLow = i
              print("Low intensity to be used: ", useIntensityLow/bins)
              print("High intensity to be used: ", (1.0-(1.0-useIntensityLow/bins)/2))

              break  

    print(outputPath)
    if outputPath != None:
        plt.figure(figsize=(6, 4))
        plt.plot(centers, counts, color='black')
        plt.axvline(useIntensityLow/bins, 0, 1, label='Low', color="red")
        plt.axvline((1.0-(1.0-useIntensityLow/bins)/2), 0, 1, label='High', color="blue")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.title("The automatically calculated hysteresis thresholding values")
        plt.tight_layout()
        plt.savefig(outputPath)
        print("Saved histogram")

    return (useIntensityLow/bins, (1.0-(1.0-useIntensityLow/bins)/2))

def contrastStretch(stack, l=2, h=100):
    p2, p98 = np.percentile(stack, (l, h))
    outStack = rescale_intensity(stack, in_range=(p2, p98))
    return outStack

def contrastStretchSliceInStack(stack, l=2, h=100):
    #per slice contrast stretching
    slCount = 0
    for sl in stack:
        p_min = np.percentile(sl, l)
        p_max = np.percentile(sl, h)
        stack[slCount] = rescale_intensity(sl, in_range=(p_min, p_max))
        slCount += 1

    return stack

def preprocess(stack, scaleFactor, percentageSaturate=0.003, scaleIntensityFactor=4, sigma2D=1.0, radius=3, amount=3, tophatBallSize=5):
    scaled = rescaleStackXY(stack, scaleFactor=scaleFactor)

    for sl in range(0,scaled.shape[0]):
        scaled[sl] = gaussian(scaled[sl], sigma2D)

    normalized = cv2.normalize(scaled, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    counts, centers = histogram(normalized, nbins=256)
    maxVal = 250
    while counts[-1]/np.sum(counts) < percentageSaturate:
        normalized = cv2.normalize(scaled, None, 0, maxVal, cv2.NORM_MINMAX, cv2.CV_8UC1)
        counts, centers = histogram(normalized, nbins=256)
        maxVal += 50

    print("NORMALIZATION Max Val = ", maxVal)

    normalized = rescaleStackXY(normalized, scaleFactor=1)
    

    return normalized

def unsharpMask(stack, radius=2, amount=2):
    return unsharp_mask(stack, radius, amount)

def rescaleStackXY(stack, scaleFactor=2):
    imList = []
    for im in stack:
        imList.append(rescale(im, scaleFactor))

    return np.array(imList)

def rescaleStackXY_RGB(stack, scaleFactor=2):
    if scaleFactor == 1:
        return stack

    imList = []
    for im in stack:
        imList.append(rescale(im, scaleFactor, multichannel=True))

    return np.array(imList)

def rescaleImageRGB(image, scaleFactor=2):
    if scaleFactor == 1:
        return image

    return rescale(image, scaleFactor, multichannel=True)

def stackToMIP(stack):
    return np.max(stack, axis=0)

def saveCroppedImagePanel(Frame1, Frame2, EventsFrameRGB, cropXStart, cropXWidth, cropYStart, cropYHeight, outputPath=None):
    frame1StackRGB = np.stack((Frame1,Frame1,Frame1), axis=-1)
    frame2StackRGB = np.stack((Frame2,Frame2,Frame2), axis=-1)

    miniPanel = np.hstack((
        np.max(frame1StackRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],
        np.ones((cropYHeight,2,3)),
        np.max(frame2StackRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],
        np.ones((cropYHeight,2,3)),
        np.max(EventsFrameRGB, axis=0)[cropYStart:cropYStart+cropYHeight,cropXStart:cropXStart+cropXWidth,:],)
        )
    
    if type(outputPath) != type(outputPath):
        io.imsave(outputPath, (miniPanel*255).astype(np.uint8))

    return miniPanel


