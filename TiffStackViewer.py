# https://github.com/kambergjohnson/tiff-stack-viewer

import sys, os, csv
from PIL import Image
import time
import exifread
from mayavi import mlab
import numpy as np
import matplotlib.pyplot as plt


# returns a list  of channels, each of which is a list (length = Z stack depth) of images
def getImages(fN):
    tags = exifread.process_file(open(fN, 'rb'), details=False, stop_tag='ImageDescription')
    imageJ = tags[np.sort([x for x in tags if 'ImageDescription' in x])[0]].values
    settings = d = {y[0]: y[1] for y in [x.split('=') for x in imageJ.split('\n')] if len(y) == 2}
    num_channels = int(settings['channels'] if 'channels' in settings else '1')
    num_stacks = int(settings['slices'])
    num_time = int(int(settings['frames'] if 'frames' in settings else '1'))
    num_images = int(settings['images'])
    ctz = [[[] for x in range(num_time)] for x in range(num_channels)]
    print('spacing: ', settings['spacing'], settings['unit'])
    tiff = Image.open(fN)
    i = 0
    mode = np.float if tiff.mode == 'F' else np.uint

    while True:
        try:
            tiff.seek(i)
            # uint32 on windows, uint64 on mac/unix?
            npa = np.ndarray((tiff.size[1], tiff.size[0]), mode, np.array(tiff.getdata()))
            ctz[i % num_channels][(i // num_channels) % num_time].append(npa)
            i += 1
        except EOFError:
            break
    return ctz


# given a list of images, displays them
def showImages(inList):
    plt.ion()
    # close()
    for p in inList:
        fig = plt.figure()
        plt.imshow(p, figure=fig, cmap='RdBu')
        plt.colorbar()
        fig.canvas.draw()
        plt.pause(0.01)
    # fig.canvas.flush_events()
    plt.ioff()
    return 0

def renderVolume(arr):
    print(arr.shape)
    # plot as volume.
    minVal = np.percentile(arr, 95)  # arr.min()
    maxVal = np.percentile(arr, 99.95)  # arr.max()

    stackToPixelSpacing = 3  # if pixel is 0.2u and zstack is 2u, put 10
    extent = [0, arr.shape[-1], 0, arr.shape[-2], 0, stackToPixelSpacing * float(arr.shape[-3])]

    viewMode = 'v'  # toggles between volume and contour
    if viewMode == 'v':
        scale = float(stackToPixelSpacing)
        stretched = np.ndarray((arr.shape[0] * stackToPixelSpacing, arr.shape[1], arr.shape[2]))
        for zI in range(stretched.shape[0]):
            minI = int(np.floor(zI // scale))
            maxI = int(np.ceil(zI // scale))
            weight = int(zI // scale - minI)
            for yI in range(stretched.shape[1]):
                for xI in range(stretched.shape[2]):
                    stretched[zI][yI][xI] = weight * arr[minI][yI][xI] + (1.0 - weight) * arr[minI][yI][xI]
        iso = mlab.pipeline.volume(mlab.pipeline.scalar_field(stretched), vmin=minVal, vmax=maxVal)
    else:
        # plot as small blobs.  vmin and vmax set the limits of the coloring scheme
        # iso = mlab.contour3d(arr, opacity=1.0) #autoscale image extents
        iso = mlab.contour3d(arr, opacity=1.0, extent=extent)  # scale image based on coefficient

    return iso

def showVolStack(arr):
    renderVolume(arr)

    mlab.show()

def showVolume(inList):
    arr = np.array(inList[0])
    iso = renderVolume(arr)

    if len(inList) > 1:
        @mlab.animate()
        def anim():
            while True:
                for i in range(len(inList)):
                    arr = np.array(inList[i])
                    iso.mlab_source.set(scalars=arr)
                    yield
                # mlab.draw()

        anim()
    mlab.show()
    return


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fN_index = -1
        choose_channel = 0
        if len(sys.argv) > 2:
            fN_index = -2
            choose_channel = int(sys.argv[-1])
        if '.tif' in sys.argv[fN_index]:
            zStack = getImages(sys.argv[fN_index])
            showVolume(zStack[choose_channel])