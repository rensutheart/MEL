import exifread
import numpy as np

class metadata:
    def __init__(self, filename):
        if(not (filename.endswith("tif") or filename.endswith("tiff"))):
            raise RuntimeError('Filename must be of type tif or tiff')

        tags = exifread.process_file(open(filename, 'rb'), details=True)
        imageJ = tags[np.sort([x for x in tags if 'ImageDescription' in x])[0]].values
        settings = d = {y[0]: y[1] for y in [x.split('=') for x in imageJ.split('\n')] if len(y) == 2}
        self.num_channels = int(settings['channels'] if 'channels' in settings else '1')
        self.sizeS = int(settings['slices'])
        self.sizeT = int(int(settings['frames'] if 'frames' in settings else '1'))
        self.sizeZ = int(settings['images'])

        self.sizeX = tags['Image ImageWidth'].values[0]
        self.sizeY  = tags['Image ImageLength'].values[0]

        xResRatio = tags['Image XResolution'].values[0]
        yResRatio = tags['Image YResolution'].values[0]

        self.xVoxelWidth  = xResRatio.den / xResRatio.num  # to get the microns per pixel

        if (self.xVoxelWidth != yResRatio.den / yResRatio.num):
            print("WARNING! The X and Y pixel resolution is different. Using only the x-resolution")

        self.zVoxelWidth = float(settings['spacing'])

        self.unit = settings['unit']