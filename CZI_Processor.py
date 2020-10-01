import czifile
from lxml import objectify

from flowdec import psf as fd_psf
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration

import ImageAnalysis

import time

class cziFile:
    def __init__(self, filename):
        if(not (filename.endswith("czi"))):
            raise RuntimeError('Filename must be of type czi')

        file = czifile.CziFile(filename)

        ImageDocument = objectify.fromstring(file.metadata())

        self.xVoxelWidth = (ImageDocument.Metadata.Scaling.Items.getchildren()[0].getchildren()[0] + 0)*10**9 #nm
        self.yVoxelWidth = (ImageDocument.Metadata.Scaling.Items.getchildren()[1].getchildren()[0] + 0)*10**9 #nm
        self.zVoxelWidth = (ImageDocument.Metadata.Scaling.Items.getchildren()[2].getchildren()[0] + 0)*10**9 #nm

        self.unit = "nm"

        # self.sizeX = int(ImageDocument.Metadata.Information.Image.SizeX*0.25)
        # self.sizeY = int(ImageDocument.Metadata.Information.Image.SizeY*0.25)
        self.sizeX = ImageDocument.Metadata.Information.Image.SizeX
        self.sizeY = ImageDocument.Metadata.Information.Image.SizeY

        try:
            self.sizeZ = ImageDocument.Metadata.Information.Image.SizeZ
        except:
            print('Could not load SizeZ')
            self.sizeZ = 1
        try:
            self.sizeS = ImageDocument.Metadata.Information.Image.SizeS
        except:
            print('Could not load SizeS')
            self.sizeS = 0

        try:
            self.sizeT = ImageDocument.Metadata.Information.Image.SizeT
        except:
            print('Could not load SizeT')
            self.sizeT = 1

        self.RefractiveIndex = ImageDocument.Metadata.Information.Image.ObjectiveSettings.RefractiveIndex

        #self.minWavelength = ImageDocument.Metadata.Experiment.ExperimentBlocks.AcquisitionBlock.MultiTrackSetup.TrackSetup.Detectors.Detector.DetectorWavelengthRanges.DetectorWavelengthRange.WavelengthStart*10**9
        self.ExcitationWavelength = ImageDocument.Metadata.Information.Image.Dimensions.Channels.Channel.ExcitationWavelength
        self.LensNA = ImageDocument.Metadata.Information.Instrument.Objectives.Objective.LensNA
        self.NominalMagnification = ImageDocument.Metadata.Information.Instrument.Objectives.Objective.NominalMagnification

        self.imageData = file.asarray()

    def printSummary(self):
        print('X, Y, Z - voxel width {}, {}, {}'.format(self.xVoxelWidth, self.yVoxelWidth, self.zVoxelWidth))
        print('x, y, z, s, t - dimensions {}, {}, {}, {}, {}'.format(self.sizeX, self.sizeY, self.sizeZ, self.sizeS, self.sizeT))
        print('Refractive Index - {}'.format(self.RefractiveIndex))
        print('Excitation Wavelength - {}'.format(self.ExcitationWavelength))
        #print('minWavelength - {}'.format(self.minWavelength))
        print('LensNA - {}'.format(self.LensNA))
        print('NominalMagnification - {}'.format(self.NominalMagnification))

    def generatePSF(self):
        psf = fd_psf.GibsonLanni(   size_x=self.sizeX,
                                    size_y=self.sizeY,
                                    size_z=self.sizeZ,
                                    na=self.LensNA,
                                    wavelength=self.ExcitationWavelength/1000,
                                    m=self.NominalMagnification,
                                    ns=self.RefractiveIndex,
                                    #min_wavelength=self.minWavelength/1000
                                    )

        return psf.generate()

    def runDeconvolution(self, position, timePoint, numIterations=25, session_config=None):
        stack = self.getStack(position, timePoint)
        #stack = self.getScaledStack(position, timePoint, 0.5)


        if(stack.shape[0] == 1 or len(stack) < 3):
            print("The data does not seem to contain any z information. Currently this can't be deconvolved.")
            return stack

        acquisition = fd_data.Acquisition(data = stack, kernel=self.generatePSF())
        print('Loaded data with shape {} and psf with shape {}'.format(acquisition.data.shape, acquisition.kernel.shape))

        start_time = time.time()

        # Initialize deconvolution with a padding minimum of 1, which will force any images with dimensions
        # already equal to powers of 2 (which is common with examples) up to the next power of 2
        algorithm = fd_restoration.RichardsonLucyDeconvolver(n_dims=acquisition.data.ndim, pad_min=[1, 1, 1]).initialize() # , device="/GPU:1"
        print('before run')


        res = algorithm.run(acquisition, niter=numIterations, session_config=session_config)

        end_time = time.time()
        print('Deconvolution complete (in {:.3f} seconds)'.format(end_time - start_time))

        print(res.info)

        return res.data

    def getStack(self, position, timePoint):
        print(self.imageData.shape)
        if self.sizeS == 0:
            return self.imageData[0, 0, timePoint, :, :, :, 0]
        else:
            return self.imageData[0, position, 0, timePoint, :, :, :, 0]

    def getScaledStack(self, scaleFactor, position, timePoint):
        return ImageAnalysis.rescaleStackXY(self.getStack(position, timePoint), scaleFactor)


if __name__ == '__main__':
    cziData = cziFile("D:\\PhD\\Old samples original HQ 1mM\\Image 18.czi")
    cziData.printSummary()
    result = cziData.runDeconvolution(4,0)