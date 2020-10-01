'''
Install: (indented should install automatically)
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

install flowdec manually
'''

import tensorflow as tf

import ImageAnalysis
import Morphology

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from os import listdir, path, makedirs
from os.path import isfile, join


from skimage import measure
from scipy.spatial import ConvexHull
from scipy.ndimage import zoom
import pandas as pd
import csv
import math

from skimage import io

import trimesh

from CZI_Processor import cziFile

import MEL

import pyglet
import glooey
import GUI



personReviewName = "Rensu"

validate = True
saveResults = True

scaleFactor = 1.5

filePath = "G:\\PhD\\MEL_2020_CHOSEN\\Original\\"
writePath = "G:\\PhD\\MEL_2020_CHOSEN\\Output\\"

stackTimelapsePath = [f for f in listdir(filePath) if isfile(join(filePath, f)) and (f.endswith("czi"))]

positionNum = 0

for i in range(0, len(stackTimelapsePath)):
    print(i, ' ', stackTimelapsePath[i])

startFileIndex = 0
startFrame = 0
print('Start file: ', stackTimelapsePath[startFileIndex])



def nextLabel():
    global fuseFragDep
    global label
    global sceneWidget
    global imageWidget
    global displayHBox
    global scaledStack
    global currentFrame

    label += 1
    print("fuseFragDep", fuseFragDep)
    print("label", label)
    if label < len(labels[fuseFragDep]):
        
        typeEvent = MEL.transType.NOTHING
        if fuseFragDep == 0:
            typeEvent = MEL.transType.FUSE
        elif fuseFragDep == 1:
            typeEvent = MEL.transType.FRAGMENT
        elif fuseFragDep == 2:
            typeEvent = MEL.transType.DEPOLARIZE

        highlighedLabels, scene = MEL.checkEvent(typeEvent, checkEventsList[0], checkEventsList[1], checkEventsList[2],
                                                checkEventsList[3],
                                                labels[fuseFragDep][label], locations[fuseFragDep][label], frame1Stack,
                                                frame2Stack, cziData.zVoxelWidth / cziData.xVoxelWidth, scaleFactor)


        scaledStack = np.uint8(ImageAnalysis.contrastStretch(ImageAnalysis.rescaleStackXY_RGB(highlighedLabels, 2.5), 0, 100) * 255)

        currentFrame = 0

        displayHBox.remove(sceneWidget)
        sceneWidget = SceneWidget(scene)
        displayHBox.add_right(sceneWidget)
        imageWidget.update_image(scaledStack[currentFrame])

        text = "Fusion: {}/{}  Fission: {}/{}  Depolarisation: {}/{} Other: {}".format(
        fusionCorrect,len(labels[0]), fissionCorrect, len(labels[1]), depolarisationCorrect, len(labels[2]),otherIncorrect)
        print(text)

    elif fuseFragDep + 1 < 3:
        fuseFragDep += 1
        label = -1
        currentFrame = 0

        nextLabel()

    else:
        print("End of images")
        window.close()

def incrementYes(widget=None):
    global fuseFragDep
    global label
    global fusionCorrect
    global fissionCorrect
    global depolarisationCorrect
    yesList.append((fuseFragDep, label))
    if(fuseFragDep == 0): #Fusion
        fusionCorrect += 1
    elif(fuseFragDep == 1): #Fission
        fissionCorrect += 1
    elif(fuseFragDep == 2): #Depolarisation
        depolarisationCorrect += 1
    nextLabel()
    print("yes")

def incrementNo(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    noList.append((fuseFragDep, label))
    locationsCopy[fuseFragDep][label] = False
    otherIncorrect += 1
    nextLabel()
    print("no")

def incrementUnclear(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    unclearList.append((fuseFragDep, label))
    otherIncorrect += 1
    nextLabel()
    print("unclear")

def incrementThreshold(widget=None):
    global fuseFragDep
    global label
    global otherIncorrect
    thresholdList.append((fuseFragDep, label))
    locationsCopy[fuseFragDep][label] = False
    otherIncorrect += 1
    nextLabel()
    print("threshold")


#############################






for fileNameIndex in range(startFileIndex, len(stackTimelapsePath)):
    fileName = stackTimelapsePath[fileNameIndex]
    print("Processing: " + fileName)
    cziData = cziFile(filePath + fileName)
    cellOutcomes = []
    totalStructures = []
    averageVolume = []

    deconvPath = '{}\\{}\\Deconvolution\\'.format(writePath, fileName)
    if not path.exists(deconvPath):
        makedirs(deconvPath)

    outputPath = '{}\\{}\\'.format(writePath, fileName)
    if not path.exists(outputPath):
        makedirs(outputPath)

    filteredBinaryF2 = None
    stackLabelsF2 = None
    numLabelsF2 = None
    labeledStackF2 = None
    filteredLabeledStackF2 = None
    cannyLabeledStackF2 = None

    for timeIndex in range(startFrame, cziData.sizeT - 1): # -1 since frame 1 and frame 2 pair
        print("Time index: " + str(timeIndex) + ' for ' + fileName)

        frame1Index = timeIndex
        frame2Index = timeIndex + 1

        frame1IndexText = str(frame1Index)
        if(frame1Index < 10):
            frame1IndexText = "0" + frame1IndexText

        frame2IndexText = str(frame2Index)
        if (frame2Index < 10):
            frame2IndexText = "0" + frame2IndexText

        cziData.printSummary()

        if isfile("{}{}.tif".format(deconvPath, frame1IndexText)):
            frame1Deconv = io.imread("{}{}.tif".format(deconvPath, frame1IndexText))
            # this is only necessary for some cases, where the dimensions are swapped
            if frame1Deconv.shape[0] > frame1Deconv.shape[-1]:
                frame1Deconv = frame1Deconv.swapaxes(0, 2)
            print("loaded deconvolution from memory for Frame 1: ", frame1Deconv.shape)
        else:
            frame1Deconv = cziData.runDeconvolution(positionNum, frame1Index)
            ImageAnalysis.saveTifStack("{}{}.tif".format(deconvPath,  frame1IndexText), frame1Deconv/np.max(frame1Deconv))

        if isfile("{}{}.tif".format(deconvPath, frame2IndexText)):
            frame2Deconv = io.imread("{}{}.tif".format(deconvPath, frame2IndexText))
            # this is only necessary for some cases, where the dimensions are swapped
            if frame2Deconv.shape[0] > frame2Deconv.shape[-1]:
                frame2Deconv = frame2Deconv.swapaxes(0, 2)
            print("loaded deconvolution from memory for Frame 2: ", frame2Deconv.shape)
        else:
            frame2Deconv = cziData.runDeconvolution(positionNum, frame2Index)
            ImageAnalysis.saveTifStack("{}{}.tif".format(deconvPath, frame2IndexText), frame2Deconv/np.max(frame2Deconv))

        if (frame1Deconv.shape[0] == 1 or len(frame1Deconv) < 3):
            print("It appears as if the frame is 2D and not a z-stack. Padded with blank slices at the top and bottom")
            frame1Deconv = ImageAnalysis.padImageTo3D(frame1Deconv)
            frame2Deconv = ImageAnalysis.padImageTo3D(frame2Deconv)

        print('frame1Deconv.shape', frame1Deconv.shape)
        print('frame2Deconv.shape', frame2Deconv.shape)


        frame1Stack = ImageAnalysis.preprocess(frame1Deconv, scaleFactor=scaleFactor) # .copy()
        frame2Stack = ImageAnalysis.preprocess(frame2Deconv, scaleFactor=scaleFactor) #.copy()

        (low, high) = ImageAnalysis.determineHysteresisThresholds(frame1Stack, "{}\\Hist{}.png".format(outputPath, frame1IndexText))



        try:
            (outputStack, filteredBinaryF1, outcomes, locations, labels, dupOutcomes, dupLocations, dupLabels, checkEventsList, totalStruc, averageStrucVolume,
            filteredBinaryF2, stackLabelsF2, numLabelsF2, labeledStackF2, filteredLabeledStackF2, cannyLabeledStackF2) = MEL.runMEL(frame1Stack, frame2Stack, '{}-{}'.format(frame1IndexText, frame2IndexText), 10, filteredBinaryF2, stackLabelsF2, numLabelsF2, labeledStackF2, filteredLabeledStackF2, cannyLabeledStackF2)
            
            print("Outcomes: ", outcomes)
            print("Total Structures: ", totalStruc)
            print("Average Structure volume: ", averageStrucVolume)
            print()
            
            if type(outputStack) == bool: # something went wrong
                break


            if not validate:
                if saveResults:
                    ImageAnalysis.saveTifStack("{}\\{}.tif".format(outputPath, frame1IndexText), (outputStack * 255).astype(np.uint8))
                    print("SHAPE: ", filteredBinaryF1.shape)
                    ImageAnalysis.saveTifStack("{}\\T{}.tif".format(outputPath, frame1IndexText), (np.stack((filteredBinaryF1,filteredBinaryF1,filteredBinaryF1),axis=-1)*255).astype(np.uint8))        

                    cellOutcomes.append(outcomes)
                    totalStructures.append(totalStruc)
                    averageVolume.append(averageStrucVolume)
            else:
                print("VALIDATING RESULT")
                ####################
                # LOG which labels
                ####################

                # I need a start item (in this case -1) in order to allow for the case where no events are reported
                yesList = [(-1,-1)]
                noList = [(-1,-1)]
                unclearList = [(-1,-1)]
                thresholdList = [(-1,-1)]

                fusionCorrect = 0
                fissionCorrect = 0
                depolarisationCorrect = 0
                otherIncorrect = 0

                fuseFragDep = 0
                label = -1 # start by -1 since first display is test screen
                currentFrame = 0

                scaledStack = None

                # store whether the event should be retained or discareded
                locationsCopy = [[],[],[]]
                for i in range(0,3):
                    for loc in locations[i]:
                        locationsCopy[i].append(True)


                

                window = pyglet.window.Window(width=1400, height=650, caption="Check MEL events")
                gui = glooey.Gui(window)

                @window.event
                def on_key_press(symbol, modifiers):
                    global currentFrame
                    global scaledStack
                    global imageWidget

                    if (modifiers == 16 or modifiers == 0) and symbol == 65363 and (currentFrame + 1) < outputStack.shape[0]: # right arrow
                        currentFrame += 1

                        imageWidget.update_image(scaledStack[currentFrame])


                    elif (modifiers == 16 or modifiers == 0) and symbol == 65361 and (currentFrame - 1) >= 0: # left arrow
                        currentFrame -= 1

                        imageWidget.update_image(scaledStack[currentFrame])

                    elif (modifiers == 16 or modifiers == 0) and symbol == 65362: # up arrow
                        incrementYes()
                        #print("YES")

                    elif (modifiers == 16 or modifiers == 0) and symbol == 65364: # down arrow
                        incrementNo()
                        #print("NO")

                    elif (modifiers == 16 or modifiers == 0) and symbol == 51539607552: # center
                        incrementUnclear()
                        #print("UNCLEAR")

                    elif (modifiers == 16 or modifiers == 0) and symbol == 65365: # 9
                        incrementThreshold()
                        #print("THRESHOLD")
                    
                    elif symbol == 65293:
                        nextLabel()

                #############################


                vBox = glooey.VBox()

                from trimesh.viewer import SceneWidget
                sc = trimesh.Scene()
                sphere = trimesh.creation.icosphere(1, 1, [0.0, 1.0, 0.0, 0.5])
                sc.add_geometry(sphere)
                sceneWidget = SceneWidget(sc)

                displayHBox = glooey.HBox()
                displayHBox.add_right(sceneWidget, size=650)


                im = np.dstack((np.ones((250, 750)) * 255, np.zeros((250, 750)), np.zeros((250, 750)))).astype(np.uint8)
                imageWidget = GUI.ImageWidget(im)
                displayHBox.add_left(imageWidget, size=750)


                vBox.add_top(displayHBox)

                hbox = glooey.HBox()
                hbox.alignment = 'bottom'


                buttons = [
                    GUI.MyButton("Yes", "Yes", height=50, on_click=incrementYes),
                    GUI.MyButton("No", "No", height=50, on_click=incrementNo),
                    GUI.MyButton("Unclear", "Unclear", height=50, on_click=incrementUnclear),
                    GUI.MyButton("Threshold Mistake", "Threshold", height=50, on_click=incrementThreshold),
                ]


                for button in buttons:
                    hbox.add(button)

                vBox.add_bottom(hbox, size=50)
                gui.add(vBox)

                nextLabel()
                pyglet.app.run()

                print('Done')


                yesListNP = np.array(yesList)
                noListNP = np.array(noList)
                unclearListNP = np.array(unclearList)
                thresholdListNP = np.array(thresholdList)


                fusionDict = {}
                fissionDict = {}
                depDict = {}
                columnNames = ['YES', 'NO', 'UNCLEAR', 'THRESHOLD']
                typeDicts = [fusionDict, fissionDict, depDict]
                for t in range(0,3): # the type of event fusion fission depolarisation
                    lists = [yesListNP[np.where(yesListNP[:,0]==t),1][0],
                            noListNP[np.where(noListNP[:,0]==t),1][0],
                            unclearListNP[np.where(unclearListNP[:,0]==t),1][0],
                            thresholdListNP[np.where(thresholdListNP[:,0]==t),1][0]]
                    for i in range(0,4):
                        for l in lists[i]:
                            typeDicts[t][l] =  columnNames[i]


                if saveResults:
                    fusion_df = pd.DataFrame(sorted(typeDicts[0].items()))
                    fission_df = pd.DataFrame(sorted(typeDicts[1].items()))
                    dep_df = pd.DataFrame(sorted(typeDicts[2].items()))

                    fusion_df.to_csv('{}-{}({})-{}-{}_fusion_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))
                    fission_df.to_csv('{}-{}({})-{}-{}_fission_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))
                    dep_df.to_csv('{}-{}({})-{}-{}_depolarisation_classification.csv'.format(personReviewName, fileName, positionNum, frame1Index, frame2Index))

                #output new TIF and outcomes
                for i in range(0,3):
                    while False in locationsCopy[i]:
                        for index in range(0, len(locations[i])):
                            if not locationsCopy[i][index]:
                                del locations[i][index]
                                del locationsCopy[i][index]
                                break;

                if saveResults:
                    newStack = MEL.generateGradientImageFromLocation(frame1Stack, locations)
                    newStack = np.clip(newStack, 0, 1)
                    ImageAnalysis.saveTifStack("{}\\R{}.tif".format(outputPath, frame1IndexText), (newStack * 255).astype(np.uint8))

                outcomesDf = pd.DataFrame()
                outcomesDf['0'] = [len(locations[0])]
                outcomesDf['1'] = [len(locations[1])]
                outcomesDf['2'] = [len(locations[2])]
                outcomesDf.index += timeIndex
                
                if saveResults:
                    try:
                        existingDf = pd.read_csv("{}outcomesHuman.csv".format(outputPath))
                        existingDf = existingDf.drop('Unnamed: 0', 1)
                        existingDf = existingDf.append(outcomesDf, ignore_index=False)
                        existingDf.to_csv("{}outcomesHuman.csv".format(outputPath))
                        print("Appended to outcomesHuman")
                    except:
                        print("outcomesHuman probably doesn't exist. CREATED")
                        outcomesDf.to_csv("{}outcomesHuman.csv".format(outputPath))

            
        except Exception as e:
            print(e)
            print("SOMETHING WENT WRONG")

    startFrame = 0

    if not validate:
        if saveResults:
            df = pd.DataFrame(cellOutcomes)
            df['Total'] = totalStructures
            df['AverageVol'] = averageVolume
            df.to_csv("{}outcomes.csv".format(outputPath))

    