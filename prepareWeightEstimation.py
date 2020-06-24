import os
import glob
from darknetpy.detector import Detector
from PIL import Image
from determineWeight import getEstimatedWeight
from matplotlib import image, patches, pyplot as plt
import configparser

imgPath = ''
rotationLimit = 180
rotationFolder = "rotationImages/"
analyseFolder = "analyzeImages/"

class ImgARatio:
    def __init__(self, img, aspectRatio):
        self.img = img
        self.aspectRatio = aspectRatio

class ImgABox:
    def __init__(self, img, bbox):
        self.img = img
        self.bbox = bbox

class CameraSettings:
    def __init__(self, focalLength, dist, sensor_height, sensor_width):
        self.focalLength = focalLength
        self.dist = dist
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width

class NeuralNetworkSettings:
    def __init__(self, data, config, weight):
        self.data = data
        self.config = config
        self.weight = weight

def getCameraConfigSettings():
    Config = configparser.ConfigParser()
    Config.read('config.ini')
    settings = CameraSettings(Config.get("Camera",'focalLength'), Config.get("Camera",'distance'), Config.get("Camera",'sensor_height'), Config.get("Camera",'sensor_width'))
    return settings

def getNeuralNetworkSettings():
    Config = configparser.ConfigParser()
    Config.read('config.ini')
    settings = NeuralNetworkSettings(Config.get("Network",'data'),Config.get("Network",'config'),Config.get("Network",'weight'))
    return settings

def checkFolder():
    if not os.path.exists(rotationFolder):
        os.makedirs(rotationFolder)
    if not os.path.exists(analyseFolder):
        os.makedirs(analyseFolder)

def createRotatedImages():
    orgImg = Image.open(imgPath)
    for i in range(10,rotationLimit,10):
        print("Rotating Image. Current degree {}/{}".format(i,rotationLimit))
        rotated = orgImg.rotate(i)
        imgName = imgPath.split('/')[-1].split('.')[0] + "-" + str(i) + "R.png"
        rotated.save(rotationFolder+imgName)
        rotated.close()
    orgImg.close()

def setupDetector(networkSettings):
    detector = Detector(
        networkSettings.data,
        networkSettings.config,
        networkSettings.weight    
    )
    return detector

def saveImageWithBbox(img, targetbox):
    plt.clf()
    imgPath = os.getcwd() + '/' + img
    savePath = os.getcwd() + '/' + analyseFolder + img.split('/')[-1]
    img = image.imread(imgPath)
    figure, ax = plt.subplots(1)
    rect = patches.Rectangle((targetbox['left'], targetbox['top']),targetbox['right'] - targetbox['left'], targetbox['bottom'] - targetbox['top'], edgecolor='r', facecolor='None')
    ax.imshow(img)
    ax.add_patch(rect)
    plt.savefig(savePath)
    print("Successfully saved analyzed Image to {}".format(savePath))

def getBoundingBoxes(detector):
    searchPattern = rotationFolder + '*' + imgPath.split('/')[-1].split('.')[0] + "*.png"
    bboxes = []
    counter = 1
    for img in glob.glob(searchPattern):
        boxes = detector.detect(os.getcwd() + '/' + img)
        targetbox = boxes[0]
        if len(boxes) > 1:
            for i, box in enumerate(boxes):
                if boxes[i]['prob'] > targetbox['prob']:
                    targetbox = boxes[i]
        boxes = ""
        bboxes.append(ImgABox(img, targetbox))
        saveImageWithBbox(img, targetbox)
        print("Successfully analyzed picture {}/{}".format(counter, len(glob.glob(searchPattern))))
        if counter == 15:
            x = 5
        counter += 1

    return bboxes

def getAspectRatios(bboxes):
    aspectRatios = []
    for i, box in enumerate(bboxes):
        width = bboxes[i].bbox['right'] - bboxes[i].bbox['left']
        height = bboxes[i].bbox['bottom'] - bboxes[i].bbox['top']
        aspectRatio = width / height
        aspectRatios.append(ImgARatio(bboxes[i].img, aspectRatio))
    return aspectRatios


def main():
    cameraConfig = getCameraConfigSettings()
    neuralNetworkConfig = getNeuralNetworkSettings()
    detector = setupDetector(neuralNetworkConfig)
    checkFolder()
    # createRotatedImages()
    bBoxes = getBoundingBoxes(detector)
    aspectRatios = getAspectRatios(bBoxes)
    bestImg = ImgARatio("Image", 0.000000001)
    for img in aspectRatios:
        if img.aspectRatio > bestImg.aspectRatio:
            bestImg = img

    print("The best image is : {}, with an Aspect Ratio of : {}".format(bestImg.img, bestImg.aspectRatio))
    print("Calculating estimated weight of Object!")
    getEstimatedWeight(detector, os.getcwd() + '/' + bestImg.img, cameraConfig)

main()