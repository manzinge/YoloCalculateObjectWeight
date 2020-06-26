import os
import glob
from PIL import Image
from determineWeight import getEstimatedWeight
from matplotlib import image, patches, pyplot as plt
from pythonAnalyser import analyseImage
import configparser

class ImgARatio:
    def __init__(self, img, aspectRatio):
        self.img = img
        self.aspectRatio = aspectRatio

class ImgABox:
    def __init__(self, img, bbox):
        self.img = img
        self.bbox = bbox

class Settings:
    def __init__(self, imgPath, rotationLimit, rotationFolder, analyseFolder, saveAnalyzedImages):
        self.imgPath = imgPath
        self.rotationLimit = rotationLimit
        self.rotationFolder = rotationFolder
        self.analyseFolder = analyseFolder
        self.saveAnalyzedImages = saveAnalyzedImages

class CameraSettings:
    def __init__(self, focalLength, dist, sensor_height, sensor_width):
        self.focalLength = focalLength
        self.dist = dist
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width

class NeuralNetworkSettings:
    def __init__(self, data, config, weight,names):
        self.data = data
        self.config = config
        self.weight = weight
        self.names = names

def getConfig():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config

def getGeneralSettings():
    Config = getConfig()
    settings = Settings(Config.get("General",'imgPath'),Config.get("General",'rotationLimit'),Config.get("General",'rotationFolder'),Config.get("General",'analyseFolder'),Config.get("General",'saveAnalyzedImages'))
    return settings

def getCameraConfigSettings():
    Config = getConfig()
    settings = CameraSettings(Config.get("Camera",'focalLength'), Config.get("Camera",'distance'), Config.get("Camera",'sensor_height'), Config.get("Camera",'sensor_width'))
    return settings

def getNeuralNetworkSettings():
    Config = getConfig()
    settings = NeuralNetworkSettings(Config.get("Network",'data'),Config.get("Network",'config'),Config.get("Network",'weight'),Config.get("Network",'names'))
    return settings

def checkFolder(settings):
    if not os.path.exists(settings.rotationFolder):
        os.makedirs(settings.rotationFolder)
    if not os.path.exists(settings.analyseFolder):
        os.makedirs(settings.analyseFolder)
    if not os.path.exists("bestCalculations"):
        os.makedirs("bestCalculations")

def createRotatedImages(settings):
    orgImg = Image.open(settings.imgPath)
    filename = settings.imgPath.split('/')[-1].split('.')[0]
    folder = settings.rotationFolder + filename
    if not os.path.exists(folder):
            os.makedirs(folder)

    for i in range(0,int(settings.rotationLimit),10):
        print("Rotating Image. Current degree {}/{}".format(i,float(settings.rotationLimit)))
        rotated = orgImg.rotate(i)
        imgName = filename +'/' + filename + "-" + str(i) + "R.png"
        if not os.path.isfile(settings.rotationFolder+imgName):
            rotated.save(settings.rotationFolder+imgName)
        rotated.close()
    orgImg.close()

def saveImageWithBbox(img, targetbox, settings):
    imgPath = os.getcwd() + '/' + img
    folder = img.split('/')[1]
    folderPath = os.getcwd() + '/' + settings.analyseFolder + folder
    if not os.path.exists(folderPath):
            os.makedirs(folderPath)
    savePath = os.getcwd() + '/' + settings.analyseFolder + folder + '/' + img.split('/')[-1]
    img = image.imread(imgPath)
    figure, ax = plt.subplots(1)
    rect = patches.Rectangle((targetbox[0], targetbox[1]),targetbox[2], targetbox[3], edgecolor='r', facecolor='None')
    ax.imshow(img)
    ax.add_patch(rect)
    plt.savefig(savePath)
    print("Successfully saved analyzed Image to {}".format(savePath))
    plt.clf()
    plt.close()

def getBoundingBoxes(settings, nnconfig):
    folder = settings.imgPath.split('/')[-1].split('.')[0]
    searchPattern = settings.rotationFolder + folder + '/*' + settings.imgPath.split('/')[-1].split('.')[0] + "*.png"
    bboxes = []
    counter = 1
    for img in glob.glob(searchPattern):
        boxescv = analyseImage(nnconfig.names,nnconfig.weight, nnconfig.config, os.getcwd() + '/' + img)
        if len(boxescv) == 0:
            print("No detection on image : {}".format(counter))
            counter +=1
            continue
        bestBox = boxescv[0][0]
        for i, det in enumerate(boxescv):
            if boxescv[0][i][4] > bestBox[4]:
                bestBox = boxescv[0][i]
        else:
            boxes = ""
            bboxes.append(ImgABox(img, bestBox))
            if settings.saveAnalyzedImages == 'True' and bestBox[4] > 0.85: 
                saveImageWithBbox(img, bestBox, settings)
            print("Successfully analyzed picture {}/{} with confidence: {}".format(counter, len(glob.glob(searchPattern)), bestBox[4]))
        counter += 1
    return bboxes

def getAspectRatios(bboxes):
    aspectRatios = []
    for i, box in enumerate(bboxes):
        width = bboxes[i].bbox[2]
        height = bboxes[i].bbox[3]
        aspectRatio = width / height
        aspectRatios.append(ImgARatio(bboxes[i].img, aspectRatio))
    return aspectRatios


def main():
    settings = getGeneralSettings()
    cameraConfig = getCameraConfigSettings()
    neuralNetworkConfig = getNeuralNetworkSettings()
    checkFolder(settings)
    createRotatedImages(settings)
    bBoxes = getBoundingBoxes(settings,neuralNetworkConfig)
    aspectRatios = getAspectRatios(bBoxes)
    bestImg = ImgARatio("Image", 0.000000001)
    lowestImage = ImgARatio("Image", 10)
    for img in aspectRatios:
        if img.aspectRatio > bestImg.aspectRatio:
            bestImg = img

    for img in aspectRatios:
        if img.aspectRatio < lowestImage.aspectRatio:
            lowestImage = img

    print("\nFinally:\nThe best image(highest AR) is : {}, with an Aspect Ratio of : {}".format(bestImg.img, bestImg.aspectRatio))
    print("The best image(lowest AR) is : {}, with an Aspect Ratio of : {}\n".format(lowestImage.img, lowestImage.aspectRatio))
    highestWeight = getEstimatedWeight(os.getcwd() + '/' + bestImg.img, cameraConfig, neuralNetworkConfig)
    lowestWeight = getEstimatedWeight(os.getcwd() + '/' + lowestImage.img, cameraConfig, neuralNetworkConfig)
    if highestWeight.height / highestWeight.width > lowestWeight.height / lowestWeight.width:
        print("Weight ist probably : {} (Alternative : {}".format(highestWeight.weight, lowestWeight.weight))
    else:
        print("Weight ist probably : {} (Alternative : {}".format(lowestWeight.weight, highestWeight.weight))


main()