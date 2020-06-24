from matplotlib import image, patches, pyplot as plt
from darknetpy.detector import Detector
from PIL import Image
from FocalLength import getFocalLength
import math

def getEstimatedWeight(detector, imgPath, CameraSettings):
    plt.clf()
    img_height, img_width = Image.open(imgPath).size

    boxes = detector.detect(imgPath)

    fig,ax = plt.subplots(1)
    ax.imshow(image.imread(imgPath))

    colors = ['r','b','y']

    for i, box in enumerate(boxes):
        if box['prob'] < 0.8:
            break
        #Get Information of Box
        l = box['left']
        t = box['top']
        b = box['bottom']
        r = box['right']
        c = box['class']
        box_width = b - t
        box_height = r - l
        color = colors[i % len(colors)]

        #Calculate real Object size and weight
        object_height = (float(CameraSettings.dist) * box_height * float(CameraSettings.sensor_height))/(float(CameraSettings.focalLength) * img_height)
        object_width = (float(CameraSettings.dist) * box_width * float(CameraSettings.sensor_width))/(float(CameraSettings.focalLength) * img_width)
        object_weight = pow(object_width/20,2) * math.pi * object_height/10

        #Draw Rectangle in Plot
        rect = patches.Rectangle(
            (l,t),
            box_height,
            box_width,
            linewidth = 1,
            edgecolor = color,
            facecolor = 'none'
        )

        print("Left = {}, Top = {}, Bottom = {}, Right = {}".format(l,t,b,r))
        print("Height of Object: {}, Width of Object: {}".format(object_height, object_width))
        print("Estimated Weight is: {}g".format(object_weight))

        #Add Rectangle to Plot
        ax.text(l, t, c, fontsize = 12, bbox = {'facecolor': color, 'pad': 2, 'ec': color})
        ax.add_patch(rect)

    plt.show()