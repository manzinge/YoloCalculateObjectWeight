from matplotlib import image, patches, pyplot as plt
from darknetpy.detector import Detector
from PIL import Image
from FocalLength import getFocalLength
import math

class EstimatedObject:
    def __init__(self, height, width, weight):
        self.height = height
        self.width = width
        self.weight = weight 

def getEstimatedWeight(detector, imgPath, CameraSettings):
    plt.clf()
    img_height, img_width = Image.open(imgPath).size

    boxes = detector.detect(imgPath)
    targetbox = boxes[0]
    for i, box in enumerate(boxes):
        if boxes[i]['prob'] > targetbox['prob']:
            targetbox = boxes[i]
    
    fig,ax = plt.subplots(1)
    ax.imshow(image.imread(imgPath))

    box = targetbox
    #Get Information of Box
    l = box['left']
    t = box['top']
    b = box['bottom']
    r = box['right']
    c = box['class']
    box_width = b - t
    box_height = r - l

    if box_width > box_height:
        calc_width, calc_height = box_height, box_width
    else:
        calc_width, calc_height = box_width, box_height

    color = 'r'

    object_height = (float(CameraSettings.dist) * calc_height * float(CameraSettings.sensor_height))/(float(CameraSettings.focalLength) * img_height)
    object_width = (float(CameraSettings.dist) * calc_width * float(CameraSettings.sensor_width))/(float(CameraSettings.focalLength) * img_width)
    
    object_weight = pow(object_width/20,2) * math.pi * object_height/10

    # Draw Rectangle in Plot
    rect = patches.Rectangle(
        (l,t),
        box_height,
        box_width,
        linewidth = 1,
        edgecolor = color,
        facecolor = 'none'
    )

    print("Height of Object: {}, Width of Object: {}".format(object_height, object_width))
    print("Estimated Weight is: {}g\n".format(object_weight))

    # Add Rectangle to Plot
    ax.text(l, t, c, fontsize = 12, bbox = {'facecolor': color, 'pad': 2, 'ec': color})
    ax.add_patch(rect)
    plt.savefig("bestCalculations/" + imgPath.split('/')[-1])

    return EstimatedObject(object_height, object_width, object_weight)