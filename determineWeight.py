from matplotlib import image, patches, pyplot as plt
from PIL import Image
from FocalLength import getFocalLength
from pythonAnalyser import analyseImage
import math

class EstimatedObject:
    def __init__(self, height, width, weight):
        self.height = height
        self.width = width
        self.weight = weight 

def getEstimatedWeight(imgPath, CameraSettings, nnconfig):
    plt.clf()
    img_height, img_width = Image.open(imgPath).size

    cv2box = analyseImage(nnconfig.names, nnconfig.weight, nnconfig.config, imgPath)
    targetbox = cv2box[0][0]
    for i, box in enumerate(cv2box):
        if cv2box[i][0][4] > targetbox[4]:
            targetbox = boxes[i]
    
    fig,ax = plt.subplots(1)
    ax.imshow(image.imread(imgPath))

    box_width = targetbox[3]
    box_height = targetbox[2]

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
        (targetbox[0],targetbox[1]),
        box_height,
        box_width,
        linewidth = 1,
        edgecolor = color,
        facecolor = 'none'
    )

    print("Height of Object: {}, Width of Object: {}".format(object_height, object_width))
    print("Estimated Weight is: {}g\n".format(object_weight))

    # Add Rectangle to Plot
    ax.text(targetbox[0], targetbox[1], 'Object', fontsize = 12, bbox = {'facecolor': color, 'pad': 2, 'ec': color})
    ax.add_patch(rect)
    plt.savefig("bestCalculations/" + imgPath.split('/')[-1])

    return EstimatedObject(object_height, object_width, object_weight)