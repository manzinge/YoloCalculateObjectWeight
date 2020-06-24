import subprocess
import re

def getFocalLength(imgPath):
    result = subprocess.Popen(["identify","-verbose", imgPath], stdout=subprocess.PIPE)
    (output, err) = result.communicate()
    x = re.findall("exif:FocalLength: \d{1,5}\/\d{1,4}", str(output))
    focalLength = float(x[0].split(' ')[-1].split('/')[0]) /float(x[0].split(' ')[-1].split('/')[1])
    return focalLength