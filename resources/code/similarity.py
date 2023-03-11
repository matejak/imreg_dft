
import cv2
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/imreg_dft"))
from imreg import (similarity, imshow)


basedir = os.path.join('..', 'examples')
# the TEMPLATE
im0_color = cv2.imread(os.path.join(basedir, "sample1.png"))
im0 = cv2.cvtColor(im0_color, cv2.COLOR_RGB2GRAY)
# the image to be transformed
im1_color = cv2.imread(os.path.join(basedir, "sample3.png"))
im1 = cv2.cvtColor(im1_color, cv2.COLOR_RGB2GRAY)
result = similarity(im0, im1, numiter=3)

assert "timg" in result
# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    imshow(im0, im1, result['timg'])
    plt.show()
