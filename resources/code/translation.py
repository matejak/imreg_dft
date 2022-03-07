
import cv2
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/imreg_dft"))
from imreg import (translation, similarity, transform_img, transform_img_dict, imshow)

# the TEMPLATE
im0_color = cv2.imread("../examples/sample1.png")
im0 = cv2.cvtColor(im0_color, cv2.COLOR_RGB2GRAY)

# the image to be transformed
im1_color = cv2.imread("../examples/sample2.png")
im1 = cv2.cvtColor(im1_color, cv2.COLOR_RGB2GRAY)

result = translation(im0, im1)
tvec = result["tvec"].round(4)
# the Transformed IMaGe.
timg = transform_img(im1, tvec=tvec)

# Maybe we don't want to show plots all the time
if os.environ.get("IMSHOW", "yes") == "yes":
    import matplotlib.pyplot as plt
    imshow(im0, im1, timg)
    plt.show()

print("Translation is {}, success rate {:.4g}"
      .format(tuple(tvec), result["success"]))
