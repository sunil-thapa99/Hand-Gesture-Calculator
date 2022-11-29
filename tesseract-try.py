import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import Image
import imutils

# Path were tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/Cellar/tesseract/5.2.0/bin/tesseract'

# image = Image.open("3.jpeg")
# img = np.array(image)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# image = Image.fromarray(thresh)
# image.show()
# cv2.imshow('img', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# image_to_text = str(pytesseract.image_to_string('test.png'))
# Converting image to string
# config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'

image = cv2.imread('total.jpeg')
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)

# rotate the image to correct the orientation
rotated = imutils.rotate_bound(image, angle=results["rotate"])
# show the original image and output image after orientation
# correction
cv2.imshow("Original", image)
cv2.imshow("Output", rotated)
cv2.waitKey(0)

# config = r'--oem 3 --psm 6 outputbase digits'
# image_to_text = pytesseract.image_to_string(image, config=config)
# print(image_to_text)

