# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
# sample_image = cv2.imread('test1.jpg')
# img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
# # plt.imshow(img)
# cv2.imshow("before segmentation", img)
# cv2.waitKey(1000)
#
# twoDimage = img.reshape((-1,3))
# twoDimage = np.float32(twoDimage)
#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 3
# attempts=10
#
# ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv.KMEANS_RANDOM_CENTERS)
# print("label:", label)
# center = np.uint8(center)
# print("center:",center)
# res = center[label.flatten()]
# print("res:",res)
# result_image = res.reshape((img.shape))
# #
# # plt.axis('off')
# # plt.imshow(result_image)
#
# cv2.imshow("after segmentation", result_image)
# cv2.waitKey(8000)

#
# import numpy as np
# # import matplotlib.pyplot as plt
# from skimage.filters import threshold_otsu
# import cv2
#
# sample_image = cv2.imread('test1.jpg')
# img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)
#
# cv2.imshow("before segmentation", img)
# cv2.waitKey(1000)
#
# img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#
# thresh = threshold_otsu(img_gray)
# img_otsu  = img_gray < thresh
#
# def filter_image(image, mask):
#
#     r = image[:,:,0] * mask
#     g = image[:,:,1] * mask
#     b = image[:,:,2] * mask
#
#     return np.dstack([r,g,b])
#
# filtered = filter_image(img, img_otsu)
#
# cv2.imshow("after segmentation", img)
# cv2.waitKey(8000)

import cv2

# Read image from which text needs to be extracted
img = cv2.imread("test1.jpg")

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)

# Creating a copy of image
im2 = img.copy()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # invCanvas = cv2.cvtColor(rect, cv2.COLOR_GRAY2BGR)

    # Use bitwise and and or to append the image and the canvas
    img = cv2.bitwise_and(img, rect)


cv2.imshow("after segmentation", img)
cv2.waitKey(8000)