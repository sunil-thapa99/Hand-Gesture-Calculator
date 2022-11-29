# import cv2
# from tensorflow import keras
# import numpy as np

# import warnings
# warnings.filterwarnings('ignore')

# import os
# x = []
# y = []
# datadir = 'dataset'
# model = keras.models.load_model('digit_model.h5')

# for folder in os.listdir(datadir):
#     path = os.path.join(datadir, folder)
#     for images in os.listdir(path):
#         img = cv2.imread(os.path.join(path, images))
#         print(os.path.join(path, images))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = cv2.resize(img, (32, 32))
#         # x.append(img)
#         y.append(folder)
#         thresh = np.expand_dims(img, axis=0)
#         pred = model.predict(thresh)
#         ypred = np.argmax(pred, axis=1)
#         print(pred, ypred)
#         cv2.imshow('img', img)
#         cv2.waitKey(0)
#         break
        
# print(y)
# labels = ['div', 'mul', '9', '0', '7', '6', '1', '8', 'sub', 'add', '4', '3', '2', '5']



# # img = cv2.imread('3.jpeg')
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # thresh = cv2.resize(thresh, (32, 32))
# # thresh = np.expand_dims(thresh, axis=0)

# # model = keras.models.load_model('digit_model.h5')

# # for i in x:
# #     thresh = np.expand_dims(i, axis=0)
# #     pred = model.predict(thresh)
# #     ypred = np.argmax(pred, axis=1)
# #     print(pred, ypred)
# # cv2.imshow('test', thresh)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()



# ----------------------------------------

import cv2
import imutils
from imutils.contours import sort_contours
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
model = keras.models.load_model('digit_model.h5')

def test_pipeline_equation(image_path):
    chars = []
    img = cv2.imread(image_path)
    img = cv2.resize(img, (800, 800))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edged = cv2.Canny(img_gray, 30, 150)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sort_contours(contours, method="left-to-right")[0]
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub']

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if 20<=w and 30<=h:
            roi = img_gray[y:y+h, x:x+w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (th, tw) = thresh.shape
            if tw > th:
                thresh = imutils.resize(thresh, width=32)
            if th > tw:
                thresh = imutils.resize(thresh, height=32)
            (th, tw) = thresh.shape
            dx = int(max(0, 32 - tw)/2.0)
            dy = int(max(0, 32 - th) / 2.0)
            padded = cv2.copyMakeBorder(thresh, top=dy, bottom=dy, left=dx, right=dx, borderType=cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = np.array(padded)
            padded = padded/255.
            padded = np.expand_dims(padded, axis=0)
            padded = np.expand_dims(padded, axis=-1)
            pred = model.predict(padded)
            pred = np.argmax(pred, axis=1)
    #         print(pred)
            label = labels[pred[0]]
            chars.append(label)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, label, (x-5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    figure = plt.figure(figsize=(10, 10))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    e = ''
    for i in chars:
        if i=='add':
            e += '+'
        elif i=='sub':
            e += '-'
        elif i=='mul':
            e += '*'
        elif i=='div':
            e += '/'
        else:
            e += i
    v = eval(e)
    print('Value of the expression {} : {}'.format(e, v)) 

test_pipeline_equation('test_equation4.jpg')
