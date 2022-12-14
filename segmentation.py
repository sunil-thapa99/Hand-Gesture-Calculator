import numpy as np
import cv2 as cv
from tensorflow import keras
model = keras.models.load_model('digit_model.h5')
def segment(heighty, widthx, img):
    chars = []
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub']
    # c = []
    i = 0
    flag_dot = 0
    i_low = 0
    i_high = 0
    j_left = 0
    j_right = 0
    i_nhigh = 0
    i_nlow = 0

    while (i <= heighty - 1):
        count_dot = 0
        for j in range(0, widthx - 1):
            avalue = np.mean(img[i, j])
            if (avalue == 0):
                count_dot = count_dot + 1

        if count_dot >= 4:
            if flag_dot == 0:
                flag_dot = 1
                i_high = i - 1
        else:
            if flag_dot == 1:
                flag_dot = 0
                i_low = i
            else:
                flag_dot = 0

        if ((i_low != 0) & (i_high != 0)):
            flag_dots = 0
            for line_j in range(0, widthx - 1):
                # lj = line_j
                count_cr = 0
                for line_i in range(i_high, i_low):
                    avalue1 = np.mean(img[line_i, line_j])
                    if (avalue1 == 0):
                        count_cr = count_cr + 1

                if count_cr >= 1:
                    if flag_dots == 0:
                        flag_dots = 1
                        j_left = line_j - 1

                else:
                    if flag_dots == 1:
                        flag_dots = 0
                        j_right = line_j
                    else:
                        flag_dots = 0

                if ((j_left != 0) & (j_right != 0)):
                    flag_dotd = 0
                    for v_i in range(i_high, i_low + 1):
                        count_cv = 0
                        for h_j in range(j_left, j_right + 1):
                            avalue2 = np.mean(img[v_i, h_j])
                            if (avalue2 == 0):
                                count_cv = count_cv + 1
                        if count_cv >= 1:
                            if flag_dotd == 0:
                                flag_dotd = 1
                                i_nhigh = v_i
                                for drawn_j in range(j_left, j_right + 1):
                                    img[v_i - 1, drawn_j] = 255
                            else:
                                img[v_i - 1, j_left] = 255
                                img[v_i - 1, j_right] = 255


                        else:
                            if flag_dotd == 1:
                                flag_dotd = 0
                                i_nlow = v_i
                                for drawn_j in range(j_left, j_right + 1):
                                    img[v_i, drawn_j] = 255

                                img[v_i - 1, j_left] = 255
                                img[v_i - 1, j_right] = 255


                    hs = abs(i_nlow - i_nhigh)
                    ws = abs(j_right - j_left)
                    img1 = np.zeros((hs, ws, 1), np.uint8)
                    img1[:] = (255)
                    for i_s in range(1, hs):
                        for j_s in range(1, ws):
                            img1[i_s, j_s] = img[i_nhigh + i_s, j_left + j_s]

                    # cv.imshow("before resize image", img1)
                    # cv.waitKey(1000)
                    # img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
                    # img1 = cv.resize(img1, (32, 32), interpolation=cv.INTER_CUBIC)


                    img1 = cv.resize(img1, (28, 28))
                    img1 = cv.threshold(img1, 128, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
                    # cv.imshow("after resize image", img1)
                    # cv.waitKey(1000)
                    img1 = cv.copyMakeBorder(img1, 4, 4, 4, 4, cv.BORDER_CONSTANT, value=[0, 0, 0])
                    img1 = cv.resize(img1, (32, 32), interpolation=cv.INTER_CUBIC)

                    # cv.imshow("after padding image", img1)
                    # cv.waitKey(1000)
                    padded = np.array(img1)
                    padded = padded / 255.
                    padded = np.expand_dims(padded, axis=0)
                    padded = np.expand_dims(padded, axis=-1)
                    print("shape of padded:", padded.shape)
                    pred = model.predict(padded)
                    pred = np.argmax(pred, axis=1)

                    label = labels[pred[0]]
                    print("Prediction values:", label)
                    chars.append(label)

                    j_right = 0
                    j_left = 0

            i_high = 0
            i_low = 0
        i = i + 1

    e = ''
    for i in chars:
        if i == 'add':
            e += '+'
        elif i == 'sub':
            e += '-'
        elif i == 'mul':
            e += '*'
        elif i == 'div':
            e += '/'
        else:
            e += i
    try:
        v = eval(e)
    except:
        v = ''
    print('Value of the expression {} : {}'.format(e, v))
    return (e, v)

def segment_digits(img):
    # Convert the canvas to grayscale image
    grayCanvas = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # kernel = np.ones((3, 3), np.uint8)
    # img1 = cv.dilate(grayCanvas, kernel, iterations=1)

    # Convert the values 0-255 to either 0 or 255 making it black or white instead of gray
    _, invCanvas = cv.threshold(grayCanvas, 48, 255, cv.THRESH_BINARY)

    height, weight = invCanvas.shape
    images = segment(height, weight, invCanvas)
    return images

#
# from tensorflow import keras
# model = keras.models.load_model('digit_model.h5')
#
# img = cv.imread('i.jpeg')
# imgs = segment_digits(img)
# # for ig in imgs:
# #     cv.imshow('img', ig)
# #     cv.waitKey(0)
#
# labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'div', 'mul', 'sub']
# for padded in imgs:
#     cv.imshow('img', padded)
#     cv.waitKey(0)
#     padded = np.array(padded)
#     padded = padded/255.
#     padded = np.expand_dims(padded, axis=0)
#     padded = np.expand_dims(padded, axis=-1)
#     # img = np.expand_dims(img, axis=0)
#     pred = model.predict(padded)
#     pred = np.argmax(pred, axis=1)
#     label = labels[pred[0]]
#     print(pred, label)
#
# cv.destroyAllWindows()
if __name__ == '__main__':
    sample_img = cv.imread("i.jpeg")
    # cv.imshow("before resize image", sample_img)
    # cv.waitKey(1000)
    result = segment_digits(sample_img)
    print("result:", result)