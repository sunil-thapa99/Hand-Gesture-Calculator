import cv2

def segment_digits(img):
    # Convert the canvas to grayscale image
    grayCanvas = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert the values 0-255 to either 0 or 255 making it black or white instead of gray
    _, invCanvas = cv2.threshold(grayCanvas, 48, 255, cv2.THRESH_BINARY_INV)

    invCanvas = cv2.cvtColor(invCanvas, cv2.COLOR_GRAY2BGR)
    cv2.imshow('threshold image', invCanvas)
    cv2.waitKey(0)


img = cv2.imread('sample1.JPG')
segment_digits(img)