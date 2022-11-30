import cv2
import time
import numpy as np
import math
import detector as dt

from test import test_pipeline_equation

pTime = 0
cTime = 0

# Get video from webcam
cap = cv2.VideoCapture(0)
# Set width and height of video frame
cap.set(3, 640)
cap.set(4, 360)

# Create a black canvas 
imgCanvas = np.zeros((360, 640, 3), np.uint8)


hand_detector = dt.Detector(mode=False)

# Define some needy global variables
temp = False
x1, y1, x2, y2 = 0, 0, 0, 0
prevx, prevy = 0, 0
is_calculator = False
count = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (640, 360))
    h, w, _ = img.shape
    # Flip the image vertically
    img = cv2.flip(img, 1)

    # Button for calculation
    cv2.rectangle(img, (450, 0), (610, 30), (0, 0, 0), -1)
    cv2.putText(img, "Enable Calculation", (460, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 250, 127), 2)
    key = cv2.waitKey(1)

    # Detect hand from the frame
    img = hand_detector.detect_hand(img)
    landmarks = hand_detector.finger_position(img)

    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[4][1:]
        x3, y3 = landmarks[12][1:]
        x1, y1, x2, y2, x3, y3 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)
        
        # Check if any finger positions are pointed upwards
        finger_list = hand_detector.is_finger_up()

        # Check if there are two fingers pointing upward
        if finger_list[1] and finger_list[2]:
            # prevx, prevy = 0, 0
            if y1 < 150:
                if 500 < x1 < 600:
                    is_calculator = True
        print(finger_list)
        if (finger_list[3] and finger_list[1] and finger_list[2] ):
            count = 0
        #     cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
        #     if prevx == 0 and prevy == 0:
        #         prevx, prevy = x1, y1

        #     # Draw a line on image and canvas while the marker is moving
        #     cv2.line(img, (prevx, prevy), (x1, y1), (0, 0, 255), 5)
        #     cv2.line(imgCanvas, (prevx, prevy), (x1, y1), (0, 0, 255), 5)
        #     prevx, prevy = x1, y1
        # else:
        #     prevx, prevy = 0, 0

        # # Calculate the distance between two circumference points
        distance = math.hypot(x2 - x1, y2 - y1)
        # cv2.putText(img, str(distance),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),1)

        # Check if the tip of thumb and index fingers are close
        # print(distance)
        if distance < 15:
            temp = True
        else:
            temp = False
            prevx, prevy = 0, 0
        if temp:
            # Get the center of the circumference of tip of index and thumb
            x, y = int((x1+x2)/2), int((y1+y2)/2)
            
            # Insert a circle to show marker
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
            if prevx == 0 and prevy == 0:
                prevx, prevy = x, y

            # Draw a line on image and canvas while the marker is moving
            cv2.line(img, (prevx, prevy), (x, y), (0, 0, 255), 3)
            cv2.line(imgCanvas, (prevx, prevy), (x, y), (0, 0, 255), 3)
            prevx, prevy = x, y
    
    if is_calculator and count == 0:
        calculation = test_pipeline_equation(invCanvas)
        is_calculator = False
        cv2.putText(imgCanvas, f'{calculation[0]} = {calculation[1]}',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255),2)
        count += 1
    if count == 0:
        cv2.rectangle(imgCanvas, (10, 30), (110, 70), (0, 0, 0), -1)

    # Convert the canvas to grayscale image
    grayCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # Convert the values 0-255 to either 0 or 255 making it black or white instead of gray
    _, invCanvas = cv2.threshold(grayCanvas, 48,255, cv2.THRESH_BINARY_INV)

    # Convert the canvas back to BGR 
    invCanvas = cv2.cvtColor(invCanvas, cv2.COLOR_GRAY2BGR)

    # Use bitwise and and or to append the image and the canvas
    img = cv2.bitwise_and(img, invCanvas)
    img = cv2.bitwise_or(img, imgCanvas)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime

    # cv2.putText(img, str(int(fps)),(10,130), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),2)
    
    cv2.imshow("img", img)
    cv2.imshow("invCanvas", invCanvas)
    cv2.imshow('test', imgCanvas)
    # Esc to exit the loop and break the frame
    if key & 0xFF == 99:
        imgCanvas = np.zeros((360, 640, 3), np.uint8)
    if key & 0xFF == 27:
        break

cv2.destroyAllWindows()
cap.release()