import cv2
import time
import numpy as np
import math
import detector as dt

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

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (640, 360))
    h, w, _ = img.shape
    # Flip the image vertically
    img = cv2.flip(img, 1)
    key = cv2.waitKey(1)

    # Detect hand from the frame
    img = hand_detector.detect_hand(img)
    landmarks = hand_detector.finger_position(img)

    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[4][1:]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Check if any finger positions are pointed upwards
        finger_list = hand_detector.is_finger_up()
        

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.putText(img, 'Press S to start/stop',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),1)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            # if cv2.waitKey(1) == ord('p'):
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)
            x1, y1 = handlms.landmark[8].x*w, handlms.landmark[8].y*h
            x2, y2 = handlms.landmark[4].x*w, handlms.landmark[4].y*h
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        distance = math.hypot(x2 - x1, y2 - y1)
        # cv2.putText(img, str(distance),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,255),1)
        if distance > 15 and distance < 40:
            temp = True
        else:
            temp = False
            prevx, prevy = 0, 0
        if temp:
            x, y = int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
            if prevx == 0 and prevy == 0:
                prevx, prevy = x, y
            cv2.line(img, (prevx, prevy), (x, y), (0, 0, 255), 5)
            cv2.line(imgCanvas, (prevx, prevy), (x, y), (0, 0, 255), 5)
            prevx, prevy = x, y

    grayCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, invCanvas = cv2.threshold(grayCanvas, 48,255, cv2.THRESH_BINARY_INV)
    invCanvas = cv2.cvtColor(invCanvas, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, invCanvas)
    img = cv2.bitwise_or(img, imgCanvas)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime

    # cv2.putText(img, str(int(fps)),(10,130), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),2)

    cv2.imshow("img", img)
    if key & 0xFF == 27:
            break

cap.release()