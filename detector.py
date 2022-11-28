import cv2
import mediapipe 
import time

class Detector:
    def __init__(self, mode=False) -> None:
        '''
            Initate all the required arguments for mediapipe
        '''
        self.mode = mode
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands(self.mode)
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.fingerTips = [4, 8, 12]

    def detect_hand(self, img):
        '''
            Detect hand from the given frame
            :input img: multi-dimensional image composed of BGR

            :return: img 
        '''
        # Convert image from BGR to RGB fomat
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)
        # Check if the processed image contains the multiple landmark for the hands
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                # Draw the landmarks on the image
                self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        
        return img

    def finger_position(self, img, hands_num=0):
        '''
            Determine the position of tip of thumb, index and middle fingers
            :input img: multi-dimensional image composed of BGR
            :input hands_num [int]: index for the landmark
            :input draw [bool]: if true, draw on the frame
            
            :return: list of detected landmarks
        '''
        self.landmark_list = []
        # Check if the results have landmarks of hand
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[hands_num]
            for id, lm in enumerate(myHand.landmark):
                # Get the size of frame
                h, w, c = img.shape

                # Convert the co-ordinate x, y for the detected point with respect to frame size
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.landmark_list.append([id, cx, cy])
        return self.landmark_list
    
    def is_finger_up(self):
        '''
            Detect whether combination of fingers are at upward direction

            :return: list of fingers pointing the top
        '''
        list_fingers = []
        if len(self.landmark_list) != 0:
            # Check for the thumb position
            if self.landmark_list[self.fingerTips[0]][1] < self.landmark_list[self.fingerTips[0] - 1][1]:
                list_fingers.append(1)
            else:
                list_fingers.append(0)

            # Check for other finger tips landmark
            for i in range(0, 3):
                if self.landmark_list[self.fingerTips[i]][2] < self.landmark_list[self.fingerTips[i] - 2][2]:
                    list_fingers.append(1)
                else:
                    list_fingers.append(0)
        return list_fingers

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while True:
        isTrue, frame = cap.read()
        
        # Detect hand and the position of the joints
        frame = detector.findHands(frame)
        landmark_list = detector.findPos(frame)

        # Check if fingers are positioned up
        f = detector.fingerUp()

        # Compute FPS
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xff == 27:
            break

    cv2.destroyAllWindows()
    cap.release()



