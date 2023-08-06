import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for eachHand in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, eachHand, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        landmarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, Landmark in enumerate(myHand.landmark):
                height, width, channels = img.shape
                cx, cy = int(Landmark.x * width), int(Landmark.y * height)
                landmarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return landmarkList

def main():
    video = cv2.VideoCapture(0)
    previousTime = 0
    currentTime = 0
    detect = HandDetector()
    while True:
        ret, img = video.read()
        img = detect.findHands(img)
        landmarklist = detect.findPosition(img)

        if len(landmarklist) != 0 :
            print(landmarklist[6])
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
