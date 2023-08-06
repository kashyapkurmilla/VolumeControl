import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)

previousTime = 0
currentTime = 0

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils #to connect the dots
hands = mpHands.Hands() # default parameter no need to enter any parameters


while True:
    ret,img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks :
        for eachHand in results.multi_hand_landmarks:
            for id, Landmark in enumerate(eachHand.landmark):
                #print(id,Landmark)
                height , width , channels = img.shape
                cx , cy = int(Landmark.x * width),int(Landmark.y * height)
                #print(id,cx,cy)
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img,eachHand,mpHands.HAND_CONNECTIONS)

    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    # print(fps)
    previousTime = currentTime
    cv2.putText(img,
                str(int(fps)),
                (10,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (189,33,255),#color
                3)

    cv2.imshow("image", img)
    cv2.waitKey(1)