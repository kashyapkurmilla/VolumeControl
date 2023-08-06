import cv2
import time
import numpy as np
import handtrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

camWidth, camHeight = 640, 480
video = cv2.VideoCapture(0)
video.set(3, camWidth)
video.set(4, camHeight)
currentTime = 0
previousTime = 0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volumeRange = volume.GetVolumeRange()
print(volumeRange)
minVol = volumeRange[0]
maxVol = volumeRange[1]

vol = 0
volBar = 400
volPer = 0

detector = htm.HandDetector()
while True:
    ret, img = video.read()
    img = detector.findHands(img)
    # flipFrame = cv2.flip(ret, 1)

    landmarkList = detector.findPosition(img, draw=False)
    # if len(landmarkList) != 0:  # making sure that there are values present in landmark list
    #     print(landmarkList[4], landmarkList[8])
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    if len(landmarkList) != 0:
        x4, y4 = landmarkList[4][1], landmarkList[4][2]
        x8, y8 = landmarkList[8][1], landmarkList[8][2]
        cx, cy = (x4 + x8) // 2, (y4 + y8) // 2

        cv2.circle(img, (x4, y4), 8, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x8, y8), 8, (255, 0, 255), cv2.FILLED)

        cv2.line(img, (x4, y4), (x8, y8), (255, 0, 255), 2)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x8 - x4, y8 - y4)
        # print(length)

        vol = np.interp(length, [21, 140], [minVol, maxVol])
        volBar = np.interp(length, [21, 140], [400, 150])
        volPer = np.interp(length, [21, 140], [0, 100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 21:
            cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)
    cv2.putText(img,
                f'fps:{str(int(fps))}',
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (189, 33, 255),  # color
                2)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
