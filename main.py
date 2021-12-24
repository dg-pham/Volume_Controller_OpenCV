import math
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import cv2
import hand

# import numpy as np

cap = cv2.VideoCapture(0)

pTime = 0

detector = hand.handDetector(detectionCon=0.55)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volume.GetMute()

minVol = volume.GetVolumeRange()[0]
maxVol = volume.GetVolumeRange()[0] + 45

fingerID = [4, 8, 12, 16, 20]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame, draw=False)

    fingers = []

    # Algorithm
    if len(lmlist) != 0:
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        x3, y3 = (x1 + x2) // 2, (y1 + y2) // 2
        d = math.hypot(x2 - x1, y2 - y1)
        # 35 -> 360

        cv2.circle(frame, (x1, y1), 10, (203, 192, 255), -1)
        cv2.circle(frame, (x2, y2), 10, (203, 192, 255), -1)
        cv2.circle(frame, (x3, y3), 10, (203, 192, 255), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (203, 192, 255), 4)

        vol = np.interp(d, [35, 360], [minVol, maxVol])
        volBar = int(np.interp(d, [35, 360], [400, 150]))
        vol_ratio = int(np.interp(d, [35, 360], [0, 25]))
        volume.SetMasterVolumeLevel(vol, None)
        if d < 35:
            cv2.circle(frame, (x3, y3), 10, (0, 0, 0), -1)

        cv2.rectangle(frame, (50, 150), (100, 400), (0, 0, 0), 2)
        cv2.rectangle(frame, (50, volBar), (100, 400), (0, 0, 0), -1)
        cv2.putText(frame, str(vol_ratio), (55, 115), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (150, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 192, 0), 2)

    cv2.imshow('', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
