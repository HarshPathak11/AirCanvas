import cv2
import mediapipe as mp
import time
import handTrackerModule as htm
cTime=0
pTime=0
cap=cv2.VideoCapture(0)
detector=htm.HandDetector()
while True:
    success, img=cap.read()
    img=detector.detectHand(img)
    posList=detector.findPosition(img)
    if len(posList)!=0:
        print(posList[4])
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)