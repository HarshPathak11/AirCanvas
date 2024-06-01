import cv2
import numpy as np
import os
import handTrackerModule as htm
import time

folderPath="header"
myList=os.listdir(folderPath)
# print(myList)

drawColor=(0,0,255)
brushThickness=7
eraserThickness=150
overlayList=[]

for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header=overlayList[0]

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

xp=0
yp=0
detector=htm.HandDetector(detectionCon=0.95)

imgCanvas= np.zeros((720,1280,3),np.uint8)

while True:
    success, img=cap.read()
    # flipping the image so that it becomes easier to draw
    img =cv2.flip(img,1)

    #Merging the two Canvas
    imgGray=cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv= cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)

    img=cv2.bitwise_and(img,imgInv)
    img=cv2.bitwise_or(img,imgCanvas)
    
    #Setting the default header image
    img[0:125,0:1280]=header
    # img=cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # Steps:
    # 1. Find Hand LandMarks
    img=detector.detectHand(img)
    lmlist=detector.findPosition(img,Draw=False)

    if len(lmlist)!=0:
        # print(lmlist)
        # tip of middle and index finger
        x1,y1=lmlist[8][1:]
        x2,y2=lmlist[12][1:]

        fingers=detector.fingersUp()
        # print(fingers)

    # 2. Check Which Fingers are Up
    # 3. Determine, if its Draw Mode, if one finger(index raised), or Selection Mode , if two fingers(index, middle) are raised
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            if y1<125:
                if 250<x1<450:
                    header=overlayList[0]
                    drawColor=(0,0,255)
                elif 550<x1<750:
                    header=overlayList[1]
                    drawColor=(255,0,0)
                if 800<x1<950:
                    header=overlayList[2]
                    drawColor=(0,255,255)
                if 1050<x1<1200:
                    header=overlayList[3]
                    drawColor=(0,0,0)
            cv2.rectangle(img,(x1,y1-20),(x2,y2+20),drawColor,cv2.FILLED)
            



        if fingers[1] and fingers[2]==False:
            
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)  
            else: 
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1


    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)

    cv2.waitKey(1)