import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpHands= mp.solutions.hands
        self.mpDraw= mp.solutions.drawing_utils
        self.hands= self.mpHands.Hands(self.mode,self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)

        self.tipIds=[4,8,12,16,20]

    def detectHand(self,img, draw=True):
        imageRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results= self.hands.process(imageRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hnd in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(img,hnd,self.mpHands.HAND_CONNECTIONS)
        return img   

    def findPosition(self,img,handNo=0,Draw=True):
        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand=self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    self.lmlist.append([id,cx,cy])
                    if Draw:
                        cv2.circle(img,(cx,cy),10,(25,243,32),cv2.FILLED)
        return self.lmlist               

    def fingersUp(self):
        fingers=[]
        #thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 4 fingers

        for id in range(1,5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cTime=0
    pTime=0
    cap=cv2.VideoCapture(0)
    detector=HandDetector()
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


if __name__ == "__main__":
    main()