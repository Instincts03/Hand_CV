# %%
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__ (self,mode=False,maxHands=2,model_complexity = 1,detectionCon=0.5,trackCon=0.5): 
        self.mode=mode
        self.maxHands=maxHands
        self.model_complexity = model_complexity
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.model_complexity,self.detectionCon,self.trackCon)
        self.mpDraw=mp.solutions.drawing_utils
        

    def findHands(self,img,draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img , handLms ,self.mpHands.HAND_CONNECTIONS)  
                    #This displays the dots and connection on image

        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmlist=[]
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):   #print inbuilt ids for each hand-posture
                #print(id,lm)
                h ,w ,c=img.shape                           # height,width,channel
                cx,cy= int(lm.x *w) , int(lm.y*h)           #middle points of img
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)     #highlights paticular lm, here id=0
        return lmlist
    
    
    
def main():
    pTime=0
    cTime=0
    cap=cv2.VideoCapture(0)
    detector=handDetector()

    while True:
        sucess, img= cap.read()
        img=detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist)!=0:
            print(lmlist[4])
        
        cTime=time.time()       #displaying the fps
        fps=1/(cTime-pTime)
        pTime=cTime

        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
        cv2.imshow("Image" , img)
        cv2.waitKey(1)
    
    
    
    
if __name__ == "__main__":
    main()


# %%
#in this file we updated the min file , in the sense we will not use the in built class, we will make our own class
#its 640*480 image


