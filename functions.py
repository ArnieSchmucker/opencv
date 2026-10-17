import cv2
import time

class Functions:
    import mediapipe as mp
    def __init__(self,W,H,piCam):
        self.width = W
        self.height = H
        self.piCam = piCam
        self.tStart = time.time()
        self.fps = 0
        self.textLowerLeft = (int(W*.01),int(H*.05))
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontThickness = int(W/425)
        self.fontScale = H*.0015
        self.fontColor = (0,0,255)
        self.handFinder = self.mp.solutions.hands.Hands(
            model_complexity =  0,
            min_detection_confidence = .5,
            min_tracking_confidence = .5,
            max_num_hands =2)
        self.mpDraw = self.mp.solutions.drawing_utils
        
    def buildFrame(self):
        self.frame= self.piCam.capture_array()
        #self.frame=cv2.flip(frame,-1)
        self.frame = cv2.flip(self.frame,1)
        #cv2.imshow("Camera", frame)
        #cv2.moveWindow("Camera",0,60)
        return self.frame

    def frameSpeed(self):
        self.deltaT = time.time() - self.tStart
        self.tStart = time.time()
        self.fps = self.fps*.95 + (1/self.deltaT)*.05
        myText = "FPS: "+str(round(self.fps,1))
        cv2.putText(self.frame,myText,self.textLowerLeft,
                    self.fontFace,self.fontScale,
                    self.fontColor,self.fontThickness)
    
class ParseHandsGeometry(Functions):
    def parseHandsGeometry(self):
        rgbFrame= cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.handFinder.process(rgbFrame)
        if results.multi_hand_landmarks:
            for handLM in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    self.frame,
                    handLM,
                    self.mp.solutions.hands.HAND_CONNECTIONS,
                    self.mpDraw.DrawingSpec(color=(100, 0, 255),thickness=6,circle_radius=8),
                    self.mpDraw.DrawingSpec(color=(0,255,0),thickness=5))
        return results
         

class ParseHandsNumbers(Functions):
    def parseHandsNumbers(self):
        rgbFrame= cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.handFinder.process(rgbFrame)
        if results.multi_hand_landmarks:
            for handLM in results.multi_hand_landmarks:
                for index, landmark in enumerate(handLM.landmark):
                    pixelX = int(landmark.x*self.width)
                    pixelY = int(landmark.y*self.height)
                    cv2.putText(
                        self.frame,str(index),(pixelX,pixelY),
                        cv2.FONT_HERSHEY_SIMPLEX, self.width*.0007,(255,0,255),int(self.width/425),)
        return results