import cv2
import time


class Functions:
    import mediapipe as mp 
    def __init__(self,width,height,piCam):
        self.width = width
        self.height = height
        self.piCam = piCam
        self.tStart = time.time()
        self.fps = 0
        
        self.textLowerLeft = (int(self.width*.01),int(self.height*.05))
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontThickness = int(self.width/425)
        self.fontScale = self.height*.0015
        self.fontColor = (0,0,255)
        self.hands = self.mp.solutions.hands.Hands(
            model_complexity =  0,
            min_detection_confidence = .5,
            min_tracking_confidence = .5,
            max_num_hands =2)
        self.mpDraw = self.mp.solutions.drawing_utils
        self.handStyle = self.mpDraw.DrawingSpec(
            color = (100, 0, 255),
            thickness = 6,
            circle_radius = 8)
        self.connectionStyle = self.mpDraw.DrawingSpec(
                color = (0,255,0),
                thickness = 5)
        

    def buildFrame(self):  
        frame = self.piCam.capture_array()
        frame = cv2.flip(frame,1)
        return frame
        
    def frameSpeed(self,frame):
        deltaT = time.time() - self.tStart
        self.tStart=time.time()
        self.fps = self.fps*.95 + (1/deltaT)*.05
        myText = "FPS: "+str(round(self.fps,1))
        cv2.putText(frame,myText,
                    self.textLowerLeft,
                    self.fontFace,
                    self.fontScale,
                    self.fontColor,
                    self.fontThickness)
        
        
class ParseHandsGeometry(Functions):
    def parseHandsGeometry(self,frame): 
        rgbFrame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgbFrame)
        if results.multi_hand_landmarks:
            for handLM in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(
                    frame,
                    handLM,
                    self.mp.solutions.hands.HAND_CONNECTIONS,
                    self.handStyle,
                    self.connectionStyle)
        return results
    
class ParseHandsNumbers(Functions):
    def parseHandsNumbers(self,frame):
        rgbFrame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgbFrame)
        if results.multi_hand_landmarks:
            for handLM in results.multi_hand_landmarks:
                for index, landmark in enumerate(handLM.landmark):
                    pixelX = int(landmark.x*self.width)
                    pixelY = int(landmark.y*self.height)
                    cv2.putText(frame,str(index),
                                (pixelX,pixelY),
                                self.fontFace,
                                self.width*.0007,
                                (255,0,255),
                                self.fontThickness)
        return results
                    
