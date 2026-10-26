import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class Functions:
    tStart = time.time()
    fps = 0
    def __init__(self,width,height,piCam):
        self.width = width
        self.height = height
        self.piCam = piCam
        self.textLowerLeft = (int(self.width*.01),int(self.height*.05))
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontThickness = int(self.width/425)
        self.fontScale = self.height*.0015
        self.fontColor = (0,0,255)
        
    def buildFrame(self):  
        frame = self.piCam.capture_array()
        frame = cv2.flip(frame,1)
        return frame
    
    def frameSpeed(self,frame):
        deltaT = time.time() - Functions.tStart
        Functions.tStart = time.time()
        Functions.fps = Functions.fps*.95 + (1/deltaT)*.05
        myText = "FPS: "+str(round(Functions.fps,1))
        cv2.putText(frame,myText,
                    self.textLowerLeft,
                    self.fontFace,
                    self.fontScale,
                    self.fontColor,
                    self.fontThickness)
        
class DetectImage(Functions):
    
    
    def set_up_detectors(self):
        modelPath = "/home/pi/ai-lab-kit/mediapipe/efficientdet_lite0.tflite"
        scoreThreshold = .35
        maxObj = 3
        baseOptions = python.BaseOptions(model_asset_path = modelPath)
        options = vision.ObjectDetectorOptions(base_options = baseOptions,
                                               score_threshold=scoreThreshold,
                                               max_results=maxObj)
        myDetector = vision.ObjectDetector.create_from_options(options)
        return myDetector

        
    def detectImage(self,frame,myDetector):
        myImage = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
        myDetections = myDetector.detect(myImage)
        for detection in myDetections.detections:
            box = detection.bounding_box
            x = box.origin_x
            y = box.origin_y
            w = box.width
            h = box.height
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), self.fontThickness)
        if detection.categories:
            category = detection.categories[0]
            myLabel = category.category_name
            labelText = myLabel+" "+ str(round(category.score*100))+"%"
            cv2.putText(frame,labelText, (x, max(y-10,25)),self.fontFace,
                                          self.fontScale*.8,
                                          (0,255,0),
                                          2)

        return myImage