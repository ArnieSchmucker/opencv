# ====================================================================
# DISCLAIMER:
# This code is provided as-is for educational and experimental 
# purposes only. The author makes no representations or warranties of 
# any kind concerning the safety, suitability, or accuracy of this 
# code. Use at your own risk. The author assumes no liability for any 
# damages, system failures, security breaches, or network issues 
# resulting from the use or implementation of this script.
# ====================================================================
 
import cv2
import time
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

piCam = Picamera2(0)
W=1280
H=720
tStart = time.time()
fps = 0
RES = (W,H)
piCam.preview_configuration.main.size = RES
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.controls.FrameRate=60
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

textLowerLeft = (int(W*.01),int(H*.05))
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontThickness = int(W/425)
fontScale = H*.0015
fontColor = (0,0,255)
topBar = 65
windowPadding = 0
cv2.namedWindow('Camera', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Camera', W, H)
cv2.moveWindow('Camera', windowPadding, topBar)
tSettle = .1
time.sleep(tSettle)


modelPath = "/home/pi/ai-lab-kit/mediapipe/efficientdet_lite0.tflite"
scoreThreshold = .35
maxObj = 3

baseOptions = python.BaseOptions(model_asset_path = modelPath)
options = vision.ObjectDetectorOptions(base_options = baseOptions,
                                       score_threshold=scoreThreshold,
                                       max_results=maxObj)
myDetector = vision.ObjectDetector.create_from_options(options)

while True:
    deltaT = time.time() - tStart
    tStart=time.time()
    fps = fps*.95 + (1/deltaT)*.05
    frame= piCam.capture_array()
    #frame=cv2.flip(frame,-1)
    myImage = mp.Image(image_format = mp.ImageFormat.SRGB, data=frame)
    myDetections = myDetector.detect(myImage)
    #print(myDetections.detections)
    cv2.rectangle(frame,(540,260),(740,460),(0,0,255),3)
    for detection in myDetections.detections:
        box = detection.bounding_box
        x = box.origin_x
        y = box.origin_y
        w = box.width
        h = box.height
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), fontThickness)


    if detection.categories:
        category = detection.categories[0]
        myLabel = category.category_name
        labelText = myLabel+" "+ str(round(category.score*100))+"%"
        cv2.putText(frame,labelText, (x, max(y-10,25)),fontFace,
                                      fontScale*.8,
                                      (0,255,0),
                                      2)
        if detection.categories[0].category_name == 'bottle':
            print('bottle detected',x,y)
            if x >= 540 and x<=740 and y >= 260 and y <= 460 :
                print("In ROI")
            else:
                print("Not in ROI")
        
                                          
    myText = "FPS: "+str(round(fps,1))
   
    cv2.putText(frame,myText,textLowerLeft,fontFace,fontScale,fontColor,fontThickness)
    cv2.imshow("Camera", frame)
    #cv2.moveWindow("Camera",0,60)
    if cv2.waitKey(1)==ord('q'):
        break
myDetector.close()
cv2.destroyAllWindows()
print('Program Terminated')