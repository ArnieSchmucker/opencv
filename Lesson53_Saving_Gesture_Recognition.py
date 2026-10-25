# ====================================================================
# DISCLAIMER:
# This code is provided as-is for educational and experimental 
# purposes only. The author makes no representations or warranties of 
# any kind concerning the safety, suitability, or accuracy of this 
# code. Use at your own risk. The author assumes no liability for any 
# damages, system failures, security breaches, or network issues 
# resulting from the use or implementation of this script.
# ====================================================================

#####Not finished##############

import cv2
import time
from picamera2 import Picamera2
from functions import ParseHandsNumbers
from training_4 import Files,TipDistance,FindGesture
#import mediapipe as mp


W=1280
H=720

piCam = Picamera2(0)
piCam.preview_configuration.main.size = (W,H)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.controls.FrameRate=60
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()


    
DATA_FILE = 'gesture_data.pkl'
files = Files(DATA_FILE)

#f = ParseHandsGeometry(W,H,piCam)
f = ParseHandsNumbers(W,H,piCam)

#trainIndex = 0
trainedData,trainIndex,gestureNames = files.loadTrainedData()


#gestureNames = []

if not trainedData:
    print("HAND GESTURE TRAINING SETUP")
    numGestures = int(input("Enter Number of Gestures to Train On: "))
    gestureNames = []
    for index in range(numGestures):
        name = input("Enter Name for Gesture #" + str(index+1) + ": ")
        gestureNames.append(name)

while True:
    frame = f.buildFrame()
    fps = f.frameSpeed()    
    results = f.parseHandsNumbers()         
    if trainIndex < len(gestureNames):
        targetName = gestureNames[trainIndex]
        promptText = "Show " + targetName + " & Press Space"
        cv2.putText(frame,promptText,
                   (int(W*.01),int(H*.21)),
                     cv2.FONT_HERSHEY_SIMPLEX,
                   H*.0015,(0,255,255),int(W/425))
        
    detectedGesture = "Unknown"
    if results.multi_hand_landmarks:
        dist = TipDistance()
        liveDistances = dist.getTipDistances(results.multi_hand_landmarks[0])
        gest = FindGesture(liveDistances,trainedData,3.5)
        detectedGesture = gest.findGesture()
    labelText = "Gesture: " + detectedGesture
    cv2.putText(frame,labelText,(int(W*.01),int(H*.14)),cv2.FONT_HERSHEY_SIMPLEX,H*.0015,(0,255,0),int(W/425))
    cv2.imshow("Camera", frame)
    cv2.moveWindow("Camera",0,60)

    key = cv2.waitKey(1)
    if key == ord('t'):
        newGestureName = input('Enter name for new gesture: ')
        gestureNames.append(newGestureName)
        files.saveTrainedData(trainedData)
    if key == ord('d'):
        delName = input('Enter Name of Gesture to Remove: ')
        if delName not in trainedData:
            print('[WARNING] Gesture '+ delName + 'Not found')
        if delName in trainedData:
            del trainedData[delName]
            files.saveTrainedData(trainedData)
            print('[INFO] Removed '+ delName)
            if delName in gestureNames:
                gestureNames.remove(delName)
            trainIndex = min(trainIndex,len(gestureNames))
    if key == ord(' '):
        if results.multi_hand_landmarks:
            #print(results.multi_hand_landmarks)
            targetHand = results.multi_hand_landmarks[0]
            #print(targetHand)
            trainedData[targetName] = dist.getTipDistances(targetHand)
            trainIndex = trainIndex + 1
    if key == ord('q'):
        break
cv2.destroyAllWindows()
files.saveTrainedData(trainedData)
print('Program Terminated')
