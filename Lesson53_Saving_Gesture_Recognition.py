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
from functions import ParseHandsNumbers,ParseHandsGeometry
#import mediapipe as mp
import math
import os
import pickle

W=1280
H=720

piCam = Picamera2(0)
piCam.preview_configuration.main.size = (W,H)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.controls.FrameRate=60
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

def saveTrainedData(filename,data):
    with open(filename,'wb') as f:
        pickle.dump(data,f)
    print('[INFO] Save Data to '+str(filename))
        
def loadTrainedData(filename):
    if os.path.exists(filename):
        with open(filename,'rb') as f:
            data = pickle.load(f)
            print('[INFO] Loaded '+str(len(data))+' gestures from '+str(filename))
            return data
    if not os.path.exists(filename):
        return{}
    
def findDistance(pt1,pt2):
    dX = pt1.x - pt2.x
    dY = pt1.y - pt2.y
    dZ = pt1.z - pt2.z
    dist = math.sqrt(dX*dX + dY*dY + dZ*dZ)
    return dist

def getTipDistances(handLM):
    lm = handLM.landmark
    wrist = lm[0]
    scaleFactor = findDistance(wrist,lm[5])
    tips = [lm[4],lm[8],lm[12],lm[16],lm[20]]
    tipDistances = []
    #calculate distance between tips and wrist
    for tip in tips: 
        dist = findDistance(wrist,tip)/scaleFactor
        tipDistances.append(dist)
    #calculate distance between fingers
    for startTip in range(len(tips)):
        for endTip in range(startTip+1,len(tips)):
            dist = findDistance(tips[startTip],tips[endTip])/scaleFactor
            tipDistances.append(dist)
    return tipDistances

def calculateError(tipDist, trainDist):
    totalError = 0
    for index in range(len(tipDist)):
        dif = tipDist[index] - trainDist[index]
        totalError = totalError + abs(dif)
    return totalError

def findGesture(tipLive, trainedData, maxAllowedError):
    bestMatch = "Unknown"
    lowestError = 99999999
    for gestureName, trainedDist in trainedData.items():
        error = calculateError(tipLive, trainedDist)
        if error < lowestError:
            lowestError = error
            bestMatch = gestureName
    if lowestError > maxAllowedError:
        return "Unknown"
    if lowestError <= maxAllowedError:
        return bestMatch
    
DATA_FILE = 'gesture_data.pkl'
trainIndex = 0
trainedData = loadTrainedData(DATA_FILE)
gestureNames = []

if not trainedData:
    print("HAND GESTURE TRAINING SETUP")
    numGestures = int(input("Enter Number of Gestures to Train On: "))
    gestureNames = []
    for index in range(numGestures):
        name = input("Enter Name for Gesture #" + str(index+1) + ": ")
        gestureNames.append(name)

#f = ParseHandsGeometry(W,H,piCam)
f = ParseHandsNumbers(W,H,piCam)

while True:
    frame = f.buildFrame()
    fps = f.frameSpeed()
    #rgbFrame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    
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
        liveDistances = getTipDistances(results.multi_hand_landmarks[0])
        detectedGesture = findGesture(liveDistances,trainedData,3.5)
    labelText = "Gesture: " + detectedGesture
    cv2.putText(frame,labelText,(int(W*.01),int(H*.14)),cv2.FONT_HERSHEY_SIMPLEX,H*.0015,(0,255,0),int(W/425))
    cv2.imshow("Camera", frame)
    cv2.moveWindow("Camera",0,60)

    key = cv2.waitKey(1)
    if key == ord('t'):
        newGestureName = input('Enter name for new gesture: ')
        gestureNames.append(newGestureName)
        saveTrainedData(DATA_FILE,trainedData)
    if key == ord('d'):
        delName = input('Enter Name of Gesture to Remove: ')
        if delName not in trainedData:
            print('[WARNING] Gesture '+ delName + 'Not found')
        if delName in trainedData:
            del trainedData[delName]
            saveTrainedData(DATA_FILE,trainedData)
            print('[INFO] Removed '+ delName)
            if delName in gestureNames:
                gestureNames.remove(delName)
            trainIndex = min(trainIndex,len(gestureNames))
    if key == ord(' '):
        if results.multi_hand_landmarks:
            #print(results.multi_hand_landmarks)
            targetHand = results.multi_hand_landmarks[0]
            #print(targetHand)
            trainedData[targetName] = getTipDistances(targetHand)
            trainIndex = trainIndex + 1
    if key == ord('q'):
        break
cv2.destroyAllWindows()
saveTrainedData(DATA_FILE,trainedData)
print('Program Terminated')
