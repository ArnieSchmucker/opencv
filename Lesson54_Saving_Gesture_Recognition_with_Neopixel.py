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


# ====================================================================
# NEOPIXEL CONTROL INITIALIZATION (Adafruit neopixel_spi via SPI)
# ====================================================================

import board
import neopixel_spi as neopixel

NUM_LEDS = 12  # Set to match your NeoPixel ring size (e.g., 12)
PO = neopixel.GRB

# Initialize NeoPixel ring using SPI
spi = board.SPI()
strip = neopixel.NeoPixel_SPI(
    spi, 
    NUM_LEDS, 
    pixel_order=PO, 
    auto_write=False
)

# Turn off all LEDs on startup
strip.fill(0)
strip.show()

NEOPIXEL_ENABLED = True
print("[INFO] NeoPixel SPI initialized successfully.")

# Animation state variables (non-blocking updates)
animation_step = 0
lastAnimUpdate = time.time()
# ================================

# ====================================================================
# NEOPIXEL CONTROL (COLOR HELPER & ANIMATION DISPATCHER)
# ====================================================================
def hsv2rgb(h):
    """ Converts HSV float (0.0-1.0) to RGB integer tuple (0-255) """
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    return (int(r * 255), int(g * 255), int(b * 255))

def updateNeoPixels(gesture):
    """ Non-blocking LED pattern update based on the current detected gesture """
    global animation_step, lastAnimUpdate
    if not NEOPIXEL_ENABLED:
        return

    now = time.time()
    gestureClean = gesture.lower()
    pattern_matched = False

    # 1. FIST / CHARGE (Red pulse)
    if gestureClean == "fist":
        pulse = int(127 + 127 * math.sin(now * 5.0))
        strip.fill((pulse, 0, 0))
        strip.show()
        pattern_matched = True

    # 2. OPEN PALM / BEACON (Solid Bright White)
    if gestureClean == "palm":
        strip.fill((255, 255, 255))
        strip.show()
        pattern_matched = True

    # 3. PEACE SIGN / DUAL SPINNER (Rotating Blue/Cyan dots)
    if gestureClean == "peace":
        if now - lastAnimUpdate > 0.08:
            animation_step = (animation_step + 1) % NUM_LEDS
            lastAnimUpdate = now
        strip.fill((40, 0, 0))  # Background red
        pos1 = animation_step
        pos2 = (animation_step + (NUM_LEDS // 2)) % NUM_LEDS
        strip[pos1] = (0, 255, 0)   # Green spinner dot
        strip[pos2] = (0, 0, 255)   # Blue spinner dot
        strip.show()
        pattern_matched = True

    # 4. THUMBS UP / SUCCESS (Solid Green)
    if gestureClean == "thumbsup":
        strip.fill((0, 255, 0))
        strip.show()
        pattern_matched = True

    # 5. POINTING / CHASER (Radar sweep)
    if gestureClean in "point":
        if now - lastAnimUpdate > 0.06:
            animation_step = (animation_step + 1) % NUM_LEDS
            lastAnimUpdate = now
        strip.fill((255, 0, 0))
        strip[animation_step] = (0, 255, 0)
        strip.show()
        pattern_matched = True

    # 6. ROCK / FIRE (Sinusoidal Green-Cyan pulse)
    if gestureClean == "rock":
        step = int((now * 100) % 360)
        brightness = (math.sin(step * math.pi / 180) + 1) / 2
        g = int(255 * brightness)
        b = int(50 * brightness)
        strip.fill((0, g, b))
        strip.show()
        pattern_matched = True

    # 7. OK SIGN / RAINBOW WHEEL (Rotating RGB spectrum)
    if gestureClean in "ok":
        if now - lastAnimUpdate > 0.03:
            animation_step = (animation_step + 1) % 100
            lastAnimUpdate = now
        hue = animation_step / 100.0
        strip.fill(hsv2rgb(hue))
        strip.show()
        pattern_matched = True

    # DEFAULT / UNKNOWN (LEDs OFF)
    if not pattern_matched:
        strip.fill((0, 0, 0))
        strip.show()
# ====================================================================

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
    
    # ====================================================================
    # EXECUTE HARDWARE PATTERN UPDATE
    # ====================================================================
    updateNeoPixels(detectedGesture)
    # ====================================================================
   
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
