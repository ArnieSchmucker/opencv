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
#import mediapipe as mp
from picamera2 import Picamera2
from functions import ParseFaceMesh

piCam = Picamera2()
W=1280
H=720

piCam.preview_configuration.main.size = (W,H)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.controls.FrameRate=60
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()
   
#f = ParseHandsGeometry(W,H,piCam)
#f = ParseHandsNumbers(W,H,piCam)
f = ParseFaceMesh(W,H,piCam)

while True:
    frame = f.buildFrame()
    fps = f.frameSpeed()    
    #results = f.parseHandsGeometry()
    #results = f.parseHandsNumbers()
    results = f.parseFaceMesh()
    cv2.imshow("Camera", frame)
    cv2.moveWindow("Camera",0,60)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
print('Program Terminated')