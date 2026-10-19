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
from functions import ObjectDetection

piCam = Picamera2(0)

W=1280
H=720

piCam.preview_configuration.main.size = (W,H)
piCam.preview_configuration.main.format = "RGB888"
piCam.preview_configuration.controls.FrameRate=60
piCam.preview_configuration.align()
piCam.configure("preview")
piCam.start()

cv2.namedWindow('Camera', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Camera', W, H)
cv2.moveWindow('Camera', 0, 65)
time.sleep(.1)
       
f = ObjectDetection(W,H,piCam)

while True:
    frame = f.buildFrame()
    fps = f.frameSpeed() 
    f.myDetector()                                  
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1)==ord('q'):
        break
    
#f.myDetector().close()
cv2.destroyAllWindows()
print('Program Terminated')