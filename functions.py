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
        self.faceMesh = self.mp.solutions.face_mesh.FaceMesh(   
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
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
        cv2.putText(self.frame,myText,
                    self.textLowerLeft,
                    self.fontFace,
                    self.fontScale,
                    self.fontColor,
                    self.fontThickness)
    
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
                        self.frame,str(index),
                        (pixelX,pixelY),
                        self.fontFace,
                        self.fontScale,
                        (255,0,255),
                        self.fontThickness,)
        return results
    
class ObjectDetection(Functions):
    

    def myDetector(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision 
        modelPath = "/home/pi/ai-lab-kit/mediapipe/efficientdet_lite0.tflite"
        scoreThreshold = .35
        maxObj = 3
        baseOptions = python.BaseOptions(model_asset_path = modelPath)
        options = vision.ObjectDetectorOptions(base_options = baseOptions,
                                       score_threshold=scoreThreshold,
                                       max_results=maxObj)
        myDetector = vision.ObjectDetector.create_from_options(options)
        myImage = mp.Image(image_format = mp.ImageFormat.SRGB, data=self.frame)
        myDetections = myDetector.detect(myImage)
        cv2.rectangle(self.frame,(540,260),(740,460),(0,0,255),3)
        if myDetections != None:
            for detection in myDetections.detections:
                box = detection.bounding_box
                x = box.origin_x
                y = box.origin_y
                w = box.width
                h = box.height
                cv2.rectangle(self.frame, (x,y), (x+w,y+h), (255,0,0),self.fontThickness )
            
                if detection.categories:
                    category = detection.categories[0]
                    myLabel = category.category_name
                    labelText = myLabel+" "+ str(round(category.score*100))+"%"
                    cv2.putText(self.frame,labelText,
                                (x, max(y-10,25)),
                                self.fontFace,
                                self.fontScale*.8,
                                (0,255,0),
                                2)
                if detection.categories[0].category_name == 'bottle':
                    #print('bottle detected',x,y)
                    if x >= 540 and x<=740 and y >= 260 and y <= 460 :
                        #print("In ROI")
                        cv2.putText(self.frame,'In ROI',(x, max(y+25,25)),
                                    self.fontFace,
                                    self.fontScale,
                                    (0,255,0),
                                    2)
                    else:
                        #print("Not in ROI")
                        cv2.putText(self.frame,'Not In ROI',(x, max(y+25,25)),
                                    self.fontFace,
                                    self.fontScale,
                                    (0,255,0),
                                    2)

class ParseFaceMesh(Functions):
    def parseFaceMesh(self):
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(rgb)
        if results.multi_face_landmarks:     
            for faceLandmarks in results.multi_face_landmarks:
                # Optional second pass for glow effect
                self.mp.solutions.drawing_utils.draw_landmarks(
                    image=self.frame,
                    landmark_list=faceLandmarks,
                    connections=self.mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(
                        color=(100, 100, 100),
                        thickness=1
                    )
                )
                # === Best looking translucent mesh ===
                self.mp.solutions.drawing_utils.draw_landmarks(
                    image=self.frame,
                    landmark_list=faceLandmarks,
                    connections=self.mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,                    # Hide dots
                    connection_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 0, 180),      # Bright cyan/teal
                        thickness=7
                    )
                )
                
                # Optional: Extra detail on irises (makes eyes look better)
                self.mp.solutions.drawing_utils.draw_landmarks(
                    image=self.frame,
                    landmark_list=faceLandmarks,
                    connections=self.mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 0, 0),
                        thickness=7
                    )
                )

#             return results