import os
import pickle
import math

class Files:
    def __init__(self,filename):
        self.filename = filename
    def saveTrainedData(self,data):
        with open(self.filename,'wb') as f:
            pickle.dump(data,f)
        print('[INFO] Save Data to '+str(self.filename))
            
    def loadTrainedData(self):
        if os.path.exists(self.filename):
            with open(self.filename,'rb') as f:
                data = pickle.load(f)
                print('[INFO] Loaded '+str(len(data))+' gestures from '+str(self.filename))
                trainIndex = len(data.keys())
                gestureNames = list(data.keys())
                print(gestureNames)
                return data,trainIndex,gestureNames
        if not os.path.exists(self.filename):
            return{},0,[]
        
class TipDistance:
    @staticmethod
    def findDistance(pt1,pt2):
        dX = pt1.x - pt2.x
        dY = pt1.y - pt2.y
        dZ = pt1.z - pt2.z
        dist = math.sqrt(dX*dX + dY*dY + dZ*dZ)
        return dist

    def getTipDistances(self,handLM):
        lm = handLM.landmark
        wrist = lm[0]
        scaleFactor = self.findDistance(wrist,lm[5])
        tips = [lm[4],lm[8],lm[12],lm[16],lm[20]]
        tipDistances = []
        #calculate distance between tips and wrist
        for tip in tips: 
            dist = self.findDistance(wrist,tip)/scaleFactor
            tipDistances.append(dist)
        #calculate distance between fingers
        for startTip in range(len(tips)):
            for endTip in range(startTip+1,len(tips)):
                dist = self.findDistance(tips[startTip],tips[endTip])/scaleFactor
                tipDistances.append(dist)
        return tipDistances

class FindGesture:
    def __init__(self,tipLive,trainedData,maxAllowedError):
        self.tipLive = tipLive
        self.trainedData = trainedData
        self.maxAllowedError = maxAllowedError
        
    @staticmethod
    def calculateError(tipDist, trainDist):
        totalError = 0
        for index in range(len(tipDist)):
            dif = tipDist[index] - trainDist[index]
            totalError = totalError + abs(dif)
        return totalError

    def findGesture(self):
        bestMatch = "Unknown"
        lowestError = 99999999
        for gestureName, trainedDist in self.trainedData.items():
            error = self.calculateError(self.tipLive, trainedDist)
            if error < lowestError:
                lowestError = error
                bestMatch = gestureName
        if lowestError > self.maxAllowedError:
            return "Unknown"
        if lowestError <= self.maxAllowedError:
            return bestMatch