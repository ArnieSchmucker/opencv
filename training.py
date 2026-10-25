import os
import pickle

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