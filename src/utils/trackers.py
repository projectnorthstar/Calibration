import abc
from scipy.spatial.transform import Rotation as R
import time
from ctypes import *
import os
import numpy as np

class Tracker:
    
    @abc.abstractmethod
    def getPose(self):
        return

class Alt(Tracker):
    
    def __init__(self, environmentData, placementData):
        self.lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\antilatency\AntilatencyService.dll"))
        self.lib.GetPose.restype = POINTER(c_float * 7)
        self.lib.Start.argtypes = [c_char_p, c_char_p]
        self.lib.Start.restype = c_int
        self.lib.Start(
            environmentData.encode(),
            placementData.encode()
        )
        return
        
    def getPose(self):
        pose = list(self.lib.GetPose().contents)
        pos = np.array(pose[:3])
        rot = R.from_quat(pose[3:]).as_matrix()
        valid = not np.allclose(0, pos)
        if valid is True:
            pos = (np.zeros(3) - pos) @ rot #origin 0, 0, 0
            pos[1] *= -1 #Flipped y Unity to OpenCV
        return valid, rot, pos
        
    def release(self):
        self.lib.Stop()
        
if __name__ == "__main__":
    tracker = Alt()
    for i in range(100):
        valid, rot, pos = tracker.getPose()
        time.sleep(0.1)
        print(valid)
        print(pos)
        print(rot)
        print()
    tracker.release()
        