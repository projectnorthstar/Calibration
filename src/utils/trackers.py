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
    
    def __init__(self):
        self.lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\antilatency\AntilatencyService.dll"))
        self.lib.GetPose.restype = POINTER(c_float * 7)
        self.lib.Start()
        return
        
    def getPose(self):
        pose = list(self.lib.GetPose().contents)
        pos = np.array(pose[:3])
        rot = R.from_quat(pose[3:]).as_matrix()
        valid = not np.allclose(0, pos)
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
        