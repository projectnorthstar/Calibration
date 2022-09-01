import pyrealsense2 as rs
import abc
import typing
import numpy as np
import cv2
import threading
from ctypes import *
import time
import os

def backproject(cameraMatrix, x, y):
    fx, _, cx = cameraMatrix[0]
    _, fy, cy = cameraMatrix[1]
    return np.array(((x - cx) / fx, (y - cy) / fy), dtype=np.float32)

class CameraThread(threading.Thread, metaclass=abc.ABCMeta):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.scheduledStop = False
        self.frameMutex = threading.Lock()
        self.leftRightImage = None
        self.newFrame = False
        self.undistort = False
        return
    
    @property
    @abc.abstractmethod
    def resolution(self):
        return
    
    @property
    @abc.abstractmethod
    def exposure(self):
        return
        
    @exposure.setter
    @abc.abstractmethod
    def exposure(self, value):
        return
        
    @abc.abstractmethod
    def _tinit(self):
        return
        
    @abc.abstractmethod
    def _tstop(self):  
        return
    
    @abc.abstractmethod
    def _readLeftRightImage(self, undistort):  
        return
    
    def run(self):
        self._tinit()
        while(True):
            leftRightImage = self._readLeftRightImage(self.undistort)
            if leftRightImage is not None:
                self.frameMutex.acquire()
                self.leftRightImage = leftRightImage
                self.newFrame = True
                self.frameMutex.release()
            if self.scheduledStop:
                self._tstop()
                break
        return
        
    def stop(self):  
        self.scheduledStop = True
        return
        
    def read(self, peek=False):
        ret = False, None
        self.frameMutex.acquire()
        if self.leftRightImage is not None:
            ret = self.newFrame, np.hstack(self.leftRightImage)
            if peek is False:
                self.newFrame = False
        self.frameMutex.release()
        return ret

class IntelCameraThread(CameraThread):
    
    def __init__(self):
        super().__init__()
        self.calibration = None
        self._resolution = (800, 1696)
        self._exposure = 10000 #microseconds
        self._gain = 2
        self.pipe = None
        self.sensor = None
        return
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def exposure(self):
        return self._exposure
        
    @exposure.setter
    def exposure(self, value):
        if self._exposure != value:
            self._exposure = value
            if self.sensor is not None:
                self.sensor.set_option(rs.option.exposure, self._exposure)
        return
        
    def _tinit(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        height, width = self.resolution
        width >>= 1
        cfg.enable_stream(rs.stream.fisheye, 1, width, height, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.fisheye, 2, width, height, rs.format.y8, 30)
        
        #needs to be done before start
        profile = cfg.resolve(self.pipe)
        sensor = profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, self.exposure)
        sensor.set_option(rs.option.gain, self._gain)
        
        profile = self.pipe.start(cfg)
        self.sensor = profile.get_device().query_sensors()[0]
        
        streams = {
            "left"  : profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right" : profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
        }
        intrinsics = {
            "left"  : streams["left"].get_intrinsics(),
            "right" : streams["right"].get_intrinsics()
        }
        self.calibration = {}
        for key in intrinsics:
            i = intrinsics[key]
            cm = self.calibration[key + "CameraMatrix"] = np.array([
                [i.fx, 0, i.ppx],
                [0, i.fy, i.ppy],
                [0, 0, 1]
            ])
            dc = self.calibration[key + "DistCoeffs"] = np.array(i.coeffs[:4])
            self.calibration[key + "Map"] = cv2.fisheye.initUndistortRectifyMap(cm, dc, None, cm, (i.width, i.height), cv2.CV_32FC1)
        extrinsics = streams["left"].get_extrinsics_to(streams["right"])
        R = np.reshape(extrinsics.rotation, (3, 3)).T
        T = np.array(extrinsics.translation)
        self.calibration["R1"] = np.eye(3)
        self.calibration["R2"] = R
        self.calibration["baseline"] = np.linalg.norm(T)
        return
        
    def _readLeftRightImage(self, undistort):
        frame = self.pipe.wait_for_frames()
        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            leftRightImage = [np.asanyarray(f1.get_data()), np.asanyarray(f2.get_data())]
            if undistort is True:
                for i, side in enumerate(("left", "right")):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.calibration[f"{side}Map"][0], self.calibration[f"{side}Map"][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            return leftRightImage
        return None
        
    def _tstop(self):
        self.pipe.stop()
        return
    
class LeapMotionThread(CameraThread):
    
    def __init__(self):
        super().__init__()
        self.imageSize = 0
        self.distortionMapSize = 0
        self.ntries = 50
        self.lib = None
        self.baseline = 0
        return
        
    @property
    def resolution(self):
        return (self.lib.GetImageHeight(0), self.lib.GetImageWidth(0) * 2)
        
    @property
    def exposure(self):
        return 0 #TODO
        
    @exposure.setter
    def exposure(self, value):
        #TODO
        return
        
    def _tinit(self):
        self.lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\ultraleap\LeapService.dll"))
        
        self.lib.PixelToRectilinear.argtypes = [c_uint, c_float, c_float, c_bool]
        self.lib.PixelToRectilinear.restype = POINTER(c_float * 3)
        self.lib.GetBaseline.restype = c_float
        
        self.lib.Start()
        
        triesRemaining = self.ntries
        while(self.lib.GetImageSize() == 0 or triesRemaining > 1):
            time.sleep(0.1)
            triesRemaining -= 1
        if triesRemaining == 0:
            raise Exception("No frames received")
            
        self.baseline = self.lib.GetBaseline()
        self.undistortionMaps = [self._getDistortionMap(0), self._getDistortionMap(1)]
        return
        
    def pixelToRectilinear(self, sideId, x, y, undistorted=False):
        return np.array(self.lib.PixelToRectilinear(sideId, x, y, undistorted).contents, dtype=np.float32)
        
    def getCameraMatrix(self, sideId):
        return np.eye(3) #TODO
        
    def _getDistortionMap(self, sideId):
        if self.distortionMapSize != self.lib.GetDistortionMapSize():
            self.distortionMapSize = self.lib.GetDistortionMapSize()
            self.lib.GetDistortionMap.restype = POINTER(c_float * self.distortionMapSize)
        dm = np.array(self.lib.GetDistortionMap(sideId).contents, dtype=np.float32).reshape(64, 64, 2)
        dm[:,:, 1] = 1 - dm[:, :, 1]
        dm *= 383 #TODO
        dm = cv2.resize(dm, (self.lib.GetImageHeight(sideId), self.lib.GetImageWidth(sideId)))
        return dm
        
    def _readLeftRightImage(self, undistort):
        if self.imageSize != self.lib.GetImageSize():
            self.imageSize = self.lib.GetImageSize()
            self.lib.GetImageData.restype = POINTER(c_byte * self.imageSize * 2)
        data = self.lib.GetImageData()
        if data:
            leftRightImage = np.array(self.lib.GetImageData().contents, dtype=np.uint8).reshape(2, self.lib.GetImageHeight(0), self.lib.GetImageWidth(0))
            if undistort is True:
                for i, side in enumerate(("left", "right")):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.undistortionMaps[i][:,:, 0], self.undistortionMaps[i][:,:, 1], cv2.INTER_LINEAR)
            return leftRightImage
        return None
        
    def _tstop(self):
        self.lib.Stop()
        return
    
class XvisioCameraThread(CameraThread):
    
    def __init__(self):
        super().__init__()
        self.imageSize = 0
        self.distortionMapSize = 0
        self.ntries = 50
        self.lib = None
        self.baseline = 0
        self.depthParameter = 0
        return
        
    @property
    def resolution(self):
        return (self.lib.GetImageHeight(0), self.lib.GetImageWidth(0) * 2)
        
    @property
    def exposure(self):
        return 0 #TODO
        
    @exposure.setter
    def exposure(self, value):
        #TODO
        return
        
    def _tinit(self):
        self.lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\xvisio\XvisioService.dll"))
        
        self.lib.PixelToRectilinear.argtypes = [c_uint, c_float, c_float, c_bool]
        self.lib.PixelToRectilinear.restype = POINTER(c_float * 3)
        self.lib.GetBaseline.restype = c_float
        self.lib.GetDepthParameter.restype = c_float
        self.lib.GetCameraMatrix.restype = POINTER(c_float * 9)
        self.lib.GetCameraExtrinsic.restype = POINTER(c_float * 12)
        
        self.lib.Start()
        
        triesRemaining = self.ntries
        while(self.lib.GetImageSize() == 0 or triesRemaining > 1):
            time.sleep(0.1)
            triesRemaining -= 1
        if triesRemaining == 0:
            raise Exception("No frames received")
            
        self.baseline = self.lib.GetBaseline()
        self.depthParameter = self.lib.GetDepthParameter()
        
        self.undistortionMaps = [self._getDistortionMap(0), self._getDistortionMap(1)]
        return
        
    def pixelToRectilinear(self, sideId, x, y, undistorted=False):
        return np.array(self.lib.PixelToRectilinear(sideId, x, y, undistorted).contents, dtype=np.float32)
        
    def getCameraMatrix(self, sideId):
        cameraMatrix = np.array(self.lib.GetCameraMatrix(sideId).contents, dtype=np.float32).reshape(3, 3)
        if self.undistort is True:
            cameraMatrix[0, 0] = cameraMatrix[1, 1] = self.depthParameter
        return cameraMatrix
        
    def _getDistortionMap(self, sideId):
        if self.distortionMapSize != self.lib.GetDistortionMapSize():
            self.distortionMapSize = self.lib.GetDistortionMapSize()
            self.lib.GetDistortionMap.restype = POINTER(c_float * self.distortionMapSize)
        return np.array(self.lib.GetDistortionMap(sideId).contents, dtype=np.float32).reshape(self.lib.GetImageHeight(sideId), self.lib.GetImageWidth(sideId), 2)
        
    def _readLeftRightImage(self, undistort):
        if self.imageSize != self.lib.GetImageSize():
            self.imageSize = self.lib.GetImageSize()
            self.lib.GetImageData.restype = POINTER(c_byte * self.imageSize * 2)
        data = self.lib.GetImageData()
        if data:
            leftRightImage = np.array(self.lib.GetImageData().contents, dtype=np.uint8).reshape(2, self.lib.GetImageHeight(0), self.lib.GetImageWidth(0))
            if undistort is True:
                for i, side in enumerate(("left", "right")):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.undistortionMaps[i][:,:, 0], self.undistortionMaps[i][:,:, 1], cv2.INTER_LINEAR)
            return leftRightImage
        return None
        
    def getExtrinsic(self, sideId):
        cameraExtrinsic = np.array(self.lib.GetCameraExtrinsic(sideId).contents, dtype=np.float32)
        return cameraExtrinsic[:9].reshape((3, 3)), cameraExtrinsic[9:12]
        
    def _tstop(self):
        self.lib.Stop()
        return

class Camera(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def read(self, peek=False) -> typing.Tuple[bool, np.ndarray]:
        return
        
    @abc.abstractmethod
    def release(self):
        return
    
    @abc.abstractmethod
    def getCameraMatrix(self, sideId: int) -> np.ndarray:
        return
    
    @abc.abstractmethod
    def pixelToRectilinear(self, sideId: int, x: float, y: float) -> np.ndarray:
        return
        
    @abc.abstractmethod
    def pixelsToRectilinear(self, sideId: int, coordinates: np.ndarray) -> np.ndarray:
        return
    
    @property
    @abc.abstractmethod
    def ready(self) -> bool:
        return
    
    @property
    @abc.abstractmethod
    def resolution(self) -> typing.Tuple[int, int]:
        return
    
    @property
    @abc.abstractmethod
    def undistort(self) -> bool:
        return
        
    @property
    @abc.abstractmethod
    def baseline(self) -> float:
        return
        
    @undistort.setter
    @abc.abstractmethod
    def undistort(self, value: bool):
        return
        
    @abc.abstractmethod
    def leftRightToDevice(self, leftRightPosition: np.ndarray) -> np.ndarray:
        return
        
class XvisioCamera(Camera):
    
    def __init__(self, undistort = True):
        self.cameraThread = XvisioCameraThread()
        self.undistort = undistort
        self.cameraThread.start()
        return

    @property
    def isAlive(self):
        return self.cameraThread.is_alive()

    @property
    def ready(self):
        return self.read(peek=True)[1] is not None
    
    @property
    def baseline(self):
        return self.cameraThread.baseline
        
    @property
    def undistort(self):
        return self.cameraThread.undistort
        
    @undistort.setter
    def undistort(self, value):
        self.cameraThread.undistort = value
        return
       
    @property
    def exposure(self):
        return self.cameraThread.exposure
        
    @exposure.setter
    def exposure(self, value):
        self.cameraThread.exposure = value
        return
        
    @property
    def resolution(self):
        return self.cameraThread.resolution
        
    def read(self, peek=False):
        return self.cameraThread.read(peek)
        
    def release(self):
        self.cameraThread.stop()
        self.cameraThread.join()
        return

    def getCameraMatrix(self, sideId):
        return self.cameraThread.getCameraMatrix(sideId)

    def pixelToRectilinear(self, sideId, x, y):
        return self.cameraThread.pixelToRectilinear(sideId, x, y, self.undistort)[:2]
        
    def leftPixelToRectilinear(self, coordinate):
        return self.pixelToRectilinear(0, coordinate[0], coordinate[1])
        
    def rightPixelToRectilinear(self, coordinate):
        return self.pixelToRectilinear(1, coordinate[0], coordinate[1])
        
    def pixelsToRectilinear(self, sideId, coordinates): #TODO
        func = self.leftPixelToRectilinear if sideId == 0 else self.rightPixelToRectilinear
        return np.apply_along_axis(func, -1, coordinates).reshape(coordinates.shape) #[1,N,2] or [N,1,2]
        
    def getExtrinsic(self, sideId):
        return self.cameraThread.getExtrinsic(sideId)
        
    def leftRightToDevice(self, leftRightPosition):
        mean = np.zeros(3, dtype=np.float32)
        for i, position in enumerate(leftRightPosition):
            rme, te = self.getExtrinsic(i)
            mean += position @ rme.T + te
        return mean / len(leftRightPosition)
        
class T265Camera(Camera):

    def __init__(self, undistort = True):
        self.sides = ("left", "right")
        self.cameraThread = IntelCameraThread()
        self.undistort = undistort
        self.cameraThread.start()
        return

    @property
    def isAlive(self):
        return self.cameraThread.is_alive()

    @property
    def ready(self):
        return self.read(peek=True)[1] is not None
    
    @property
    def baseline(self):
        return self.calibration["baseline"]
        
    @property
    def undistort(self):
        return self.cameraThread.undistort
        
    @undistort.setter
    def undistort(self, value):
        self.cameraThread.undistort = value
        return
       
    @property
    def exposure(self):
        return self.cameraThread.exposure
        
    @exposure.setter
    def exposure(self, value):
        self.cameraThread.exposure = value
        return
       
    @property
    def calibration(self):
        return self.cameraThread.calibration
        
    @property
    def resolution(self):
        return self.cameraThread.resolution
        
    def read(self, peek=False):
        return self.cameraThread.read(peek)
        
    def release(self):
        self.cameraThread.stop()
        self.cameraThread.join()
        return

    def getCameraMatrix(self, sideId):
        return self.calibration[self.sides[sideId] + "CameraMatrix"]

    def pixelToRectilinear(self, sideId, x, y):
        coordinates = np.array((((x, y),),), dtype=np.float32)
        return self.pixelsToRectilinear(sideId, coordinates)[0, 0]
        
    def pixelsToRectilinear(self, sideId, coordinates):
        cameraMatrix = self.getCameraMatrix(sideId)
        r = self.calibration[f"R{sideId + 1}"]
        if self.undistort is True:
            return cv2.undistortPoints(coordinates, cameraMatrix, np.zeros(5), R=r).reshape(coordinates.shape) #same as backproject, reshape added to match input shape
        else:
            distCoeffs = self.calibration[self.sides[sideId] + "DistCoeffs"]
            return cv2.fisheye.undistortPoints(coordinates, cameraMatrix, distCoeffs, R=r).reshape(coordinates.shape)
        return
        
    def leftRightToDevice(self, leftRightPosition):
        return np.mean(leftRightPosition, axis=0)

class LeapCamera(Camera):
    
    def __init__(self, undistort = False):
        self.cameraThread = LeapMotionThread()
        self.undistort = undistort
        self.cameraThread.start()
        return

    @property
    def isAlive(self):
        return self.cameraThread.is_alive()

    @property
    def ready(self):
        return self.read(peek=True)[1] is not None
    
    @property
    def baseline(self):
        return self.cameraThread.baseline
        
    @property
    def undistort(self):
        return self.cameraThread.undistort
        
    @undistort.setter
    def undistort(self, value):
        self.cameraThread.undistort = value
        return
       
    @property
    def exposure(self):
        return self.cameraThread.exposure
        
    @exposure.setter
    def exposure(self, value):
        self.cameraThread.exposure = value
        return
        
    @property
    def resolution(self):
        return self.cameraThread.resolution
        
    def read(self, peek=False):
        return self.cameraThread.read(peek)
        
    def release(self):
        self.cameraThread.stop()
        self.cameraThread.join()
        return

    def getCameraMatrix(self, sideId):
        return self.cameraThread.getCameraMatrix(sideId)

    def pixelToRectilinear(self, sideId, x, y):
        return self.cameraThread.pixelToRectilinear(sideId, x, y, self.undistort)[:2]
        
    def leftPixelToRectilinear(self, coordinate):
        return self.pixelToRectilinear(0, coordinate[0], coordinate[1])
        
    def rightPixelToRectilinear(self, coordinate):
        return self.pixelToRectilinear(1, coordinate[0], coordinate[1])
        
    def pixelsToRectilinear(self, sideId, coordinates): #TODO
        func = self.leftPixelToRectilinear if sideId == 0 else self.rightPixelToRectilinear
        return np.apply_along_axis(func, -1, coordinates).reshape(coordinates.shape) #[1,N,2] or [N,1,2]
        
    def leftRightToDevice(self, leftRightPosition):
        return np.mean(leftRightPosition, axis=0)

if __name__=="__main__":
    #cam = T265Camera()
    #cam = XvisioCamera()
    cam = LeapCamera(True)
    try:
        print(cam.ready)
        while cam.ready is False:
            time.sleep(0.1)
        for _ in range(100):
            ret, frame = cam.read(peek=True)
            if frame is not None:
                cv2.imshow("test", frame)
            cv2.waitKey(10)
        print(cam.pixelToRectilinear(0, 100, 100))
        print(cam.pixelToRectilinear(1, 100, 100))
        pointsMat = np.array((((100, 100), (200, 100), (100, 100)),), dtype=np.float32)
        print(cam.pixelsToRectilinear(0, pointsMat))
        print(cam.pixelsToRectilinear(1, pointsMat))
        cam.undistort = False
        for _ in range(100):
            ret, frame = cam.read(peek=True)
            if frame is not None:
                cv2.imshow("test", frame)
            cv2.waitKey(10)
        print(cam.pixelToRectilinear(0, 100, 100))
        print(cam.pixelToRectilinear(1, 100, 100))
        print(cam.pixelsToRectilinear(0, pointsMat))
        print(cam.pixelsToRectilinear(1, pointsMat))
        print()
        print(cam.undistort)
        print(cam.resolution)
        print(cam.ready)
        print(cam.baseline)
    finally:
        cam.release()