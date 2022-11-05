import pyrealsense2 as rs
import abc
import typing
import numpy as np
import cv2
import threading
from ctypes import *
import time
import os
import json

def backproject(cameraMatrix, x, y):
    fx, _, cx = cameraMatrix[0]
    _, fy, cy = cameraMatrix[1]
    return np.array(((x - cx) / fx, (y - cy) / fy), dtype=np.float32)

class CameraThread(threading.Thread, metaclass=abc.ABCMeta):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.undistort = False
        self._scheduledStop = False
        self._frameMutex = threading.Lock()
        self._leftRightImage = None
        self._newFrame = False
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
                self._frameMutex.acquire()
                self._leftRightImage = leftRightImage
                self._newFrame = True
                self._frameMutex.release()
            if self._scheduledStop:
                self._tstop()
                break
        return
        
    def stop(self):  
        self._scheduledStop = True
        return
        
    def read(self, peek=False):
        ret = False, None
        self._frameMutex.acquire()
        if self._leftRightImage is not None:
            ret = self._newFrame, np.hstack(self._leftRightImage)
            if peek is False:
                self._newFrame = False
        self._frameMutex.release()
        return ret

class IntelCameraThread(CameraThread):
    
    def __init__(self, exposure):
        super().__init__()
        self.calibration = None
        self.sides = ("left", "right")
        self._resolution = (800, 1696)
        self._exposure = exposure #microseconds
        self._gain = 2
        self._pipe = None
        self._sensor = None
        return
    
    @property
    def resolution(self):
        return self._resolution
    
    @property
    def exposure(self):
        return self._exposure
        
    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        if self._sensor is not None:
            self._sensor.set_option(rs.option.exposure, self._exposure)
        return
        
    def _tinit(self):
        self._pipe = rs.pipeline()
        cfg = rs.config()
        height, width = self.resolution
        width >>= 1
        cfg.enable_stream(rs.stream.fisheye, 1, width, height, rs.format.y8, 30)
        cfg.enable_stream(rs.stream.fisheye, 2, width, height, rs.format.y8, 30)
        
        #sensor options - needs to be done before start
        profile = cfg.resolve(self._pipe)
        sensor = profile.get_device().query_sensors()[0]
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, self.exposure)
        sensor.set_option(rs.option.gain, self._gain)
        
        profile = self._pipe.start(cfg)
        self._sensor = profile.get_device().query_sensors()[0]
        streams = {
            "left"  : profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
            "right" : profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()
        }
        
        #calibration data
        self.calibration = {}
        intrinsics = {
            "left"  : streams["left"].get_intrinsics(),
            "right" : streams["right"].get_intrinsics()
        }
        extrinsics = streams["left"].get_extrinsics_to(streams["right"])
        R = np.reshape(extrinsics.rotation, (3, 3)).T
        T = np.array(extrinsics.translation)
        self.calibration["R1"] = np.eye(3)
        self.calibration["R2"] = R
        self.calibration["baseline"] = np.linalg.norm(T)
        for ii, key in enumerate(intrinsics):
            i = intrinsics[key]
            cm = self.calibration[key + "CameraMatrix"] = np.array([
                [i.fx, 0, i.ppx],
                [0, i.fy, i.ppy],
                [0, 0, 1]
            ])
            dc = self.calibration[key + "DistCoeffs"] = np.array(i.coeffs[:4])
            self.calibration[key + "Map"] = cv2.fisheye.initUndistortRectifyMap(cm, dc, self.calibration[f"R{ii + 1}"], cm, (i.width, i.height), cv2.CV_32FC1)
        return
        
    def _readLeftRightImage(self, undistort):
        frame = self._pipe.wait_for_frames()
        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            leftRightImage = [np.asanyarray(f1.get_data()), np.asanyarray(f2.get_data())]
            if undistort is True:
                for i, side in enumerate(self.sides):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.calibration[f"{side}Map"][0], self.calibration[f"{side}Map"][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            return leftRightImage
        return None
        
    def _tstop(self):
        self._pipe.stop()
        return
        
class Cv2CameraThread(CameraThread):
    
    def __init__(self, index, fisheye, calibrationFile, autoExposure, exposure):
        super().__init__()
        self.calibration = None
        self.sides = ("left", "right")
        self._cap = None
        self._index = index
        self._calibrationFile = calibrationFile
        self._fisheye = fisheye
        self._autoExposure = autoExposure
        self._exposure = exposure
        return
    
    @property
    def fisheye(self):
        return self._fisheye
    
    @property
    def resolution(self):
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    
    @property
    def exposure(self):
        return self._cap.get(cv2.CAP_PROP_EXPOSURE)
        
    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure)
        return
        
    def _tinit(self):
        self._cap = cv2.VideoCapture(self._index)
        path = self._calibrationFile
        if os.path.isabs(path) is False:
            path = os.path.join(os.path.dirname(__file__), path)
        with open(path) as f:
            self.calibration = json.load(f)
        ufunc = cv2.fisheye.initUndistortRectifyMap if self.fisheye is True else cv2.initUndistortRectifyMap
        for key in self.calibration:
            if isinstance(self.calibration[key], list):
                self.calibration[key] = np.array(self.calibration[key])
        for i, side in enumerate(self.sides):
            cm = self.calibration[side + "CameraMatrix"]
            dc = self.calibration[side + "DistCoeffs"]
            self.calibration[side + "Map"] = ufunc(cm, dc, self.calibration[f"R{i + 1}"], cm, (self.calibration["imageWidth"], self.calibration["imageHeight"]), cv2.CV_32FC1)

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calibration["imageWidth"] << 1)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calibration["imageHeight"])
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self._autoExposure)
        self.exposure = self._exposure
        return
        
    def _readLeftRightImage(self, undistort):
        _, frame = self._cap.read()
        if len(frame.shape) > 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        leftRightImage = np.hsplit(frame, 2)
        if undistort is True:
            for i, side in enumerate(self.sides):
                leftRightImage[i] = cv2.remap(leftRightImage[i], self.calibration[f"{side}Map"][0], self.calibration[f"{side}Map"][1], cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        return leftRightImage
        
    def _tstop(self):
        self._cap.release()
        return
    
class LeapMotionThread(CameraThread):
    
    def __init__(self):
        super().__init__()
        self.baseline = 0
        self._imageSize = 0
        self._distortionMapSize = 0
        self._ntries = 50
        self._lib = None
        return
        
    @property
    def resolution(self):
        return (self._lib.GetImageHeight(0), self._lib.GetImageWidth(0) * 2)
        
    @property
    def exposure(self):
        return 0 #TODO
        
    @exposure.setter
    def exposure(self, value):
        #TODO
        return
        
    def _tinit(self):
        self._lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\ultraleap\LeapService.dll"))
        
        self._lib.PixelToRectilinear.argtypes = [c_uint, c_float, c_float]
        self._lib.PixelToRectilinear.restype = POINTER(c_float * 3)
        self._lib.GetBaseline.restype = c_float
        
        self._lib.Start()
        
        triesRemaining = self._ntries
        while(self._lib.GetImageSize() == 0 or triesRemaining > 1):
            time.sleep(0.1)
            triesRemaining -= 1
        if triesRemaining == 0:
            raise Exception("No frames received")
            
        self.baseline = self._lib.GetBaseline()
        self.undistortionMaps = [self._getDistortionMap(0), self._getDistortionMap(1)]
        return
        
    def pixelToRectilinear(self, sideId, x, y, undistorted=False):
        #TODO currently undistorted ignored
        return np.array(self._lib.PixelToRectilinear(sideId, x, y).contents, dtype=np.float32)
        
    def getCameraMatrix(self, sideId):
        return np.eye(3) #TODO
        
    def _getDistortionMap(self, sideId):
        if self._distortionMapSize != self._lib.GetDistortionMapSize():
            self._distortionMapSize = self._lib.GetDistortionMapSize()
            self._lib.GetDistortionMap.restype = POINTER(c_float * self._distortionMapSize)
        dm = np.array(self._lib.GetDistortionMap(sideId).contents, dtype=np.float32).reshape(64, 64, 2)
        dm[:,:, 1] = 1 - dm[:, :, 1]
        dm[:,:, 0] *= self._lib.GetImageWidth(sideId) - 1
        dm[:,:, 1] *= self._lib.GetImageHeight(sideId) - 1
        dm = cv2.resize(dm, (self._lib.GetImageWidth(sideId), self._lib.GetImageHeight(sideId)))
        return dm
        
    def _readLeftRightImage(self, undistort):
        if self._imageSize != self._lib.GetImageSize():
            self._imageSize = self._lib.GetImageSize()
            self._lib.GetImageData.restype = POINTER(c_byte * self._imageSize * 2)
        data = self._lib.GetImageData()
        if data:
            leftRightImage = np.array(self._lib.GetImageData().contents, dtype=np.uint8).reshape(2, self._lib.GetImageHeight(0), self._lib.GetImageWidth(0))
            if undistort is True:
                for i in range(len(leftRightImage)):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.undistortionMaps[i][:,:, 0], self.undistortionMaps[i][:,:, 1], cv2.INTER_LINEAR)
            return leftRightImage
        return None
        
    def _tstop(self):
        self._lib.Stop()
        return
    
class XvisioCameraThread(CameraThread):
    
    def __init__(self, exposure):
        super().__init__()
        self.baseline = 0
        self._imageSize = 0
        self._distortionMapSize = 0
        self._ntries = 50
        self._lib = None
        self._depthParameter = 0
        self._exposure = exposure
        return
        
    @property
    def resolution(self):
        return (self._lib.GetImageHeight(0), self._lib.GetImageWidth(0) * 2)
        
    @property
    def exposure(self):
        return self._exposure
        
    @exposure.setter
    def exposure(self, value):
        self._exposure = value
        if self._lib is not None:
            self._lib.SetGainExposure(25, int(self._exposure / 1000.0))
        return
        
    def _tinit(self):
        self._lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), r"libs\xvisio\XvisioService.dll"))
        
        self._lib.PixelToRectilinear.argtypes = [c_uint, c_float, c_float, c_bool]
        self._lib.PixelToRectilinear.restype = POINTER(c_float * 3)
        self._lib.GetBaseline.restype = c_float
        self._lib.GetDepthParameter.restype = c_float
        self._lib.GetCameraMatrix.restype = POINTER(c_float * 9)
        self._lib.GetCameraExtrinsic.restype = POINTER(c_float * 12)
        self._lib.SetGainExposure.argtypes = [c_int, c_int]
        self._lib.Start()
        self.exposure = self._exposure #set value from init once initialized
        
        triesRemaining = self._ntries
        while(self._lib.GetImageSize() == 0 or triesRemaining > 1):
            time.sleep(0.1)
            triesRemaining -= 1
        if triesRemaining == 0:
            raise Exception("No frames received")
            
        self.baseline = self._lib.GetBaseline()
        self._depthParameter = self._lib.GetDepthParameter()
        
        self.undistortionMaps = [self._getDistortionMap(0), self._getDistortionMap(1)]
        return
        
    def pixelToRectilinear(self, sideId, x, y, undistorted=False):
        return np.array(self._lib.PixelToRectilinear(sideId, x, y, undistorted).contents, dtype=np.float32)
        
    def getCameraMatrix(self, sideId):
        cameraMatrix = np.array(self._lib.GetCameraMatrix(sideId).contents, dtype=np.float32).reshape(3, 3)
        if self.undistort is True:
            cameraMatrix[0, 0] = cameraMatrix[1, 1] = self._depthParameter
        return cameraMatrix
        
    def _getDistortionMap(self, sideId):
        if self._distortionMapSize != self._lib.GetDistortionMapSize():
            self._distortionMapSize = self._lib.GetDistortionMapSize()
            self._lib.GetDistortionMap.restype = POINTER(c_float * self._distortionMapSize)
        return np.array(self._lib.GetDistortionMap(sideId).contents, dtype=np.float32).reshape(self._lib.GetImageHeight(sideId), self._lib.GetImageWidth(sideId), 2)
        
    def _readLeftRightImage(self, undistort):
        if self._imageSize != self._lib.GetImageSize():
            self._imageSize = self._lib.GetImageSize()
            self._lib.GetImageData.restype = POINTER(c_byte * self._imageSize * 2)
        data = self._lib.GetImageData()
        if data:
            leftRightImage = np.array(self._lib.GetImageData().contents, dtype=np.uint8).reshape(2, self._lib.GetImageHeight(0), self._lib.GetImageWidth(0))
            if undistort is True:
                for i in range(len(leftRightImage)):
                    leftRightImage[i] = cv2.remap(leftRightImage[i], self.undistortionMaps[i][:,:, 0], self.undistortionMaps[i][:,:, 1], cv2.INTER_LINEAR)
            return leftRightImage
        return None
        
    def getExtrinsic(self, sideId):
        cameraExtrinsic = np.array(self._lib.GetCameraExtrinsic(sideId).contents, dtype=np.float32)
        return cameraExtrinsic[:9].reshape((3, 3)), cameraExtrinsic[9:12]
        
    def _tstop(self):
        self._lib.Stop()
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
    
    def __init__(self, undistort=True, exposure=10000):
        self.cameraThread = XvisioCameraThread(exposure)
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

    def __init__(self, undistort=True, exposure=10000):
        self.cameraThread = IntelCameraThread(exposure)
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
        return self.calibration[self.cameraThread.sides[sideId] + "CameraMatrix"]

    def pixelToRectilinear(self, sideId, x, y):
        coordinates = np.array((((x, y),),), dtype=np.float32)
        return self.pixelsToRectilinear(sideId, coordinates)[0, 0]
        
    def pixelsToRectilinear(self, sideId, coordinates):
        cameraMatrix = self.getCameraMatrix(sideId)
        r = self.calibration[f"R{sideId + 1}"]
        if self.undistort is True:
            return cv2.undistortPoints(coordinates, cameraMatrix, np.zeros(5)).reshape(coordinates.shape) #same as backproject, reshape added to match input shape
        else:
            distCoeffs = self.calibration[self.cameraThread.sides[sideId] + "DistCoeffs"]
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
        
class Cv2Camera(Camera):
    
    def __init__(self, index, autoExposure=0.25, exposure=-3, undistort=True, fisheye=False, calibrationFile="../data/Cv2CameraCalibration.json"):
        self.cameraThread = Cv2CameraThread(index, fisheye, calibrationFile, autoExposure, exposure)
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
        return self.calibration[self.cameraThread.sides[sideId] + "CameraMatrix"]

    def pixelToRectilinear(self, sideId, x, y):
        coordinates = np.array((((x, y),),), dtype=np.float32)
        return self.pixelsToRectilinear(sideId, coordinates)[0, 0]
        
    def pixelsToRectilinear(self, sideId, coordinates):
        cameraMatrix = self.getCameraMatrix(sideId)
        r = self.calibration[f"R{sideId + 1}"]
        if self.undistort is True:
            return cv2.undistortPoints(coordinates, cameraMatrix, np.zeros(5)).reshape(coordinates.shape) #same as backproject, reshape added to match input shape
        else:
            ufunc = cv2.fisheye.undistortPoints if self.cameraThread._fisheye is True else cv2.undistortPoints
            distCoeffs = self.calibration[self.cameraThread.sides[sideId] + "DistCoeffs"]
            return ufunc(coordinates, cameraMatrix, distCoeffs, R=r).reshape(coordinates.shape)
        return
        
    def leftRightToDevice(self, leftRightPosition):
        return np.mean(leftRightPosition, axis=0)

if __name__=="__main__":
    #cam = T265Camera()
    #cam = XvisioCamera()
    #cam = LeapCamera(True)
    cam = Cv2Camera(1)
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