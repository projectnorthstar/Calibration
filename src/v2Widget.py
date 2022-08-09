import os
import numpy as np
import cv2
import math
import time
import abc
import typing
from wand.image import Image
import sys



from PySide6.QtWidgets import QApplication, QWidget, QSizePolicy
from PySide6.QtCore import QTimer, Qt, QRect
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QVBoxLayout, QLabel
from forms.v2Form import Ui_V2Form

from utils.polyHelpers import calcCoeffs
from utils.cameras import T265Camera
from utils.lut import LookupTable

def wait(timeInMs):
    timeInNs = timeInMs * 1000000
    t = time.time_ns()
    while(time.time_ns() - t < timeInNs):
        cv2.waitKey(10)
    return

def cachedArray(func):
    def inner(self, height, width, *args, **kwargs):
        if(hasattr(self, "_cache") is not True):
            self._cache = {}
        cachedValue = self._cache.get(func.__name__)
        if(cachedValue is None or cachedValue.shape != (height, width)):
            self._cache[func.__name__] = func(self, height, width, *args, **kwargs)
        return self._cache[func.__name__]
    return inner

class Borg:
    
    _shared_state = {}
    
    def __init__(self):
        self.__dict__ = self._shared_state
        if(getattr(self, "_initialized", False) is False):
            self.initialize()
            self._initialized = True
        return
    
    def initialize(self):
        return

class CalibrationHelpers(Borg):
    
    _shared_state = {}
    
    def __init__(self):
        Borg.__init__(self)
        self.continuum = self._continuum()
        return
    
    @cachedArray
    def allWhite(self, height, width):
        return np.ones((height, width), dtype=np.uint8) * 255
    
    @cachedArray
    def allDark(self, height, width):
        return np.zeros((height, width), dtype=np.uint8)
    
    def _continuum(self):
        c = np.arange(0, 256, dtype=np.uint8)
        c = np.bitwise_xor(c, c//2) # Binary to Gray
        return c
    
    @cachedArray
    def widthContinuum(self, height, width, splitscreen):
        wc = self.allDark(height, width)
        c = self.continuum
        if splitscreen is False:
            wc = cv2.resize(c[None, :], (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            wc[:, : int(width / 2)] = cv2.resize(c[None, :], (int(width / 2), height), interpolation=cv2.INTER_NEAREST)
            wc[:, int(width / 2) :] = wc[:, : int(width / 2)]
        return wc
        
    @cachedArray
    def heightContinuum(self, height, width):
        return cv2.resize(self.continuum[:, None], (width, height), interpolation=cv2.INTER_NEAREST)
        
    @cachedArray
    def widthBits(self, height, width, splitscreen=False):
        wc = self.widthContinuum(height, width, splitscreen)
        return np.unpackbits(wc[: , :, None].astype(np.uint8), axis=-1)
        
    @cachedArray
    def heightBits(self, height, width):
        hc = self.heightContinuum(height, width)
        return np.unpackbits(hc[:, :, None].astype(np.uint8), axis=-1)
        
    @staticmethod
    def calibration2GLSL(cal):
        glslStrs = []
        for key, coeffs in cal.items():
            glslStr = f"float[] {key} = float[{len(coeffs)}] ("
            glslStr += ", ".join(map(str, coeffs))
            glslStr += ");"
            glslStrs.append(glslStr)
        return "\n".join(glslStrs)

class CalibrationManager(Borg):

    _shared_state = {}

    def __init__(self):
        Borg.__init__(self)
        return
        
    def initialize(self):
        self.helpers = CalibrationHelpers()
        self.frameBuffer = []
        return
        
    def captureNewFrameRoutine(self, camera):
        frame = None
        while(True):
            ret, frame = camera.read()
            if ret is True:
                self.frameBuffer.append(frame)
                break
            else:
                yield
        return frame
                
    def clearFrameBuffer(self):
        self.frameBuffer.clear()
        return
        
    def createMonitorMaskRoutine(self, camera, screen, target, threshold):
        self.clearFrameBuffer()
        aw = self.helpers.allWhite(*screen.resolution)
        ad = self.helpers.allDark(*screen.resolution)
        screen.setImage(aw)
        yield
        yield from self.captureNewFrameRoutine(camera)
        screen.setImage(ad)
        yield
        yield from self.captureNewFrameRoutine(camera)
        self.screenMask = self.createMask(self.frameBuffer, threshold, True)
        target.setImage(self.screenMask * 255)
        return
        
    def erodeMask(self, mask, ksize=(21,21)):
        maskcopy = mask.copy()
        for img in np.hsplit(maskcopy, 2):
            cv2.rectangle(img, (0,0), img.shape[::-1], (0, 0, 0), 1)
        kernel = np.ones(ksize, np.uint8)
        return cv2.erode(maskcopy, kernel)
    
    def measureWidthBitsRoutine(self, mask, camera, screen, target, brightness=127):
        self.widthBits = yield from self.measureBitsRoutine(self.helpers.widthBits(*camera.resolution, True), mask, camera, screen, brightness)
        target.setImage(self.widthBits)
        return
    
    def measureHeightBitsRoutine(self, mask, camera, screen, target, brightness=127):
        self.heightBits = yield from self.measureBitsRoutine(self.helpers.heightBits(*camera.resolution), mask, camera, screen, brightness, True)
        target.setImage(self.heightBits)
        return
    
    def measureBitsRoutine(self, bits, mask, camera, screen, brightness, invert=False):
        self.clearFrameBuffer()
        displayedBuffer = bits[:, :, 0] * brightness
        
        for i in range(15):
            bitIndex = (i + 1) // 2
            screen.setImage(displayedBuffer)
            yield
            frame = yield from self.captureNewFrameRoutine(camera)
            if i % 2 is 0:
                displayedBuffer = (1 - bits[:, :, bitIndex]) * brightness
            else:
                displayedBuffer = bits[:, :, bitIndex] * brightness
        return self.createGradient(camera.resolution, self.frameBuffer, mask, invert)
        
    def createGradient(self, resolution, frames, mask, invert):
        measuredBits = np.ones((resolution) + (8, ), dtype=np.uint8)
        lastResult = np.full(resolution, invert, dtype=np.uint8)
        
        for i in range(15):
            bitIndex = (i + 1) // 2
            frame = frames[i]
            if i % 2 is 0:
                darkFrameBuffer = frame
            else:
                bitmask = self.createMask((frame, darkFrameBuffer), 1)
                lastResult = bitmask == lastResult # xor with last bitmask - Grey -> binary
                measuredBits[:, :, bitIndex - 1] = lastResult
        return np.packbits(measuredBits, axis=-1)[:, :, 0] * mask
        
    def createMask(self, frames, threshold, erode=False):
        frame, darkFrame = frames
        mask = cv2.threshold(cv2.subtract(frame, darkFrame), thresh=threshold, maxval=1, type=cv2.THRESH_BINARY)[1]
        if erode is True:
            mask = self.erodeMask(mask)
        return mask
        
    def calibrateGreycodes(self, camera, widthData, heightData, calibration=None):
        rawData    = np.zeros((*widthData.shape, 3), dtype=np.uint8)
        rawData[..., 2] = widthData; rawData[..., 1] = heightData
        leftData   = rawData [:, : int(rawData.shape[1] / 2)  ]
        rightData  = rawData [:,   int(rawData.shape[1] / 2) :]
        leftCoeffs = calcCoeffs(0, leftData, camera.pixelsToRectilinear)
        rightCoeffs = calcCoeffs(1, rightData, camera.pixelsToRectilinear)
        return {
            'left_uv_to_rect_x' : leftCoeffs[0].flatten().tolist(),  'left_uv_to_rect_y': leftCoeffs[1].flatten().tolist(),
            'right_uv_to_rect_x': rightCoeffs[0].flatten().tolist(), 'right_uv_to_rect_y': rightCoeffs[1].flatten().tolist()
        }

class ImageArea(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def setImage(self, image: np.ndarray):
        return

class Screen(ImageArea):
    
    @abc.abstractmethod
    def setImage(self, image: np.ndarray):
        return
        
    @abc.abstractmethod
    def show(self):
        return
    
    @abc.abstractmethod
    def close(self):
        return
    
    @abc.abstractmethod
    def setGeometry(self, rect: QRect):
        return
    
    @property
    @abc.abstractmethod
    def resolution(self) -> typing.Tuple[int, int]:
        return
        
class QImageArea(ImageArea):
    
    def __init__(self, label):
        self.label = label
        
    def setImage(self, image):
        height, width = image.shape
        qImg = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        self.label.setPixmap(QPixmap(qImg))
        return
        
class QPatternScreen(QImageArea, Screen):
    
    def __init__(self):
        QImageArea.__init__(self, QLabel())
        self.widget = QWidget()
        self.widget.setWindowState(Qt.WindowFullScreen)
        layout = QVBoxLayout()
        self.label.setPixmap(QPixmap(u"blank.png"))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label.setSizePolicy(sizePolicy)
        self.label.setScaledContents(True)
        layout.addWidget(self.label)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget.setLayout(layout)
        self._resolution = (0, 0)
        return
        
    def show(self):
        self.widget.show()
        return
    
    def close(self):
        self.widget.close()
        return
    
    def setGeometry(self, rect: QRect):
        self.widget.setGeometry(rect)
        self._resolution = (rect.height(), rect.width())
        return
        
    @property
    def resolution(self):
        return self._resolution

class CalibrationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_V2Form()
        self.ui.setupUi(self)
        
        self.selectedCamera = None
        self.supportedCameras = (
            ("T265/1", T265Camera),
        )
        self.coroutines = {}
        
        deviceCB = self.ui.deviceComboBox
        deviceCB.addItems((cam[0] for cam in self.supportedCameras))
        deviceCB.setCurrentIndex(-1)
        deviceCB.currentIndexChanged.connect(self.setupCamera)
        
        self.ui.createMaskPushButton.pressed.connect(self.onCreateMaskPressed)
        self.ui.widthBitsPushButton.pressed.connect(self.onWidthBitsPressed)
        self.ui.heightBitsPushButton.pressed.connect(self.onHeightBitsPressed)
        self.ui.polyFitPushButton.pressed.connect(self.onPolyFitPressed)
        self.ui.undistortCheckBox.stateChanged.connect(self.onUndistortStateChanged)
        self.ui.maskThresholdSlider.valueChanged.connect(self.onMaskThresholdChanged)
        self.ui.exposureSlider.valueChanged.connect(self.onExposureValueChanged)
        self.ui.displayIndexSpinBox.valueChanged.connect(self.onDisplayIndexChanged)

        self.appTimer = QTimer(self)
        self.appTimer.timeout.connect(self.update)
        self.appTimer.start(20)
        
        QTimer.singleShot(self.ui.displayDelaySpinBox.value(), self.coroutineUpdate)
        
        self.patternScreen = QPatternScreen()
        self.patternScreen.show()
        
        self.cameraFeed = QImageArea(self.ui.cameraFeedLabel)
        self.resultArea = QImageArea(self.ui.resultAreaLabel)
        
        self.app = QApplication.instance()
        
        #init according to GUI
        self.onDisplayIndexChanged(self.ui.displayIndexSpinBox.value())
        
        #misc
        self.calibrationManager = CalibrationManager()
        return
        
    def coroutineUpdate(self):
        for key in tuple(self.coroutines.keys()):
            try:
                next(self.coroutines[key])
            except StopIteration:
                self.coroutines.pop(key)
        QTimer.singleShot(self.ui.displayDelaySpinBox.value(), self.coroutineUpdate)
        return
        
    def update(self):
        self.updateCameraFeed()
        self.ui.displayIndexSpinBox.setMaximum(len(app.screens()) - 1)
        return
        
    def updateCameraFeed(self):
        if self.selectedCamera is not None:
            if self.selectedCamera.isAlive is True:
                if self.ui.liveFeedCheckBox.isChecked() is True:                
                    ret, frame = self.selectedCamera.read(peek=True)
                    if ret:
                        self.cameraFeed.setImage(frame)
            else:
                #camera died
                self.ui.deviceComboBox.blockSignals(True)
                self.ui.deviceComboBox.setCurrentIndex(-1)
                self.ui.deviceComboBox.blockSignals(False)
                self.selectedCamera = None
        return
        
    def setupCamera(self, cameraIndex):
        if self.selectedCamera is not None:
            self.selectedCamera.release()
        self.selectedCamera = self.supportedCameras[cameraIndex][1]()
        #init camera according to GUI
        self.onUndistortStateChanged(self.ui.undistortCheckBox.checkState())
        self.onExposureValueChanged(self.ui.exposureSlider.value())
        return
        
    def closeEvent(self, event):
        self.patternScreen.close()
        event.accept()
        if self.selectedCamera is not None:
            self.selectedCamera.release()
        return
        
    def onCreateMaskPressed(self):
        self.coroutines["createMask"] = self.calibrationManager.createMonitorMaskRoutine(self.selectedCamera, self.patternScreen, self.resultArea, self.ui.maskThresholdSlider.value())
        return
        
    def onUndistortStateChanged(self, value):
        if self.selectedCamera is not None:
            self.selectedCamera.undistort = value == 2
        return
        
    def onExposureValueChanged(self, value):
        if self.selectedCamera is not None:
            self.selectedCamera.exposure = value
        return
        
    def onDisplayIndexChanged(self, value):
        screen = self.app.screens()[value]
        self.displayResolution = (screen.size().width(), screen.size().height())
        self.patternScreen.setGeometry(screen.geometry())
        return
        
    def onMaskThresholdChanged(self, value):
        self.calibrationManager.screenMask = self.calibrationManager.createMask(self.calibrationManager.frameBuffer, value, True)
        self.resultArea.setImage(self.calibrationManager.screenMask * 255)
        return
        
    def onWidthBitsPressed(self):
        self.coroutines["measureWidthBits"] = self.calibrationManager.measureWidthBitsRoutine(self.calibrationManager.screenMask, self.selectedCamera, self.patternScreen, self.resultArea)
        return
        
    def onHeightBitsPressed(self):
        self.coroutines["measureHeightBits"] = self.calibrationManager.measureHeightBitsRoutine(self.calibrationManager.screenMask, self.selectedCamera, self.patternScreen, self.resultArea)
        return
        
    def onPolyFitPressed(self):
        cal = self.calibrationManager.calibrateGreycodes(self.selectedCamera, self.calibrationManager.widthBits, self.calibrationManager.heightBits)
        lut = LookupTable()
        lut.loadCameraProperties(r"data\CameraProperties.json")
        lut.fillLuT(cal)
        
        targetResolution = (1440, 1600)
        targetWidth, targetHeight = targetResolution
        pixelRect = lut.cameraProperties["pixelRect"]
        import cv2
        img = cv2.imread("imgs\charuco.png", cv2.IMREAD_GRAYSCALE)
        #remap from checker to region captured by left / right camera
        ry, rx = np.indices(targetResolution[::-1]).astype(np.float32)
        scaleX = abs(pixelRect["r11"][0] - pixelRect["r01"][0]) / targetWidth
        scaleY = abs(pixelRect["r00"][1] - pixelRect["r01"][1]) / targetHeight
        rx = rx * scaleX + pixelRect["r01"][0]
        ry = ry * scaleY + pixelRect["r01"][1]
        imgl = cv2.remap(img, rx - 32, ry, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        imgr = cv2.remap(img, rx + 32, ry, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        
        lrx = lut.lut[0, :, :, 2].astype(np.float32)
        lry = (65535 - lut.lut[0, :, :, 1].astype(np.float32)) #flip y bcs opencv...
        rrx = lut.lut[1, :, :, 2].astype(np.float32)
        rry = (65535 - lut.lut[1, :, :, 1].astype(np.float32))
        
        lrx = cv2.resize(lrx, targetResolution) #resize LuT to output res
        lry = cv2.resize(lry, targetResolution)
        rrx = cv2.resize(rrx, targetResolution)
        rry = cv2.resize(rry, targetResolution)
        
        lrx = lrx / 65535 * (targetWidth - 1) #scale for mapping in input image
        lry = lry / 65535 * (targetHeight - 1)
        rrx = rrx / 65535 * (targetWidth - 1)
        rry = rry / 65535 * (targetHeight - 1)
        
        img1 = cv2.remap(imgl, lrx, lry, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        img2 = cv2.remap(imgr, rrx, rry, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        
        self.resultArea.setImage(np.hstack((img1, img2)))
        self.patternScreen.setImage(np.hstack((img1, img2)))
        print(cal)
        print(CalibrationHelpers.calibration2GLSL(cal))
        return

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    widget = CalibrationWidget()
    widget.show()
    sys.exit(app.exec())
