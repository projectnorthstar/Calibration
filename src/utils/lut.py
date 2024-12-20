from utils.polyHelpers import polyfit2d
from utils.transformHelpers import worldToViewport
import json
import numpy as np
import sys
import imageio
import os
import cv2

class ARRaytracer:
    
    def __init__(self):
        self.calibrationData = {}
        return
        
    def matmul(self, v1, v2, pad=False, padBy=1.0):
        if(pad is True):
            return np.matmul(v1, np.append(v2, padBy))[:-1]
        return np.matmul(v1, v2)
        
    def getPoint(self, ray, distance):
        return ray[0] + ray[1] * distance
        
    def normalize(self, v):
        return v / self.magnitude(v)
        
    def loadV1Calibration(self, path):
        data = None
        with open(path, "r") as f:
            data = json.load(f)
        self.calibrationData = {}
        for k1 in ("leftEye", "rightEye"):
            self.calibrationData[k1] = {}
            for k2 in ("screenForward", "screenPosition", "eyePosition"):
                self.calibrationData[k1][k2] = np.array([data[k1][k2]["x"], data[k1][k2]["y"], data[k1][k2]["z"]])
            for k2 in ("sphereToWorldSpace", "worldToScreenSpace"):
                self.calibrationData[k1][k2] = np.array([[data[k1][k2][f"e{i}{j}"] for j in range(4)] for i in range(4)]).reshape((4,4))
            for k2 in ("eyeRotation", ):
                self.calibrationData[k1][k2] = np.array([data[k1][k2]["x"], data[k1][k2]["y"], data[k1][k2]["z"], data[k1][k2]["w"]])
            self.calibrationData[k1]["worldToSphereSpace"] = np.linalg.inv(self.calibrationData[k1]["sphereToWorldSpace"])
            self.calibrationData[k1]["ellipseMinorAxis"] = data[k1]["ellipseMinorAxis"]
            self.calibrationData[k1]["ellipseMajorAxis"] = data[k1]["ellipseMajorAxis"]
        return
        
    def reflect(self, inDirection, inNormal):
        factor = -2.0 * self.matmul(inNormal, inDirection)
        return np.array([factor * inNormal[0] + inDirection[0], factor * inNormal[1] + inDirection[1], factor * inNormal[2] + inDirection[2]]) 
    
    def magnitude(self, v):
        return np.sqrt(self.sqrMagnitude(v))
        
    def sqrMagnitude(self, v):
        return self.matmul(v, v)
    
    def project(self, v1, v2):
        v2Norm = self.normalize(v2)
        return self.matmul(v1, v2Norm) * v2Norm
    
    def intersectLineSphere(self, origin, direction, spherePos, sphereRadiusSqrd, frontSide = True):
        l = spherePos - origin
        offsetFromSphereCenterToRay = self.project(l, direction) - l
        sqrmOffsetFromSphereCenterToRay = self.sqrMagnitude(offsetFromSphereCenterToRay)
        return (self.matmul(l, direction) - np.sqrt(sphereRadiusSqrd - sqrmOffsetFromSphereCenterToRay) * (1.0 if frontSide else -1.0)) if (sqrmOffsetFromSphereCenterToRay <= sphereRadiusSqrd) else -1.0
    
    def intersectPlane(self, n, p0, l0, l):
        denom = self.matmul(-n, l)
        if (denom > sys.float_info.min):
            p0l0 = p0 - l0
            t = self.matmul(p0l0, -n) / denom
            return t
        return -1.0
    
    #for eye ray direction in !world space! returns screen UV
    def renderUVToDisplayUV(self, inputUV, optics):
        sphereSpaceRayOrigin = self.matmul(optics["worldToSphereSpace"], optics["eyePosition"], True)
        sphereSpaceRayDirection = self.matmul(optics["worldToSphereSpace"], optics["eyePosition"] + inputUV, True) - sphereSpaceRayOrigin
        sphereSpaceRayDirection = self.normalize(sphereSpaceRayDirection)
        intersectionTime = self.intersectLineSphere(sphereSpaceRayOrigin, sphereSpaceRayDirection, np.array([0.0, 0.0, 0.0]), 0.5 * 0.5, False)
        if (intersectionTime < 0.0):
            return -np.ones(2)
        sphereSpaceIntersection = sphereSpaceRayOrigin + (intersectionTime * sphereSpaceRayDirection)
        
        #Ellipsoid  Normals
        sphereSpaceNormal = -self.normalize(sphereSpaceIntersection)
        sphereSpaceNormal[0] /= np.power(optics["ellipseMinorAxis"] / 2.0, 2.0)
        sphereSpaceNormal[1] /= np.power(optics["ellipseMinorAxis"] / 2.0, 2.0)
        sphereSpaceNormal[2] /= np.power(optics["ellipseMajorAxis"] / 2.0, 2.0)
        sphereSpaceNormal = self.normalize(sphereSpaceNormal)
        
        worldSpaceIntersection = self.matmul(optics["sphereToWorldSpace"], sphereSpaceIntersection, True)
        worldSpaceNormal = self.matmul(optics["sphereToWorldSpace"], sphereSpaceNormal, True, 0.0)
        worldSpaceNormal = self.normalize(worldSpaceNormal)
        
        firstBounce = (worldSpaceIntersection, self.normalize(self.reflect(inputUV, worldSpaceNormal)))
        intersectionTime = self.intersectPlane(optics["screenForward"], optics["screenPosition"], firstBounce[0], firstBounce[1])
        if (intersectionTime < 0.0):
            return -np.ones(2)
        planeIntersection = self.getPoint(firstBounce, intersectionTime)

        screenUV = self.matmul(optics["worldToScreenSpace"], planeIntersection, True)
        screenUV = np.array([0.5 + screenUV[1], 0.5 + screenUV[0]])
        return screenUV
        
    def fit(self, optics, polynomialDegree=3):
        displayUVs = []
        cameraRays = []
        for rx in np.linspace(-1, 1, num=32):
            for ry in np.linspace(-1, 1, num=32):
                ray = np.array([rx, ry, 1.0])
                uv = self.renderUVToDisplayUV(ray, optics)
                uv = 1 - uv
                if uv[0] < 0 or uv[0] > 1 or uv[1] < 0 or uv[1] > 1:
                    continue
                displayUVs.append(uv)
                cameraRays.append(ray[:-1])
        cameraRays = np.array(cameraRays)
        displayUVs = np.array(displayUVs)
        x_coeffs = polyfit2d(
            displayUVs[:, 0], # "Input"  X Axis
            displayUVs[:, 1], # "Input"  Y Axis
            cameraRays[:, 0], # "Output" X Axis
            np.asarray([polynomialDegree, polynomialDegree])
        )
        y_coeffs = polyfit2d(
            displayUVs[:, 0], # "Input"  X Axis
            displayUVs[:, 1], # "Input"  Y Axis
            -cameraRays[:, 1], # "Output" Y Axis - is there because we need to match flip in standard V2 (OpenCV vs Unity)
            np.asarray([polynomialDegree, polynomialDegree])
        )
        return x_coeffs, y_coeffs

class LookupTable:

    def __init__(self, resolution=(400, 360)):
        self.lut = np.zeros((2, *resolution, 3), dtype=np.float32)
        self.cameraProperties = None
        return
        
    def loadCameraProperties(self, path):
        data = None
        with open(path, "r") as f:
            self.cameraProperties = json.load(f)
        return
        
    def loadV2Calibration(self, path):
        data = None
        with open(path, "r") as f:
            data = json.load(f)
        self.fillLuT(data)
        return
        
    def fillLuT(self, data):
        _, height, width, _ = self.lut.shape
        xData = (np.array(data["left_uv_to_rect_x"]), np.array(data["right_uv_to_rect_x"]))
        yData = (np.array(data["left_uv_to_rect_y"]), np.array(data["right_uv_to_rect_y"]))
        pm = np.array(self.cameraProperties["projectionMatrix"]).reshape((4,4))
        v, u = np.indices((height, width))
        u = u.ravel() / (width - 1)
        v = 1.0 - v.ravel() / (height - 1)
        for i in range(2):
            points = np.ones((u.shape[0], 3))
            points[:, 0] = np.polynomial.polynomial.polyval2d(u, v, xData[i].reshape((4,4)))
            #flip y to convert it from opencv to unity coordinate space, flip z bcs camera's forward is the negative Z axis, see: https://docs.unity3d.com/ScriptReference/Camera-worldToCameraMatrix.html
            points[:, 1] = -np.polynomial.polynomial.polyval2d(u, v, yData[i].reshape((4,4)))
            points[:, 2] *= -1
            rg = worldToViewport(pm, points)
            rg = rg.reshape((height, width, 2))
            self.lut[i, :, :, 2] = rg[:, :, 0]
            self.lut[i, :, :, 1] = rg[:, :, 1]
        return
        
    def loadV1Calibration(self, path):
        rt = ARRaytracer()
        rt.loadV1Calibration(path)

        coeffs = [rt.fit(rt.calibrationData[f"leftEye"]), rt.fit(rt.calibrationData[f"rightEye"])]
        data = {
            "left_uv_to_rect_x": coeffs[0][0].ravel().tolist(),
            "left_uv_to_rect_y": coeffs[0][1].ravel().tolist(),
            "right_uv_to_rect_x": coeffs[1][0].ravel().tolist(),
            "right_uv_to_rect_y": coeffs[1][1].ravel().tolist()
        }
        print(json.dumps(data, indent=4))
        
        self.fillLuT(data)
        return

    def export(self, path, side=-1):
        ext = os.path.splitext(path)[-1]
        ext = ext[1:].lower()
        funcs = {
            "png": self.exportPNG,
            "exr": self.exportEXR
        }
        if ext not in funcs:
            raise ValueError(f"Invalid file extension, supported formats are: {tuple(funcs)}")
        data = None
        if side == -1:
            data = np.hstack(self.lut)
        elif -1 < side < 2:
            data = self.lut[side]
        else:
            raise ValueError("Invalid argument provided, side value must be from <-1, 1>")
        funcs[ext](path, data)
        return

    def exportEXR(self, path, data):
        imageio.plugins.freeimage.download()
        imageio.imwrite(path, data[:,:,::-1])
        return
        
    def exportPNG(self, path, data):
        data = (data * 65535).astype(np.uint16)
        cv2.imwrite(path, data)
        return

if __name__=="__main__":
    lut = LookupTable()
    lut.loadCameraProperties(r"data\CameraProperties.json")
    lut.loadV2Calibration(r"data\V2Out.json")
    #lut.loadV1Calibration(r"data\V1Out.json")
    cv2.imshow('lut_left', lut.lut[0])
    cv2.imshow('lut_right', lut.lut[1])
    stacked = np.hstack(lut.lut)
    cv2.imshow('lut_stacked', stacked)
    lut.export('lut_left.exr', 0)
    lut.export('lut_right.exr', 1)
    lut.export('lut.exr')
    lut.export('lut_left.png', 0)
    lut.export('lut_right.png', 1)
    lut.export('lut.png')
    cv2.waitKey(0)
