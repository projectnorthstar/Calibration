import numpy as np
import abc
import typing
from transformHelpers import composeTRS, composeTR, eulerToRot, pad
import json
import cv2

class SurfaceRayTracer:
    
    @abc.abstractmethod
    def raysToIntersections(self, directions: np.ndarray, origin: np.ndarray):
        return
    
    @abc.abstractmethod
    def reflectRays(self, directions: np.ndarray, origin: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        return

class DisplayRayTracer(SurfaceRayTracer):
    
    def __init__(self, position, rotation):
        self.position = np.array(position)
        self.rotationMatrix = eulerToRot(rotation)
        self.normal = np.array((0, 0, 1)) @ self.rotationMatrix.T
        self.localToWorld = composeTR(self.position, self.rotationMatrix)
        self.worldToLocal = np.linalg.inv(self.localToWorld)
        return
    
    def raysToIntersections(self, directions, origins, local=False):
        denoms = np.sum(self.normal * directions, axis = 1)
        denoms[np.abs(denoms) < 1e-6] = np.nan
        segments = self.position - origins
        t = np.sum(segments * self.normal, axis = 1) / denoms
        t[t < 0] = np.nan
        intersections = origins + directions * t[:, np.newaxis]
        if local is True:
            intersections = (pad(intersections) @ self.worldToLocal.T)[:, 0:3]
        return intersections
        
    def reflectRays(self, directions, origin):
        raise NotImplementedError #might be interesting though
        return

class SphereRayTracer(SurfaceRayTracer):
    
    def __init__(self, center, radius = 0.5):
        self.radius = radius
        self.center = np.array(center)
        return
        
    def raysToIntersections(self, directions, origin = (0, 0, 0), local=False):
        ndirs = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        orientedSegments = self.center - origin
        t = np.sum(orientedSegments * ndirs, axis = 1)
        pe = origin + ndirs * t[:, np.newaxis]
        d = np.linalg.norm(pe - self.center, axis = 1)
        i = (self.radius ** 2 - d ** 2) ** 0.5
        ps = origin + ndirs * (t + i)[:, np.newaxis]
        return ps
        
    def intersectionsToNormals(self, intersections):
        normals = intersections - self.center
        normals = normals / np.linalg.norm(normals, axis = 1)[:, np.newaxis]
        return normals
        
    def reflectRays(self, directions, origin = (0, 0, 0)):
        intersections = self.raysToIntersections(directions, origin)
        normals = self.intersectionsToNormals(intersections)
        factors = -2.0 * np.sum(normals * directions, axis = 1)
        reflected = normals * factors[:, np.newaxis] + directions
        reflected = reflected / np.linalg.norm(reflected, axis = 1)[:, np.newaxis]
        return intersections, reflected

class EllipsoidRayTracer(SurfaceRayTracer):
    
    def __init__(self, center, rotation, minorAxis, majorAxis):
        self.sphereRayTracer = SphereRayTracer((0, 0, 0))
        self.center = center
        self.minorAxis = minorAxis
        self.majorAxis = majorAxis
        self.rotationMatrix = eulerToRot(rotation)
        self.localToWorld = composeTRS(self.center, self.rotationMatrix, (self.minorAxis, self.minorAxis, self.majorAxis))
        self.worldToLocal = np.linalg.inv(self.localToWorld)
        return
        
    def raysToIntersections(self, directions, origin = (0, 0, 0), local=False):
        #local origin and rays
        ldirections = directions @ self.worldToLocal[0:3, 0:3].T #dir world to local
        lorigin = (self.worldToLocal @ pad(np.array(origin)))[0:3]
        intersections = self.sphereRayTracer.raysToIntersections(ldirections, lorigin) #local sphere intersection
        if local is False:
            intersections = (pad(intersections) @ self.localToWorld.T)[:, 0:3]
        return intersections
        
    def reflectRays(self, directions, origin = (0, 0, 0)): #should probably return (origins, directions)
        intersections = self.raysToIntersections(directions, origin, True) #local sphere intersection
        #get local normal, center (0, 0, 0)
        normals = self.sphereRayTracer.intersectionsToNormals(intersections)
        #scale normal from sphere space to ellipsoid
        normals[:, 0] /= np.power(self.minorAxis / 2.0, 2.0)
        normals[:, 1] /= np.power(self.minorAxis / 2.0, 2.0)
        normals[:, 2] /= np.power(self.majorAxis / 2.0, 2.0)
        normals / np.linalg.norm(normals, axis = 1)[:, np.newaxis]
        #apply localToWorld to normal (vector) - only scale and rotation
        wNormals = normals @ self.localToWorld[0:3, 0:3].T
        wNormals = wNormals / np.linalg.norm(wNormals, axis = 1)[:, np.newaxis]
        #apply localToWorld to intersections
        wIntersections = (pad(intersections) @ self.localToWorld.T)[:, 0:3]
        #reflect
        factors = -2.0 * np.sum(wNormals * directions, axis = 1)
        reflected = wNormals * factors[:, np.newaxis] + directions
        reflected = reflected / np.linalg.norm(reflected, axis = 1)[:, np.newaxis]
        return wIntersections, reflected

if __name__ == "__main__":
    
    displaySize = (0.066, 0.0594) #height, width
    minor = 0.24494899809360505
    major = 0.3099985122680664

    rays = None
    uvs = None
    samples = None
    with open("v2sample.json") as f:
        samples = json.load(f)["samples"]
    rays = np.ones((len(samples), 3))
    uvs = np.ones((len(samples), 2))
    for i, sample in enumerate(samples):
        rays[i, 0:2] = sample["ray"]
        uvs[i, 0:2] = sample["uv"]
    
    epos = [0.08926964, 0, -0.03249149]
    erot = [0, 110, 0]
    dpos = [0.05, -0.003, 0.05]
    drot = [0, -45, -10]
    
    imsize = (330, 297)
    iuvs = np.array(uvs)
    iuvs[:, 0] *= imsize[1]
    iuvs[:, 1] *= imsize[0]
    iuvs = iuvs.astype(np.uint16)
        
    waitedKey = 0
    while(True):
        #manual adjustments just for testing
        if waitedKey == ord('q'):
            break
        elif waitedKey == ord('a'):
            dpos[0] += 0.0001
        elif waitedKey == ord('s'):
            dpos[0] -= 0.0001
        elif waitedKey == ord('d'):
            dpos[1] += 0.0001
        elif waitedKey == ord('f'):
            dpos[1] -= 0.0001
        elif waitedKey == ord('g'):
            dpos[2] += 0.0001
        elif waitedKey == ord('h'):
            dpos[2] -= 0.0001
        elif waitedKey == ord('z'):
            drot[0] += 0.5
        elif waitedKey == ord('x'):
            drot[0] -= 0.5
        elif waitedKey == ord('c'):
            drot[1] += 0.5
        elif waitedKey == ord('v'):
            drot[1] -= 0.5
        elif waitedKey == ord('b'):
            drot[2] += 0.5
        elif waitedKey == ord('n'):
            drot[2] -= 0.5
        elif waitedKey == ord('A'):
            epos[0] += 0.0001
        elif waitedKey == ord('S'):
            epos[0] -= 0.0001
        elif waitedKey == ord('D'):
            epos[1] += 0.0001
        elif waitedKey == ord('F'):
            epos[1] -= 0.0001
        elif waitedKey == ord('G'):
            epos[2] += 0.0001
        elif waitedKey == ord('H'):
            epos[2] -= 0.0001
        elif waitedKey == ord('Z'):
            erot[0] += 0.5
        elif waitedKey == ord('X'):
            erot[0] -= 0.5
        elif waitedKey == ord('C'):
            erot[1] += 0.5
        elif waitedKey == ord('V'):
            erot[1] -= 0.5
        elif waitedKey == ord('B'):
            erot[2] += 0.5
        elif waitedKey == ord('N'):
            erot[2] -= 0.5
        
        #init ellipsoid ray tracer and reflect v2 rays
        rt = EllipsoidRayTracer(epos, erot, minor, major)
        rsi, rr = rt.reflectRays(rays)
        
        #init display ray tracer and get local intersections
        drt = DisplayRayTracer(dpos, drot)
        drsi = drt.raysToIntersections(rr, rsi, local=True)[:,:2]
        
        #convert to uvs
        drsi[:, 0] /= displaySize[1]
        drsi[:, 1] /= displaySize[0]
        drsi += 0.5
        #error
        print(np.sum((drsi - uvs)**2))
        
        #visualize
        drsi = np.clip(drsi, 0, 1)
        drsi[:, 0] *= imsize[1] - 1
        drsi[:, 1] *= imsize[0] - 1
        drsi = drsi.astype(np.uint16)
        uvimg = np.zeros((imsize[0], imsize[1], 3), dtype=np.uint8)
        uvimg[tuple((drsi[:, 1],drsi[:, 0]))] = 255
        uvimg[tuple((iuvs[:, 1],iuvs[:, 0], 2))] = 255
        cv2.imshow("t", uvimg)
        
        waitedKey = cv2.waitKey(0)
