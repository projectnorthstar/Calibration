import numpy as np
import math
from scipy.spatial.transform import Rotation

def worldToViewport(projectionMatrix, points):
    points4 = np.ones((points.shape[0], 4), np.float32)
    points4[:, 0:3] = points
    rg = points4 @ projectionMatrix.T
    rg[:, 0] = rg[:, 0] / rg[:, 3] * 0.5 + 0.5
    rg[:, 1] = rg[:, 1] / rg[:, 3] * 0.5 + 0.5
    return rg[:, 0:2]
    
def viewportToWorld(projectionMatrix, points, z = 1.0):
    points4 = np.ones((points.shape[0], 4), np.float32)
    points4[:, 0:2] = points * 2.0 - 1.0
    rg = points4 @ np.linalg.inv(projectionMatrix).T
    rg[:, :] /= rg[:, 3]
    scale = z / rg[:, 2][0]
    rg[:, 0] *= scale
    rg[:, 1] *= scale
    rg[:, 2] = z
    return rg[:, 0:3]
    
def worldToPixel(points, wh=2.048, dpm=1000, flipY=False):
    points += wh * 0.5
    if flipY is True:
        points[:, 1] = wh - points[:, 1]
    return points[:, 0:2] * dpm
    
def rotToEuler(rot, degrees=True) :
    r = Rotation.from_matrix(rot)
    euler = r.as_euler("zxy", degrees)
    return euler[[1, 2, 0]]
    
def eulerToRot(theta, degrees=True) :
    r = Rotation.from_euler("zxy", (theta[2], theta[0], theta[1]), degrees)
    return r.as_matrix()
    
def composeTR(pos, rot):
    tr = np.eye(4)
    tr[0:3,0:3] = rot
    tr[0:3, 3] = pos
    return tr
    
def composeTRS(pos, rot, scale):
    scaleMat = np.eye(4)
    scaleMat[0,0] = scale[0]
    scaleMat[1,1] = scale[1]
    scaleMat[2,2] = scale[2]
    return np.matmul(composeTR(pos, rot), scaleMat)
    
def pad(a, val = 1.0, n = 1, axis = -1):
    axes = [(0, 0)] * len(a.shape)
    axes[axis] = (0, n)
    return np.pad(a, axes, 'constant', constant_values=(val))
    
if __name__ == "__main__":
    rm = eulerToRot((0, 90, 45))
    print(rm)
    print(rotToEuler(rm))
    p = np.array((1, 2, 3, 1))
    tr = composeTR((1, 2, 3), rm)
    print(tr)
    res = (p @ tr.T)[0:3]
    print(res)
    tr = composeTRS((1, 2, 3), rm, (2, 3, 4)) #local to world
    print(tr)
    res = (p @ tr.T)[0:3]
    print(res)
    
