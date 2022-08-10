import numpy as np

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