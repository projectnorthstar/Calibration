from utils.cameras import T265Camera, XvisioCamera, LeapCamera, Cv2Camera
from utils.trackers import Alt
import time
import cv2
import sys
import numpy as np
from utils.transformHelpers import rotToEuler
import argparse

def kabsch(canonical_points, predicted_points):
    """
    canonical @ rotation.T + translation are close to predicted
    """
    canonical_mean = np.mean(canonical_points, axis=0)
    predicted_mean = np.mean(predicted_points, axis=0)

    canonical_centered = canonical_points - np.expand_dims(canonical_mean, axis=0)
    predicted_centered = predicted_points - np.expand_dims(predicted_mean, axis=0)

    cross_correlation = predicted_centered.T @ canonical_centered

    u, s, vt = np.linalg.svd(cross_correlation)

    rotation = u @ vt

    det = np.linalg.det(rotation)

    if det < 0.0:
        vt[-1, :] *= -1.0
        rotation = np.dot(u, vt)

    translation = predicted_mean - canonical_mean
    translation = np.dot(rotation, translation) - np.dot(rotation, predicted_mean) + predicted_mean

    return rotation, translation
    
def affine3D(canonical_points, predicted_points):
    res, scale = cv2.estimateAffine3D(np.array(canonical_points), np.array(predicted_points), force_rotation = True)
    return res[0:3, 0:3], res[0:3, 3:4].ravel() * scale

if __name__ == "__main__":
    
    alignmentFunc = affine3D #affine3D (uses scaling) or kabsch (no scaling)
    
    supportedCameras = {
        "T26x": {
            "cls": T265Camera,
            "kwargs": {
                "undistort": True,
                "exposure": 25000
            }
        },
        "Xvisio": {
            "cls": XvisioCamera,
            "kwargs": {
                "undistort": True,
                "exposure": 15000
            }
        },
        "Leap": {
            "cls": LeapCamera,
            "kwargs": {
                "undistort": False
            },
            "undistortCorners": True
        },
        "ELP": {
            "cls": Cv2Camera,
            "kwargs": {
                "index": 1,
                "exposure": -3
            }
        }
    }
    
    supportedTrackers = {
        "Alt": {
            "cls": Alt,
            "kwargs" : {}
        }
    }
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', help='Show help message and exit', action='store_true')
    parser.add_argument('-r', '--record', help='Record to csv', action='store_true')
    parser.add_argument('-s', '--sensors', help=f"List of sensors ({'|'.join(list(supportedCameras.keys()) + list(supportedTrackers.keys()))})...", nargs='+', required=True)
    parser.add_argument('-l', '--length', help='Marker length', type=float, default=0.1)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    trackersId = []
    camerasId = []
    sensors = []
    undistortCorners = []
    for i, key in enumerate(args.sensors):
        sensor = None
        if key in supportedCameras:
            sc = supportedCameras[key]
            sensor = sc["cls"](**sc["kwargs"])
            camerasId.append(i)
            undistortCorners.append(sc.get("undistortCorners") is True)
        elif key in supportedTrackers:
            sc = supportedTrackers[key]
            sensor = sc["cls"](**sc["kwargs"])
            trackersId.append(i)
        else:
            raise RuntimeError(f"Sensor {key} not supported")
        sensors.append(sensor)
    camerasId = tuple(camerasId)
    trackersId = tuple(trackersId)
    sensors = tuple(sensors)
    
    while not all(sensors[camId].ready for camId in camerasId):
        time.sleep(1)
        
    allPositions = [[] for _ in range(len(sensors))]
    while True:
        waitedKey = cv2.waitKey(10)
        frames = [sensors[camId].read(peek=True)[1] for camId in camerasId]
        poses = [sensors[tId].getPose() for tId in trackersId]
        perSensorPosition = [None] * len(sensors)
        for i, cid in enumerate(camerasId):
            cam = sensors[cid]
            positions = []
            for j, frame in enumerate(np.hsplit(frames[i], 2)):
                color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
                corners = [corner for ic, corner in enumerate(corners) if ids[ic] == 5]
                if len(corners) > 0:
                    color = cv2.aruco.drawDetectedMarkers(color, corners, borderColor=(0, 0, 255))
                    if undistortCorners[i] is True:
                        func = cam.leftPixelToRectilinear if j == 0 else cam.rightPixelToRectilinear
                        corners = np.apply_along_axis(func, -1, corners)
                    cm = cam.getCameraMatrix(j)
                    dc = np.zeros(5)
                    rvecs, tvecs, objpts = cv2.aruco.estimatePoseSingleMarkers(corners, args.length, cm, dc)
                    for k, tvec in enumerate(tvecs):
                        rvec = rvecs[k]
                        positions.append(tvec[0])
                    if undistortCorners[i] is not True: #would not work
                        color = cv2.drawFrameAxes(color, cm, dc, rvec, tvec, 0.05)
                cv2.imshow(f"{type(cam).__name__}_{j}", color)
            if len(positions) == 2:
                perSensorPosition[cid] = cam.leftRightToDevice(positions)
        for i, tid in enumerate(trackersId):
            tracker = sensors[tid]
            valid, rot, pos = poses[i]
            if valid is True:
                position = (np.zeros(3) - pos) @ rot #origin 0, 0, 0
                position[1] *= -1 #Flipped y Unity to OpenCV #TODO move to trackers, each sensor should follow OpenCV
                perSensorPosition[tid] = position
        if all(pos is not None for pos in perSensorPosition):
            for i, position in enumerate(perSensorPosition):
                allPositions[i].append(position)
        if waitedKey == ord("q"):
            break
            
    for i, sensor in enumerate(sensors):
        sensor.release()
    
    print(f"{len(allPositions[0])} samples recorded")
    if len(allPositions[0]) > 0:
        for i, sensor in enumerate(sensors):
            if args.record is True:
                np.savetxt(f"points_{type(sensor).__name__}.csv", allPositions[i], delimiter=",")

            if i > 0:
                r, t = alignmentFunc(allPositions[i], allPositions[0])
                print("From:", type(sensors[0]).__name__, "to:", type(sensors[i]).__name__)
                print("Unity")
                ru = rotToEuler(r) * (-1, 1, -1) #Flipped y OpenCV to Unity
                tu = t * (1, -1, 1) #Flipped y OpenCV to Unity
                print("r: ", f"Vector3({','.join(str(x) for x in ru)})")
                print("t: ", f"Vector3({','.join(str(x) for x in tu)})")
                print("OpenCV")
                print("r =", np.array2string(r, separator=','))
                print("t =", np.array2string(t, separator=','))
    