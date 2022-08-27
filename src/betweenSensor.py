from utils.cameras import T265Camera, XvisioCamera
import time
import cv2
import sys
import numpy as np
from utils.transformHelpers import rotToEuler
import argparse

def kabsch(canonical_points, predicted_points):
    """
    rotation from preditcted to canonical
    translation from predicted to canonical
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

if __name__ == "__main__":
    
    supportedCameras = {
        "T26x": T265Camera,
        "Xvisio": XvisioCamera
    }
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', help='Show help message and exit', action='store_true')
    parser.add_argument('-r', '--record', help='Record to csv', action='store_true')
    parser.add_argument('-s', '--sensors', help=f"List of sensors ({'|'.join(supportedCameras.keys())})...", nargs='+', required=True)
    parser.add_argument('-l', '--length', help='Marker length', type=float, default=0.1)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    arucoParams = cv2.aruco.DetectorParameters_create()
    arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    cameras = tuple(supportedCameras[key](undistort=True) for key in args.sensors)
    
    while not all(cam.ready for cam in cameras):
        time.sleep(1)
        
    allPositions = [[] for _ in range(len(cameras))]
    while True:
        waitedKey = cv2.waitKey(10)
        frames = [None] * len(cameras)
        for i, cam in enumerate(cameras):
            _, frames[i] = cam.read(peek=True)
        perCamPosition = []
        for i, cam in enumerate(cameras):
            positions = []
            for j, frame in enumerate(np.hsplit(frames[i], 2)):
                color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
                color = cv2.aruco.drawDetectedMarkers(color, corners, borderColor=(0, 0, 255))
                if len(corners) > 0:
                    cm = cam.getCameraMatrix(j)
                    dc = np.zeros(5)
                    rvecs, tvecs, objpts = cv2.aruco.estimatePoseSingleMarkers(corners, args.length, cm, dc)
                    for k, tvec in enumerate(tvecs):
                        rvec = rvecs[k]
                        positions.append(tvec[0])
                    color = cv2.drawFrameAxes(color, cm, dc, rvec, tvec, 0.05)
                cv2.imshow(f"{type(cam).__name__}_{j}", color)
            if len(positions) == 2:
                perCamPosition.append(cam.leftRightToDevice(positions))
        if len(perCamPosition) == len(cameras):
            for i, position in enumerate(perCamPosition):
                allPositions[i].append(position)
        if waitedKey == ord("q"):
            break
            
    for i, cam in enumerate(cameras):
        cam.release()
        
        if args.record is True:
            np.savetxt(f"points_{type(cam).__name__}.csv", allPositions[i], delimiter=",")

        if i > 0:
            r, t = kabsch(allPositions[0], allPositions[i])
            print("From:", type(cameras[0]).__name__, "to:", type(cameras[i]).__name__)
            print("R (parent): ", rotToEuler(r))
            print("T (local): ", t * (1, -1, 1)) #OpenCV to Unity