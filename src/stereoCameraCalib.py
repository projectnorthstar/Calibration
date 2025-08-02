from collections import deque
import numpy as np
import cv2
import json
import os

class Calibrator:
    def __init__(self):
        self.frame_queue = deque()
        self.img_size = None
        self.left_pts = []
        self.right_pts = []
        self.pattern_size = (8, 7)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
        self.cal_res = {}
        self.square_size = 0.02
        self.subpix_win_size = (11, 11)
        return

    def process_frame(self, record, skip=True):
        frame = None

        while len(self.frame_queue) > 0:
            frame = self.frame_queue.popleft()
            if skip is False:
                break

        if frame is not None:
            img_l, img_r = np.hsplit(frame, 2)
            left_img = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
            if self.img_size is None:
                self.img_size = (left_img.shape[1], left_img.shape[0])

            res_left, corners_left = cv2.findChessboardCorners(left_img, self.pattern_size)
            if res_left is True:
                res_right, corners_right = cv2.findChessboardCorners(right_img, self.pattern_size)
                if res_right:
                    corners_left = cv2.cornerSubPix(left_img, corners_left, self.subpix_win_size, (-1,-1), self.criteria)
                    corners_right = cv2.cornerSubPix(right_img, corners_right, self.subpix_win_size, (-1,-1), self.criteria)
                    s = self.img_size
                    cv2.drawChessboardCorners(frame[0:s[1], 0:s[0]], self.pattern_size, corners_left, True)
                    cv2.drawChessboardCorners(frame[0:s[1], s[0]:s[0]<<1], self.pattern_size, corners_right, True)
                    if record:
                        self.left_pts.append(corners_left)
                        self.right_pts.append(corners_right)
                        print(f"{len(self.left_pts)} samples recorded")
        return frame

    def calibrate(self):
        if len(self.left_pts) == 0:
            return False
        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(self.pattern_size).T.reshape(-1, 2)
        pattern_points = [pattern_points * self.square_size] * len(self.left_pts)

        err, lcm, ldc, rcm, rdc, rm, t, _, _ = cv2.stereoCalibrate(pattern_points, self.left_pts, self.right_pts, None, None, None, None, self.img_size, flags=cv2.CALIB_FIX_TANGENT_DIST)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(lcm, ldc, rcm, rdc, self.img_size, rm, t)

        self.cal_res["leftCameraMatrix"] = lcm.tolist()
        self.cal_res["rightCameraMatrix"] = rcm.tolist()
        self.cal_res["leftDistCoeffs"] = ldc.tolist()
        self.cal_res["rightDistCoeffs"] = rdc.tolist()
        self.cal_res["R"] = rm.tolist()
        self.cal_res["T"] = t.ravel().tolist()
        self.cal_res["R1"] = R1.tolist()
        self.cal_res["R2"] = R2.tolist()
        self.cal_res["P1"] = P1.tolist()
        self.cal_res["P2"] = P2.tolist()
        self.cal_res["imageWidth"] = self.img_size[0]
        self.cal_res["imageHeight"] = self.img_size[1]
        self.cal_res["baseline"] = np.linalg.norm(t)
        return True

    def export(self, path):
        with open(path, "w") as f:
            json.dump(self.cal_res, f, indent=4)
        return

if __name__ == "__main__":
    camera_index = 0
    cap_fps = 60
    cap_width = 2560
    cap_height = 800
    relative_path = "data/out.json"
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap.set(cv2.CAP_PROP_FPS, cap_fps)

    calib = Calibrator()

    while True:
        record = False
        waitedKey = cv2.waitKey(10) & 0xFF
        if waitedKey == 32: #space
            record = True
        elif waitedKey == 27: #escif frame is not None:
            break        
        ret, frame = cap.read()
        if ret is True:
            calib.frame_queue.append(frame)
            frame = calib.process_frame(record)
            cv2.imshow("frame", frame)
        
    cv2.destroyAllWindows()

    if calib.calibrate():
        calib.export(relative_path)
        absolute_path = os.path.abspath(relative_path)
        print(f"Calibration written to file: {absolute_path}")