import math
import numpy as np
import cv2
from dataclasses import dataclass, field

@dataclass
class HeadPoseResult:
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    cal_x: float = 0.0
    cal_y: float = 0.0
    raw_x: float = 0.0
    raw_y: float = 0.0


class HeadPoseModule:
    _WORLD_PTS = np.array([
        [285, 528, 200],
        [285, 371, 152],
        [197, 574, 128],
        [173, 425, 108],
        [360, 574, 128],
        [391, 425, 108],
    ], dtype=np.float64)
    _LANDMARK_IDS_POSE = [1, 9, 57, 130, 287, 359]
    _LANDMARK_IDS_EYE = [33, 263, 1, 61, 291, 199]

    def __init__(self):
        self._calibrated_x = 6.0
        self._calibrated_y = 0.0
        self._needs_calibration = False
        self.last: HeadPoseResult = HeadPoseResult()

    def request_calibration(self):
        self._needs_calibration = True

    def process(self, face_landmarks, img_h: int, img_w: int) -> HeadPoseResult:
        lm = face_landmarks.landmark
        cam, dist = self._camera_matrix(img_h, img_w)

        pitch, yaw, roll = self._compute_pitch_yaw_roll(lm, img_h, img_w, cam, dist)
        raw_x, raw_y, cal_x, cal_y = self._compute_calibrated_xy(lm, img_h, img_w, cam, dist)

        self.last = HeadPoseResult(
            pitch=pitch, yaw=yaw, roll=roll,
            raw_x=raw_x, raw_y=raw_y,
            cal_x=cal_x, cal_y=cal_y,
        )
        return self.last

    @staticmethod
    def _camera_matrix(img_h, img_w):
        focal = float(img_w)
        cam = np.array([[focal, 0, img_w / 2],
                        [0, focal, img_h / 2],
                        [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)
        return cam, dist

    def _compute_pitch_yaw_roll(self, lm, img_h, img_w, cam, dist):
        pts_2d = np.array(
            [[int(lm[idx].x * img_w), int(lm[idx].y * img_h)]
             for idx in self._LANDMARK_IDS_POSE],
            dtype=np.float64,
        )
        ok, rvec, _ = cv2.solvePnP(self._WORLD_PTS, pts_2d, cam, dist)
        if not ok:
            return self.last.pitch, self.last.yaw, self.last.roll

        R, _ = cv2.Rodrigues(rvec)
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        z = math.atan2(R[1, 0], R[0, 0])
        angles = np.array([x, y, z]) * 180.0 / math.pi
        return float(angles[0]), float(angles[1]), float(angles[2])

    def _compute_calibrated_xy(self, lm, img_h, img_w, cam, dist):
        pts_2d = np.array(
            [[int(lm[idx].x * img_w), int(lm[idx].y * img_h)]
             for idx in self._LANDMARK_IDS_EYE],
            dtype=np.float64,
        )
        pts_3d = np.array(
            [[int(lm[idx].x * img_w), int(lm[idx].y * img_h), lm[idx].z]
             for idx in self._LANDMARK_IDS_EYE],
            dtype=np.float64,
        )

        ok, rvec, _ = cv2.solvePnP(pts_3d, pts_2d, cam, dist,
                                    flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return self.last.raw_x, self.last.raw_y, self.last.cal_x, self.last.cal_y

        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        raw_x = float(angles[0] * 360)
        raw_y = float(angles[1] * 360)

        if self._needs_calibration:
            self._calibrated_x = raw_x
            self._calibrated_y = raw_y
            self._needs_calibration = False

        return raw_x, raw_y, raw_x - self._calibrated_x, raw_y - self._calibrated_y
