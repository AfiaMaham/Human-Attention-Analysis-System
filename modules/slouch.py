import time
import numpy as np

class SlouchModule:
    def __init__(
        self,
        slouch_threshold: float = 0.08,
        calibration_frames: int = 15,
        alert_cooldown: float = 1.5,
    ):
        self.slouch_threshold = slouch_threshold
        self.calibration_frames = calibration_frames
        self.alert_cooldown = alert_cooldown

        self.is_calibrated = False
        self._cal_data: list = []
        self.baseline = None

        self.current_distance = None
        self.slouch_count = 0
        self._last_alert = 0.0

    def recalibrate(self):
        self.is_calibrated = False
        self._cal_data = []
        self.baseline = None
        self.slouch_count = 0

    def process(self, landmarks, mp_pose) -> bool:
        d = self._neck_distance(landmarks, mp_pose)
        if d is None:
            return False

        if not self.is_calibrated:
            self._cal_data.append(d)
            if len(self._cal_data) >= self.calibration_frames:
                self.baseline      = float(np.median(self._cal_data))
                self.is_calibrated = True
            return False

        self.current_distance = d
        slouch_amount = d - self.baseline

        if slouch_amount > self.slouch_threshold:
            now = time.time()
            if now - self._last_alert > self.alert_cooldown:
                self.slouch_count += 1
                self._last_alert   = now
            return True

        return False

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return len(self._cal_data), self.calibration_frames

    @staticmethod
    def _neck_distance(landmarks, mp_pose):
        try:
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            shoulder_mid_y = (ls.y + rs.y) / 2.0
            return nose.y - shoulder_mid_y
        except Exception:
            return None
