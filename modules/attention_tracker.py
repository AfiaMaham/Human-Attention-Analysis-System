import time


class AttentionTracker:
    NO_EYE_CONTACT_SEC = 5.0
    SLOUCH_SEC = 3.0
    HEAD_AWAY_SEC = 3.0
    YAW_THRESHOLD = 30
    PITCH_THRESHOLD = 25

    def __init__(self):
        self._no_eye_start = None
        self._slouch_start = None
        self._head_away_start = None

    def update(
        self,
        eye_contact: bool,
        is_slouching: bool,
        pitch: float,
        yaw: float) -> tuple[bool, list[str]]:

        now = time.time()
        reasons: list[str] = []

        if not eye_contact:
            if self._no_eye_start is None:
                self._no_eye_start = now
            elapsed = now - self._no_eye_start
            if elapsed >= self.NO_EYE_CONTACT_SEC:
                reasons.append(f"No eye contact for {elapsed:.0f}s")
        else:
            self._no_eye_start = None

        if is_slouching:
            if self._slouch_start is None:
                self._slouch_start = now
            elapsed = now - self._slouch_start
            if elapsed >= self.SLOUCH_SEC:
                reasons.append(f"Slouching for {elapsed:.0f}s")
        else:
            self._slouch_start = None

        head_away = (abs(yaw) > self.YAW_THRESHOLD or
                     abs(pitch) > self.PITCH_THRESHOLD)
        if head_away:
            if self._head_away_start is None:
                self._head_away_start = now
            elapsed = now - self._head_away_start
            if elapsed >= self.HEAD_AWAY_SEC:
                reasons.append(
                    f"Head turned away for {elapsed:.0f}s "
                    f"(yaw={yaw:+.0f}° pitch={pitch:+.0f}°)"
                )
        else:
            self._head_away_start = None

        return len(reasons) == 0, reasons
