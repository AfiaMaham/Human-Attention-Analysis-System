import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_hud(
    frame: np.ndarray,
    *,
    eye_contact: bool,
    gaze_text: str,
    head_text: str,
    is_slouching: bool,
    slouch_calibrated: bool,
    slouch_cal_progress: int,
    slouch_cal_total: int,
    pitch: float,
    yaw: float,
    roll: float,
    is_attentive: bool,
    reasons: list[str]) -> np.ndarray:

    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 52), (0, 0, 0), -1)

    if not slouch_calibrated:
        banner_text  = f"CALIBRATING POSTURE  {slouch_cal_progress}/{slouch_cal_total} : sit upright"
        banner_color = (0, 220, 220)
    elif is_attentive:
        banner_text  = "ATTENTIVE"
        banner_color = (0, 210, 60)
    else:
        banner_text  = "NOT ATTENTIVE"
        banner_color = (30, 60, 230)

    cv2.putText(frame, banner_text, (16, 36), _FONT, 1.05,
                banner_color, 3, cv2.LINE_AA)

    for i, r in enumerate(reasons):
        cv2.putText(frame, f"   {r}", (16, 56 + 30 * i),
                    _FONT, 0.68, (60, 130, 255), 2, cv2.LINE_AA)

    panel_top = h - 108
    cv2.rectangle(frame, (0, panel_top - 6), (w, h - 28), (0, 0, 0), -1)

    status_lines = [
        (
            f"Eye contact : {'YES' if eye_contact else 'NO '}   "
            f"gaze={gaze_text:<8} head={head_text}",
            (0, 200, 80) if eye_contact else (60, 80, 220),
        ),
        (
            f"Slouch      : {'YES' if is_slouching else 'NO '}",
            (60, 80, 220) if is_slouching else (0, 200, 80),
        ),
        (
            f"Head pose   : pitch={pitch:+.1f}   yaw={yaw:+.1f}   roll={roll:+.1f}",
            (200, 200, 200),
        ),
    ]
    for i, (txt, col) in enumerate(status_lines):
        cv2.putText(frame, txt, (12, panel_top + i * 26),
                    _FONT, 0.60, col, 2, cv2.LINE_AA)

    cv2.putText(frame, "q = quit    c = calibrate gaze    r = recalibrate posture",
                (12, h - 8), _FONT, 0.48, (140, 140, 140), 1, cv2.LINE_AA)

    return frame
