import numpy as np
from modules.head_pose import HeadPoseResult

class EyeContactModule:
    EAR_THRESHOLD = 0.18

    def __init__(self):
        self.blinking    = False
        self.eye_contact = False

    def process(self, face_landmarks, head_pose: HeadPoseResult):
        lm = face_landmarks.landmark
        x = head_pose.cal_x
        y = head_pose.cal_y

        if y < -8:
            head_text = "Left"
        elif y >  8:
            head_text = "Right"
        elif x < -15:
            head_text = "Down"
        elif x >  8:
            head_text = "Up"
        else:
            head_text = "Forward"

        l_eye = [lm[i] for i in [33,  160, 158, 133, 153, 144]]
        r_eye = [lm[i] for i in [362, 385, 387, 263, 373, 380]]
        avg_ear = (self._ear(l_eye) + self._ear(r_eye)) / 2.0

        if avg_ear < self.EAR_THRESHOLD and head_text != "Up":
            self.blinking = True
            head_text = "Blinking"
        else:
            self.blinking = False

        l_iris = [lm[468]]
        r_iris = [lm[473]]
        l_eye2 = [lm[i] for i in [33,  173]]
        r_eye2 = [lm[i] for i in [398, 263]]

        lg = self._gaze_vec(l_iris, l_eye2)
        rg = self._gaze_vec(r_iris, r_eye2)
        gaze = (lg + rg) / 2.0 + np.array([0.0, 0.31, 0.0])

        if gaze[0] < -0.27:
            gaze_text = "Left"
        elif gaze[0] >  0.27:
            gaze_text = "Right"
        elif gaze[1] < -0.25:
            gaze_text = "Up"
        elif gaze[1] >  0.25:
            gaze_text = "Down"
        else:
            gaze_text = "Forward"

        contact = (
            not self.blinking and (
                (head_text == "Forward" and gaze_text == "Forward") or
                ( 3 <= y <=  8 and -0.9 <= gaze[0] <= -0.2) or
                (-8 <= y <= -3 and  0.2 <= gaze[0] <=  0.9) or
                (-20 <= x <= -12 and -0.4 <= gaze[1] <= -0.14) or
                (head_text == "Up"    and gaze_text in ("Down", "Forward")) or
                (head_text == "Down"  and gaze_text == "Up") or
                (head_text == "Left"  and gaze_text == "Right") or
                (head_text == "Right" and gaze_text == "Left")
            )
        )
        self.eye_contact = contact
        return contact, gaze_text, head_text

    @staticmethod
    def _gaze_vec(iris_lms, eye_lms):
        iris = np.mean([(l.x, l.y, l.z) for l in iris_lms], axis=0)
        eye  = np.mean([(l.x, l.y, l.z) for l in eye_lms],  axis=0)
        v = iris - eye
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    @staticmethod
    def _ear(eye_lms):
        def pt(l): return np.array([l.x, l.y])
        A = np.linalg.norm(pt(eye_lms[1]) - pt(eye_lms[5]))
        B = np.linalg.norm(pt(eye_lms[2]) - pt(eye_lms[4]))
        C = np.linalg.norm(pt(eye_lms[0]) - pt(eye_lms[3]))
        return (A + B) / (2.0 * C + 1e-9)
