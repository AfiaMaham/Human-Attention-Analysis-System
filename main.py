import cv2
import mediapipe as mp
from video_stream import VideoStream
from modules import HeadPoseModule, EyeContactModule, SlouchModule, AttentionTracker
from utils import draw_hud


def main():
    print("Attention Detector")
    print("q - quit")
    print("c - calibrate gaze (look straight at camera, then press c)")
    print("r - recalibrate posture (sit upright, then press r)")
    print()

    stream = VideoStream(src=0, width=1280, height=720)

    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,)
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    head_mod = HeadPoseModule()
    eye_mod = EyeContactModule()
    slouch_mod = SlouchModule(slouch_threshold=0.08, calibration_frames=15)
    tracker = AttentionTracker()

    eye_contact = False
    gaze_text = "N/A"
    head_text = "N/A"
    is_slouching = False
    pitch = yaw = roll = 0.0
    is_attentive = True
    reasons: list[str] = []

    try:
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            face_results = face_mesh.process(rgb)
            pose_results = pose.process(rgb)
            rgb.flags.writeable = True

            if face_results.multi_face_landmarks:
                fl = face_results.multi_face_landmarks[0]
                head_pose = head_mod.process(fl, img_h, img_w)
                pitch, yaw, roll = head_pose.pitch, head_pose.yaw, head_pose.roll

                eye_contact, gaze_text, head_text = eye_mod.process(fl, head_pose)

            if pose_results.pose_landmarks:
                is_slouching = slouch_mod.process(
                    pose_results.pose_landmarks.landmark, mp_pose)

            is_attentive, reasons = tracker.update(
                eye_contact, is_slouching, pitch, yaw)

            cal_prog, cal_total = slouch_mod.calibration_progress
            frame = draw_hud(
                frame,
                eye_contact=eye_contact,
                gaze_text=gaze_text,
                head_text=head_text,
                is_slouching=is_slouching,
                slouch_calibrated=slouch_mod.is_calibrated,
                slouch_cal_progress=cal_prog,
                slouch_cal_total=cal_total,
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                is_attentive=is_attentive,
                reasons=reasons,
            )

            cv2.imshow("Attention Detector", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                head_mod.request_calibration()
                print("Gaze calibration queued: look straight ahead.")
            elif key == ord('r'):
                slouch_mod.recalibrate()
                print("Posture recalibration started: sit upright.")

    finally:
        stream.stop()
        face_mesh.close()
        pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
