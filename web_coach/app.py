import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Setze die Seitenkonfiguration ganz oben (nur einmal!)
st.set_page_config(page_title="PoseCoach", layout="centered")

st.title("PoseCoach – Squats & Bizepscurls Erkennung")
st.sidebar.markdown("### 📸 Aktiviere deine Kamera unten")
run = st.checkbox("Kamera starten")
frame_window = st.image([])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Wiederholungszähler
counter_squats = 0
counter_biceps = 0
stage_squats = None
stage_biceps = None

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# ---- Video Transformer Class ----
class PoseCoach(VideoTransformerBase):
    def __init__(self):
        self.counter_squats = 0
        self.counter_biceps = 0
        self.stage_squats = None
        self.stage_biceps = None

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Squats
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            squat_angle = calculate_angle(hip, knee, ankle)

            if squat_angle < 90:
                self.stage_squats = "down"
            if squat_angle > 160 and self.stage_squats == "down":
                self.stage_squats = "up"
                self.counter_squats += 1

            # Bizeps
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            bicep_angle = calculate_angle(shoulder, elbow, wrist)

            if bicep_angle < 40:
                self.stage_biceps = "up"
            if bicep_angle > 150 and self.stage_biceps == "up":
                self.stage_biceps = "down"
                self.counter_biceps += 1

            # Anzeigen
            cv2.putText(image, f"Squats: {self.counter_squats}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (39, 64, 139), 3)
            cv2.putText(image, f"Bizepscurls: {self.counter_biceps}", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (39, 64, 139), 3)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image

# ---- Streamlit WebRTC Setup ----
if run:
    webrtc_streamer(key="pose-coach", video_transformer_factory=PoseCoach)
