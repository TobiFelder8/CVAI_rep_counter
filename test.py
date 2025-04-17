import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Hilfsfunktion zur Winkelberechnung
def calculate_angle(a, b, c):
    a = np.array(a)  # Hüfte
    b = np.array(b)  # Knie
    c = np.array(c)  # Knöchel

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Zählvariablen
counter = 0
stage = None  # "down" oder "up"

# Kamera starten
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Bild verarbeiten
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Rechtes Bein: Hüfte, Knie, Knöchel
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        angle = calculate_angle(hip, knee, ankle)

        # Squat Logik
        if angle < 90:
            stage = "down"
        if angle > 160 and stage == "down":
            stage = "up"
            counter += 1

        # Feedback anzeigen
        cv2.putText(image, f"Angle: {int(angle)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        if angle < 90:
            cv2.putText(image, "Tief genug!", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        else:
            cv2.putText(image, "Beuge die Knie", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    except:
        pass

    # Zeichne Pose
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Anzeigen
    cv2.imshow('PoseCoach – Squats', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
