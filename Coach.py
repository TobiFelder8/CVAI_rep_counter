import cv2
import mediapipe as mp
import numpy as np

# ---- MediaPipe Setup ----
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


# ---- Variablen für Wiederholungen ----
counter_squats = 0
counter_biceps = 0
stage_squats = None  # "up" / "down"
stage_biceps = None  # "up" / "down"

# ---- Kamera starten ----
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        # Kniebeugen (Squats) Logik
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        squat_angle = calculate_angle(hip, knee, ankle)

        # Bizepscurls Logik
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        bicep_angle = calculate_angle(shoulder, elbow, wrist)

        # Erkennung: Wenn der Winkel bei den Beinen sehr klein ist, könnte es ein Squat sein.
        if squat_angle < 90:
            stage_squats = "down"
        if squat_angle > 160 and stage_squats == "down":
            stage_squats = "up"
            counter_squats += 1

        # Erkennung: Wenn der Winkel des Ellbogens beim Curlen klein ist, könnte es ein Bizepscurl sein.
        if bicep_angle < 40:
            stage_biceps = "up"
        if bicep_angle > 150 and stage_biceps == "up":
            stage_biceps = "down"
            counter_biceps += 1

        # Anzeigen von Squats oder Bizepscurls (je nach Erkennung)
        if squat_angle < 90:
            label_squats = "Squats"
            feedback_squats = "Tiefer gehen!" if squat_angle > 100 else "Gut so!"
        else:
            label_squats = "Squats"
            feedback_squats = "Gut gemacht!"

        if bicep_angle < 40:
            label_biceps = "Bizepscurls"
            feedback_biceps = "Arm beugen!" if bicep_angle > 90 else "Top!"
        else:
            label_biceps = "Bizepscurls"
            feedback_biceps = "Gut gemacht!"

        # Anzeigen der Zähler und Feedback für beide Übungen
        cv2.putText(image, f"Squats: {counter_squats}", (frame.shape[1] - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(image, f"Bizepscurls: {counter_biceps}", (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Optional: Anzeige des Winkels für jedes Gelenk
        cv2.putText(image, f"Squats-Winkel: {int(squat_angle)}°", (frame.shape[1] - 250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(image, f"Bizeps-Winkel: {int(bicep_angle)}°", (40, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    except:
        pass

    # Pose zeichnen
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Fenster zeigen
    cv2.imshow('PoseCoach – Übungserkennung', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
