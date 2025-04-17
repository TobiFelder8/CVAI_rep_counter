import cv2
import mediapipe as mp
import numpy as np

# ---- USER WÄHLT ÜBUNG ----
print("Welche Übung willst du tracken?")
print("1 – Kniebeugen (Squats)")
print("2 – Bizepscurls (rechts)")
choice = input(">> Gib 1 oder 2 ein: ")

exercise = "squats" if choice == "1" else "bicep_curls"
print(f"\n📹 Starte PoseCoach für: {exercise}\nDrücke Q zum Beenden.\n")

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
counter = 0
stage = None  # "up" / "down" oder "open" / "closed"

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

        if exercise == "squats":
            # Rechte Seite: Hüfte, Knie, Knöchel
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

            label = "Knie-Winkel"
            feedback = "Tiefer gehen!" if angle > 100 else "Gut so!"

        elif exercise == "bicep_curls":
            # Rechte Seite: Schulter, Ellbogen, Handgelenk
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            angle = calculate_angle(shoulder, elbow, wrist)

            # Bizepscurl Logik
            if angle < 40:
                stage = "up"
            if angle > 150 and stage == "up":
                stage = "down"
                counter += 1

            label = "Ellbogen-Winkel"
            feedback = "Arm beugen!" if angle > 90 else "Top!"

        # Anzeige
        cv2.putText(image, f"{label}: {int(angle)}°", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(image, f"Reps: {counter}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(image, feedback, (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 200), 2)

    except:
        pass

    # Pose zeichnen
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Fenster zeigen
    cv2.imshow(f'PoseCoach – {exercise.capitalize()}', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
