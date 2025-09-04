import os
import cv2
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance
from imutils import face_utils
import mediapipe as mp
import time
from datetime import datetime

# === Face Registration & Recognition ===

ENCODINGS_FILE = "known_encodings.npy"
NAMES_FILE = "known_names.npy"
FACE_DIR = "registered_faces"

os.makedirs(FACE_DIR, exist_ok=True)

if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
    known_encodings = list(np.load(ENCODINGS_FILE, allow_pickle=True))
    known_names = list(np.load(NAMES_FILE, allow_pickle=True))
else:
    known_encodings = []
    known_names = []

def save_data():
    np.save(ENCODINGS_FILE, known_encodings)
    np.save(NAMES_FILE, known_names)

def register():
    name = input("Enter your name to register: ").strip()
    if name in known_names:
        print(f"[INFO] {name} is already registered.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Look at the camera to register your face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            known_encodings.append(face_encoding)
            known_names.append(name)
            save_data()

            # Save face image for reference
            top, right, bottom, left = face_locations[0]
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(os.path.join(FACE_DIR, f"{name}.jpg"), face_image)

            print(f"[INFO] {name} registered successfully!")
            break
        elif len(face_locations) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only your face is visible.")
        else:
            print("[INFO] No face detected. Please look at the camera.")

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

def login():
    if not known_encodings or not known_names:
        print("[ERROR] No registered users found. Please register first.")
        return None

    cap = cv2.VideoCapture(0)
    print("[INFO] Look at the camera to login...")

    user_name = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if True in matches:
                best_match_index = np.argmin(face_distances)
                user_name = known_names[best_match_index]
                print(f"[INFO] Welcome, {user_name}!")
            else:
                print("[INFO] Face not recognized. Access denied.")
            break
        elif len(face_locations) > 1:
            print("[WARNING] Multiple faces detected. Please ensure only your face is visible.")
        else:
            print("[INFO] No face detected. Please look at the camera.")

        cv2.imshow("Login Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Login cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return user_name

# === Seatbelt Detection (OpenCV line detection) ===

def detect_seatbelt(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.4):int(h*0.9), int(w*0.4):int(w*0.9)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    seatbelt_lines = []
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if 0.5 < abs(slope) < 1.5:
                seatbelt_lines.append(line[0])
                cv2.line(roi, (x1,y1), (x2,y2), (0,0,255), 3)

    seatbelt_status = len(seatbelt_lines) >= 1
    return seatbelt_status

# === Driver Monitoring System with Logging & Thresholds ===

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
DISTRACTION_THRESHOLD = 75
FACE_MISSING_THRESHOLD = 15

DROWSINESS_ALERT_INTERVAL = 3
DISTRACTION_ALERT_INTERVAL = 3
HAND_ALERT_INTERVAL = 2
SEATBELT_ALERT_INTERVAL = 3

SEATBELT_CHECK_INTERVAL = 60  # 1 minutes
FACE_CHECK_INTERVAL = 60     # 1 minutes

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Hand Detection Robustness Thresholds ===
HAND_DETECTION_CONFIDENCE = 0.7  # Increase for stricter detection, decrease for more sensitivity
HAND_TRACKING_CONFIDENCE = 0.7   # Same as above

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(shape, frame_size):
    image_points = np.array([
        shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    size = frame_size
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles

def start_driver_monitoring(user_name):
    print(f"\n[INFO] Starting driver monitoring for {user_name}...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)

    # Alert timers and logs
    last_drowsy_alert = 0
    last_distraction_alert = 0
    last_hand_alert = 0
    last_seatbelt_alert = 0
    last_seatbelt_check = 0
    last_face_check = 0
    event_log = []

    counter = 0
    face_missing_frames = 0

    mp_hands_instance = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE
    )

    # --- Immediate seatbelt check after login ---
    ret, frame = cap.read()
    if ret:
        seatbelt_on = detect_seatbelt(frame)
        if seatbelt_on:
            print("[INFO] Seatbelt detected immediately after login.")
        else:
            msg = "[ALERT] Seatbelt NOT detected immediately after login!"
            print(msg)
            event_log.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}")
    last_seatbelt_check = time.time()
    last_face_check = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to grab frame.")
            continue

        now = time.time()
        alerts_this_frame = []

        # Seatbelt check every 5 minutes
        if now - last_seatbelt_check >= SEATBELT_CHECK_INTERVAL:
            seatbelt_on = detect_seatbelt(frame)
            last_seatbelt_check = now
            if not seatbelt_on and now - last_seatbelt_alert > SEATBELT_ALERT_INTERVAL:
                msg = "[ALERT] Seatbelt NOT detected (periodic check)!"
                print(msg)
                alerts_this_frame.append(msg)
                last_seatbelt_alert = now
            elif seatbelt_on:
                print("[INFO] Seatbelt detected (periodic check).")

        # Face re-authentication every 5 minutes
        if now - last_face_check >= FACE_CHECK_INTERVAL:
            print("[INFO] Performing face re-authentication...")
            cap_face = cv2.VideoCapture(0)
            authenticated = False
            for _ in range(100):
                ret_face, frame_face = cap_face.read()
                if not ret_face:
                    continue
                rgb_frame = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if len(face_locations) == 1:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if known_names[best_match_index] == user_name:
                            print("[INFO] Face re-authentication successful.")
                            authenticated = True
                            break
            cap_face.release()
            if not authenticated:
                msg = "[ALERT] Face re-authentication failed!"
                print(msg)
                alerts_this_frame.append(msg)
            last_face_check = now

            cap.release()
            time.sleep(1)  # Give the camera a moment to reset
            cap = cv2.VideoCapture(0)

        # Hand detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands_instance.process(img_rgb)
        if results.multi_hand_landmarks:
            if now - last_hand_alert > HAND_ALERT_INTERVAL:
                msg = "[ALERT] Driver's hand detected - possible distraction or activity."
                print(msg)
                alerts_this_frame.append(msg)
                last_hand_alert = now
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Face/drowsiness/distraction detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            face_missing_frames += 1
            if face_missing_frames >= FACE_MISSING_THRESHOLD:
                if now - last_distraction_alert > DISTRACTION_ALERT_INTERVAL:
                    msg = "[ALERT] Distraction detected! (No face visible)"
                    print(msg)
                    alerts_this_frame.append(msg)
                    last_distraction_alert = now
        else:
            face_missing_frames = 0
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftHull = cv2.convexHull(leftEye)
                rightHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES:
                        if now - last_drowsy_alert > DROWSINESS_ALERT_INTERVAL:
                            msg = "[ALERT] Drowsiness detected!"
                            print(msg)
                            alerts_this_frame.append(msg)
                            last_drowsy_alert = now
                else:
                    counter = 0

                pitch, yaw, roll = get_head_pose(shape, frame.shape)
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

                if abs(yaw) > DISTRACTION_THRESHOLD:
                    if now - last_distraction_alert > DISTRACTION_ALERT_INTERVAL:
                        msg = "[ALERT] Distraction detected! (Yaw angle)"
                        print(msg)
                        alerts_this_frame.append(msg)
                        last_distraction_alert = now

        # Log alerts with timestamp
        if alerts_this_frame:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            for alert in alerts_this_frame:
                event_log.append(f"{timestamp} {alert}")

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Monitoring ended.")

    # Save log prompt
    if event_log:
        print("\nSession Event Log:")
        for entry in event_log:
            print(entry)
        save = input("\nDo you want to save this session log? (y/n): ").strip().lower()
        if save == 'y':
            fname = f"monitor_log_{user_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(fname, "w") as f:
                for entry in event_log:
                    f.write(entry + "\n")
            print(f"Log saved as {fname}")
        else:
            print("Log not saved.")

# === Main Program ===

def main():
    while True:
        action = input("\nType 'register' to register, 'login' to login and monitor, or 'exit' to quit: ").strip().lower()
        if action == "register":
            register()
        elif action == "login":
            user = login()
            if user:
                start_driver_monitoring(user)
        elif action == "exit":
            print("Exiting program.")
            break
        else:
            print("[ERROR] Invalid input. Please type 'register', 'login', or 'exit'.")

if __name__ == "__main__":
    main()
