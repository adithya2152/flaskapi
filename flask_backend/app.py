from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import dlib
from scipy.spatial import distance
from imutils import face_utils
import mediapipe as mp
import time
from datetime import datetime
import base64
import os

app = Flask(__name__)
CORS(app)

# === Configuration ===
ENCODINGS_FILE = "known_encodings.npy"
NAMES_FILE = "known_names.npy"
FACE_DIR = "registered_faces"

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20
DISTRACTION_THRESHOLD = 75
FACE_MISSING_THRESHOLD = 15

# Alert intervals
DROWSINESS_ALERT_INTERVAL = 3
DISTRACTION_ALERT_INTERVAL = 3
HAND_ALERT_INTERVAL = 2
SEATBELT_ALERT_INTERVAL = 3

# Initialize directories and data
os.makedirs(FACE_DIR, exist_ok=True)

if os.path.exists(ENCODINGS_FILE) and os.path.exists(NAMES_FILE):
    known_encodings = list(np.load(ENCODINGS_FILE, allow_pickle=True))
    known_names = list(np.load(NAMES_FILE, allow_pickle=True))
else:
    known_encodings = []
    known_names = []

# Initialize dlib and mediapipe
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global variables for monitoring state
user_sessions = {}
global_alerts = []

def save_data():
    np.save(ENCODINGS_FILE, known_encodings)
    np.save(NAMES_FILE, known_names)

def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

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

    return len(seatbelt_lines) >= 1

@app.route('/api/register', methods=['POST'])
def register_user():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        image_base64 = data.get('image', '')

        if not name or not image_base64:
            return jsonify({'success': False, 'message': 'Name and image are required'})

        if name in known_names:
            return jsonify({'success': False, 'message': f'{name} is already registered'})

        # Convert base64 to OpenCV image
        frame = base64_to_cv2(image_base64)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) != 1:
            return jsonify({'success': False, 'message': 'Please ensure exactly one face is visible'})

        # Extract face encoding
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        known_encodings.append(face_encoding)
        known_names.append(name)
        save_data()

        # Save face image for reference
        top, right, bottom, left = face_locations[0]
        face_image = frame[top:bottom, left:right]
        cv2.imwrite(os.path.join(FACE_DIR, f"{name}.jpg"), face_image)

        return jsonify({'success': True, 'message': f'{name} registered successfully'})

    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'success': False, 'message': 'Registration failed'})

@app.route('/api/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        image_base64 = data.get('image', '')

        if not image_base64:
            return jsonify({'success': False, 'message': 'Image is required'})

        if not known_encodings or not known_names:
            return jsonify({'success': False, 'message': 'No registered users found'})

        # Convert base64 to OpenCV image
        frame = base64_to_cv2(image_base64)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if len(face_locations) != 1:
            return jsonify({'success': False, 'message': 'Please ensure exactly one face is visible'})

        # Check face recognition
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_match_index = np.argmin(face_distances)
            username = known_names[best_match_index]
            
            # Initialize user session
            user_sessions[username] = {
                'counter': 0,
                'face_missing_frames': 0,
                'last_drowsy_alert': 0,
                'last_distraction_alert': 0,
                'last_hand_alert': 0,
                'last_seatbelt_alert': 0,
                'alerts': []
            }
            
            return jsonify({
                'success': True, 
                'message': f'Welcome, {username}!',
                'data': {'username': username}
            })
        else:
            return jsonify({'success': False, 'message': 'Face not recognized'})

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed'})

@app.route('/api/monitor', methods=['POST'])
def monitor_frame():
    try:
        data = request.get_json()
        image_base64 = data.get('image', '')
        username = data.get('username', '')

        if not image_base64 or not username:
            return jsonify({'success': False, 'message': 'Image and username are required'})

        if username not in user_sessions:
            return jsonify({'success': False, 'message': 'User session not found'})

        session = user_sessions[username]
        frame = base64_to_cv2(image_base64)
        now = time.time()
        alerts_this_frame = []

        # Seatbelt detection
        seatbelt_on = detect_seatbelt(frame)
        if not seatbelt_on and now - session['last_seatbelt_alert'] > SEATBELT_ALERT_INTERVAL:
            alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Seatbelt NOT detected!"
            alerts_this_frame.append(alert_msg)
            session['last_seatbelt_alert'] = now

        # Hand detection
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            if now - session['last_hand_alert'] > HAND_ALERT_INTERVAL:
                alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Hand detected - possible distraction!"
                alerts_this_frame.append(alert_msg)
                session['last_hand_alert'] = now

        # Face detection and drowsiness/distraction analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects) == 0:
            session['face_missing_frames'] += 1
            if session['face_missing_frames'] >= FACE_MISSING_THRESHOLD:
                if now - session['last_distraction_alert'] > DISTRACTION_ALERT_INTERVAL:
                    alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: No face visible - distraction detected!"
                    alerts_this_frame.append(alert_msg)
                    session['last_distraction_alert'] = now
        else:
            session['face_missing_frames'] = 0
            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Eye aspect ratio for drowsiness
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    session['counter'] += 1
                    if session['counter'] >= CONSEC_FRAMES:
                        if now - session['last_drowsy_alert'] > DROWSINESS_ALERT_INTERVAL:
                            alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Drowsiness detected!"
                            alerts_this_frame.append(alert_msg)
                            session['last_drowsy_alert'] = now
                else:
                    session['counter'] = 0

                # Head pose for distraction
                pitch, yaw, roll = get_head_pose(shape, frame.shape)
                if abs(yaw) > DISTRACTION_THRESHOLD:
                    if now - session['last_distraction_alert'] > DISTRACTION_ALERT_INTERVAL:
                        alert_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ALERT: Head distraction detected!"
                        alerts_this_frame.append(alert_msg)
                        session['last_distraction_alert'] = now

        # Store alerts
        session['alerts'].extend(alerts_this_frame)
        global_alerts.extend(alerts_this_frame)

        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts_this_frame,
                'seatbelt_status': seatbelt_on,
                'face_detected': len(rects) > 0
            }
        })

    except Exception as e:
        print(f"Monitoring error: {e}")
        return jsonify({'success': False, 'message': 'Monitoring failed'})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        return jsonify({'alerts': global_alerts[-20:]})  # Return last 20 alerts
    except Exception as e:
        print(f"Error fetching alerts: {e}")
        return jsonify({'alerts': []})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting Flask Driver Monitoring API...")
    print("Make sure you have the shape_predictor_68_face_landmarks.dat file in the same directory")
    app.run(host='0.0.0.0', port=5000, debug=True)</parameter>