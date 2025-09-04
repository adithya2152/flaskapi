# Driver Monitoring React Native App

A React Native Expo application that uses the phone's camera for driver monitoring, working with a Flask backend for face recognition and safety analysis.

## Features

- **Face Registration**: Register your face for authentication
- **Face Login**: Secure login using facial recognition
- **Real-time Monitoring**: Continuous monitoring for:
  - Drowsiness detection (eye aspect ratio)
  - Distraction detection (head pose and face visibility)
  - Hand detection (possible distraction)
  - Seatbelt detection
- **Live Alerts**: Real-time safety alerts displayed on mobile

## Setup Instructions

### Backend Setup

1. Navigate to the `flask_backend` directory
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dlib face landmarks file:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2
   ```
4. Start the Flask server:
   ```bash
   python app.py
   ```

### Mobile App Setup

1. Navigate to the `DriverMonitorApp` directory
2. Install dependencies:
   ```bash
   npm install
   ```
3. Update the API base URL in `src/services/api.ts` with your Flask server's IP address
4. Start the Expo development server:
   ```bash
   npx expo start
   ```

## Configuration

### API Configuration

Update the `BASE_URL` in `src/services/api.ts` to match your Flask server:

```typescript
const BASE_URL = 'http://YOUR_FLASK_SERVER_IP:5000';
```

### Network Setup

- Ensure your mobile device and Flask server are on the same network
- The Flask server runs on port 5000 by default
- Use your computer's local IP address (not localhost) for the BASE_URL

## Usage

1. **Register**: Use the "Register Face" option to register your face with a name
2. **Login**: Use "Login with Face" to authenticate using facial recognition
3. **Monitor**: Once logged in, start monitoring to begin real-time safety analysis
4. **Alerts**: View live alerts for drowsiness, distraction, hand detection, and seatbelt status

## Technical Details

- **Frontend**: React Native with Expo, TypeScript
- **Backend**: Flask with OpenCV, dlib, face_recognition, mediapipe
- **Camera**: Expo Camera API for real-time image capture
- **Communication**: RESTful API with base64 image transmission
- **Storage**: AsyncStorage for user session persistence

## Safety Features

- Eye aspect ratio monitoring for drowsiness
- Head pose analysis for distraction detection
- Hand detection using MediaPipe
- Seatbelt detection using line detection algorithms
- Periodic face re-authentication
- Real-time alert system with configurable thresholds</parameter>