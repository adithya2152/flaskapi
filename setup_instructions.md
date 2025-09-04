# Driver Monitoring System Setup Guide

## Overview
This project consists of a React Native Expo mobile app that communicates with a Flask backend for real-time driver monitoring using computer vision.

## Backend Setup (Flask)

### 1. Install Python Dependencies
```bash
cd flask_backend
pip install -r requirements.txt
```

### 2. Download Required Model File
You need the dlib facial landmarks predictor file:
```bash
# Download the file (about 99MB)
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

### 3. Start Flask Server
```bash
python app.py
```
The server will start on `http://0.0.0.0:5000`

## Mobile App Setup (React Native Expo)

### 1. Install Dependencies
```bash
cd DriverMonitorApp
npm install
```

### 2. Configure API Endpoint
Edit `src/services/api.ts` and update the BASE_URL with your computer's IP address:
```typescript
const BASE_URL = 'http://YOUR_COMPUTER_IP:5000'; // Replace with actual IP
```

To find your IP address:
- **Windows**: `ipconfig` (look for IPv4 Address)
- **Mac/Linux**: `ifconfig` or `ip addr show`

### 3. Start Expo Development Server
```bash
npx expo start
```

### 4. Run on Device
- Install Expo Go app on your phone
- Scan the QR code from the terminal
- Or use an emulator/simulator

## Important Notes

### Network Configuration
- Ensure your phone and computer are on the same WiFi network
- The Flask server must be accessible from your phone
- Test connectivity by visiting `http://YOUR_COMPUTER_IP:5000/health` in your phone's browser

### Camera Permissions
- The app will request camera permissions on first use
- Grant permissions for face registration and monitoring to work

### File Requirements
- The `shape_predictor_68_face_landmarks.dat` file is essential for facial landmark detection
- This file should be in the same directory as your Flask app

## Usage Flow

1. **Register**: Open the app and register your face with a name
2. **Login**: Use face authentication to log in
3. **Monitor**: Start monitoring session to begin real-time analysis
4. **Alerts**: View live safety alerts on your phone

## Troubleshooting

### Common Issues

1. **"No module named 'dlib'"**
   - Install dlib: `pip install dlib`
   - On some systems, you may need: `pip install cmake` first

2. **"shape_predictor_68_face_landmarks.dat not found"**
   - Download the file as shown in step 2 of backend setup

3. **Network connection issues**
   - Verify both devices are on same network
   - Check firewall settings
   - Test the `/health` endpoint

4. **Camera not working**
   - Ensure camera permissions are granted
   - Try restarting the Expo app

### Performance Tips

- Use a good lighting environment for better face detection
- Keep the phone steady during registration and login
- Ensure stable network connection for real-time monitoring</parameter>