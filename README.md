# driver-drowsiness-system-detection-
Real-time driver drowsiness detection system using multi-signal fusion (EAR, MAR, head pose, LSTM) with 14 modules, live dashboard, and intelligent alert system.
It is designed as a practical safety solution for road-accident prevention.
1. Overview
**Dataset used:** The model was trained using the dataset https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
Drowsy driving is a major global safety issue. This system monitors the driver's face through a camera feed and analyzes multiple bio-signals including:

1.Eye Aspect Ratio (EAR)
2.PERCLOS (percentage of eye closure)
3.Yawn detection via MAR
4.Head pose estimation
5.Iris gaze direction
6.Blink rate

The system then fuses all signals through a weighted scoring engine and generates graded alerts (Low → Critical).
2. Key Features
 Real-Time Facial Landmark Tracking
MediaPipe Face Mesh (468+ points)
Iris tracking for gaze detection
Accurate EAR, MAR & PERCLOS computation
 Deep Learning Temporal Modeling
LSTM/GRU for temporal fatigue patterns
ONNX + TensorRT support for fast inference
 14-Module Architecture
Preprocessing (CLAHE, FPS control)
Head pose estimation using PnP
Multi-signal fusion engine
Alert escalation logic
Dashboard + session analytics
 Intelligent Alert System
Low → Medium → High → Critical
Voice alerts, buzzer beeps, warnings
Logging for all alert events
 Analytics & Reports
Live dashboard (Flask)
CSV/JSON logging, PDF session report
Yawn frequency, blink rate, PERCLOS graphing
 3. System Architecture
Video Input → Preprocessing → Face Landmarks → EAR/MAR/Gaze/Head-Pose
              ↓
       LSTM Temporal Model
              ↓
    Multi-Signal Scoring Engine
              ↓
   Alert System + Dashboard + Logs
 4. Technologies Used
Computer Vision
OpenCV
CLAHE enhancement
FPS throttling
Face AI
MediaPipe 0.10+
3D landmark mapping
Iris tracking
Deep Learning
PyTorch
LSTM/GRU
ONNX export
TensorRT acceleration
Backend
Flask (REST API + MJPEG feed)
SQLite database
Reports & Storage
CSV logs
JSON summary
Optional PDF generation
 5. Signals & Thresholds
Signal	Description	Threshold/Logic
EAR	Eye closure	0.21, 3 consecutive frames
PERCLOS	Eye closed % over 60s	> 15%
MAR	Mouth opening (yawns)	prolonged > 1.5s
Head Pose	Pitch/Yaw/Roll	15°/30°/25°
Gaze	Off-road detection	> 2s
Blink Rate	Abnormal frequency	adaptive
 6. Alert Levels
Level	Score Range	Action
Low	30–54	gentle beep
Medium	55–74	two-tone alert
High	75–89	triple beep + voice
Critical	90+	emergency alarm

Cooldown: 5s
3 dismissals → automatic escalation.

 7. Dashboard & API
Endpoints
/video_feed → MJPEG stream
/data → JSON metrics
/start → start camera
/stop → stop camera
Dashboard Shows
EAR, MAR, Score, FPS
Rolling 60s graphs
Session summary
Alert logs
 8. Installation
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install -r requirements.txt

Dependencies include:

opencv-python
mediapipe
torch
flask
numpy
matplotlib (optional)
 9. Running the System
python app.py

Open in browser:

http://127.0.0.1:5000
 10. Project Structure (Recommended)
 driver-drowsiness-detection
├── app.py
├── models/
│   ├── lstm_model.onnx
│   └── calibration.json
├── utils/
│   ├── preprocessing.py
│   ├── landmarks.py
│   ├── ear_mar.py
│   ├── head_pose.py
│   ├── scoring.py
│   └── alerts.py
├── dashboard/
│   ├── templates/
│   └── static/
├── reports/
│   ├── session_logs/
├── requirements.txt
└── README.md
