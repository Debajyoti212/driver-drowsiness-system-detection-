"""
Centralized configuration for the Driver Drowsiness Detection System.
All thresholds, weights, paths, and feature toggles.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "driver_data.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "fatigue_lstm.pth")

# ─── Module 01: Video Processing ─────────────────────────────────────────────
CAMERA_INDEX = 0
TARGET_FPS = 20
MAX_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FRAME_SKIP_THRESHOLD = 15          # skip frames if FPS drops below this
CAMERA_RECONNECT_DELAY = 2.0       # seconds before reconnect attempt
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)

# ─── Module 02: Eye State Detection ──────────────────────────────────────────
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
IRIS_LEFT = [468, 469, 470, 471, 472]
IRIS_RIGHT = [473, 474, 475, 476, 477]

EAR_THRESHOLD = 0.21               # default, overridden by calibration
EAR_CONSEC_FRAMES = 3              # min consecutive frames for blink
MICROSLEEP_THRESHOLD_MS = 500      # full closure < 500ms
PERCLOS_WINDOW_SEC = 60            # rolling window for PERCLOS
PERCLOS_ALERT_THRESHOLD = 0.15     # 15% eye closure = drowsy

# Kalman filter parameters for EAR smoothing
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2

# Gaze estimation
GAZE_OFF_ROAD_THRESHOLD = 0.3      # normalized gaze offset
GAZE_DISTRACTION_TIME = 2.0        # seconds off-road before alert

# ─── Module 03: Mouth / Yawn Detection ────────────────────────────────────────
UPPER_LIP = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
LOWER_LIP = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82]
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
INNER_TOP = 82
INNER_BOTTOM = 87

MAR_THRESHOLD = 0.6                # default, overridden by calibration
YAWN_MIN_DURATION_SEC = 1.5        # distinguish genuine yawn from speaking
YAWN_FREQUENCY_WINDOW_SEC = 300    # 5 minute rolling window
YAWN_FREQUENCY_ALERT = 3           # 3+ yawns in window = drowsy

# ─── Module 04: Head Pose Estimation ──────────────────────────────────────────
# 3D model points for PnP solver (nose tip, chin, left/right eye corners, mouth corners)
FACE_3D_MODEL = [
    (0.0, 0.0, 0.0),              # Nose tip
    (0.0, -330.0, -65.0),         # Chin
    (-225.0, 170.0, -135.0),      # Left eye left corner
    (225.0, 170.0, -135.0),       # Right eye right corner
    (-150.0, -150.0, -125.0),     # Left mouth corner
    (150.0, -150.0, -125.0),      # Right mouth corner
]
# Corresponding MediaPipe landmark indices
FACE_2D_LANDMARKS = [1, 152, 33, 263, 61, 291]

PITCH_NOD_THRESHOLD = -15.0        # degrees, head nod forward
YAW_TURN_THRESHOLD = 30.0          # degrees, looking sideways
ROLL_TILT_THRESHOLD = 25.0         # degrees, head tilting
HEAD_NOD_TIME = 1.0                # seconds of nod = microsleep onset
HEAD_TURN_TIME = 3.0               # seconds looking away

# Head movement rate
HEAD_MOVEMENT_WINDOW_SEC = 10      # smoothing window
HEAD_MOVEMENT_LOW_THRESHOLD = 2.0  # too little movement = drowsy
HEAD_MOVEMENT_HIGH_THRESHOLD = 20.0  # erratic movement = fatigue

# ─── Module 05: Driver Identity & Personalization ─────────────────────────────
CALIBRATION_DURATION_SEC = 30      # baseline calibration at session start
DEFAULT_DRIVER_NAME = "Default"

# ─── Module 06: Environmental Context Awareness ──────────────────────────────
HIGH_RISK_HOURS = (2, 4)           # 2 AM - 4 AM
MEDIUM_RISK_HOURS = [(0, 2), (4, 6), (13, 15)]  # early morning, post-lunch
TRIP_LONG_DURATION_MIN = 180       # 3 hours
TRIP_THRESHOLD_REDUCTION = 0.15    # reduce alert thresholds by 15% after long drive

NIGHT_BRIGHTNESS_THRESHOLD = 60    # frame mean brightness below this = night
GLARE_BRIGHTNESS_THRESHOLD = 220   # region brightness above this = glare

# ─── Module 07: Temporal Fatigue Analysis (LSTM/GRU) ──────────────────────────
SEQUENCE_LENGTH = 30               # 10s at ~3 features/sec
INPUT_FEATURES = 6                 # EAR, MAR, pitch, yaw, blink_rate, yawn_freq
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
FOCAL_LOSS_GAMMA = 2.0
CONFIDENCE_THRESHOLD = 0.7        # below this = low-confidence prediction

# ─── Module 08: Multi-Signal Drowsiness Scoring Engine ────────────────────────
SCORE_WEIGHTS = {
    "ear": 0.20,
    "perclos": 0.20,
    "blink_rate": 0.10,
    "yawn_freq": 0.15,
    "head_pose": 0.15,
    "fps_penalty": 0.05,
    "trip_duration": 0.05,
    "time_of_day": 0.10,
}

ALERT_THRESHOLDS = {
    "low": 30,
    "medium": 55,
    "high": 75,
    "critical": 90,
}

TREND_WINDOW_SEC = 60              # seconds of history for trend prediction
TREND_SLOPE_WARNING = 0.5          # score increase rate per second

# ─── Module 09: Intelligent Alert System ──────────────────────────────────────
ALERT_COOLDOWN_SEC = 5             # min seconds between alerts
ESCALATION_WINDOW_SEC = 600        # 10 minutes
ESCALATION_DISMISSALS = 3          # 3 dismissals in window = escalate

# Alert frequencies (Hz) for different levels
ALERT_FREQUENCIES = {
    "low": [(800, 200)],
    "medium": [(1000, 300), (1200, 300)],
    "high": [(1500, 500), (1800, 500), (2000, 500)],
    "critical": [(2000, 1000), (2500, 1000)],
}

# ─── Module 10: Occlusion & Edge Case Handling ───────────────────────────────
SUNGLASSES_EAR_VARIANCE_THRESHOLD = 0.0005  # near-zero EAR variance = sunglasses
FACE_VISIBILITY_THRESHOLD = 0.3    # min face landmarks visible ratio
MIN_DETECTION_FOR_SCORING = 2      # at least 2 signals needed for valid score

# ─── Module 11: Performance Dashboard ─────────────────────────────────────────
DASHBOARD_UPDATE_INTERVAL_MS = 100
GRAPH_HISTORY_SEC = 60             # 60s rolling graph
DASHBOARD_WIDTH = 1400
DASHBOARD_HEIGHT = 900

# ─── Feature Toggles ─────────────────────────────────────────────────────────
ENABLE_DEEP_LEARNING = False       # set True when LSTM model is trained
ENABLE_DASHBOARD = True
ENABLE_GPS = False                 # stub for REST recommendation
ENABLE_SOUND_ALERTS = True
