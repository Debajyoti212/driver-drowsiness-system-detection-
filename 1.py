import cv2
import time
import math
import mediapipe as mp
import winsound

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

EAR_THRESHOLD = 0.18
EYE_TIME = 1.0

TURN_THRESHOLD = 0.25
TURN_TIME = 3.0

eye_start = None
turn_start = None

speed = 40
road_sign = "CLEAR"
speed_limit = None


def dist(p1, p2):
    return math.dist(p1, p2)


def eye_aspect_ratio(landmarks, eye, w, h):

    p1 = (landmarks[eye[0]].x * w, landmarks[eye[0]].y * h)
    p2 = (landmarks[eye[1]].x * w, landmarks[eye[1]].y * h)
    p3 = (landmarks[eye[2]].x * w, landmarks[eye[2]].y * h)
    p4 = (landmarks[eye[3]].x * w, landmarks[eye[3]].y * h)
    p5 = (landmarks[eye[4]].x * w, landmarks[eye[4]].y * h)
    p6 = (landmarks[eye[5]].x * w, landmarks[eye[5]].y * h)

    return (dist(p2, p6) + dist(p3, p5)) / (2 * dist(p1, p4))


cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    alert = None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:

        face = result.multi_face_landmarks[0]

        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)

        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:

            if eye_start is None:
                eye_start = time.time()

            elif time.time() - eye_start >= EYE_TIME:
                alert = "OPEN YOUR EYES"
                winsound.Beep(1000, 300)

        else:
            eye_start = None

        nose_x = face.landmark[NOSE].x * w
        left_x = face.landmark[LEFT_CHEEK].x * w
        right_x = face.landmark[RIGHT_CHEEK].x * w

        center = (left_x + right_x) / 2
        width = abs(right_x - left_x)

        offset = abs(nose_x - center) / width

        if offset > TURN_THRESHOLD:

            if turn_start is None:
                turn_start = time.time()

            elif time.time() - turn_start >= TURN_TIME and alert is None:
                alert = "LOOK FORWARD"
                winsound.Beep(1200, 300)

        else:
            turn_start = None

    if speed_limit and speed > speed_limit and alert is None:
        alert = f"SLOW DOWN - {road_sign}"
        winsound.Beep(900, 300)

    cv2.putText(frame, f"Speed: {speed} km/h", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Road: {road_sign}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if alert:
        cv2.putText(frame, alert, (60, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

    cv2.imshow("Final Driver Safety System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('w'):
        speed += 5

    elif key == ord('s'):
        speed = max(0, speed - 5)

    elif key == ord('1'):
        road_sign = "SCHOOL AHEAD"
        speed_limit = 20

    elif key == ord('2'):
        road_sign = "BUMP AHEAD"
        speed_limit = 25

    elif key == ord('3'):
        road_sign = "BAD ROAD"
        speed_limit = 30    

    elif key == ord('0'):
        road_sign = "CLEAR"
        speed_limit = None

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()