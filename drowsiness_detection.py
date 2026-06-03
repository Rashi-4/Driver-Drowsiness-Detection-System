import cv2
import numpy as np
import dlib
from imutils import face_utils
import time

# Initialize Camera and Models

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "/Users/jiyaarora/Desktop/open cv project/shape_predictor_68_face_landmarks.dat"
)

# Status Variables

sleep = 0
drowsy = 0
active = 0

status = ""
color = (0, 255, 0)


# Head Nod Detection Variables

prev_nose_y = None
nod_count = 0

nod_threshold = 25
last_nod_time = 0
nod_cooldown = 2


# Helper Functions

def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(a, b, c, d, e, f):
    up = compute(b, d) + compute(e, f)
    down = compute(a, f)

    if down == 0:
        return 0

    return up / (2.0 * down)


# Main Loop

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    status = ""
    color = (0, 255, 0)


    # Face Detection

    if len(faces) == 0:
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    for face in faces:

        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


        # Facial Landmark Detection

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)


        # EAR Calculation

        left_ear = eye_aspect_ratio(
            landmarks[36],
            landmarks[37],
            landmarks[38],
            landmarks[41],
            landmarks[40],
            landmarks[39]
        )

        right_ear = eye_aspect_ratio(
            landmarks[42],
            landmarks[43],
            landmarks[44],
            landmarks[47],
            landmarks[46],
            landmarks[45]
        )

        ear = (left_ear + right_ear) / 2


        # Head Nod Detection

        nose_y = landmarks[30][1]
        current_time = time.time()

        if prev_nose_y is not None:

            movement = nose_y - prev_nose_y

            if (
                movement > nod_threshold
                and (current_time - last_nod_time) > nod_cooldown
            ):
                nod_count += 1
                last_nod_time = current_time

        prev_nose_y = nose_y


        # EAR State Classification
        # Adjust thresholds according
        # to your webcam testing

        if ear <= 0.26:

            sleep += 1
            drowsy = 0
            active = 0

        elif 0.26 < ear < 0.32:

            drowsy += 1
            sleep = 0
            active = 0

        else:

            active += 1
            sleep = 0
            drowsy = 0


        # Priority Decision Logic
        # Sleepy > Head Nod > Drowsy > Active

        if sleep > 10:

            status = "SLEEPY"
            color = (0, 0, 255)

            nod_count = 0

        elif nod_count >= 2 and (current_time - last_nod_time) < 8:

            status = "HEAD NOD DROWSY"
            color = (0, 0, 255)

            nod_count = 0

        elif drowsy > 8:

            status = "DROWSY"
            color = (0, 255, 255)

        elif active > 8:

            status = "ACTIVE"
            color = (0, 255, 0)


        # Draw Facial Landmarks

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)


    # Display Driver Status

    if status:
        cv2.putText(
            frame,
            status,
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break


# Cleanup

cap.release()
cv2.destroyAllWindows()
