# import cv2
# import numpy as np
# import dlib
# from imutils import face_utils

# cap = cv2.VideoCapture(0)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("/Users/jiyaarora/Desktop/open cv project/shape_predictor_68_face_landmarks.dat")

# # status counters
# sleep = 0
# drowsy = 0
# active = 0
# status = ""
# color = (0, 0, 0)

# # for head nod detection
# prev_nose_y = None
# nod_count = 0
# nod_threshold = 20  # pixels of vertical movement to consider as a head nod

# def compute(ptA, ptB):
#     return np.linalg.norm(ptA - ptB)

# def blinked(a, b, c, d, e, f):
#     up = compute(b, d) + compute(e, f)
#     down = compute(a, f)
#     ratio = up / (2.0 * down)
#     return ratio

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
    
#     for face in faces:
#         x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
#         face_frame = frame.copy()
#         cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         landmarks = predictor(gray, face)
#         landmarks = face_utils.shape_to_np(landmarks)

#         # EAR calculation
#         left_ratio = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
#         right_ratio = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
#         ear = (left_ratio + right_ratio) / 2

#         # --- HEAD NOD DETECTION ---
#         nose_y = landmarks[30][1]  # nose tip position

#         if prev_nose_y is not None:
#             movement = nose_y - prev_nose_y
#             if movement > nod_threshold:
#                 nod_count += 1
#                 print("Head Nod Detected! Count:", nod_count)

#                 if nod_count > 2:
#                     status = "Head Nodding - Drowsy!"
#                     color = (0, 0, 255)
#                     nod_count = 0  # reset after detection
#         prev_nose_y = nose_y
#         # --- END HEAD NOD DETECTION ---

#         # --- EYE ASPECT RATIO (EAR) BASED STATES ---
#         if ear <= 0.26:
#             sleep += 1
#             drowsy = 0
#             active = 0
#             if sleep > 6:
#                 status = "sleepy!"
#                 color = (255, 0, 0)
#         elif 0.27 <= ear < 0.30:
#             drowsy += 1
#             sleep = 0
#             active = 0
#             if drowsy > 4:
#                 status = "DROWSY "
#                 color = (0, 0, 255)
#         else:
#             active += 1
#             drowsy = 0
#             sleep = 0
#             if active > 6:
#                 status = "ACTIVE "
#                 color = (0, 255, 0)
#         cv2.putText(frame, status, (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
#         for (x, y) in landmarks:
#             cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Result of detector", face_frame)

#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import dlib
from imutils import face_utils
import time

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/jiyaarora/Desktop/open cv project/shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 255, 0)

# Head nod detection variables
prev_nose_y = None
nod_count = 0
nod_threshold = 25  # pixel movement to consider as a nod
last_nod_time = 0
nod_cooldown = 2  # seconds

# Helper functions
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(e, f)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    return ratio

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    status = ""
    color = (0, 255, 0)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # EAR calculation
        left_ratio = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_ratio = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])
        ear = (left_ratio + right_ratio) / 2
        print("EAR:", round(ear,3))


        # HEAD NOD DETECTION
        nose_y = landmarks[30][1]
        now = time.time()
        if prev_nose_y is not None:
            movement = nose_y - prev_nose_y
            if movement > nod_threshold and (now - last_nod_time) > nod_cooldown:
                nod_count += 1
                last_nod_time = now
                print(f"Head Nod Detected: {nod_count}")
        prev_nose_y = nose_y

        # EAR-based detection logic
        if ear <= 0.260:
            sleep += 1
            drowsy = 0
            active = 0
        elif 0.260 < ear < 0.320:
            drowsy += 1
            sleep = 0
            active = 0
        else:
            active += 1
            drowsy = 0
            sleep = 0

        # Decide status — prioritize Sleepy > Head Nod > Drowsy > Active
        if sleep > 10:
            status = "SLEEPY 😴"
            color = (255, 0, 0)
            nod_count = 0
        elif nod_count >= 2 and (now - last_nod_time) < 8:
            status = "HEAD NOD DROWSY"
            color = (0, 0, 255)
            nod_count = 0
        elif drowsy > 8:
            status = "DROWSY "
            color = (0, 0, 255)
        elif active > 8:
            status = "ACTIVE "
            color = (0, 255, 0)

        # Draw facial landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)

    # Display the current status clearly
    if status:
        cv2.putText(frame, status, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
