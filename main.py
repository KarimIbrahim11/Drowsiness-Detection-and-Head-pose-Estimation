import cv2
import mediapipe as mp
import numpy as np
import time
import math
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def calculate_ear(landmarks, refer_idxs, frame_width, frame_height):
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y,
                                            frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def draw_axes(forehead_2d, x, y, z, image):
    p1 = (int(forehead_2d[0]), int(forehead_2d[1]))
    p2 = (int(forehead_2d[0] + 5 * y), int(forehead_2d[1] - x * 5))
    cv2.line(image, p1, p2, (255, 0, 0), 3)

    p3 = (int(forehead_2d[0]), int(forehead_2d[1]))
    p4 = (
        int(forehead_2d[0] + 50 * math.cos(math.radians(x))),
        int(forehead_2d[1] + 50 * math.sin(math.radians(x))))
    cv2.line(image, p3, p4, (0, 255, 0), 3)

    p5 = (int(forehead_2d[0]), int(forehead_2d[1]))
    p6 = (
        int(forehead_2d[0] + 50 * math.sin(math.radians(z))),
        int(forehead_2d[1] - 50 * math.cos(math.radians(z))))
    cv2.line(image, p5, p6, (0, 0, 255), 3)


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    left_ear, left_lm_coordinates = calculate_ear(
        landmarks,
        left_eye_idxs,
        image_w,
        image_h
    )
    right_ear, right_lm_coordinates = calculate_ear(
        landmarks,
        right_eye_idxs,
        image_w,
        image_h
    )
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

eye_idxs = {
    "left": [362, 385, 387, 263, 373, 380],
    "right": [33, 160, 158, 133, 153, 144],
}

# For tracking counters and sharing states in and out of callbacks.
state_tracker = {
    "start_time": time.perf_counter(),
    "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
    "COLOR": (255, 0, 0),
    "play_alarm": False,
}

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 255))

cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    start = time.time()
    success, image = cap.read()
    if success:
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape

        face_3d = []
        face_2d = []

        # Printing Props
        EAR_txt_pos = (10, 30)
        DROWSY_TIME_txt_pos = (10, int(img_h // 2 * 1.7))
        ALM_txt_pos = (10, int(img_h // 2 * 1.85))

        if results.multi_face_landmarks:
            # Assessing ERA
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, eye_idxs["left"], eye_idxs["right"], img_w,
                                                 img_h)
            frame = plot_eye_landmarks(image, coordinates[0], coordinates[1], state_tracker["COLOR"])

            if EAR < 0.27:
                # Increase DROWSY_TIME to track the time period with EAR less than threshold nd reset the start_time
                # for the next iteration.
                end_time = time.perf_counter()

                state_tracker["DROWSY_TIME"] += end_time - state_tracker["start_time"]
                state_tracker["start_time"] = end_time
                state_tracker["COLOR"] = (255, 0, 0)

                if state_tracker["DROWSY_TIME"] >= 2.0:
                    plot_text(image, "WAKE UP! WAKE UP", ALM_txt_pos, (0, 0, 255))

            else:
                state_tracker["start_time"] = time.perf_counter()
                state_tracker["DROWSY_TIME"] = 0.0
                state_tracker["COLOR"] = (0, 255, 0)

            EAR_txt = f"EAR: {round(EAR, 2)}"
            DROWSY_TIME_txt = f"DROWSY: {round(state_tracker['DROWSY_TIME'], 3)} Secs"
            plot_text(image, EAR_txt, EAR_txt_pos, state_tracker["COLOR"])
            plot_text(image, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, state_tracker["COLOR"])

            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199 or idx == 8:
                        if idx == 8:
                            forehead_2d = (lm.x * img_w, lm.y * img_h)
                            forehead_3d = (lm.x * img_w, lm.y * img_h, lm.z * 100)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # into an upper triangular matrix (mtxR) and
                # an orthogonal matrix (mtxQ), such that rmat =
                # mtxR * mtxQ.

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = "looking left"
                elif y > 7:
                    text = "looking Right"
                elif x < -5:
                    text = "looking Down"
                elif x > 10:
                    text = "looking Up"
                else:
                    text = "forward"

                forehead_3d_v2 = cv2.projectPoints(forehead_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                # # Draw axes on the forehead
                if forehead_3d is not None:
                    draw_axes(forehead_2d, x, y, z, image)

                plot_text(image, text, (10, int(0.15 * img_h)), (0, 255, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                plot_text(image, 'x: ' + str(np.round(x, 2)), (500, 50), (0, 0, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                plot_text(image, 'y: ' + str(np.round(y, 2)), (500, 100), (0, 0, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                plot_text(image, 'z: ' + str(np.round(z, 2)), (500, 150), (0, 0, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

                end = time.time()
                totalTime = end - start
                fps = 1 / totalTime
                # print('FPS', fps)
                plot_text(image, f'FPS: {int(fps)}', (int(0.78 * img_w), int(0.4 * img_h)), (0, 255, 0), cv2.FONT_HERSHEY_SIMPLEX,
                            1, 1)

                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     landmark_drawing_spec=drawing_spec,
                #     connection_drawing_spec=drawing_spec)

            # Display the result
            cv2.imshow("Head Pose Estimation and Drowsiness Detection", image)
        else:
            state_tracker["start_time"] = time.perf_counter()
            state_tracker["DROWSY_TIME"] = 0.0
            state_tracker["COLOR"] = (0, 255, 0)
            state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(image, 1)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
