# Predefined posture types:
# - Upright - default
# - Slouching Forward
# - Leaning Back
# - Left Side Slouch
# - Right Side Slouch
# - Forward Head Posture
# - No User

import cv2
import mediapipe as mp
import math
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def calculateDistance(point1, point2):
    if point1 is None or point2 is None:
        return None
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculateAngle(line1, line2):
    if None in (line1, line2):
        return None
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x4 - x3, y4 - y3)

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    if magnitude1 == 0 or magnitude2 == 0:
        return None

    angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))
    return math.degrees(angle_radians)


cap = cv2.VideoCapture(0)

my_drawing_specs = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)

calibration_start = None
recording_start = None
distance_ranges = {
    "chin_shoulders": [float("inf"), float("-inf")],
    "chin_eyebrows": [float("inf"), float("-inf")],
    "shoulder_to_shoulder": [float("inf"), float("-inf")],
    "medial_angle": [float("inf"), float("-inf")],
}

deviations = {}

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        pose_landmarks = results.pose_landmarks
        face_landmarks = results.face_landmarks

        if face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=my_drawing_specs,
            )

        if pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=pose_landmarks,
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=None,
            )

            # Countdown before calibration
            if calibration_start is None:
                countdown_start = time.time()
                for i in range(5, -1, -1):
                    while time.time() - countdown_start < 1:
                        success, image = cap.read()
                        if not success:
                            break
                        image = cv2.flip(image, 1)
                        cv2.putText(
                            image,
                            str(i),
                            (image.shape[1] // 2 - 100, image.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4,
                            (255, 255, 255),
                            5,
                        )
                        cv2.imshow("Camera View", image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()
                    countdown_start = time.time()
                calibration_start = time.time()

            if calibration_start is not None:
                # Small countdown in the corner during calibration
                if recording_start is None and time.time() - calibration_start <= 10:
                    countdown_time = 10 - int(time.time() - calibration_start)
                    cv2.putText(
                        image,
                        f"Calibration: {countdown_time}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                if face_landmarks and pose_landmarks:
                    # Get landmarks
                    chin_landmark = face_landmarks.landmark[152]  # Chin landmark
                    nose_tip = face_landmarks.landmark[1]  # Nose tip for medial line
                    left_shoulder = pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.LEFT_SHOULDER
                    ]
                    right_shoulder = pose_landmarks.landmark[
                        mp_holistic.PoseLandmark.RIGHT_SHOULDER
                    ]
                    left_eyebrow = face_landmarks.landmark[
                        65
                    ]  # Approx. left eyebrow landmark
                    right_eyebrow = face_landmarks.landmark[
                        295
                    ]  # Approx. right eyebrow landmark

                    # Convert normalized coordinates to pixel values
                    h, w, _ = image.shape
                    chin_point = (int(chin_landmark.x * w), int(chin_landmark.y * h))
                    nose_point = (int(nose_tip.x * w), int(nose_tip.y * h))
                    left_shoulder_point = (
                        int(left_shoulder.x * w),
                        int(left_shoulder.y * h),
                    )
                    right_shoulder_point = (
                        int(right_shoulder.x * w),
                        int(right_shoulder.y * h),
                    )
                    left_eyebrow_point = (
                        int(left_eyebrow.x * w),
                        int(left_eyebrow.y * h),
                    )
                    right_eyebrow_point = (
                        int(right_eyebrow.x * w),
                        int(right_eyebrow.y * h),
                    )

                    # Calculate midpoints and distances
                    shoulders_midpoint = (
                        (left_shoulder_point[0] + right_shoulder_point[0]) // 2,
                        (left_shoulder_point[1] + right_shoulder_point[1]) // 2,
                    )
                    eyebrows_midpoint = (
                        (left_eyebrow_point[0] + right_eyebrow_point[0]) // 2,
                        (left_eyebrow_point[1] + right_eyebrow_point[1]) // 2,
                    )
                    chin_shoulders_distance = calculateDistance(
                        chin_point, shoulders_midpoint
                    )
                    chin_eyebrows_distance = calculateDistance(
                        chin_point, eyebrows_midpoint
                    )
                    shoulder_to_shoulder_distance = calculateDistance(
                        left_shoulder_point, right_shoulder_point
                    )

                    # Calculate the facial medial line (chin to nose) and shoulder line
                    medial_line = (
                        chin_point[0],
                        chin_point[1],
                        nose_point[0],
                        nose_point[1],
                    )
                    shoulder_line = (
                        left_shoulder_point[0],
                        left_shoulder_point[1],
                        right_shoulder_point[0],
                        right_shoulder_point[1],
                    )
                    medial_angle = calculateAngle(medial_line, shoulder_line)

                if recording_start is None:
                    # Update calibration ranges
                    for key, value in [
                        ("chin_shoulders", chin_shoulders_distance),
                        ("chin_eyebrows", chin_eyebrows_distance),
                        ("shoulder_to_shoulder", shoulder_to_shoulder_distance),
                        ("medial_angle", medial_angle),
                    ]:
                        if value is not None:
                            distance_ranges[key][0] = min(
                                distance_ranges[key][0], value
                            )
                            distance_ranges[key][1] = max(
                                distance_ranges[key][1], value
                            )

            # End calibration after 10 seconds
            if (
                calibration_start is not None
                and recording_start is None
                and time.time() - calibration_start > 10
            ):
                print("Calibration completed:")
                for key, (min_val, max_val) in distance_ranges.items():
                    print(f"{key}: Min = {min_val:.2f}, Max = {max_val:.2f}")
                recording_start = time.time()

            # Check for bad posture based on deviation (20%)
            bad_posture = False
            deviations.clear()

            # Check deviations for each metric
            for key, value in [
                ("chin_shoulders", chin_shoulders_distance),
                ("chin_eyebrows", chin_eyebrows_distance),
                ("shoulder_to_shoulder", shoulder_to_shoulder_distance),
                ("medial_angle", medial_angle),
            ]:
                if value is not None:
                    min_val, max_val = distance_ranges[key]
                    lower_bound = min_val - 0.2 * ((max_val + min_val) / 2)
                    upper_bound = max_val + 0.2 * ((max_val + min_val) / 2)
                    deviations[key] = (lower_bound, upper_bound, value)

                    # Check if it's out of bounds
                    if not (lower_bound <= value <= upper_bound):
                        bad_posture = True

            # Display the red caption if bad posture is detected
            if bad_posture:
                cv2.putText(
                    image,
                    "BAD POSTURE DETECTED",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 0, 255),
                    3,
                )

            # Display distances and angles
            cv2.putText(
                image,
                f"Chin-Shoulders: {chin_shoulders_distance:.2f}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                image,
                f"Chin-Eyebrows: {chin_eyebrows_distance:.2f}",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            cv2.putText(
                image,
                f"Shoulder-Shoulder: {shoulder_to_shoulder_distance:.2f}",
                (50, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
            if medial_angle is not None:
                cv2.putText(
                    image,
                    f"Medial Angle: {medial_angle:.2f}",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),  # Orange
                    2,
                )

        # Display the updated video feed
        cv2.imshow("Camera View", image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
