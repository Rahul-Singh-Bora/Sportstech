import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), 'w', buffering=1)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def get_coord(landmark, w, h):
    """Convert landmark to pixel coordinates with depth"""
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

def calculate_angle(a, b, c):
    """Calculate angle at point b between points a-b-c in degrees"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def calculate_velocity(prev_pos, curr_pos, time_diff):
    """Calculate velocity in pixels per second"""
    if time_diff <= 0 or prev_pos is None:
        return 0.0
    distance = np.linalg.norm(curr_pos[:2] - prev_pos[:2])
    return distance / time_diff

# Game-specific metric calculators
def cricket_bowling_metrics(landmarks, prev_data, w, h, current_time):
    """Calculate bowling arm speed, elbow angle, and shoulder rotation"""
    metrics = []
    ids = mp.solutions.holistic.PoseLandmark

    if landmarks and len(landmarks) > max(ids.RIGHT_WRIST.value, ids.RIGHT_ELBOW.value, ids.RIGHT_SHOULDER.value):
        wrist = get_coord(landmarks[ids.RIGHT_WRIST.value], w, h)
        elbow = get_coord(landmarks[ids.RIGHT_ELBOW.value], w, h)
        shoulder = get_coord(landmarks[ids.RIGHT_SHOULDER.value], w, h)
        hip = get_coord(landmarks[ids.RIGHT_HIP.value], w, h)

        # Calculate arm speed
        prev_wrist = prev_data.get('wrist')
        time_diff = current_time - prev_data.get('time', current_time)
        arm_speed = calculate_velocity(prev_wrist, wrist, time_diff) if prev_wrist is not None else 0.0

        # Calculate elbow angle
        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        # Calculate shoulder-hip alignment
        shoulder_hip_angle = calculate_angle(np.array([shoulder[0] + 100, shoulder[1], shoulder[2]]), shoulder, hip)

        metrics.append(f"Cricket Bowling")
        metrics.append(f"Arm Speed: {arm_speed:.1f} px/s")
        metrics.append(f"Elbow Angle: {elbow_angle:.1f} deg")
        metrics.append(f"Posture: {'Good' if 160 < elbow_angle < 180 else 'Check form'}")

        prev_data['wrist'] = wrist
        prev_data['time'] = current_time
    else:
        metrics.append("Cricket - Landmarks not detected")

    return metrics, prev_data

def basketball_shooting_metrics(landmarks, prev_data, w, h, current_time):
    """Calculate shooting arm angle, wrist speed, and knee bend"""
    metrics = []
    ids = mp.solutions.holistic.PoseLandmark

    if landmarks and len(landmarks) > max(ids.RIGHT_WRIST.value, ids.RIGHT_KNEE.value):
        wrist = get_coord(landmarks[ids.RIGHT_WRIST.value], w, h)
        elbow = get_coord(landmarks[ids.RIGHT_ELBOW.value], w, h)
        shoulder = get_coord(landmarks[ids.RIGHT_SHOULDER.value], w, h)
        hip = get_coord(landmarks[ids.RIGHT_HIP.value], w, h)
        knee = get_coord(landmarks[ids.RIGHT_KNEE.value], w, h)
        ankle = get_coord(landmarks[ids.RIGHT_ANKLE.value], w, h)

        # Wrist speed
        prev_wrist = prev_data.get('wrist')
        time_diff = current_time - prev_data.get('time', current_time)
        wrist_speed = calculate_velocity(prev_wrist, wrist, time_diff) if prev_wrist is not None else 0.0

        # Shooting arm angle
        shooting_angle = calculate_angle(shoulder, elbow, wrist)

        # Knee bend
        knee_angle = calculate_angle(hip, knee, ankle)

        metrics.append(f"Basketball Shooting")
        metrics.append(f"Wrist Speed: {wrist_speed:.1f} px/s")
        metrics.append(f"Elbow Angle: {shooting_angle:.1f} deg")
        metrics.append(f"Knee Bend: {knee_angle:.1f} deg")
        metrics.append(f"Form: {'Good' if 90 < knee_angle < 130 else 'Adjust stance'}")

        prev_data['wrist'] = wrist
        prev_data['time'] = current_time
    else:
        metrics.append("Basketball - Landmarks not detected")

    return metrics, prev_data

def yoga_pose_metrics(landmarks, prev_data, w, h, current_time):
    """Calculate balance, alignment, and joint angles"""
    metrics = []
    ids = mp.solutions.holistic.PoseLandmark

    if landmarks and len(landmarks) > max(ids.LEFT_ANKLE.value, ids.RIGHT_ANKLE.value):
        left_hip = get_coord(landmarks[ids.LEFT_HIP.value], w, h)
        left_knee = get_coord(landmarks[ids.LEFT_KNEE.value], w, h)
        left_ankle = get_coord(landmarks[ids.LEFT_ANKLE.value], w, h)

        right_hip = get_coord(landmarks[ids.RIGHT_HIP.value], w, h)
        right_knee = get_coord(landmarks[ids.RIGHT_KNEE.value], w, h)
        right_ankle = get_coord(landmarks[ids.RIGHT_ANKLE.value], w, h)

        left_shoulder = get_coord(landmarks[ids.LEFT_SHOULDER.value], w, h)
        right_shoulder = get_coord(landmarks[ids.RIGHT_SHOULDER.value], w, h)

        # Balance metric (ankle distance normalized by shoulder width)
        ankle_distance = abs(left_ankle[0] - right_ankle[0])
        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]) + 1e-8
        balance_ratio = ankle_distance / shoulder_width

        # Knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Hip alignment
        hip_center = (left_hip + right_hip) / 2
        prev_hip = prev_data.get('hip_center')
        time_diff = current_time - prev_data.get('time', current_time)
        hip_stability = calculate_velocity(prev_hip, hip_center, time_diff) if prev_hip is not None else 0.0

        metrics.append(f"Yoga Pose Analysis")
        metrics.append(f"Balance: {balance_ratio:.2f}")
        metrics.append(f"L Knee: {left_knee_angle:.1f} deg | R Knee: {right_knee_angle:.1f} deg")
        metrics.append(f"Hip Stability: {hip_stability:.1f} px/s")
        metrics.append(f"Posture: {'Stable' if hip_stability < 50 else 'Adjusting'}")

        prev_data['hip_center'] = hip_center
        prev_data['time'] = current_time
    else:
        metrics.append("Yoga - Landmarks not detected")

    return metrics, prev_data

def boxing_punch_metrics(landmarks, prev_data, w, h, current_time):
    """Calculate punch speed, arm extension, and stance"""
    metrics = []
    ids = mp.solutions.holistic.PoseLandmark

    if landmarks and len(landmarks) > max(ids.RIGHT_WRIST.value, ids.LEFT_WRIST.value):
        right_wrist = get_coord(landmarks[ids.RIGHT_WRIST.value], w, h)
        right_elbow = get_coord(landmarks[ids.RIGHT_ELBOW.value], w, h)
        right_shoulder = get_coord(landmarks[ids.RIGHT_SHOULDER.value], w, h)

        left_wrist = get_coord(landmarks[ids.LEFT_WRIST.value], w, h)
        left_elbow = get_coord(landmarks[ids.LEFT_ELBOW.value], w, h)
        left_shoulder = get_coord(landmarks[ids.LEFT_SHOULDER.value], w, h)

        # Punch speed for both hands
        prev_r_wrist = prev_data.get('right_wrist')
        prev_l_wrist = prev_data.get('left_wrist')
        time_diff = current_time - prev_data.get('time', current_time)

        r_punch_speed = calculate_velocity(prev_r_wrist, right_wrist, time_diff) if prev_r_wrist is not None else 0.0
        l_punch_speed = calculate_velocity(prev_l_wrist, left_wrist, time_diff) if prev_l_wrist is not None else 0.0

        # Arm angles
        r_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        l_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        metrics.append(f"Boxing Punch Analysis")
        metrics.append(f"R Punch Speed: {r_punch_speed:.1f} px/s")
        metrics.append(f"L Punch Speed: {l_punch_speed:.1f} px/s")
        metrics.append(f"R Arm: {r_arm_angle:.1f} deg | L Arm: {l_arm_angle:.1f} deg")
        metrics.append(f"Power: {'High' if max(r_punch_speed, l_punch_speed) > 200 else 'Moderate'}")

        prev_data['right_wrist'] = right_wrist
        prev_data['left_wrist'] = left_wrist
        prev_data['time'] = current_time
    else:
        metrics.append("Boxing - Landmarks not detected")

    return metrics, prev_data

def squat_metrics(landmarks, prev_data, w, h, current_time):
    """Calculate squat depth, knee angles, and hip movement"""
    metrics = []
    ids = mp.solutions.holistic.PoseLandmark

    if landmarks and len(landmarks) > max(ids.LEFT_KNEE.value, ids.RIGHT_KNEE.value):
        left_hip = get_coord(landmarks[ids.LEFT_HIP.value], w, h)
        left_knee = get_coord(landmarks[ids.LEFT_KNEE.value], w, h)
        left_ankle = get_coord(landmarks[ids.LEFT_ANKLE.value], w, h)

        right_hip = get_coord(landmarks[ids.RIGHT_HIP.value], w, h)
        right_knee = get_coord(landmarks[ids.RIGHT_KNEE.value], w, h)
        right_ankle = get_coord(landmarks[ids.RIGHT_ANKLE.value], w, h)

        # Knee angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Hip vertical movement
        hip_center = (left_hip + right_hip) / 2
        prev_hip = prev_data.get('hip_center')
        time_diff = current_time - prev_data.get('time', current_time)
        hip_speed = calculate_velocity(prev_hip, hip_center, time_diff) if prev_hip is not None else 0.0

        # Squat depth assessment
        if avg_knee_angle < 90:
            depth = "Deep"
        elif avg_knee_angle < 110:
            depth = "Parallel"
        else:
            depth = "Partial"

        metrics.append(f"Squat Analysis")
        metrics.append(f"L Knee: {left_knee_angle:.1f} deg | R Knee: {right_knee_angle:.1f} deg")
        metrics.append(f"Depth: {depth} ({avg_knee_angle:.1f} deg)")
        metrics.append(f"Hip Speed: {hip_speed:.1f} px/s")
        metrics.append(f"Form: {'Good' if abs(left_knee_angle - right_knee_angle) < 10 else 'Uneven'}")

        prev_data['hip_center'] = hip_center
        prev_data['time'] = current_time
    else:
        metrics.append("Squat - Landmarks not detected")

    return metrics, prev_data

# Game selection menu
GAMES = {
    '1': ('Cricket Bowling', cricket_bowling_metrics),
    '2': ('Basketball Shooting', basketball_shooting_metrics),
    '3': ('Yoga', yoga_pose_metrics),
    '4': ('Boxing', boxing_punch_metrics),
    '5': ('Squats/Weightlifting', squat_metrics)
}

def choose_game():
    """Display menu and get user's game choice"""
    print("\n=== Sports Performance Tracker ===", flush=True)
    print("Choose your sport/activity:", flush=True)
    for key, (name, _) in GAMES.items():
        print(f"{key}. {name}", flush=True)
    print("==================================", flush=True)
    print("", flush=True)  # Extra newline for clarity

    while True:
        choice = input("Enter your choice (1-5): ").strip()
        if choice in GAMES:
            game_name, metric_function = GAMES[choice]
            print(f"\nStarting {game_name} tracking...", flush=True)
            print("Press ESC to exit\n", flush=True)
            return game_name, metric_function
        else:
            print("Invalid choice. Please enter a number between 1 and 5.", flush=True)

def main():
    game_name, metric_function = choose_game()

    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        prev_data = {}

        while cap.isOpened():
            current_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

            # Calculate and display game-specific metrics
            landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
            metrics_text, prev_data = metric_function(landmarks, prev_data, w, h, current_time)

            # Overlay metrics on frame
            y_offset = 30
            for i, text in enumerate(metrics_text):
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow(f"Body Tracking - {game_name}", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
