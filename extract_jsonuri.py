import cv2
import mediapipe as mp
import json
import os

VIDEOS_DIR = "workouts_page/tutorials"
OUTPUT_DIR = "workouts_page/tutorials/landmarks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

video_files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(".mp4")]

for video_file in video_files:
    video_path = os.path.join(VIDEOS_DIR, video_file)
    plan_name = os.path.splitext(video_file)[0]  # Fără extensie
    safe_name = plan_name.lower().replace(" ", "_")
    output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")

    cap = cv2.VideoCapture(video_path)
    landmark_sequence = []

    print(f"[🔄] Procesăm: {video_file}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in result.pose_landmarks.landmark
            ]

            if len(landmarks) == 33:
                landmark_sequence.append(landmarks)

        # Sare câteva frame-uri (ex: salvează doar 1 din 3)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 != 0:
            continue

    cap.release()

    with open(output_path, "w") as f:
        json.dump(landmark_sequence, f, indent=2)

    print(f"[✅] Salvat: {output_path} | Frames: {len(landmark_sequence)}")

print("\n🎉 Toate JSON-urile au fost generate cu succes!")
