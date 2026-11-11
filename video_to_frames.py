import cv2
import os

video_path = "demo/test_video.mp4"
frames_dir = "outputs/frames"

os.makedirs(frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f"{frames_dir}/frame_{frame_id:05d}.jpg", frame)
    frame_id += 1

cap.release()
print(f"Saved {frame_id} frames")
