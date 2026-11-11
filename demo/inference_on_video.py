import cv2
import torch
import time
import numpy as np
import os

# ---------------- HARD SAFETY LOCK ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""   
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
os.environ["FORCE_CPU"] = "1"

# ---------------- GroundingDINO ----------------
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)

# ---------------- CONFIG ----------------
CONFIG_PATH = "../groundingdino/config/GroundingDINO_SwinB_cfg.py"
CHECKPOINT_PATH = "../weights/groundingdino_swinb_cogcoor.pth"

VIDEO_PATH = "foggy_traffic_people.mp4"
OUTPUT_VIDEO_PATH = "../outputs/foggy_traffic_people.mp4"

TEXT_PROMPT = "person . car . bus . scooter . truck . rickshaw . rain"

BASE_BOX_THRESHOLD = 0.30
BASE_TEXT_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.35

FRAME_SKIP = 2
OUT_W, OUT_H = 640, 360
MAX_MISSING_FRAMES = 3
# ----------------------------------------

print("✅ Running in CPU MODE")
DEVICE = "cpu"

torch.set_grad_enabled(False)
torch.set_num_threads(6)   

# ---------------- LOAD MODEL ----------------
model = load_model(
    CONFIG_PATH,
    CHECKPOINT_PATH,
    device="cpu"
)
model.eval()

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open video")

fps_video = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps_video / FRAME_SKIP,
    (OUT_W, OUT_H)
)

frame_id = 0
saved_frames = 0

last_boxes = None
last_logits = None
last_phrases = None
missing_count = 0

prev_time = time.time()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    if frame_id % FRAME_SKIP != 0:
        continue

    print(f"Processing frame {frame_id}")

    frame = cv2.resize(frame, (OUT_W, OUT_H))

    # ---------- Adaptive threshold ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()

    if contrast < 30:
        BOX_THRESHOLD = 0.22
        TEXT_THRESHOLD = 0.18
        visibility = "LOW (Foggy)"
    else:
        BOX_THRESHOLD = BASE_BOX_THRESHOLD
        TEXT_THRESHOLD = BASE_TEXT_THRESHOLD
        visibility = "NORMAL"

    # ---------- INFERENCE ----------
    try:
        temp_path = "_temp_frame.jpg"
        cv2.imwrite(temp_path, frame)

        image_source, image = load_image(temp_path)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
                device="cpu"
            )

    except Exception as e:
        print(f" Frame {frame_id} skipped:", e)
        boxes, logits, phrases = None, None, None

    # ---------- Confidence filtering ----------
    if boxes is not None and len(boxes) > 0:
        keep = logits > CONFIDENCE_THRESHOLD
        boxes = boxes[keep]
        logits = logits[keep]
        phrases = [phrases[i] for i in range(len(phrases)) if keep[i]]

    # ---------- Temporal stability ----------
    if boxes is not None and len(boxes) > 0:
        last_boxes = boxes
        last_logits = logits
        last_phrases = phrases
        missing_count = 0
    else:
        missing_count += 1
        if missing_count > MAX_MISSING_FRAMES:
            last_boxes = None
            last_logits = None
            last_phrases = None

    # ---------- Draw ----------
    if last_boxes is not None:
        annotated = annotate(
            frame[:, :, ::-1],
            last_boxes,
            last_logits,
            last_phrases
        )
    else:
        annotated = frame[:, :, ::-1]

    annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

    # ---------- FPS ----------
    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    cv2.putText(
        annotated_bgr,
        f"FPS: {fps:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_bgr,
        f"Visibility: {visibility}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    out.write(annotated_bgr)
    saved_frames += 1


cap.release()
out.release()

print("✅ Video processing complete!")
print("Frames written:", saved_frames)
print("Saved at:", OUTPUT_VIDEO_PATH)
