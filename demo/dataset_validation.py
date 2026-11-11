import os
import torch
import numpy as np
from groundingdino.util.inference import load_model, load_image, predict

# ===============================
# CONFIG
# ===============================
CONFIG_PATH = "../groundingdino/config/GroundingDINO_SwinB_cfg.py"
CHECKPOINT_PATH = "../weights/groundingdino_swinb_cogcoor.pth"

DATASET_PATH = "../cityscapes_foggy_val"

TEXT_PROMPT = "person . pedestrian . car . bus . truck . bicycle"

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
CONF_THRESHOLD = 0.50   # ⬅️ IMPORTANT (higher)

DEVICE = "cpu"
MAX_IMAGES = 500

# ===============================
print("🔹 Loading GroundingDINO model...")
model = load_model(CONFIG_PATH, CHECKPOINT_PATH, device=DEVICE)
print("✅ Model loaded successfully")

total_images = 0
valid_detections = 0
confidence_scores = []

# ===============================
# VALIDATION LOOP
# ===============================
for city in os.listdir(DATASET_PATH):
    city_path = os.path.join(DATASET_PATH, city)

    print(f"📁 Processing city folder: {city}")

    for img_name in os.listdir(city_path):
        img_path = os.path.join(city_path, img_name)

        image_source, image = load_image(img_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE
        )

        total_images += 1

        # ✅ STRICT validation
        if boxes is not None and len(boxes) > 0:
            high_conf = logits[logits >= CONF_THRESHOLD]

            if len(high_conf) > 0:
                valid_detections += 1
                confidence_scores.append(float(high_conf.mean()))

        if total_images % 10 == 0:
            print(f"▶ Processed {total_images} images...")

        if total_images >= MAX_IMAGES:
            break

    if total_images >= MAX_IMAGES:
        break

# ===============================
# FINAL METRICS
# ===============================
detection_accuracy = (valid_detections / total_images) * 100
avg_confidence = np.mean(confidence_scores) * 100 if confidence_scores else 0

print("\n==============================")
print("📊 DATASET VALIDATION RESULT")
print("==============================")
print(f"Total images tested      : {total_images}")
print(f"Valid detections         : {valid_detections}")
print(f"Detection Accuracy (%)   : {detection_accuracy:.2f}%")
print(f"Avg Confidence (%)       : {avg_confidence:.2f}%")
print("==============================")
