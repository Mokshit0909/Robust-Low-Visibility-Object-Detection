import cv2
import os
import numpy as np

INPUT_DIR = "../cityscapes dataset/leftImg8bit_trainvaltest/leftImg8bit/val"
OUTPUT_DIR = "../cityscapes dataset/foggy_val"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_fog(image, intensity=0.6):
    h, w = image.shape[:2]
    fog = np.full((h, w, 3), 255, dtype=np.uint8)
    foggy = cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    return foggy

def add_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

count = 0

for city in os.listdir(INPUT_DIR):
    city_path = os.path.join(INPUT_DIR, city)
    out_city = os.path.join(OUTPUT_DIR, city)
    os.makedirs(out_city, exist_ok=True)

    for img_name in os.listdir(city_path):
        img_path = os.path.join(city_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        foggy = add_fog(img)
        foggy = add_blur(foggy)

        out_path = os.path.join(out_city, img_name)
        cv2.imwrite(out_path, foggy)

        count += 1
        if count % 50 == 0:
            print(f"Processed {count} images")

print("✅ Foggy dataset created successfully!")
