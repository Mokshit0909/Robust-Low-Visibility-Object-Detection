import cv2
import numpy as np
import os

INPUT_DIR = "../cityscapes dataset/leftImg8bit_trainvaltest/leftImg8bit/val"
OUTPUT_DIR = "../cityscapes_foggy_val"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def add_fog(image, beta=0.08):
    h, w = image.shape[:2]
    fog = np.ones((h, w, 3), dtype=np.uint8) * 255
    foggy = cv2.addWeighted(image, 1 - beta, fog, beta, 0)
    return foggy

count = 0
for city in os.listdir(INPUT_DIR):
    city_path = os.path.join(INPUT_DIR, city)
    out_city = os.path.join(OUTPUT_DIR, city)
    os.makedirs(out_city, exist_ok=True)

    for img in os.listdir(city_path):
        img_path = os.path.join(city_path, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = add_fog(image)
        image = cv2.GaussianBlur(image, (7, 7), 0)

        cv2.imwrite(os.path.join(out_city, img), image)

        count += 1
        if count >= 500:
            break
    if count >= 500:
        break

print("✅ Foggy & blurry images generated:", count)
