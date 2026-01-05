import os
import cv2
from tqdm import tqdm

# BASE_DIR = "/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/driveway_walk_done" // done
# BASE_DIR = "/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/falling_done"
# BASE_DIR = "/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/jay_walk_done"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/001_wide_field"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/002_long_load"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/003_alley_way"
BASE_DIR = "/data/ARGOS/face_detector/Annotation/Youtube_done"
CROP_DIR = os.path.join(BASE_DIR, "crop")

os.makedirs(CROP_DIR, exist_ok=True)

IMG_EXTS = [".jpg", ".jpeg", ".png"]

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    """
    YOLO normalized bbox → pixel xyxy
    """
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    # clip
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    return x1, y1, x2, y2


def process_one_image(img_path, txt_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] Cannot read image: {img_path}")
        return

    h, w = img.shape[:2]
    base = os.path.splitext(os.path.basename(img_path))[0]

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls = parts[0]
        xc, yc, bw, bh = map(float, parts[1:])

        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

        if x2 <= x1 or y2 <= y1:
            continue

        crop_img = img[y1:y2, x1:x2]

        out_name = (
            f"{base}_{cls}_"
            f"{xc:.6f}_{yc:.6f}_{bw:.6f}_{bh:.6f}.jpg"
        )
        out_path = os.path.join(CROP_DIR, out_name)

        cv2.imwrite(out_path, crop_img)


def main():
    print("=== START CROP ===")

    for fname in tqdm(os.listdir(BASE_DIR)):
        if not any(fname.lower().endswith(ext) for ext in IMG_EXTS):
            continue

        img_path = os.path.join(BASE_DIR, fname)
        txt_path = os.path.join(BASE_DIR, os.path.splitext(fname)[0] + ".txt")

        if not os.path.exists(txt_path):
            print(f"[SKIP] No label: {fname}")
            continue

        process_one_image(img_path, txt_path)

    print("=== DONE ===")


if __name__ == "__main__":
    main()