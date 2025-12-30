import os
import shutil
from PIL import Image, ImageDraw
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/001_wide_field"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/002_long_load"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/003_alley_way"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/004_driveway_sidewalk"

BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250901_VALID_TODO/004_driveway_sidewalk"
CROPPED_DIR = "/data/ARGOS/005.outdoor_detector/20250901_VALID_TODO/004_driveway_sidewalk/crop_pred"
VALID_TXT_PATH = os.path.join(CROPPED_DIR, "crop_valid.txt")
VALID_CHECK_DIR = os.path.join(BASE_DIR, "validcheck")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

SCALE_FACTOR = 1.5  # bbox 1.5배 확대

COLOR_DELETE = (255, 0, 0)        # RED
COLOR_MODIFY = (255, 165, 0)      # ORANGE

# 요청사항: status=1(DELETE)도 1.5배 확대 + "얇은" 느낌 유지
WIDTH_DELETE = 5
WIDTH_MODIFY = 5

# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def parse_crop_filename(fname):
    """
    crop 파일명 규칙:
      <orig_base>_<cls>_<x>_<y>_<w>_<h>.<ext>
    return: orig_base, cls, x,y,w,h
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    parts = base.split("_")
    if len(parts) < 6:
        return None
    try:
        cls = int(float(parts[-5]))
        x = float(parts[-4])
        y = float(parts[-3])
        w = float(parts[-2])
        h = float(parts[-1])
    except Exception:
        return None
    orig_base = "_".join(parts[:-5])
    return orig_base, cls, x, y, w, h

def find_original_image_path(orig_base):
    for ext in IMG_EXTS:
        p = os.path.join(BASE_DIR, orig_base + ext)
        if os.path.isfile(p):
            return p
    return None

def yolo_to_xyxy_scaled(x, y, w, h, img_w, img_h, scale=1.0):
    cx = x * img_w
    cy = y * img_h
    bw = (w * img_w) * scale
    bh = (h * img_h) * scale

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))

    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

def normalize_relpath(p):
    return p.replace("\\", "/").strip()

def float_close(a, b, eps=1e-6):
    return abs(a - b) <= eps

def remove_matching_label_line(txt_path, cls, x, y, w, h):
    """
    status=1(DELETE): txt에서 동일 bbox 라인 제거
    """
    if not os.path.isfile(txt_path):
        return 0

    removed = 0
    out_lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 5:
                out_lines.append(line)
                continue
            try:
                c = int(float(parts[0]))
                fx = float(parts[1])
                fy = float(parts[2])
                fw = float(parts[3])
                fh = float(parts[4])
            except Exception:
                out_lines.append(line)
                continue

            if (c == cls and
                float_close(fx, x) and float_close(fy, y) and
                float_close(fw, w) and float_close(fh, h)):
                removed += 1
                continue
            out_lines.append(line)

    if removed > 0:
        tmp = txt_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        os.replace(tmp, txt_path)

    return removed

def draw_bbox_on_image(img_path, cls, x, y, w, h, status):
    """
    img_path(=VALID_CHECK_DIR의 이미지)에 bbox를 "추가로" 그린다.
    (이미 그려진 bbox 유지)
    """
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size

    x1, y1, x2, y2 = yolo_to_xyxy_scaled(x, y, w, h, iw, ih, SCALE_FACTOR)

    draw = ImageDraw.Draw(img)
    if status == 1:
        color = COLOR_DELETE
        width = WIDTH_DELETE
    else:
        color = COLOR_MODIFY
        width = WIDTH_MODIFY

    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    img.save(img_path)

# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isfile(VALID_TXT_PATH):
        print(f"[ERR] crop_valid.txt not found: {VALID_TXT_PATH}")
        return 1

    safe_mkdir(VALID_CHECK_DIR)

    # classes.txt 복사
    classes_src = os.path.join(BASE_DIR, "classes.txt")
    classes_dst = os.path.join(VALID_CHECK_DIR, "classes.txt")
    if os.path.isfile(classes_src):
        shutil.copy2(classes_src, classes_dst)
        print(f"[OK] copied classes.txt -> {classes_dst}")

    # ✅ 같은 orig_base는 최초 1회만 copy하기 위한 set
    processed = set()

    total = 0
    copied_unique = 0
    not_found = 0
    removed_labels = 0

    with open(VALID_TXT_PATH, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing valid check"):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            rel = normalize_relpath(parts[0])
            try:
                status = int(parts[1])
            except Exception:
                continue

            if status not in (1, 2):
                continue

            total += 1

            crop_fname = os.path.basename(rel)
            parsed = parse_crop_filename(crop_fname)
            if not parsed:
                print(f"[SKIP] cannot parse crop filename: {rel}")
                continue

            orig_base, cls, x, y, w, h = parsed

            orig_img_path = find_original_image_path(orig_base)
            if not orig_img_path:
                not_found += 1
                print(f"[MISS] original image not found for base: {orig_base}")
                continue

            orig_txt_path = os.path.join(BASE_DIR, orig_base + ".txt")

            dst_img_path = os.path.join(VALID_CHECK_DIR, os.path.basename(orig_img_path))
            dst_txt_path = os.path.join(VALID_CHECK_DIR, orig_base + ".txt")

            # ✅ 핵심 수정: 같은 orig_base면 copy로 덮어쓰지 않는다
            if orig_base not in processed:
                shutil.copy2(orig_img_path, dst_img_path)
                if os.path.isfile(orig_txt_path):
                    shutil.copy2(orig_txt_path, dst_txt_path)
                else:
                    print(f"[WARN] label txt not found: {orig_txt_path}")

                processed.add(orig_base)
                copied_unique += 1

            # ✅ bbox는 validcheck 이미지에 "누적"으로 그린다
            draw_bbox_on_image(dst_img_path, cls, x, y, w, h, status)

            # status=1이면 해당 라벨 라인 삭제
            if status == 1 and os.path.isfile(dst_txt_path):
                removed = remove_matching_label_line(dst_txt_path, cls, x, y, w, h)
                removed_labels += removed

    print("===================================")
    print(f"TARGET(status in 1,2): {total}")
    print(f"COPIED unique images:  {copied_unique}")
    print(f"MISS originals:        {not_found}")
    print(f"REMOVED labels:        {removed_labels}")
    print(f"OUTPUT DIR:            {VALID_CHECK_DIR}")
    print("===================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
