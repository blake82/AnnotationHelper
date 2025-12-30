import os
import shutil
import time

# -----------------------------
# CONFIG
# -----------------------------
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/001_wide_field"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/002_long_load"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/003_alley_way"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/004_driveway_sidewalk"

BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250901_VALID_TODO/004_driveway_sidewalk"
CROPPED_DIR = os.path.join(BASE_DIR, "crop_pred")
CROP_DIR = os.path.join(BASE_DIR, "crop")

VALID_TXT_PATH = os.path.join(CROPPED_DIR, "crop_valid.txt")

CF_DIR = os.path.join(BASE_DIR, "classfication")  # 요청하신 spelling 그대로 사용
DST_MAP = {
    0: os.path.join(CF_DIR, "O_Person"),
    1: os.path.join(CF_DIR, "O_Vehicle"),
    2: os.path.join(CF_DIR, "O_Bike"),
}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 안전장치: 먼저 True로 테스트 후, 실제 실행시 False로 변경
DRY_RUN = False

# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def normalize_relpath(p):
    # crop_valid.txt는 "animal\xxx.jpg" 형태일 수 있어 정규화
    return p.replace("\\", "/").strip()

def parse_crop_filename(fname):
    """
    crop 파일명 규칙:
      <orig_base>_<cls>_<x>_<y>_<w>_<h>.<ext>
    return:
      orig_base(str), cls(int), x,y,w,h(float)
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

def is_image_file(p):
    return p.lower().endswith(IMG_EXTS)

def remove_tree(path, log):
    if not os.path.exists(path):
        log(f"[SKIP] dir not found: {path}")
        return
    if DRY_RUN:
        log(f"[DRY] rmtree: {path}")
        return
    shutil.rmtree(path)
    log(f"[OK] rmtree: {path}")

def safe_unlink(path, log):
    if not os.path.exists(path):
        log(f"[SKIP] not found: {path}")
        return False
    if DRY_RUN:
        log(f"[DRY] delete: {path}")
        return True
    os.unlink(path)
    log(f"[OK] delete: {path}")
    return True

def safe_move(src, dst, log):
    safe_mkdir(os.path.dirname(dst))
    if DRY_RUN:
        log(f"[DRY] move: {src} -> {dst}")
        return True

    # 이름 충돌 방지: 동일 파일명이 있으면 _dupN 추가
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        n = 1
        while True:
            cand = f"{base}_dup{n}{ext}"
            if not os.path.exists(cand):
                dst = cand
                break
            n += 1

    shutil.move(src, dst)
    log(f"[OK] move: {src} -> {dst}")
    return True

# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isdir(CROPPED_DIR):
        print(f"[ERR] CROPPED_DIR not found: {CROPPED_DIR}")
        return 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_mkdir(CF_DIR)
    log_path = os.path.join(CF_DIR, f"step_final_log_{ts}.txt")

    def log(msg):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("===================================")
    log(f"DRY_RUN={DRY_RUN}")
    log(f"CROPPED_DIR={CROPPED_DIR}")
    log(f"CROP_DIR={CROP_DIR}")
    log(f"CF_DIR={CF_DIR}")
    log("===================================")

    # 1) crop_valid.txt 기반으로 state 1/2 crop 파일 삭제
    if not os.path.isfile(VALID_TXT_PATH):
        log(f"[ERR] crop_valid.txt not found: {VALID_TXT_PATH}")
        return 1

    to_delete = set()  # CROPPED_DIR 내 상대경로(정규화)
    total_lines = 0
    del_targets = 0

    with open(VALID_TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            rel = normalize_relpath(parts[0])
            try:
                st = int(parts[1])
            except Exception:
                continue

            total_lines += 1
            if st in (1, 2):
                to_delete.add(rel)
                del_targets += 1

    log(f"[INFO] crop_valid lines={total_lines}, delete_targets(state1/2)={del_targets}")

    deleted_ok = 0
    deleted_miss = 0

    for rel in sorted(to_delete):
        abs_path = os.path.join(CROPPED_DIR, rel)  # rel includes subdir/name.jpg
        if os.path.isfile(abs_path):
            if safe_unlink(abs_path, log):
                deleted_ok += 1
        else:
            deleted_miss += 1
            log(f"[MISS] delete target not found: {abs_path}")

    log(f"[INFO] deleted_ok={deleted_ok}, deleted_miss={deleted_miss}")

    # 2) 남은 crop 이미지들을 O_Person/O_Vehicle/O_Bike로 move
    for k, d in DST_MAP.items():
        safe_mkdir(d)

    moved_ok = 0
    moved_skip_parse = 0
    moved_skip_nonimg = 0
    moved_skip_unknowncls = 0

    # CROPPED_DIR 내부 모든 서브디렉터리 순회
    for root, dirs, files in os.walk(CROPPED_DIR):
        # crop_valid.txt는 건드리지 않음(원하시면 마지막에 삭제 가능)
        for fn in files:
            if fn == "crop_valid.txt":
                continue
            full = os.path.join(root, fn)
            if not is_image_file(full):
                moved_skip_nonimg += 1
                continue

            parsed = parse_crop_filename(fn)
            if not parsed:
                moved_skip_parse += 1
                log(f"[SKIP] parse fail: {full}")
                continue

            _orig_base, cls, _x, _y, _w, _h = parsed
            if cls not in DST_MAP:
                moved_skip_unknowncls += 1
                log(f"[SKIP] unknown cls={cls}: {full}")
                continue

            dst = os.path.join(DST_MAP[cls], fn)
            if safe_move(full, dst, log):
                moved_ok += 1

    log("===================================")
    log(f"[RESULT] moved_ok={moved_ok}")
    log(f"[RESULT] skip_parse={moved_skip_parse}")
    log(f"[RESULT] skip_nonimg={moved_skip_nonimg}")
    log(f"[RESULT] skip_unknowncls={moved_skip_unknowncls}")
    log("===================================")

    # 3) crop_pred / crop 디렉터리 삭제
    # - crop_pred에는 이미 이미지가 빠져나갔지만, 빈 폴더/기타 파일이 남을 수 있어 통째로 삭제
    # - crop도 통째로 삭제
    remove_tree(CROPPED_DIR, log)
    remove_tree(CROP_DIR, log)

    log("[DONE]")
    log(f"log saved: {log_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
