import os
import csv
from collections import defaultdict

def iter_label_files(label_dir: str, recursive: bool = False):
    if recursive:
        for root, _dirs, files in os.walk(label_dir):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(label_dir):
            if fn.lower().endswith(".txt"):
                yield os.path.join(label_dir, fn)

def count_yolo_classes(label_dir: str, recursive: bool = False):
    class_count = defaultdict(int)
    total_files = 0
    total_boxes = 0

    for label_path in iter_label_files(label_dir, recursive=recursive):
        if not os.path.isfile(label_path):
            continue

        total_files += 1

        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                # YOLO: cls x y w h (기본 5개). 뒤에 뭐가 더 붙어도 cls만 읽으면 됨.
                if len(parts) < 5:
                    continue

                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    continue

                class_count[cls] += 1
                total_boxes += 1

    return class_count, total_files, total_boxes

def save_counts_csv(out_csv_path: str, label_dir: str, counts: dict, total_files: int, total_boxes: int):
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label_dir", "total_txt_files", "total_bboxes", "class_id", "count", "ratio"])

        if total_boxes <= 0:
            # 라벨이 하나도 없을 때도 CSV는 남김
            w.writerow([label_dir, total_files, total_boxes, "", 0, 0.0])
            return

        for cls in sorted(counts.keys()):
            c = counts[cls]
            ratio = c / float(total_boxes)
            w.writerow([label_dir, total_files, total_boxes, cls, c, ratio])

if __name__ == "__main__":
    # 변경해서 사용
    LABEL_DIR = "/home2/ai/data/detection/outdoor_corridor/BEFORE_20251231/train"
    # 필요하면 True로: 하위 디렉터리까지 재귀 탐색
    RECURSIVE = False

    # 저장할 CSV 경로
    OUT_CSV = "./output/count_of_object.csv"

    counts, num_files, num_boxes = count_yolo_classes(LABEL_DIR, recursive=RECURSIVE)

    print("======================================")
    print(f"Label directory : {LABEL_DIR}")
    print(f"Total txt files : {num_files}")
    print(f"Total bboxes    : {num_boxes}")
    print("--------------------------------------")
    print("Class ID : Count")
    print("--------------------------------------")
    for cls in sorted(counts.keys()):
        print(f"{cls:>7} : {counts[cls]}")
    print("======================================")

    save_counts_csv(OUT_CSV, LABEL_DIR, counts, num_files, num_boxes)
    print(f"[OK] CSV saved: {OUT_CSV}")



