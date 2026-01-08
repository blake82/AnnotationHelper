import os
from collections import defaultdict

def count_yolo_classes(label_dir):
    class_count = defaultdict(int)
    total_files = 0
    total_boxes = 0

    for filename in os.listdir(label_dir):
        if not filename.lower().endswith(".txt"):
            continue

        label_path = os.path.join(label_dir, filename)
        if not os.path.isfile(label_path):
            continue

        total_files += 1

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                try:
                    cls = int(float(parts[0]))
                except ValueError:
                    continue

                class_count[cls] += 1
                total_boxes += 1

    return class_count, total_files, total_boxes


if __name__ == "__main__":
    #LABEL_DIR = "/data/ARGOS/face_detector/Annotation/outdoor_PersonData_20251231_BEFORE/119_IR_Data_Person"  # 변경해서 사용
    LABEL_DIR = "/home2/ai/data/detection/outdoor_corridor/BEFORE_20251231/train"  # 변경해서 사용

    counts, num_files, num_boxes = count_yolo_classes(LABEL_DIR)

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
