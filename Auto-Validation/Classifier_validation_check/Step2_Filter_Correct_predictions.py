import os

# BASE_DIR = "/data/ARGOS/face_detector/Annotation/Outdoor_simple_Quickly/driveway_walk_done/crop_pred"
# # BASE_DIR = "/data/ARGOS/005.outdoor_detector/"
# # BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/002_long_load"
# # BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/003_alley_way"
# BASE_DIR = "/data/ARGOS/005.outdoor_detector/20250831/004_driveway_sidewalk"
BASE_DIR = "/data/ARGOS/face_detector/Annotation/Youtube_done/crop_pred"

# True면 삭제하지 않고 로그만 출력
DRY_RUN = False

# 26 class
# CLASS_DIR_TO_DET_IDX = {
#     "person_L_done":      [0],

#     "car_L_done":         [1],
#     "bus_L_done":         [1],
#     "truck_L_done":       [1],

#     "bicycle_L_done":     [2],
#     "motorbike_L_done":   [2],
# }
# 3 class
CLASS_DIR_TO_DET_IDX = {
    "unknown":         [3],
    "O_Bike":      [2],
    "O_Person":         [0],
    "O_Vehicle":         [1],
}

def extract_det_class_from_filename(fname):
    """
    filename 규칙:
    xxx_<cls>_<x>_<y>_<w>_<h>.jpg
    """
    name = os.path.splitext(fname)[0]
    parts = name.split("_")
    try:
        return int(parts[-5])
    except Exception:
        return None

def main():
    total_checked = 0
    total_deleted = 0

    for class_dir, valid_det_indices in CLASS_DIR_TO_DET_IDX.items():
        dir_path = os.path.join(BASE_DIR, class_dir)
        if not os.path.isdir(dir_path):
            continue

        for fname in os.listdir(dir_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue

            total_checked += 1
            det_idx = extract_det_class_from_filename(fname)
            if det_idx is None:
                continue

            if det_idx in valid_det_indices:
                full_path = os.path.join(dir_path, fname)
                total_deleted += 1

                if DRY_RUN:
                    print(f"[DELETE-CANDIDATE] {full_path}")
                else:
                    os.remove(full_path)
                    print(f"[DELETED] {full_path}")

    print("\n================ RESULT ================")
    print(f"Checked files : {total_checked}")
    print(f"Deleted files : {total_deleted}")
    print("========================================")

if __name__ == "__main__":
    main()
