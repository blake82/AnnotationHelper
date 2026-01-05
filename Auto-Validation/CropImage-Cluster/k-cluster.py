import os
import shutil
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# -----------------------------
# CONFIG (여기만 수정)
# -----------------------------
SRC_DIR = "/data/ARGOS/face_detector/Annotation/classfication/O_Vehicle"                 # crop 이미지 폴더
OUT_DIR = "/data/ARGOS/face_detector/Annotation/classfication/O_Vehicle_k_Cluster/"        # 결과 폴더
MOVE_FILES = False   # True: move / False: copy
SEED = 12345

# 그룹 수(K) 자동결정 파라미터
MIN_GROUP_RATIO = 0.10
MAX_GROUP_RATIO = 0.30
SIGNATURE_SAMPLE_MAX = 2000
REP_POOL_MAX = 5000

# Feature 설정: HOG + HSV hist
HOG_RESIZE = (96, 96)
HOG_WIN = (96, 96)
HOG_BLOCK = (16, 16)
HOG_STRIDE = (8, 8)
HOG_CELL = (8, 8)
HOG_NBINS = 9

HSV_RESIZE = (128, 128)
H_BINS, S_BINS, V_BINS = 24, 8, 8

HOG_WEIGHT = 1.0
COLOR_WEIGHT = 1.0

# output layout
DUPS_DIRNAME = "DUPS"
GLOBAL_KEEP_DIRNAME = "KEEP_DIR"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# -----------------------------
# Utils
# -----------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def list_images(src: Path):
    return [p for p in src.rglob("*") if p.is_file() and is_image(p)]

def l2_normalize(v: np.ndarray, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def copy_or_move(src: Path, dst: Path, move: bool):
    safe_mkdir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def imread_bgr(p: Path):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)

def safe_copy2(src: Path, dst: Path):
    safe_mkdir(dst.parent)
    shutil.copy2(str(src), str(dst))


# -----------------------------
# Feature: HOG
# -----------------------------
_HOG = cv2.HOGDescriptor(
    _winSize=HOG_WIN,
    _blockSize=HOG_BLOCK,
    _blockStride=HOG_STRIDE,
    _cellSize=HOG_CELL,
    _nbins=HOG_NBINS,
)

def hog_feature(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, HOG_RESIZE, interpolation=cv2.INTER_AREA)
    feat = _HOG.compute(gray)
    if feat is None:
        return np.zeros((0,), np.float32)
    feat = feat.reshape(-1).astype(np.float32)
    return l2_normalize(feat)

# -----------------------------
# Feature: HSV hist
# -----------------------------
def hsv_color_hist(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_bgr, HSV_RESIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=[H_BINS, S_BINS, V_BINS],
        ranges=[0, 180, 0, 256, 0, 256],
    ).astype(np.float32)

    s = float(hist.sum())
    if s > 0:
        hist /= s
    feat = hist.flatten()
    return l2_normalize(feat)

def extract_feature(img_path: Path) -> np.ndarray:
    bgr = imread_bgr(img_path)
    if bgr is None:
        return None

    h = hog_feature(bgr) * HOG_WEIGHT
    c = hsv_color_hist(bgr) * COLOR_WEIGHT
    feat = np.concatenate([h, c], axis=0).astype(np.float32)
    return l2_normalize(feat)

# -----------------------------
# "Histogram analysis" for K estimation
# -----------------------------
def coarse_signature(img_bgr: np.ndarray) -> bytes:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    e = cv2.Canny(g, 50, 150)
    e = (e > 0).astype(np.uint8)

    grid = 8
    cell = 128 // grid
    edge_bins = []
    for yy in range(grid):
        for xx in range(grid):
            patch = e[yy*cell:(yy+1)*cell, xx*cell:(xx+1)*cell]
            edge_bins.append(int(patch.mean() * 255))
    edge_bins = np.array(edge_bins, dtype=np.uint8)

    img = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256]).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    hist_q = np.clip(hist.flatten() * 255.0, 0, 255).astype(np.uint8)

    return edge_bins.tobytes() + hist_q.tobytes()

def estimate_k_by_histogram(img_paths, seed, min_ratio, max_ratio):
    n = len(img_paths)
    minK = max(2, int(np.ceil(n * min_ratio)))
    maxK = min(n, int(np.ceil(n * max_ratio)))

    random.seed(seed)
    sample_n = min(n, SIGNATURE_SAMPLE_MAX)
    sample_paths = random.sample(img_paths, sample_n) if sample_n < n else img_paths

    sig_set = set()
    for p in tqdm(sample_paths, desc="AnalyzeHist(K)", unit="img"):
        bgr = imread_bgr(p)
        if bgr is None:
            continue
        sig_set.add(coarse_signature(bgr))

    uniq = len(sig_set)
    uniq_ratio = uniq / max(1, sample_n)
    k_est = int(round(uniq_ratio * n))

    k = max(minK, min(maxK, k_est))
    return k, minK, maxK, k_est, uniq_ratio, sample_n, uniq

# -----------------------------
# Representative selection: FPS
# -----------------------------
def farthest_point_sampling(feats: np.ndarray, k: int, seed: int):
    M = feats.shape[0]
    k = min(k, M)
    rng = np.random.RandomState(seed)

    first = rng.randint(0, M)
    selected = [first]
    best_sim = feats @ feats[first]

    for _ in tqdm(range(1, k), desc="FPS", unit="rep"):
        nxt = int(np.argmin(best_sim))
        selected.append(nxt)
        sim_new = feats @ feats[nxt]
        best_sim = np.maximum(best_sim, sim_new)

    return np.array(selected, dtype=np.int32)

# -----------------------------
# Main
# -----------------------------
def main():
    src = Path(SRC_DIR)
    out = Path(OUT_DIR)
    safe_mkdir(out)

    imgs = list_images(src)
    n = len(imgs)
    if n == 0:
        print(f"[ERR] No images: {SRC_DIR}")
        return 1

    print("===================================")
    print(f"TOTAL: {n}")
    print("K will be chosen by histogram analysis (min 10% groups).")
    print("===================================")

    # 1) K 자동 결정
    k, minK, maxK, k_est, uniq_ratio, sample_n, uniq = estimate_k_by_histogram(
        imgs, SEED, MIN_GROUP_RATIO, MAX_GROUP_RATIO
    )

    print("---------- K decision ----------")
    print(f"sample_n={sample_n}, unique_sig={uniq}, unique_ratio={uniq_ratio:.4f}")
    print(f"k_est={k_est}, minK={minK}, maxK={maxK} -> K={k}")
    print("--------------------------------")

    # 2) 대표 선택 후보 풀
    random.seed(SEED)
    pool_n = min(n, REP_POOL_MAX)
    pool = random.sample(imgs, pool_n) if pool_n < n else imgs

    # 3) pool feature
    pool_feats = []
    pool_paths = []
    print("[1/4] Extracting features for representative pool...")
    for p in tqdm(pool, desc="PoolFeat", unit="img"):
        f = extract_feature(p)
        if f is None:
            continue
        pool_feats.append(f)
        pool_paths.append(p)

    if len(pool_feats) == 0:
        print("[ERR] Could not extract any feature.")
        return 1

    pool_feats = np.stack(pool_feats, axis=0)
    pool_paths = list(pool_paths)

    # 4) FPS 대표 선택
    print("[2/4] Selecting representatives by FPS...")
    rep_idx = farthest_point_sampling(pool_feats, k, SEED)
    rep_feats = pool_feats[rep_idx]
    rep_paths = [pool_paths[i] for i in rep_idx]

    # 5) 전체 이미지 할당
    print("[3/4] Assigning all images to nearest representative (cosine)...")
    groups = [[] for _ in range(len(rep_paths))]

    for p in tqdm(imgs, desc="Assign", unit="img"):
        f = extract_feature(p)
        if f is None:
            continue
        sims = rep_feats @ f
        best = int(np.argmax(sims))
        groups[best].append(p)

    # 6) 저장 구조 변경:
    # - group_xxxx/ : 대표 1장(KEEP 폴더 없음)
    # - group_xxxx/DUPS/ : 중복들
    # - OUT_DIR/KEEP_DIR/ : 모든 대표를 추가 copy2로 모음
    print("[4/4] Writing groups (rep at group root, DUPS only) and collecting global KEEP_DIR...")

    global_keep = out / GLOBAL_KEEP_DIRNAME
    safe_mkdir(global_keep)

    total_written = 0
    total_dups = 0
    total_reps = 0

    for gi, g in enumerate(tqdm(groups, desc="WriteGroup", unit="group")):
        if not g:
            continue

        grp_dir = out / f"group_{gi:04d}"
        dups_dir = grp_dir / DUPS_DIRNAME
        safe_mkdir(grp_dir)
        safe_mkdir(dups_dir)

        # 그룹 내부 대표 1장 선정(대표 feature와 가장 유사한 실제 이미지)
        best_p = None
        best_sim = -1.0
        rep_f = rep_feats[gi]

        for p in g:
            f = extract_feature(p)
            if f is None:
                continue
            s = float(rep_f @ f)
            if s > best_sim:
                best_sim = s
                best_p = p

        if best_p is None:
            continue

        # 대표는 group_xxxx/ 루트로
        rep_dst = grp_dir / best_p.name
        copy_or_move(best_p, rep_dst, MOVE_FILES)
        total_written += 1
        total_reps += 1

        # 대표를 GLOBAL KEEP_DIR에도 추가 복사(copy2 고정)
        # (MOVE_FILES=True여도 대표는 이미 grp_dir로 이동했으므로 그 파일을 복사)
        keep_dst = global_keep / best_p.name
        if keep_dst.exists():
            # 충돌 방지: group id prefix
            keep_dst = global_keep / f"group_{gi:04d}__{best_p.name}"
        safe_copy2(rep_dst, keep_dst)

        # 나머지는 DUPS로
        for p in g:
            if p == best_p:
                continue
            dup_dst = dups_dir / p.name
            copy_or_move(p, dup_dst, MOVE_FILES)
            total_written += 1
            total_dups += 1

    print("===================================")
    print(f"K(groups):         {len(rep_paths)}")
    print(f"REP(kept):         {total_reps}")
    print(f"TOTAL written:     {total_written}")
    print(f"DUPS moved/copied: {total_dups}")
    print(f"OUT_DIR:           {OUT_DIR}")
    print(f"GLOBAL KEEP_DIR:   {str(global_keep)}")
    print(f"MODE:              {'MOVE' if MOVE_FILES else 'COPY'}")
    print("===================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())



