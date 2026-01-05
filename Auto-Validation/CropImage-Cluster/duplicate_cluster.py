import os
import shutil
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm
import imagehash

# -----------------------------
# CONFIG (여기만 수정)
# -----------------------------
SRC_DIR = "/data/ARGOS/face_detector/Annotation/classfication/O_Person/"          # 중복 제거 대상 디렉터리(하위 포함)
CLUSTER_DIR = "/data/ARGOS/face_detector/Annotation/classfication/O_Person" # 중복(이동) 저장 디렉터리


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# 1) 완전 동일(바이트 동일) 중복
ENABLE_EXACT_DUP = True

# 2) 유사(near) 중복
ENABLE_NEAR_DUP = True
PHASH_THRESH = 11          # 4는 너무 약함. 스크린샷 케이스면 8~12 권장
AHASH_THRESH = 12          # aHash도 함께 써서 후보군 넓히기
MSE_THRESH = 50.0          # 다운샘플 MSE (작을수록 더 비슷). 20~60 범위에서 조정

# 너무 많이 남는다 → PHASH_THRESH=12, MSE_THRESH=60로 조금 더 공격적으로
# 너무 다른 것도 묶인다(오탐) → PHASH_THRESH=8, MSE_THRESH=25로 보수적으로
# 속도 이슈 → Bucket compare에서 len(idxs)>2000 상한을 더 낮추거나, NORM_SIZE=96로 낮추기

# pHash 계산 안정화
NORM_SIZE = 128            # 전처리 resize
USE_BLUR = True            # 약간의 블러로 압축노이즈 완화

# 이동(move)/복사(copy)
MOVE_DUPLICATES = True

# -----------------------------
# Utils
# -----------------------------
def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def iter_image_files(root: str):
    for r, _d, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(r, fn)

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rel_from_src(path: str, src_root: str) -> str:
    return os.path.relpath(path, src_root).replace("\\", "/")

def move_or_copy(src: str, dst: str, move: bool):
    safe_mkdir(os.path.dirname(dst))
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)

def preprocess_for_hash(im: Image.Image) -> Image.Image:
    # grayscale + resize + (optional) blur
    im = im.convert("RGB")
    im = ImageOps.grayscale(im)
    im = im.resize((NORM_SIZE, NORM_SIZE), Image.BILINEAR)
    if USE_BLUR:
        im = im.filter(ImageFilter.GaussianBlur(radius=0.6))
    return im

def compute_hashes(path: str):
    with Image.open(path) as im:
        pim = preprocess_for_hash(im)
        ph = imagehash.phash(pim)  # 64-bit
        ah = imagehash.average_hash(pim)  # 64-bit
        return ph, ah, pim  # pim은 MSE에 재사용

def mse_downsample(a: Image.Image, b: Image.Image, ds: int = 32) -> float:
    # a,b: grayscale PIL images
    a2 = a.resize((ds, ds), Image.BILINEAR)
    b2 = b.resize((ds, ds), Image.BILINEAR)
    pa = list(a2.getdata())
    pb = list(b2.getdata())
    s = 0.0
    for x, y in zip(pa, pb):
        d = float(x) - float(y)
        s += d * d
    return s / (ds * ds)

@dataclass
class Group:
    keep: str
    dups: List[str]

# -----------------------------
# Stage 1: Exact duplicates
# -----------------------------
def cluster_exact(files: List[str]) -> Tuple[List[Group], List[str]]:
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for p in tqdm(files, desc="SHA1 hashing", unit="img"):
        try:
            by_hash[sha1_file(p)].append(p)
        except Exception:
            by_hash[f"__READFAIL__:{p}"].append(p)

    groups: List[Group] = []
    uniques: List[str] = []
    for _h, paths in by_hash.items():
        if len(paths) == 1:
            uniques.append(paths[0])
        else:
            paths_sorted = sorted(paths)
            groups.append(Group(keep=paths_sorted[0], dups=paths_sorted[1:]))
            uniques.append(paths_sorted[0])
    return groups, uniques

# -----------------------------
# Stage 2: Near duplicates (improved)
# -----------------------------
def build_lsh_buckets(items: List[Tuple[str, imagehash.ImageHash, imagehash.ImageHash]]) -> Dict[str, List[int]]:
    """
    LSH 스타일로 후보를 줄이기 위해 hash의 일부(밴드)로 버킷 구성.
    - phash/ahash 각각을 여러 밴드로 쪼개서 동일 밴드면 후보로 보게 함.
    """
    buckets = defaultdict(list)

    def add_bands(tag: str, h: imagehash.ImageHash, idx: int):
        # 64bit -> hex 16 chars
        hs = str(h)
        # 4밴드(각 4 hex chars) = 16bit씩
        # 필요하면 밴드 수/길이 조절 가능
        bands = [hs[0:4], hs[4:8], hs[8:12], hs[12:16]]
        for bi, b in enumerate(bands):
            buckets[f"{tag}:{bi}:{b}"].append(idx)

    for i, (_p, ph, ah) in enumerate(items):
        add_bands("p", ph, i)
        add_bands("a", ah, i)

    return buckets

def cluster_near(files: List[str]) -> List[Group]:
    # 1) 해시/전처리 이미지 준비
    phash_list: List[Optional[imagehash.ImageHash]] = [None] * len(files)
    ahash_list: List[Optional[imagehash.ImageHash]] = [None] * len(files)
    prep_img: List[Optional[Image.Image]] = [None] * len(files)

    for i, p in enumerate(tqdm(files, desc="Compute hashes", unit="img")):
        try:
            ph, ah, pim = compute_hashes(p)
            phash_list[i] = ph
            ahash_list[i] = ah
            prep_img[i] = pim
        except Exception:
            phash_list[i] = None
            ahash_list[i] = None
            prep_img[i] = None

    items = [(files[i], phash_list[i], ahash_list[i]) for i in range(len(files)) if phash_list[i] is not None]
    idx_map = {p: i for i, p in enumerate(files)}

    # 2) LSH 버킷으로 후보군 만들기
    compact = [(p, phash_list[idx_map[p]], ahash_list[idx_map[p]]) for (p, _, _) in items]  # type: ignore
    buckets = build_lsh_buckets(compact)  # indices are compact-index

    # 3) Union-Find로 그룹화
    n = len(compact)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 후보 비교
    # 같은 버킷에 들어온 것들만 pairwise 비교 (큰 그룹은 샘플링/제한 가능)
    for _k, idxs in tqdm(buckets.items(), desc="Bucket compare", unit="bucket"):
        if len(idxs) < 2:
            continue
        # 너무 큰 버킷은 비용 급증 -> 상한
        if len(idxs) > 2000:
            idxs = idxs[:2000]

        # pairwise (O(m^2))이지만 m이 작도록 버킷을 설계했음
        for i in range(len(idxs)):
            a = idxs[i]
            pa, pha, aha = compact[a]
            ia = idx_map[pa]
            for j in range(i + 1, len(idxs)):
                b = idxs[j]
                pb, phb, ahb = compact[b]
                ib = idx_map[pb]

                # 해밍 1차 필터 (둘 중 하나라도 기준 통과하면 후보)
                if (pha - phb) > PHASH_THRESH and (aha - ahb) > AHASH_THRESH:
                    continue

                # 2차: MSE로 최종 판정
                if prep_img[ia] is None or prep_img[ib] is None:
                    continue
                m = mse_downsample(prep_img[ia], prep_img[ib], ds=32)
                if m <= MSE_THRESH:
                    union(a, b)

    # 4) 그룹 추출 (size>=2)
    groups_map = defaultdict(list)
    for i in range(n):
        groups_map[find(i)].append(i)

    groups: List[Group] = []
    for _, members in groups_map.items():
        if len(members) < 2:
            continue
        paths = [compact[i][0] for i in members]
        paths_sorted = sorted(paths)
        keep = paths_sorted[0]
        dups = paths_sorted[1:]
        groups.append(Group(keep=keep, dups=dups))

    return groups

# -----------------------------
# Apply: move duplicates
# -----------------------------
def apply_groups(groups: List[Group], src_root: str, cluster_root: str, tag: str):
    moved = 0
    total_dups = 0
    if not groups:
        return moved, total_dups

    for gi, g in enumerate(tqdm(groups, desc=f"Move {tag}", unit="grp")):
        grp_dir = os.path.join(cluster_root, tag, f"group_{gi:06d}")
        safe_mkdir(grp_dir)

        for dup in g.dups:
            total_dups += 1
            rel = rel_from_src(dup, src_root)
            dst = os.path.join(grp_dir, rel)
            try:
                move_or_copy(dup, dst, MOVE_DUPLICATES)
                moved += 1
            except Exception:
                pass

    return moved, total_dups

# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.isdir(SRC_DIR):
        print(f"[ERR] SRC_DIR not found: {SRC_DIR}")
        return 1
    safe_mkdir(CLUSTER_DIR)

    files = list(iter_image_files(SRC_DIR))
    if not files:
        print(f"[ERR] no images under: {SRC_DIR}")
        return 1

    print(f"[INFO] Found images: {len(files)}")
    moved_total = 0
    dups_total = 0

    uniques = files

    if ENABLE_EXACT_DUP:
        exact_groups, uniques = cluster_exact(files)
        moved, dups = apply_groups(exact_groups, SRC_DIR, CLUSTER_DIR, tag="exact_sha1")
        moved_total += moved
        dups_total += dups
        print(f"[OK] Exact: groups={len(exact_groups)}, dups={dups}, moved={moved}")

    if ENABLE_NEAR_DUP:
        near_groups = cluster_near(uniques)
        moved, dups = apply_groups(near_groups, SRC_DIR, CLUSTER_DIR, tag=f"near_phash{PHASH_THRESH}_ahash{AHASH_THRESH}_mse{int(MSE_THRESH)}")
        moved_total += moved
        dups_total += dups
        print(f"[OK] Near: groups={len(near_groups)}, dups={dups}, moved={moved}")

    print("===================================")
    print(f"SRC_DIR:      {SRC_DIR}")
    print(f"CLUSTER_DIR:  {CLUSTER_DIR}")
    print(f"DUP_FILES:    {dups_total}")
    print(f"MOVED/COPIED: {moved_total}")
    print("===================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
