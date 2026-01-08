import os
import csv
from collections import defaultdict

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import random


# =========================
# CONFIG
# =========================
TRAIN_DIR = "/home2/ai/data/detection/outdoor_corridor/train"
VAL_DIR   = "/home2/ai/data/detection/outdoor_corridor/validation"

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

OUT_DIR = "./output"
OUT_CSV = os.path.join(OUT_DIR, "bbox_histograms.csv")
OUT_PNG = os.path.join(OUT_DIR, "bbox_histograms.png")

# ✅ bin 간격 설정 가능 (요청: 0.02)
BIN_STEP = 0.02  # 0.02 -> 50 bins, 0.1 -> 10 bins

# class 개수(자동 추정도 가능하지만, 여기서는 라벨에서 등장한 class로 자동 수집)
# 필요하면 강제로 [0,1,2]처럼 지정해도 됩니다.
FORCE_CLASS_IDS = None  # 예: [0,1,2] 또는 None(자동)

EPS = 1e-12

# =========================
# Utils
# =========================
def find_image_for_label(txt_path: str) -> str | None:
    base = os.path.splitext(txt_path)[0]
    for ext in IMG_EXTS:
        p = base + ext
        if os.path.isfile(p):
            return p
    return None

def clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

def compute_num_bins(bin_step: float) -> int:
    # 1.0을 정확히 포함시키기 위해 ceil
    return int(np.ceil(1.0 / bin_step))

def bin_edges(bin_step: float):
    n = compute_num_bins(bin_step)
    edges = [i * bin_step for i in range(n + 1)]
    edges[-1] = 1.0  # 마지막은 1.0 고정
    return edges

def bin_index(x01: float, bin_step: float) -> int:
    """
    [0,1] 값을 bin_step으로 quantize.
    마지막 bin은 1.0 포함.
    """
    x01 = clamp01(x01)
    n = compute_num_bins(bin_step)
    if x01 >= 1.0 - EPS:
        return n - 1
    idx = int(x01 / bin_step)
    if idx < 0: idx = 0
    if idx >= n: idx = n - 1
    return idx

def iter_txt_files(root_dir: str):
    for r, _d, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                yield os.path.join(r, fn)

def parse_yolo_line(line: str):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(float(parts[0]))
        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        return cls, x, y, w, h
    except Exception:
        return None

# =========================
# Histogram Accumulator
# =========================
def init_hist(num_bins: int):
    # metrics: area, width, height
    return {
        "area":  np.zeros(num_bins, dtype=np.int64),
        "width": np.zeros(num_bins, dtype=np.int64),
        "height":np.zeros(num_bins, dtype=np.int64),
    }

def accumulate(train_dir: str, val_dir: str, bin_step: float):
    num_bins = compute_num_bins(bin_step)

    # 전체(all) + class별
    h_all = init_hist(num_bins)
    h_cls = defaultdict(lambda: init_hist(num_bins))

    meta = {
        "total_txt": 0,
        "total_boxes": 0,
        "missing_img": 0,
        "bad_img": 0,
    }

    # txt 리스트 확보 (tqdm 총량 표시 목적)
    all_txts = list(iter_txt_files(train_dir)) + list(iter_txt_files(val_dir))

    seen_classes = set()

    for txt_path in tqdm(all_txts, desc="Scan labels", unit="txt"):
        meta["total_txt"] += 1

        img_path = find_image_for_label(txt_path)
        if img_path is None:
            meta["missing_img"] += 1
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            meta["bad_img"] += 1
            continue

        if img_w <= 0 or img_h <= 0:
            meta["bad_img"] += 1
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = parse_yolo_line(line)
                if not parsed:
                    continue
                cls, _x, _y, bw, bh = parsed

                seen_classes.add(cls)

                # YOLO 정규화 가정: bw,bh가 0~1 비율
                bw01 = clamp01(bw)
                bh01 = clamp01(bh)
                area01 = clamp01(bw01 * bh01)

                ia = bin_index(area01, bin_step)
                iw = bin_index(bw01, bin_step)
                ih = bin_index(bh01, bin_step)

                h_all["area"][ia]   += 1
                h_all["width"][iw]  += 1
                h_all["height"][ih] += 1

                h_cls[cls]["area"][ia]   += 1
                h_cls[cls]["width"][iw]  += 1
                h_cls[cls]["height"][ih] += 1

                meta["total_boxes"] += 1

    # class 목록 결정
    if FORCE_CLASS_IDS is not None:
        class_ids = list(FORCE_CLASS_IDS)
    else:
        class_ids = sorted(list(seen_classes))

    return h_all, h_cls, class_ids, meta, num_bins

# =========================
# CSV Writer (single file)
# =========================
def write_csv(out_csv: str, bin_step: float, h_all, h_cls, class_ids, meta, num_bins: int):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    edges = bin_edges(bin_step)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # 메타 블록
        w.writerow(["key", "value"])
        w.writerow(["train_dir", TRAIN_DIR])
        w.writerow(["val_dir", VAL_DIR])
        w.writerow(["bin_step", bin_step])
        w.writerow(["num_bins", num_bins])
        for k in ["total_txt", "total_boxes", "missing_img", "bad_img"]:
            w.writerow([k, meta.get(k, "")])
        w.writerow([])

        # long-format 본문: metric, class_id, bin_lo, bin_hi, count
        w.writerow(["metric", "class_id", "bin_lo", "bin_hi", "count"])

        def dump_hist(metric_key: str, class_id: str, arr: np.ndarray):
            for i in range(num_bins):
                lo = edges[i]
                hi = edges[i + 1]
                # 마지막 bin은 hi=1.0
                w.writerow([metric_key, class_id, f"{lo:.4f}", f"{hi:.4f}", int(arr[i])])

        # All 먼저
        dump_hist("image_size_vs_bbox_area", "ALL", h_all["area"])
        dump_hist("image_width_vs_bbox_width", "ALL", h_all["width"])
        dump_hist("image_height_vs_bbox_height", "ALL", h_all["height"])

        # Class별
        for cid in class_ids:
            dump_hist("image_size_vs_bbox_area", str(cid), h_cls[cid]["area"])
            dump_hist("image_width_vs_bbox_width", str(cid), h_cls[cid]["width"])
            dump_hist("image_height_vs_bbox_height", str(cid), h_cls[cid]["height"])

# =========================
# PNG Renderer (single file)
# ========================= 
def render_png(out_png: str, bin_step: float, h_all, h_cls, class_ids, meta, num_bins: int):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # x축: bin 중심값(0~1)
    edges = np.array(bin_edges(bin_step), dtype=np.float32)  # (num_bins+1,)
    centers = (edges[:-1] + edges[1:]) * 0.5                 # (num_bins,)

    # -----------------------------
    # "ALL + class 3개"만 표시
    # -----------------------------
    CLASS_IDS_TO_PLOT = None  # 예: [0, 1, 2] / None이면 자동으로 3개 선택

    all_cids = sorted(list(class_ids))
    if CLASS_IDS_TO_PLOT is None:
        class_ids_plot = all_cids[:3]
    else:
        class_ids_plot = [c for c in CLASS_IDS_TO_PLOT if c in h_cls]

    metrics = [
        ("Image Size : BBox Size (area = w*h)", "area"),
        ("Image Width : BBox Width", "width"),
        ("Image Height : BBox Height", "height"),
    ]

    # ---------------------------------------------------------
    # [NEW] 상위 20% "빈(bin)" 제외 마스크 생성 (ALL 기준)
    # ---------------------------------------------------------
    def make_top_bin_mask(y_all: np.ndarray, top_ratio: float = 0.20):
        """
        y_all: (num_bins,) ALL histogram counts
        top_ratio: 제외할 bin 비율 (0.2 = 상위 20% bin)
        return: mask_keep (True=표시, False=제외)
        """
        y = np.asarray(y_all, dtype=np.float64)
        k = int(np.ceil(len(y) * top_ratio))
        if k <= 0:
            return np.ones_like(y, dtype=bool), np.array([], dtype=int)

        # count가 큰 bin 인덱스 k개 (내림차순)
        top_idx = np.argsort(-y)[:k]
        mask_keep = np.ones_like(y, dtype=bool)
        mask_keep[top_idx] = False
        return mask_keep, top_idx

    def apply_mask_as_nan(y: np.ndarray, mask_keep: np.ndarray):
        """
        제외 bin은 NaN으로 만들어 plot이 끊기게 함.
        """
        yy = np.asarray(y, dtype=np.float64).copy()
        yy[~mask_keep] = np.nan
        return yy

    # ---------------------------------------------------------
    # [NEW] 그래프 1장 그리는 내부 함수 (mask 적용 가능)
    # ---------------------------------------------------------
    def plot_one(fig_title: str, out_path: str, mask_keep_by_metric=None):
        fig = plt.figure(figsize=(14, 9), dpi=150)
        gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.35)

        for i, (title, mk) in enumerate(metrics):
            ax = fig.add_subplot(gs[i, 0])

            y_all = h_all[mk].astype(np.float64)
            if mask_keep_by_metric is not None:
                y_all_plot = apply_mask_as_nan(y_all, mask_keep_by_metric[mk])
            else:
                y_all_plot = y_all

            ax.plot(centers, y_all_plot, linewidth=2.2, label="ALL")

            for cid in class_ids_plot:
                y = h_cls[cid][mk].astype(np.float64)
                if mask_keep_by_metric is not None:
                    y = apply_mask_as_nan(y, mask_keep_by_metric[mk])
                ax.plot(centers, y, linewidth=1.3, alpha=0.95, label=f"class {cid}")

            ax.set_title(title, loc="left", fontsize=11)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks(np.arange(0.0, 1.0001, 0.1))
            ax.grid(True, which="major", axis="both", alpha=0.25)

            ax.set_ylabel("count")
            if i == 2:
                ax.set_xlabel("normalized ratio (0~1)")

            ax.legend(loc="upper right", fontsize=9, ncol=2)

        fig.suptitle(fig_title, fontsize=10, y=0.995)
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    # ---------------------------------------------------------
    # 1) 원본 그래프 저장
    # ---------------------------------------------------------
    title_base = (
        f"bbox histograms | bin_step={bin_step}, bins={num_bins} | "
        f"txt={meta.get('total_txt',0)}, boxes={meta.get('total_boxes',0)}, "
        f"missing_img={meta.get('missing_img',0)}, bad_img={meta.get('bad_img',0)} | "
        f"classes={class_ids_plot}"
    )
    plot_one(title_base, out_png, mask_keep_by_metric=None)

    # ---------------------------------------------------------
    # 2) [NEW] 상위 20% bin 제외 그래프 저장
    #    - metric 별로 top bin 계산(ALL 기준)
    # ---------------------------------------------------------
    top_ratio = 0.20  # 필요하면 외부 파라미터로 빼도 됨

    mask_keep_by_metric = {}
    removed_info = {}

    for _t, mk in metrics:
        mask_keep, top_idx = make_top_bin_mask(h_all[mk], top_ratio=top_ratio)
        mask_keep_by_metric[mk] = mask_keep
        removed_info[mk] = top_idx

    # 파일명: xxx.png -> xxx_cutTop20.png
    base, ext = os.path.splitext(out_png)
    out_png_cut = f"{base}_cutTop{int(top_ratio*100)}{ext}"

    # 제거된 bin 범위/인덱스 정보(원하면 title에 더 넣을 수 있음)
    title_cut = (
        f"{title_base} | CUT: top {int(top_ratio*100)}% bins (by ALL count)"
    )

    plot_one(title_cut, out_png_cut, mask_keep_by_metric=mask_keep_by_metric)

    print(f"[OK] saved:\n  - {out_png}\n  - {out_png_cut}")


# =========================
# export_bbox_distribution_images - PNG Renderer (single file)
# =========================
def _parse_yolo_txt(txt_path: str):
    """YOLO txt 한 파일에서 (cls, xc, yc, w, h) 리스트 반환. (float)"""
    out = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    xc = float(parts[1]); yc = float(parts[2])
                    w  = float(parts[3]); h  = float(parts[4])
                except Exception:
                    continue
                # 유효 범위 약간 벗어나도 클램프해서 처리
                out.append((cls, xc, yc, w, h))
    except Exception:
        pass
    return out


def _collect_all_bboxes(label_dirs, class_ids=None):
    """
    label_dirs: [".../train", ".../val"] 처럼 txt들이 있는 디렉터리들
    class_ids: None이면 전체 class, list면 그 class만 필터
    return: list of (cls, xc, yc, w, h)
    """
    bboxes = []
    for d in label_dirs:
        for fn in os.listdir(d):
            if not fn.lower().endswith(".txt"):
                continue
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            for rec in _parse_yolo_txt(p):
                if class_ids is None or rec[0] in class_ids:
                    bboxes.append(rec)
    return bboxes


def _draw_bbox_canvas(
    bboxes,
    out_png: str,
    canvas_size: int = 768,
    line_width: float = 0.25,
    max_boxes: int | None = 200000,
    seed: int = 1,
    title: str = "",
):
    """
    bboxes: list of (cls, xc, yc, w, h) normalized (0~1)
    out_png: 저장 경로
    canvas_size: 768
    line_width: 선 두께(얇게)
    max_boxes: 너무 많으면 랜덤 샘플링
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    if (max_boxes is not None) and (len(bboxes) > max_boxes):
        random.seed(seed)
        bboxes = random.sample(bboxes, max_boxes)

    # matplotlib: 흰 배경 캔버스
    fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # 좌표계를 [0,1]로 두고, (0,0)=좌상단 느낌으로 보기 위해 y축 반전
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # y 반전
    ax.set_aspect("equal", adjustable="box")

    # 축/눈금 제거 (분포만 보기)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # bbox 라인 그리기
    # 너무 과도한 alpha를 주면 겹침이 안 보일 수 있어, 적당히 낮춰 누적 느낌을 줌
    for (_cls, xc, yc, w, h) in tqdm(bboxes):
        # 클램프
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w  = min(max(w,  0.0), 1.0)
        h  = min(max(h,  0.0), 1.0)

        x1 = xc - w * 0.5
        y1 = yc - h * 0.5

        # bbox가 살짝 밖으로 나가도 그리긴 하되, 표시범위만 유지
        rect = plt.Rectangle(
            (x1, y1), w, h,
            fill=False,
            linewidth=line_width,
            edgecolor="black",
            alpha=0.06  # 겹치면 진해져서 분포가 보임
        )
        ax.add_patch(rect)

    if title:
        ax.set_title(title, fontsize=10, loc="left")

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    
def _draw_center_scatter_canvas(
    bboxes,
    out_png: str,
    canvas_size: int = 768,
    point_size: float = 1.0,     # 점 크기
    alpha: float = 0.03,         # 점 투명도 (낮을수록 누적될 때 진해짐)
    max_points: int | None = 400000,
    seed: int = 1,
    title: str = "",
):
    """
    bboxes: list of (cls, xc, yc, w, h) normalized (0~1)
    out_png: 저장 경로
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    if (max_points is not None) and (len(bboxes) > max_points):
        random.seed(seed)
        bboxes = random.sample(bboxes, max_points)

    xs = []
    ys = []
    for (_cls, xc, yc, _w, _h) in tqdm(bboxes):
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        xs.append(xc)
        ys.append(yc)

    fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)  # y 반전
    ax.set_aspect("equal", adjustable="box")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # scatter: 누적될수록 진해지는 heatmap 느낌
    ax.scatter(xs, ys, s=point_size, c="black", alpha=alpha, linewidths=0)

    if title:
        ax.set_title(title, fontsize=10, loc="left")

    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    

def export_bbox_distribution_images(
    train_label_dir: str,
    val_label_dir: str,
    out_dir: str,
    class_ids_to_plot=(0, 1, 2),
    canvas_size: int = 768,
    line_width: float = 0.25,
    max_boxes: int | None = 20000, 
    seed: int = 1
):
    """
    결과:
      윤곽선 4장:
        - bbox_dist_ALL.png
        - bbox_dist_class{cid}.png
      중심점 scatter 4장:
        - center_scatter_ALL.png
        - center_scatter_class{cid}.png
    """
    label_dirs = [train_label_dir, val_label_dir]
    os.makedirs(out_dir, exist_ok=True)

    # ALL
    all_boxes = _collect_all_bboxes(label_dirs, class_ids=None)

    _draw_bbox_canvas(
        all_boxes,
        out_png=os.path.join(out_dir, "bbox_dist_ALL.png"),
        canvas_size=canvas_size,
        line_width=line_width,
        max_boxes=max_boxes,
        seed=seed,
        title=f"ALL (boxes={len(all_boxes)}, sampled={min(len(all_boxes), max_boxes) if max_boxes else len(all_boxes)})"
    ) 

    # class별
    for cid in tqdm(class_ids_to_plot):
        cls_boxes = _collect_all_bboxes(label_dirs, class_ids=[cid])

        _draw_bbox_canvas(
            cls_boxes,
            out_png=os.path.join(out_dir, f"bbox_dist_class{cid}.png"),
            canvas_size=canvas_size,
            line_width=line_width,
            max_boxes=max_boxes,
            seed=seed,
            title=f"class {cid} (boxes={len(cls_boxes)}, sampled={min(len(cls_boxes), max_boxes) if max_boxes else len(cls_boxes)})"
        ) 

    print("[OK] Saved bbox outline + center scatter images to:", out_dir)


def export_bbox_distribution_images_scatter(
    train_label_dir: str,
    val_label_dir: str,
    out_dir: str,
    class_ids_to_plot=(0, 1, 2),
    canvas_size: int = 768,
    line_width: float = 0.25,
    max_boxes: int | None = 200000,
    # --- center scatter 옵션 ---
    point_size: float = 1.0,
    point_alpha: float = 0.03,
    max_points: int | None = 400000,
    seed: int = 1
):
    """
    결과:
      윤곽선 4장:
        - bbox_dist_ALL.png
        - bbox_dist_class{cid}.png
      중심점 scatter 4장:
        - center_scatter_ALL.png
        - center_scatter_class{cid}.png
    """
    label_dirs = [train_label_dir, val_label_dir]
    os.makedirs(out_dir, exist_ok=True)

    # ALL
    all_boxes = _collect_all_bboxes(label_dirs, class_ids=None)

    _draw_center_scatter_canvas(
        all_boxes,
        out_png=os.path.join(out_dir, "center_scatter_ALL.png"),
        canvas_size=canvas_size,
        point_size=point_size,
        alpha=point_alpha,
        max_points=max_points,
        seed=seed,
        title=f"ALL centers (boxes={len(all_boxes)}, sampled={min(len(all_boxes), max_points) if max_points else len(all_boxes)})"
    )

    # class별
    for cid in tqdm(class_ids_to_plot):
        cls_boxes = _collect_all_bboxes(label_dirs, class_ids=[cid])

        _draw_center_scatter_canvas(
            cls_boxes,
            out_png=os.path.join(out_dir, f"center_scatter_class{cid}.png"),
            canvas_size=canvas_size,
            point_size=point_size,
            alpha=point_alpha,
            max_points=max_points,
            seed=seed,
            title=f"class {cid} centers (boxes={len(cls_boxes)}, sampled={min(len(cls_boxes), max_points) if max_points else len(cls_boxes)})"
        )

    print("[OK] Saved bbox outline + center scatter images to:", out_dir)
 
# =========================
# Main
# =========================
def main():
    h_all, h_cls, class_ids, meta, num_bins = accumulate(TRAIN_DIR, VAL_DIR, BIN_STEP)

    write_csv(OUT_CSV, BIN_STEP, h_all, h_cls, class_ids, meta, num_bins)
    render_png(OUT_PNG, BIN_STEP, h_all, h_cls, class_ids, meta, num_bins)

    print("======================================")
    print("[OK] Done.")
    print(f"CSV: {OUT_CSV}")
    print(f"PNG: {OUT_PNG}")
    print(f"Classes: {class_ids}")
    print(f"total_txt={meta['total_txt']}, total_boxes={meta['total_boxes']}, missing_img={meta['missing_img']}, bad_img={meta['bad_img']}")
    print("======================================")
    
    export_bbox_distribution_images(
        train_label_dir=TRAIN_DIR,
        val_label_dir=VAL_DIR,
        out_dir=OUT_DIR,
        class_ids_to_plot=(0, 1, 2),
        canvas_size=768,
        # 윤곽선
        line_width=0.25,
        max_boxes=20000,
        seed=1
    )
    
    export_bbox_distribution_images_scatter(
        train_label_dir=TRAIN_DIR,
        val_label_dir=VAL_DIR,
        out_dir=OUT_DIR,
        class_ids_to_plot=(0, 1, 2),
        canvas_size=768,
        # 윤곽선
        line_width=0.25,
        max_boxes=20000,

        # 중심점 scatter
        point_size=1.0,
        point_alpha=0.03,
        max_points=400000,
        seed=1
    )


if __name__ == "__main__":
    main()
