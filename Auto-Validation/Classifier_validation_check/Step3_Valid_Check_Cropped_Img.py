import os
import glob
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = r"X:\data\ARGOS\face_detector\Annotation\rtsp_seoul_2_done"
CROPPED_DIR = r"X:\data\ARGOS\face_detector\Annotation\rtsp_seoul_2_done\crop_pred"
VALID_TXT_PATH = os.path.join(CROPPED_DIR, "crop_valid.txt")

MIN_W, MIN_H = 1280, 720
RIGHT_W = 500
TOOLBAR_H = 52

DET_LABEL = {0: "Person", 1: "Vehicle", 2: "Bike", 3: "Unknown"}
DET_COLOR = {
    0: "#2ECC71",  # Person  - Green
    1: "#F1C40F",  # Vehicle - Yellow
    2: "#3498DB",  # Bike    - Blue
    3: "#E74C3C",  # Unknown - Red
}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# -----------------------------
# Utilities
# -----------------------------
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def list_subdirs(base):
    out = []
    if not os.path.isdir(base): return out
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if os.path.isdir(p): out.append(d)
    return out

def list_images_in_dir(dir_path):
    files = []
    for f in sorted(os.listdir(dir_path)):
        if f.lower().endswith(IMG_EXTS): files.append(f)
    return files

def build_original_index(base_dir):
    idx = {}
    for ext in IMG_EXTS:
        for p in glob.glob(os.path.join(base_dir, f"*{ext}")):
            bn = os.path.splitext(os.path.basename(p))[0]
            idx[bn] = p
    return idx

def parse_crop_filename(fname):
    base, _ext = os.path.splitext(fname)
    parts = base.split("_")
    if len(parts) < 6: return None
    try:
        cls = int(float(parts[-5]))
        x, y, w, h = map(float, parts[-4:])
    except Exception:
        return None
    orig_base = "_".join(parts[:-5])
    return orig_base, cls, x, y, w, h

def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    cx, cy, bw, bh = x * img_w, y * img_h, w * img_w, h * img_h
    x1, y1 = int(round(cx - bw/2)), int(round(cy - bh/2))
    x2, y2 = int(round(cx + bw/2)), int(round(cy + bh/2))
    return max(0, min(img_w-1, x1)), max(0, min(img_h-1, y1)), \
           max(0, min(img_w-1, x2)), max(0, min(img_h-1, y2))

def fit_image_keep_ratio(img: Image.Image, max_w, max_h):
    iw, ih = img.size
    if iw <= 0 or ih <= 0: return img
    scale = min(max_w / iw, max_h / ih)
    return img.resize((max(1, int(iw * scale)), max(1, int(ih * scale))), Image.BILINEAR)

def load_valid_map(path):
    m = {}
    if not os.path.exists(path): return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 2:
                try:
                    v = int(p[1])
                    if v not in (0,1,2): v = 0
                    m[p[0]] = v
                except:
                    continue
    return m

def save_valid_map(path, m):
    lines = [f"{k} {m[k]}\n" for k in sorted(m.keys())]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    os.replace(tmp, path)

# -----------------------------
# GUI App
# -----------------------------
class ReviewApp:
    def __init__(self, root):
        self.root = root
        root.title("Crop Review Tool")
        root.geometry("1920x1080")
        root.minsize(MIN_W, MIN_H)

        # Zoom state (LEFT only)
        self.left_zoom = 1.0
        self.left_zoom_step = 0.10
        self.left_zoom_min = 0.30
        self.left_zoom_max = 3.00

        self.subdirs = list_subdirs(CROPPED_DIR)
        self.items = []
        self.subdir_ranges = {}
        self._build_items()

        if not self.items:
            messagebox.showerror("Error", "No images found.")
            root.destroy()
            return

        self.orig_index = build_original_index(BASE_DIR)
        self.valid_map = load_valid_map(VALID_TXT_PATH)
        self.cur = 0
        self.cur_status = 0
        self.orig_cache_img = None
        self.orig_cache_key = None

        # ✅ MOD: class group ranges for (0,1,2) after sorting
        self.cls_ranges = {}
        self._build_cls_ranges()

        self._build_ui()
        self._bind_keys()
        self.load_current()

    def _build_items(self):
        idx = 0
        for sd in self.subdirs:
            p = os.path.join(CROPPED_DIR, sd)
            imgs = list_images_in_dir(p)
            self.subdir_ranges[sd] = (idx, len(imgs))  # 원본 dir기준(정렬 전 의미)
            for fname in imgs:
                fullpath = os.path.join(p, fname)
                relkey = os.path.join(sd, fname)

                # ✅ MOD: det_cls 미리 파싱해서 items에 저장
                parsed = parse_crop_filename(fname)
                det_cls = parsed[1] if parsed else -1

                self.items.append({
                    "subdir": sd,
                    "fname": fname,
                    "fullpath": fullpath,
                    "relkey": relkey,
                    "det_cls": det_cls,  # ✅ MOD
                })
                idx += 1

        # ✅ MOD: Person(0) -> Vehicle(1) -> Bike(2) -> Others(-1/기타)
        def cls_order(c):
            if c == 0: return 0
            if c == 1: return 1
            if c == 2: return 2
            if c == 3: return 3
            return 9

        self.items.sort(key=lambda it: (cls_order(it["det_cls"]), it["subdir"], it["fname"]))

    # ✅ MOD: 정렬된 items 기준으로 class 그룹 범위 계산(0/1/2)
    def _build_cls_ranges(self):
        # cls_ranges[0] = (start, count) 형태
        self.cls_ranges = {0: (-1, 0), 1: (-1, 0), 2: (-1, 0), 9: (-1, 0)}
        starts = {}
        counts = {0: 0, 1: 0, 2: 0, 9: 0}

        def cls_order(c):
            if c == 0: return 0
            if c == 1: return 1
            if c == 2: return 2
            if c == 3: return 3
            return 9

        for i, it in enumerate(self.items):
            k = cls_order(it["det_cls"])
            if k not in starts:
                starts[k] = i
            counts[k] += 1

        for k in (0,1,2,9):
            self.cls_ranges[k] = (starts.get(k, -1), counts.get(k, 0))

    def _build_ui(self):
        self.toolbar = tk.Frame(self.root, height=TOOLBAR_H)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.toolbar, text="Prev (A)", command=self.prev_item).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Next (D)", command=self.next_item).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Toggle Status (S)", command=self.toggle_status).pack(side=tk.LEFT, padx=5)

        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(self.toolbar, text="Size + (Full)", command=lambda: self.set_window_size(1920, 1080)).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Size - (Small)", command=lambda: self.set_window_size(1280, 720)).pack(side=tk.LEFT, padx=5)

        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Button(self.toolbar, text="Zoom -", command=self.zoom_out).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.toolbar, text="Zoom +", command=self.zoom_in).pack(side=tk.LEFT, padx=5)
        self.lbl_zoom = ttk.Label(self.toolbar, text="Zoom: 100%")
        self.lbl_zoom.pack(side=tk.LEFT, padx=10)

        # Jump
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        ttk.Label(self.toolbar, text="Jump:").pack(side=tk.LEFT, padx=(0, 4))
        self.jump_var = tk.StringVar()
        self.entry_jump = ttk.Entry(self.toolbar, width=7, textvariable=self.jump_var)
        self.entry_jump.pack(side=tk.LEFT, padx=2)
        self.btn_jump = ttk.Button(self.toolbar, text="Go", command=self.jump_to_index)
        self.btn_jump.pack(side=tk.LEFT, padx=5)
        self.entry_jump.bind("<Return>", lambda e: self.jump_to_index())

        self.lbl_info = ttk.Label(self.toolbar, text="")
        self.lbl_info.pack(side=tk.LEFT, padx=15)

        ttk.Button(self.toolbar, text="Exit", command=self.on_close).pack(side=tk.RIGHT, padx=5)

        self.body = tk.Frame(self.root, bg="black")
        self.body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.right = tk.Frame(self.body, width=RIGHT_W, bg="gray20")
        self.right.pack(side=tk.RIGHT, fill=tk.Y)
        self.right.pack_propagate(False)

        self.right_top = tk.Frame(self.right, height=200, bg="gray15")
        self.right_top.pack(side=tk.TOP, fill=tk.X)
        self.right_top.pack_propagate(False)

        self.lbl_class = tk.Label(self.right_top, text="", font=("Arial", 24, "bold"),
                                  fg="white", bg="gray15")
        self.lbl_class.pack(pady=10)

        self.lbl_status = tk.Label(self.right_top, text="", font=("Arial", 28, "bold"),
                                   fg="white", bg="darkgreen", width=12)
        self.lbl_status.pack(pady=10)

        self.canvas_right = tk.Canvas(self.right, bg="black", highlightthickness=0)
        self.canvas_right.pack(fill=tk.BOTH, expand=True)

        self.left = tk.Frame(self.body, bg="black")
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas_left = tk.Canvas(self.left, bg="black", highlightthickness=0)
        self.canvas_left.pack(fill=tk.BOTH, expand=True)

        self._last_lwh = (0, 0)
        self._last_rwh = (0, 0)
        self.root.bind("<Configure>", self.on_window_resize)

    def on_window_resize(self, event):
        if event.widget != self.root:
            return
        lw, lh = self.canvas_left.winfo_width(), self.canvas_left.winfo_height()
        rw, rh = self.canvas_right.winfo_width(), self.canvas_right.winfo_height()
        if (lw, lh) == self._last_lwh and (rw, rh) == self._last_rwh:
            return
        self._last_lwh = (lw, lh)
        self._last_rwh = (rw, rh)
        self.display_images()

    def set_window_size(self, w, h):
        self.root.geometry(f"{w}x{h}")

    def _bind_keys(self):
        for k in ['a', 'A']: self.root.bind(f"<KeyPress-{k}>", lambda e: self.prev_item())
        for k in ['d', 'D']: self.root.bind(f"<KeyPress-{k}>", lambda e: self.next_item())
        for k in ['s', 'S']: self.root.bind(f"<KeyPress-{k}>", lambda e: self.toggle_status())
        self.root.bind("<KeyPress-plus>", lambda e: self.zoom_in())
        self.root.bind("<KeyPress-equal>", lambda e: self.zoom_in())
        self.root.bind("<KeyPress-minus>", lambda e: self.zoom_out())
        self.root.bind("<Escape>", lambda e: self.on_close())

    def zoom_in(self):
        z = min(self.left_zoom_max, self.left_zoom + self.left_zoom_step)
        if z != self.left_zoom:
            self.left_zoom = z
            self.update_zoom_ui()
            self.display_images()

    def zoom_out(self):
        z = max(self.left_zoom_min, self.left_zoom - self.left_zoom_step)
        if z != self.left_zoom:
            self.left_zoom = z
            self.update_zoom_ui()
            self.display_images()

    def update_zoom_ui(self):
        self.lbl_zoom.config(text=f"Zoom: {int(self.left_zoom * 100)}%")

    def toggle_status(self):
        self.cur_status = (self.cur_status + 1) % 3
        self.save_current_status()
        self.update_status_ui()

    def update_status_ui(self):
        states = {0: ("SUCCESS", "darkgreen"), 1: ("DELETE", "darkred"), 2: ("MODIFY", "#E67E22")}
        txt, color = states.get(self.cur_status, ("UNKNOWN", "gray"))
        self.lbl_status.config(text=txt, bg=color)

    def jump_to_index(self):
        s = self.jump_var.get().strip()
        if not s:
            return
        try:
            target = int(s)
        except ValueError:
            messagebox.showwarning("Invalid input", "숫자를 입력하세요. (예: 119)")
            return

        if target < 1 or target > len(self.items):
            messagebox.showwarning("Out of range", f"1 ~ {len(self.items)} 사이 값을 입력하세요.")
            return

        self.save_current_status()
        self.cur = target - 1
        self.load_current()
        self.entry_jump.focus_set()
        self.entry_jump.selection_range(0, tk.END)

    def load_current(self):
        item = self.items[self.cur]
        self.cur_status = self.valid_map.get(item["relkey"], 0)
        self.update_status_ui()
        self.update_zoom_ui()

        parsed = parse_crop_filename(item["fname"])
        self.current_left_img = None

        det_cls = -1
        if parsed:
            orig_base, det_cls, x, y, w, h = parsed
            cls_name = DET_LABEL.get(det_cls, f"Class {det_cls}")
            cls_color = DET_COLOR.get(det_cls, "white")

            self.lbl_class.config(text=cls_name, fg=cls_color)

            orig_path = self.orig_index.get(orig_base)
            if orig_path and os.path.exists(orig_path):
                if self.orig_cache_key != orig_path:
                    self.orig_cache_key = orig_path
                    self.orig_cache_img = Image.open(orig_path).convert("RGB")

                self.current_left_img = self.orig_cache_img.copy()
                x1, y1, x2, y2 = yolo_to_xyxy(x, y, w, h, *self.current_left_img.size)
                draw = ImageDraw.Draw(self.current_left_img)
                bbox_color = DET_COLOR.get(det_cls, "red")
                draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)

        self.current_right_img = Image.open(item["fullpath"]).convert("RGB")

        bg_color = DET_COLOR.get(det_cls, "black")
        self.canvas_right.config(bg=bg_color)

        self.display_images()

        # ✅ MOD: class group 기반 진행표시 + subdir 표시
        def cls_order(c):
            if c == 0: return 0
            if c == 1: return 1
            if c == 2: return 2
            if c == 3: return 3
            return 9

        cur_group = cls_order(item.get("det_cls", -1))
        g_start, g_count = self.cls_ranges.get(cur_group, (-1, 0))
        if g_start >= 0 and g_count > 0:
            idx_in_group = (self.cur - g_start) + 1
            group_total = g_count
            group_name = DET_LABEL.get(cur_group, "Other")
        else:
            idx_in_group = 0
            group_total = 0
            group_name = "Other"

        sd = item["subdir"]
        self.lbl_info.config(
            text=f"CLASS: {group_name} ({idx_in_group}/{group_total}) | DIR: {sd} | TOTAL: {self.cur+1}/{len(self.items)}"
        )

    def display_images(self):
        if self.current_left_img:
            lw, lh = self.canvas_left.winfo_width(), self.canvas_left.winfo_height()
            if lw > 1 and lh > 1:
                fitted_l = fit_image_keep_ratio(self.current_left_img, lw, lh)
                if self.left_zoom != 1.0:
                    zw = max(1, int(fitted_l.size[0] * self.left_zoom))
                    zh = max(1, int(fitted_l.size[1] * self.left_zoom))
                    fitted_l = fitted_l.resize((zw, zh), Image.BILINEAR)

                self.tk_l = ImageTk.PhotoImage(fitted_l)
                self.canvas_left.delete("all")
                self.canvas_left.create_image(lw//2, lh//2, anchor="center", image=self.tk_l)

        if self.current_right_img:
            rw, rh = self.canvas_right.winfo_width(), self.canvas_right.winfo_height()
            if rw > 1 and rh > 1:
                fitted_r = fit_image_keep_ratio(self.current_right_img, rw, rh)
                self.tk_r = ImageTk.PhotoImage(fitted_r)
                self.canvas_right.delete("all")
                self.canvas_right.create_image(rw//2, rh//2, anchor="center", image=self.tk_r)

    def save_current_status(self):
        self.valid_map[self.items[self.cur]["relkey"]] = self.cur_status
        save_valid_map(VALID_TXT_PATH, self.valid_map)

    def prev_item(self):
        if self.cur > 0:
            self.cur -= 1
            self.load_current()

    def next_item(self):
        if self.cur < len(self.items) - 1:
            self.cur += 1
            self.load_current()

    def on_close(self):
        self.save_current_status()
        self.root.destroy()

if __name__ == "__main__":
    safe_mkdir(CROPPED_DIR)
    root = tk.Tk()
    app = ReviewApp(root)
    root.mainloop()
