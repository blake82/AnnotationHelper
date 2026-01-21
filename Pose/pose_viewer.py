# pose_viewer.py
# YOLO Pose Viewer (PyQt5)
# - Left: Full image (all bboxes green thin, selected bbox red thick)
# - Right: Top label, 2/3 crop view, bottom 1/3 (Index list + Image list side-by-side)
# - Index list: ONLY left color chip (icon) per v (2=green,1=blue,0=red), text does NOT overlap
# - Click index: keep highlight selection, show comment/name on header, emphasize keypoint ring in crop
# - Shortcuts: A/D prev/next image, Left/Right prev/next object
# - Toolbar buttons: Prev/Next Image, Prev/Next Object, window modes (1920x1080 / 1280x720)
# - Fix: prevent weird resizing on Down key by not using arrow keys for window control; fixed window sizes.
#
# Requirements: pip install pyqt5 opencv-python

import os
import sys
import cv2
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QListWidget,
    QHBoxLayout, QVBoxLayout, QListWidgetItem, QToolBar, QAction
)
from PyQt5.QtGui import QPixmap, QImage, QKeySequence, QColor, QIcon, QPainter, QPen
from PyQt5.QtCore import Qt


# -----------------------------
# Helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def cvimg_to_qpixmap(rgb_img):
    """rgb_img: HxWx3 uint8 RGB"""
    h, w, ch = rgb_img.shape
    bytes_per_line = w * ch
    q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def kv_to_rgb_color(kv):
    """v: 2=green, 1=blue, 0=red (RGB tuple for OpenCV)"""
    try:
        v = int(round(float(kv)))
    except Exception:
        v = 0
    if v >= 2:
        return (0, 255, 0)
    elif v == 1:
        return (0, 0, 255)
    else:
        return (255, 0, 0)


def make_v_icon(v, size=10):
    """Make a small colored square icon (chip) for QListWidgetItem icon."""
    if int(v) == 2:
        color = QColor(0, 255, 0)
    elif int(v) == 1:
        color = QColor(0, 0, 255)
    else:
        color = QColor(255, 0, 0)

    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)

    p = QPainter(pm)
    p.fillRect(0, 0, size, size, color)
    p.setPen(QPen(QColor(0, 0, 0, 120)))
    p.drawRect(0, 0, size - 1, size - 1)
    p.end()

    return QIcon(pm)


# -----------------------------
# Main Viewer
# -----------------------------
class PoseViewer(QMainWindow):
    MODE_1080P = (1920, 1080)
    MODE_720P = (1280, 720)

    def __init__(self, image_path, label_info_path, start_mode="1080p"):
        super().__init__()

        self.image_dir = image_path
        self.label_info_path = label_info_path

        # metadata
        self.index_info = []      # [(name, comment)]
        self.skeleton_info = []   # [(name1, name2)]

        # runtime
        self.image_list = []
        self.current_img_idx = 0
        self.current_obj_idx = -1
        self.selected_kpt_idx = -1

        self.img_rgb = None
        self.objects = []  # list of {"cls":int, "bbox":[cx,cy,w,h], "kpts":[x,y,v,...]}

        self.load_metadata()
        self.load_image_list()
        self.init_ui()

        if str(start_mode).lower() == "720p":
            self.set_window_mode_720p()
        else:
            self.set_window_mode_1080p()

        if self.image_list:
            self.img_list_widget.setCurrentRow(0)

    # ---------------------------
    # Fixed window modes
    # ---------------------------
    def set_fixed_window_size(self, w, h):
        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)
        self.resize(w, h)

    def set_window_mode_1080p(self):
        self.set_fixed_window_size(*self.MODE_1080P)
        self.statusBar().showMessage("Window Mode: 1920x1080", 1500)

    def set_window_mode_720p(self):
        self.set_fixed_window_size(*self.MODE_720P)
        self.statusBar().showMessage("Window Mode: 1280x720", 1500)

    # ---------------------------
    # Data loading
    # ---------------------------
    def load_metadata(self):
        idx_path = os.path.join(self.label_info_path, "index.txt")
        self.index_info = []
        if os.path.exists(idx_path):
            with open(idx_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "#" in line:
                        name, comment = line.split("#", 1)
                        self.index_info.append((name.strip(), comment.strip()))
                    else:
                        self.index_info.append((line.strip(), ""))

        # ===== skeleton.txt robust parse =====
        sk_path = os.path.join(self.label_info_path, "skeleton.txt")
        self.skeleton_info = []
        if os.path.exists(sk_path):
            with open(sk_path, "r", encoding="utf-8") as f:
                text = f.read()

            # 파일 전체에서 [A,B] 패턴을 전부 추출
            pairs = re.findall(r"\[\s*([^\[\],\s]+)\s*,\s*([^\[\],\s]+)\s*\]", text)

            # 중복 제거(입력순서 유지)
            seen = set()
            for a, b in pairs:
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                self.skeleton_info.append((a, b))


    def load_image_list(self):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.image_list = []
        if os.path.isdir(self.image_dir):
            self.image_list = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(exts)])

    def load_data(self):
        if not self.image_list:
            return

        img_name = self.image_list[self.current_img_idx]
        img_path = os.path.join(self.image_dir, img_name)
        txt_path = os.path.join(self.image_dir, os.path.splitext(img_name)[0] + ".txt")

        bgr = cv2.imread(img_path)
        if bgr is None:
            self.img_rgb = None
            self.objects = []
            self.current_obj_idx = -1
            self.selected_kpt_idx = -1
            self.update_display(rebuild_list=True)
            self.update_status()
            return

        self.img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.objects = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls = int(float(parts[0]))
                        bbox = list(map(float, parts[1:5]))
                        kpts = list(map(float, parts[5:]))
                        self.objects.append({"cls": cls, "bbox": bbox, "kpts": kpts})
                    except Exception:
                        continue

        self.current_obj_idx = 0 if self.objects else -1
        self.selected_kpt_idx = -1
        self.label_header.setText("Object Selected")

        # load new image => rebuild list
        self.update_display(rebuild_list=True)
        self.update_status()

    # ---------------------------
    # UI
    # ---------------------------
    def init_ui(self):
        self.setWindowTitle("YOLO Pose Viewer")

        self.init_toolbar()

        central = QWidget()
        self.setCentralWidget(central)

        # Top-level: left (full) + right (panel)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # Left: full image only
        self.main_view = QLabel("No Image")
        self.main_view.setAlignment(Qt.AlignCenter)
        self.main_view.setStyleSheet("background-color:#1e1e1e; border:2px solid #333;")
        main_layout.addWidget(self.main_view, 7)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        main_layout.addWidget(right_widget, 3)

        # Header label (selected keypoint comment/name)
        self.label_header = QLabel("Object Selected")
        self.label_header.setAlignment(Qt.AlignCenter)
        self.label_header.setStyleSheet(
            "font-size:18pt; font-weight:bold; color:#00FF00;"
            "border-bottom:2px solid gray;"
        )
        right_layout.addWidget(self.label_header, 0)

        # Crop view (2/3)
        self.crop_view = QLabel("No Crop")
        self.crop_view.setAlignment(Qt.AlignCenter)
        self.crop_view.setStyleSheet("border:1px solid white; background-color:black;")
        self.crop_view.setMinimumSize(450, 450)
        right_layout.addWidget(self.crop_view, 2)

        # Bottom area: index list + image list side by side (1/3)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        right_layout.addWidget(bottom_widget, 1)

        # Index list (left)
        self.kpt_list_widget = QListWidget()
        self.kpt_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.kpt_list_widget.itemClicked.connect(self.on_kpt_clicked)
        self.kpt_list_widget.currentRowChanged.connect(self.on_kpt_row_changed)
        bottom_layout.addWidget(self.kpt_list_widget, 1)

        # Image list (right)
        self.img_list_widget = QListWidget()
        self.img_list_widget.addItems(self.image_list)
        self.img_list_widget.currentRowChanged.connect(self.on_list_row_changed)
        bottom_layout.addWidget(self.img_list_widget, 1)

        # Status bar
        self.status_label = QLabel("")
        self.statusBar().addPermanentWidget(self.status_label)

    def init_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)

        act_prev_img = QAction("Prev Image (A)", self)
        act_prev_img.setShortcut(QKeySequence("A"))
        act_prev_img.triggered.connect(self.go_prev_image)

        act_next_img = QAction("Next Image (D)", self)
        act_next_img.setShortcut(QKeySequence("D"))
        act_next_img.triggered.connect(self.go_next_image)

        act_prev_obj = QAction("Prev Object (<-)", self)
        act_prev_obj.setShortcut(QKeySequence(Qt.Key_Left))
        act_prev_obj.triggered.connect(self.go_prev_object)

        act_next_obj = QAction("Next Object (->)", self)
        act_next_obj.setShortcut(QKeySequence(Qt.Key_Right))
        act_next_obj.triggered.connect(self.go_next_object)

        act_1080p = QAction("1920x1080", self)
        act_1080p.triggered.connect(self.set_window_mode_1080p)

        act_720p = QAction("1280x720", self)
        act_720p.triggered.connect(self.set_window_mode_720p)

        tb.addAction(act_prev_img)
        tb.addAction(act_next_img)
        tb.addSeparator()
        tb.addAction(act_prev_obj)
        tb.addAction(act_next_obj)
        tb.addSeparator()
        tb.addAction(act_1080p)
        tb.addAction(act_720p)

        # ensure global shortcuts (avoid focusing list capturing)
        for act in (act_prev_img, act_next_img, act_prev_obj, act_next_obj):
            act.setShortcutContext(Qt.ApplicationShortcut)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # When window mode is fixed, resize won't happen often,
        # but this keeps scaled pixmaps correct.
        self.update_display(rebuild_list=False)

    # ---------------------------
    # Navigation
    # ---------------------------
    def go_prev_image(self):
        if not self.image_list:
            return
        self.current_img_idx = (self.current_img_idx - 1) % len(self.image_list)
        self.img_list_widget.setCurrentRow(self.current_img_idx)

    def go_next_image(self):
        if not self.image_list:
            return
        self.current_img_idx = (self.current_img_idx + 1) % len(self.image_list)
        self.img_list_widget.setCurrentRow(self.current_img_idx)

    def go_prev_object(self):
        if not self.objects:
            return
        self.current_obj_idx = (self.current_obj_idx - 1) % len(self.objects)
        self.selected_kpt_idx = -1
        self.label_header.setText("Object Selected")
        self.update_display(rebuild_list=True)

    def go_next_object(self):
        if not self.objects:
            return
        self.current_obj_idx = (self.current_obj_idx + 1) % len(self.objects)
        self.selected_kpt_idx = -1
        self.label_header.setText("Object Selected")
        self.update_display(rebuild_list=True)

    # ---------------------------
    # Rendering
    # ---------------------------
    def update_display(self, rebuild_list=True):
        if self.img_rgb is None:
            self.main_view.setText("No Image")
            self.crop_view.setText("No Crop")
            if rebuild_list:
                self.kpt_list_widget.clear()
            return

        h, w, _ = self.img_rgb.shape
        display = self.img_rgb.copy()

        # Draw bboxes
        for i, obj in enumerate(self.objects):
            cx, cy, bw, bh = obj["bbox"]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            x1 = clamp(x1, 0, w - 1)
            y1 = clamp(y1, 0, h - 1)
            x2 = clamp(x2, 0, w - 1)
            y2 = clamp(y2, 0, h - 1)

            if i == self.current_obj_idx:
                color = (255, 0, 0)  # red
                thick = 5
            else:
                color = (0, 255, 0)  # green
                thick = 1
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thick)

        pm = cvimg_to_qpixmap(display)
        self.main_view.setPixmap(pm.scaled(self.main_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.update_crop_and_list(rebuild_list=rebuild_list)

    def update_crop_and_list(self, rebuild_list=True):
        if self.current_obj_idx < 0 or self.current_obj_idx >= len(self.objects):
            self.crop_view.setText("No Objects Found")
            if rebuild_list:
                self.kpt_list_widget.clear()
            return

        obj = self.objects[self.current_obj_idx]
        kpts = obj["kpts"]

        h, w, _ = self.img_rgb.shape
        cx, cy, bw, bh = obj["bbox"]

        x1_px = int((cx - bw / 2) * w)
        y1_px = int((cy - bh / 2) * h)
        x2_px = int((cx + bw / 2) * w)
        y2_px = int((cy + bh / 2) * h)

        x1 = clamp(x1_px, 0, w - 1)
        y1 = clamp(y1_px, 0, h - 1)
        x2 = clamp(x2_px, 0, w - 1)
        y2 = clamp(y2_px, 0, h - 1)

        if x2 <= x1 or y2 <= y1:
            self.crop_view.setText("Invalid BBox")
            if rebuild_list:
                self.kpt_list_widget.clear()
            return

        crop = self.img_rgb[y1:y2, x1:x2].copy()
        if crop.size == 0:
            self.crop_view.setText("Empty Crop")
            if rebuild_list:
                self.kpt_list_widget.clear()
            return

        ch, cw, _ = crop.shape

        # name->idx
        name_to_idx = {name: i for i, (name, _) in enumerate(self.index_info)}

        # skeleton lines (gray)
        for n1, n2 in self.skeleton_info:
            if n1 not in name_to_idx or n2 not in name_to_idx:
                continue
            i1, i2 = name_to_idx[n1], name_to_idx[n2]
            if i1 * 3 + 2 >= len(kpts) or i2 * 3 + 2 >= len(kpts):
                continue

            v1 = float(kpts[i1 * 3 + 2])
            v2 = float(kpts[i2 * 3 + 2])
            if v1 <= 0 or v2 <= 0:
                continue

            x1n, y1n = kpts[i1 * 3], kpts[i1 * 3 + 1]
            x2n, y2n = kpts[i2 * 3], kpts[i2 * 3 + 1]
            p1 = (int(x1n * w) - x1, int(y1n * h) - y1)
            p2 = (int(x2n * w) - x1, int(y2n * h) - y1)

            if (0 <= p1[0] < cw and 0 <= p1[1] < ch and
                    0 <= p2[0] < cw and 0 <= p2[1] < ch):
                cv2.line(crop, p1, p2, (200, 200, 200), 2)

        # Index list + keypoints
        if rebuild_list:
            self.kpt_list_widget.blockSignals(True)
            self.kpt_list_widget.clear()

        for i, (name, comment) in enumerate(self.index_info):
            # v for list icon
            v = 0
            if i * 3 + 2 < len(kpts):
                try:
                    v = int(float(kpts[i * 3 + 2]))
                except Exception:
                    v = 0

            if rebuild_list:
                text = f"{i}: {name}"
                item = QListWidgetItem(text)
                item.setIcon(make_v_icon(v, size=10))  # chip only (no background)
                # store index so we can locate if needed
                item.setData(Qt.UserRole, i)
                self.kpt_list_widget.addItem(item)

            # draw keypoint on crop
            if i * 3 + 2 >= len(kpts):
                continue

            kx, ky, kvv = kpts[i * 3:i * 3 + 3]
            pt = (int(kx * w) - x1, int(ky * h) - y1)
            color = kv_to_rgb_color(kvv)

            if 0 <= pt[0] < cw and 0 <= pt[1] < ch:
                cv2.circle(crop, pt, 5, color, -1)
                if i == self.selected_kpt_idx:
                    cv2.circle(crop, pt, 16, color, 2)

        if rebuild_list:
            # restore selection if any
            if 0 <= self.selected_kpt_idx < self.kpt_list_widget.count():
                self.kpt_list_widget.setCurrentRow(self.selected_kpt_idx)
            self.kpt_list_widget.blockSignals(False)

        pm = cvimg_to_qpixmap(crop)
        self.crop_view.setPixmap(pm.scaled(self.crop_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ---------------------------
    # Events
    # ---------------------------
    def on_kpt_clicked(self, item):
        row = self.kpt_list_widget.row(item)
        self.kpt_list_widget.setCurrentRow(row)   # selection을 확정
        self.on_kpt_row_changed(row)              # 동일 효과

    # def on_kpt_clicked(self, item):
    #     idx = self.kpt_list_widget.row(item)
    #     self.selected_kpt_idx = idx

    #     if 0 <= idx < len(self.index_info):
    #         name, comment = self.index_info[idx]
    #         self.label_header.setText(comment if comment else name)
    #     else:
    #         self.label_header.setText("Object Selected")

    #     # IMPORTANT: keep list selection; do NOT rebuild list
    #     self.update_crop_and_list(rebuild_list=False)

    def on_list_row_changed(self, idx):
        if idx < 0 or idx >= len(self.image_list):
            return
        self.current_img_idx = idx
        self.load_data()

    def update_status(self):
        if not self.image_list:
            self.status_label.setText("[ 0 / 0 ]")
            return
        self.status_label.setText(f"[ {self.current_img_idx + 1} / {len(self.image_list)} ]")

    def on_kpt_row_changed(self, row):
        # ↑/↓ 이동 시에도 클릭과 같은 효과
        if row < 0:
            return

        # 내부에서 리스트 rebuild 하는 경우(currentRowChanged가 재진입) 방지
        if getattr(self, "_in_kpt_row_changed", False):
            return

        self._in_kpt_row_changed = True
        try:
            self.selected_kpt_idx = row

            if 0 <= row < len(self.index_info):
                name, comment = self.index_info[row]
                self.label_header.setText(comment if comment else name)
            else:
                self.label_header.setText("Object Selected")

            # 리스트 selection은 유지해야 하므로 rebuild_list=False
            self.update_crop_and_list(rebuild_list=False)
        finally:
            self._in_kpt_row_changed = False


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    # Update these paths for your environment
    # Windows example:
        # mpii
    # IMAGE_PATH = r"X:\data\HERMES\pose_detector\mpii\split_2000_0"
    # LABEL_INFO_PATH = r"X:\data\HERMES\pose_detector\mpii\label_info"
        # coco
    # IMAGE_PATH = r"X:\data\HERMES\pose_detector\coco\split_1000_0"
    # LABEL_INFO_PATH = r"X:\data\HERMES\pose_detector\coco\label_info"

    # WFLW
    IMAGE_PATH = r"Z:\tada\007.Face.Recognition\WFLW\archive\WFLW_images\27--Spa"
    LABEL_INFO_PATH = r"Z:\tada\007.Face.Recognition\WFLW\archive\label_info"

    app = QApplication(sys.argv)
    viewer = PoseViewer(IMAGE_PATH, LABEL_INFO_PATH, start_mode="1080p")  # or "720p"
    viewer.show()
    sys.exit(app.exec_())
