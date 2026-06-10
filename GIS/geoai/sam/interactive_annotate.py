"""
GeoAI SAM 交互式标注工具 — 图形界面版

在影像窗口上直接操作:
  - 鼠标点击放置前景/背景点 → 实时点提示分割
  - 鼠标点击两个角点绘制矩形框 → 框提示分割
  - 输入文本提示词 → Grounded-SAM 自动检测
  - 一键自动全图分割
  - 后处理 & 矢量化导出

用法:
    python interactive_annotate.py --image 西安19级.tif
    python interactive_annotate.py --image 西安19级.tif --mode point
    python interactive_annotate.py --image 西安19级.tif --mode box
    python interactive_annotate.py --image 西安19级.tif --mode text --prompt building
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ──────────────────────────────────────────────
# 主类
# ──────────────────────────────────────────────
class InteractiveAnnotator:
    """
    基于 matplotlib 的可视化交互式标注工具。

    支持四种标注模式（通过工具栏切换）:
        点标注 — 左键放置前景点，右键放置背景点
        框标注 — 左键点击两个角点绘制矩形
        文本标注 — 输入文本提示词自动检测
        自动分割 — 全图网格采样自动发现目标
    """

    MAX_DISPLAY = 2048  # 显示图像最大边长（像素）

    def __init__(
        self,
        image_path: str,
        model_type: str = "vit_l",
        sam_version: str = "sam1",
        output_dir: str = "output/interactive",
        initial_mode: str = "point",
    ):
        self.image_path = os.path.abspath(image_path)
        self.model_type = model_type
        self.sam_version = sam_version
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # SAM 模型（延迟初始化）
        self._sam = None
        self._gsam = None

        # 状态
        self.current_mask: Optional[np.ndarray] = None
        self.masks_history: list = []
        self.annotation_count = 0

        # 交互状态
        self._mode = initial_mode       # point | box | pan | text | auto
        self._fg_points: List[Tuple[int, int]] = []   # 前景点（原图坐标）
        self._bg_points: List[Tuple[int, int]] = []   # 背景点（原图坐标）
        self._box_first: Optional[Tuple[int, int]] = None
        self._boxes_done: List[List[int]] = []

        # matplotlib 对象
        self._fig = None
        self._ax = None
        self._img_obj = None
        self._overlay_obj = None
        self._artifacts: list = []      # 临时绘制对象
        self._buttons: dict = {}
        self._scale_x = 1.0
        self._scale_y = 1.0
        self._orig_h = 0
        self._orig_w = 0
        self._display_img: Optional[np.ndarray] = None
        self._text_input = None

    # ──────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────
    def _ensure_sam(self):
        """确保 SAM 模型已加载（首次分割时触发）。"""
        if self._sam is not None:
            return
        from geoai_sam import SAMWrapper
        self._status("正在加载 SAM 模型，请稍候...")
        self._sam = SAMWrapper(
            model_type=self.model_type,
            sam_version=self.sam_version,
            automatic=False,
        )
        self._sam.set_image(self.image_path)
        self._status("SAM 模型已加载，可以开始标注")

    def _ensure_gsam(self):
        """确保 GroundedSAM 模型已加载。"""
        if self._gsam is not None:
            return
        from geoai_sam import GroundedSAMWrapper
        self._status("正在加载 Grounded-SAM 模型...")
        self._gsam = GroundedSAMWrapper(sam_model_type=self.model_type)
        self._gsam.set_image(self.image_path)
        self._status("Grounded-SAM 已就绪")

    def _load_and_prepare_image(self):
        """加载影像并生成降采样显示版本（内存优化）。"""
        import matplotlib.pyplot as plt

        # 计算目标显示尺寸
        try:
            import rasterio
            with rasterio.open(self.image_path) as src:
                self._orig_w = src.width
                self._orig_h = src.height
        except Exception:
            import cv2
            probe = cv2.imread(self.image_path)
            self._orig_h, self._orig_w = probe.shape[:2]

        scale = min(self.MAX_DISPLAY / max(self._orig_h, self._orig_w), 1.0)
        dh = max(1, int(self._orig_h * scale))
        dw = max(1, int(self._orig_w * scale))

        # 策略: 尽量使用 rasterio 的 overview / 窗口读取来避免全图加载
        display_img = None
        try:
            import rasterio
            from rasterio.enums import Resampling
            with rasterio.open(self.image_path) as src:
                bands = min(src.count, 3)
                # 使用 overview 降采样读取（如果可用）
                ovr_idx = self._best_overview(src, dw, dh)
                if ovr_idx is not None:
                    # 从 overview 级别读取
                    ovr_w = src.overviews(1)[ovr_idx]
                    ovr_h = int(src.height * ovr_w / src.width)
                    img = src.read(
                        list(range(1, bands + 1)),
                        out_shape=(bands, ovr_h, ovr_w),
                        resampling=Resampling.average,
                    )
                else:
                    # 直接读取并用 rasterio 降采样
                    img = src.read(
                        list(range(1, bands + 1)),
                        out_shape=(bands, dh, dw),
                        resampling=Resampling.average,
                    )
                img = np.transpose(img, (1, 2, 0))
        except Exception:
            # 回退: 用 cv2 + skimage resize
            import cv2
            img = cv2.imread(self.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img.shape[0] != dh or img.shape[1] != dw:
                img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)

        # 归一化到 0-1 (在已降采样的数组上操作)
        img = img.astype(np.float32)
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = (img - lo) / (hi - lo)

        self._display_img = img
        self._scale_x = self._orig_w / img.shape[1]
        self._scale_y = self._orig_h / img.shape[0]

    @staticmethod
    def _best_overview(src, target_w, target_h):
        """选择最接近目标尺寸的 overview 索引。"""
        try:
            overviews = src.overviews(1)
            if not overviews:
                return None
            # overviews 是降采样因子列表，如 [2, 4, 8, 16]
            best_idx = None
            best_diff = float("inf")
            for i, factor in enumerate(overviews):
                ovr_w = src.width // factor
                diff = abs(ovr_w - target_w)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            return best_idx
        except Exception:
            return None

    def _create_figure(self):
        """创建 matplotlib 图形窗口 + 工具栏。"""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self._fig, self._ax = plt.subplots(
            figsize=(14, 10), num="GeoAI SAM — 交互式标注"
        )
        self._fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.08)

        # 显示影像
        self._img_obj = self._ax.imshow(self._display_img)
        self._ax.set_xlim(0, self._display_img.shape[1])
        self._ax.set_ylim(self._display_img.shape[0], 0)
        self._ax.axis("off")

        # ── 工具栏按钮 ──
        btn_h = 0.055
        btn_w = 0.075
        y0 = 0.925

        specs = [
            ("point",  "  点标注  ",  0.02,  self._mode_point),
            ("box",    "  框标注  ",  0.10,  self._mode_box),
            ("pan",    "  平移缩放  ", 0.18,  self._mode_pan),
            ("text",   "  文本标注  ", 0.26,  self._mode_text),
            ("auto",   " 自动分割 ",  0.34,  self._mode_auto),
            ("fg",     "前景/背景",   0.44,  self._toggle_fg_bg),
            ("segment","  执行分割  ", 0.54,  self._on_segment),
            ("clear",  " 清空输入 ",  0.63,  self._on_clear_input),
            ("clear_r"," 清空结果 ",  0.72,  self._on_clear_results),
            ("post",   "  后处理  ",  0.80,  self._on_postprocess),
            ("export", " 导出菜单 ",  0.89,  self._on_export),
        ]

        for key, label, x0, callback in specs:
            ax_b = self._fig.add_axes([x0, y0, btn_w, btn_h])
            btn = Button(ax_b, label)
            btn.on_clicked(callback)
            self._buttons[key] = btn

        # 文本输入框（用于文本标注模式）
        ax_txt = self._fig.add_axes([0.44, y0 - 0.055, 0.44, 0.04])
        from matplotlib.widgets import TextBox
        self._text_input = TextBox(ax_txt, "文本提示: ", initial="")
        self._text_input.on_submit(self._on_text_submit)

        # ── 信息文本 ──
        self._title_text = self._fig.text(
            0.50, 0.985, "", ha="center", va="top",
            fontsize=13, fontweight="bold",
            fontfamily="sans-serif",
        )
        self._info_text = self._fig.text(
            0.50, 0.965, "", ha="center", va="top",
            fontsize=9, color="#444",
            fontfamily="sans-serif",
        )
        self._status_text = self._fig.text(
            0.01, 0.005, "", ha="left", va="bottom",
            fontsize=8, color="#666",
        )
        self._legend_text = self._fig.text(
            0.01, 0.985, "", ha="left", va="top",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#ccc"),
        )

        # ── 事件连接 ──
        self._fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # 获取影像地理信息（如有）
        try:
            import rasterio
            with rasterio.open(self.image_path) as src:
                crs = src.crs
                w, h = src.width, src.height
                self._title_text.set_text(
                    f"GeoAI SAM  |  {w} x {h}  |  CRS: {crs}"
                )
        except Exception:
            self._title_text.set_text(
                f"GeoAI SAM  |  {self._orig_w} x {self._orig_h}"
            )

        self._refresh_info()
        self._fig.canvas.draw_idle()

    # ──────────────────────────────────────────
    # 模式切换
    # ──────────────────────────────────────────
    def _set_mode(self, mode: str):
        self._mode = mode
        if mode == "pan":
            self._ax.set_navigate(True)
        else:
            self._ax.set_navigate(False)
        self._refresh_info()
        self._fig.canvas.draw_idle()

    def _mode_point(self, _=None):   self._set_mode("point")
    def _mode_box(self, _=None):     self._set_mode("box")
    def _mode_pan(self, _=None):     self._set_mode("pan")
    def _mode_text(self, _=None):    self._set_mode("text")
    def _mode_auto(self, _=None):
        self._set_mode("auto")
        self._do_auto_segment()

    def _toggle_fg_bg(self, _=None):
        """切换前景/背景点（仅点标注模式）。"""
        if self._mode != "point":
            return
        # 如果有背景点，切回前景；否则切到背景
        if self._bg_points:
            pass  # 保持在前景
        self._refresh_info()
        self._fig.canvas.draw_idle()

    def _refresh_info(self):
        """刷新底部信息文本。"""
        mode_labels = {
            "point": "点标注模式",
            "box": "框标注模式",
            "pan": "平移/缩放模式",
            "text": "文本标注模式",
            "auto": "自动分割模式",
        }
        mode_name = mode_labels.get(self._mode, self._mode)

        if self._mode == "point":
            tips = ("左键: 前景点 | 右键: 背景点 | "
                    "中键/滚轮: 分割 | S: 分割 | C: 清空输入")
        elif self._mode == "box":
            tips = "左键: 点击两个角点绘制矩形框 | 中键: 分割 | S: 分割 | C: 清空"
        elif self._mode == "text":
            tips = "在下方输入框中输入提示词后按回车"
        elif self._mode == "auto":
            tips = "自动分割已执行，查看结果"
        else:
            tips = "拖动平移 | 滚轮缩放"

        self._info_text.set_text(f"[{mode_name}]  {tips}")
        self._draw_legend()

    def _draw_legend(self):
        """绘制左下角的标注状态图例。"""
        lines = []
        if self._fg_points:
            lines.append(f"前景点: {len(self._fg_points)} 个")
        if self._bg_points:
            lines.append(f"背景点: {len(self._bg_points)} 个")
        if self._box_first:
            lines.append("框: 等待第 2 个角点")
        if self._boxes_done:
            lines.append(f"已完成框: {len(self._boxes_done)} 个")
        if self.annotation_count:
            lines.append(f"已生成: {self.annotation_count} 个 Mask")

        self._legend_text.set_text("\n".join(lines) if lines else "")

    def _status(self, msg: str):
        self._status_text.set_text(msg)
        self._fig.canvas.draw_idle()
        print(f"[标注] {msg}")

    # ──────────────────────────────────────────
    # 鼠标 & 键盘事件
    # ──────────────────────────────────────────
    def _disp_to_orig(self, dx, dy):
        """显示坐标 → 原图像素坐标。"""
        ox = int(round(dx * self._scale_x))
        oy = int(round(dy * self._scale_y))
        ox = max(0, min(ox, self._orig_w - 1))
        oy = max(0, min(oy, self._orig_h - 1))
        return ox, oy

    def _on_click(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        # 在 pan 模式下不处理
        if self._mode == "pan":
            return

        ox, oy = self._disp_to_orig(event.xdata, event.ydata)

        if self._mode == "point":
            if event.button == 1:     # 左键 → 前景
                self._fg_points.append((ox, oy))
                self._plot_point(event.xdata, event.ydata, "lime", "+")
                self._status(f"前景点 ({ox}, {oy})")
            elif event.button == 3:   # 右键 → 背景
                self._bg_points.append((ox, oy))
                self._plot_point(event.xdata, event.ydata, "red", "x")
                self._status(f"背景点 ({ox}, {oy})")
            elif event.button == 2:   # 中键 → 分割
                self._do_point_segment()
            self._draw_legend()

        elif self._mode == "box":
            if event.button == 1:
                if self._box_first is None:
                    self._box_first = (ox, oy)
                    self._plot_point(event.xdata, event.ydata, "cyan", "o")
                    self._status(f"框第 1 个角点: ({ox}, {oy})，请点击第 2 个角点")
                else:
                    x1, y1 = self._box_first
                    box = [min(x1, ox), min(y1, oy), max(x1, ox), max(y1, oy)]
                    self._boxes_done.append(box)
                    # 绘制矩形
                    self._plot_rect(
                        min(x1, ox) / self._scale_x,
                        min(y1, oy) / self._scale_y,
                        abs(ox - x1) / self._scale_x,
                        abs(oy - y1) / self._scale_y,
                    )
                    self._box_first = None
                    self._status(f"框已添加: {box}  (中键执行分割)")
            elif event.button == 2:   # 中键 → 分割
                self._do_box_segment()

        self._fig.canvas.draw_idle()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key == "s":
            self._on_segment()
        elif key == "c":
            self._on_clear_input()
        elif key == "escape":
            self._on_clear_input()
            self._on_clear_results()
        elif key == "p" and self._mode == "point":
            self._toggle_fg_bg()

    # ──────────────────────────────────────────
    # 绘图辅助
    # ──────────────────────────────────────────
    def _plot_point(self, x, y, color, marker):
        (pt,) = self._ax.plot(
            x, y, marker=marker, color=color, markersize=12,
            markeredgewidth=2, zorder=10,
        )
        self._artifacts.append(pt)

    def _plot_rect(self, x, y, w, h):
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor="cyan", facecolor="none",
            zorder=10,
        )
        self._ax.add_patch(rect)
        self._artifacts.append(rect)

    # ──────────────────────────────────────────
    # 分割执行
    # ──────────────────────────────────────────
    def _on_segment(self, _=None):
        """根据当前模式执行分割。"""
        if self._mode == "point":
            self._do_point_segment()
        elif self._mode == "box":
            self._do_box_segment()
        elif self._mode == "text":
            self._do_text_segment()
        elif self._mode == "auto":
            self._do_auto_segment()

    def _do_point_segment(self):
        all_pts = self._fg_points + self._bg_points
        if not all_pts:
            self._status("请先放置至少一个点")
            return

        labels = [1] * len(self._fg_points) + [0] * len(self._bg_points)
        self._status(f"点提示分割: {len(self._fg_points)} 前景 + {len(self._bg_points)} 背景 ...")

        self._ensure_sam()
        masks = self._sam.generate_masks_by_points(
            points=all_pts, point_labels=labels,
        )
        self._finalize(masks, "point")

    def _do_box_segment(self):
        if not self._boxes_done:
            self._status("请先绘制至少一个框")
            return

        self._status(f"框提示分割: {len(self._boxes_done)} 个框 ...")
        self._ensure_sam()

        if len(self._boxes_done) == 1:
            masks = self._sam.generate_masks_by_box(box=self._boxes_done[0])
        else:
            masks = self._sam.generate_masks_by_boxes(boxes=self._boxes_done)

        self._finalize(masks, "box")

    def _do_text_segment(self):
        text = self._text_input.text.strip() if self._text_input else ""
        if not text:
            self._status("请在下方输入框中输入文本提示词")
            return

        self._status(f"文本分割: '{text}' ...")
        self._ensure_gsam()
        masks = self._gsam.segment_by_text(text)
        self._finalize(masks, "text")

    def _on_text_submit(self, text):
        """文本输入框回车回调。"""
        if text.strip():
            self._do_text_segment()

    def _do_auto_segment(self):
        self._status("自动分割: 正在加载模型...")
        self._ensure_sam()

        # 需要 automatic=True 的模型
        from geoai_sam import SAMWrapper
        auto_sam = SAMWrapper(
            model_type=self.model_type,
            sam_version=self.sam_version,
            automatic=True,
        )
        auto_sam.set_image(self.image_path)

        self._status("自动分割中（全图网格采样），请耐心等待...")
        masks = auto_sam.generate_masks_auto(
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
        )
        self._finalize(masks, "auto")

    def _finalize(self, masks, tag: str):
        """分割完成后的统一处理：保存状态 + 刷新显示。"""
        self.current_mask = masks
        if masks is not None:
            self.masks_history.append(masks)
        self.annotation_count += 1
        self._status(
            f"分割完成! Mask shape: "
            f"{masks.shape if isinstance(masks, np.ndarray) else type(masks).__name__}"
        )
        self._show_overlay()
        self._on_clear_input()

    # ──────────────────────────────────────────
    # 显示 & 管理
    # ──────────────────────────────────────────
    def _show_overlay(self):
        """在当前图像上叠加半透明 mask 显示。"""
        if self.current_mask is None:
            return

        mask = self.current_mask
        if isinstance(mask, np.ndarray):
            m = mask[0] if mask.ndim == 3 else mask
        else:
            return

        # 降采样 mask 到显示尺寸
        dh, dw = self._display_img.shape[:2]
        if m.shape != (dh, dw):
            from skimage.transform import resize
            m_disp = resize(m.astype(np.float32), (dh, dw),
                            order=0, preserve_range=True).astype(bool)
        else:
            m_disp = m.astype(bool)

        overlay = np.zeros((dh, dw, 4), dtype=np.float32)
        overlay[m_disp] = [0.2, 0.9, 0.3, 0.40]   # 绿色半透明

        if self._overlay_obj is not None:
            self._overlay_obj.remove()
        self._overlay_obj = self._ax.imshow(
            overlay, extent=self._img_obj.get_extent(),
            interpolation="nearest", zorder=5,
        )
        self._fig.canvas.draw_idle()

    def _on_clear_input(self, _=None):
        """清空当前输入的点/框。"""
        self._fg_points.clear()
        self._bg_points.clear()
        self._box_first = None
        self._boxes_done.clear()

        for art in self._artifacts:
            try:
                art.remove()
            except Exception:
                pass
        self._artifacts.clear()
        self._draw_legend()
        self._status("输入已清空")
        self._fig.canvas.draw_idle()

    def _on_clear_results(self, _=None):
        """清空分割结果叠加层。"""
        if self._overlay_obj is not None:
            self._overlay_obj.remove()
            self._overlay_obj = None
        self.annotation_count = 0
        self._draw_legend()
        self._status("结果已清空")
        self._fig.canvas.draw_idle()

    # ──────────────────────────────────────────
    # 后处理
    # ──────────────────────────────────────────
    def _on_postprocess(self, _=None):
        """对当前 Mask 执行默认后处理流水线。"""
        if self.current_mask is None:
            self._status("没有 Mask 可后处理，请先分割")
            return

        from geoai_sam import MaskPostProcessor

        mask = self.current_mask
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            mask = mask[0]

        self._status("正在后处理...")
        clean = MaskPostProcessor.default_pipeline(
            mask, min_size=200, fill_holes_flag=True,
            smooth_sigma=1.5, opening_radius=2, closing_radius=3,
        )
        self.current_mask = clean
        self._show_overlay()

        proc = MaskPostProcessor()
        proc.load(clean)
        stats = proc.get_statistics()
        self._status(
            f"后处理完成: {stats['num_objects']} 个目标, "
            f"覆盖率 {stats['coverage_ratio']:.2%}"
        )

    # ──────────────────────────────────────────
    # 导出
    # ──────────────────────────────────────────
    def _on_export(self, _=None):
        """导出当前 Mask（矢量化 + 栅格）。"""
        if self.current_mask is None:
            self._status("没有 Mask 可导出，请先分割")
            return

        mask = self.current_mask
        if isinstance(mask, np.ndarray) and mask.ndim == 3:
            mask = mask[0]

        self._status("正在导出...")

        # ── 保存 Mask 栅格 ──
        mask_tif = os.path.join(self.output_dir, "final_mask.tif")
        self._save_mask_raster(mask, mask_tif)

        # ── 矢量化 ──
        try:
            from geoai_sam import MaskVectorizer
            vec = MaskVectorizer()
            gdf = vec.vectorize(mask, reference_image=self.image_path, min_area=50)

            for ext in ("geojson", "gpkg"):
                out = os.path.join(self.output_dir, f"polygons.{ext}")
                try:
                    vec.save(out)
                except Exception:
                    pass

            n = vec.get_polygon_count()
            self._status(f"导出完成: {n} 个多边形 → {self.output_dir}/")
        except Exception as e:
            self._status(f"导出异常: {e}")

        # ── 保存可视化截图 ──
        png_path = os.path.join(self.output_dir, "annotation_result.png")
        self._fig.savefig(png_path, dpi=150, bbox_inches="tight")

        print(f"[标注] 结果已保存到: {os.path.abspath(self.output_dir)}")

    def _save_mask_raster(self, mask: np.ndarray, path: str):
        """将 mask 保存为带地理信息的 GeoTIFF。"""
        try:
            import rasterio
            with rasterio.open(self.image_path) as src:
                profile = src.profile
                profile.update(count=1, dtype="uint8")
                if mask.shape == (src.height, src.width):
                    with rasterio.open(path, "w", **profile) as dst:
                        dst.write(mask.astype(np.uint8), 1)
                    return
        except Exception:
            pass
        np.save(path.replace(".tif", ".npy"), mask.astype(np.uint8))

    # ──────────────────────────────────────────
    # 入口
    # ──────────────────────────────────────────
    def run(self):
        """启动交互式标注窗口。"""
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        print("=" * 60)
        print("  GeoAI SAM 交互式标注工具（图形界面）")
        print("=" * 60)
        print(f"  影像: {self.image_path}")
        print(f"  模型: {self.model_type} ({self.sam_version})")
        print(f"  输出: {self.output_dir}")
        print()
        print("  操作提示:")
        print("    点标注 — 左键: 前景点 | 右键: 背景点 | 中键/S: 分割")
        print("    框标注 — 左键点击两个角点绘制矩形 | 中键/S: 分割")
        print("    快捷键 — S: 分割 | C: 清空输入 | Esc: 全部清空")
        print("=" * 60)

        self._load_and_prepare_image()
        self._create_figure()

        # 设置初始模式
        self._set_mode(self._mode)

        self._status(
            f"影像 {self._orig_w}x{self._orig_h} 已加载 "
            f"(显示缩放 {self._scale_x:.2f}x)。SAM 模型将在首次分割时加载。"
        )

        plt.show()


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoAI SAM 交互式标注工具（图形界面版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 启动（默认点标注模式）
  python interactive_annotate.py --image 西安19级.tif

  # 指定框标注模式
  python interactive_annotate.py --image 西安19级.tif --mode box

  # 文本提示标注
  python interactive_annotate.py --image 西安19级.tif --mode text --prompt building

  # 使用 vit_b 快速测试
  python interactive_annotate.py --image 西安19级.tif --model-type vit_b

  # 指定输出目录
  python interactive_annotate.py --image 西安19级.tif --output results
        """,
    )

    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="输入遥感影像路径",
    )
    parser.add_argument(
        "--mode", "-m", type=str,
        choices=["point", "box", "text", "auto"],
        default="point",
        help="初始标注模式 (默认: point)",
    )
    parser.add_argument(
        "--model-type", type=str,
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_l",
        help="SAM 模型类型 (默认: vit_l, 适合 6GB 显存)",
    )
    parser.add_argument(
        "--sam-version", type=str,
        choices=["sam1", "sam2", "sam3"],
        default="sam1",
        help="SAM 版本 (默认: sam1)",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="output/interactive",
        help="输出目录 (默认: output/interactive)",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default=None,
        help="文本提示词 (仅 text 模式初始使用)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 影像文件不存在: {args.image}")
        sys.exit(1)

    annotator = InteractiveAnnotator(
        image_path=args.image,
        model_type=args.model_type,
        sam_version=args.sam_version,
        output_dir=args.output,
        initial_mode=args.mode,
    )

    # 如果指定了文本提示词，预设到输入框
    if args.prompt:
        annotator._pending_prompt = args.prompt

    annotator.run()

    print("\n" + "=" * 60)
    print("  标注会话结束!")
    print("=" * 60)
    print(f"  输出目录: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
