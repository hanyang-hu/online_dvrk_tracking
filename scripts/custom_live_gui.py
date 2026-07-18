#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

try:
    from PySide6.QtCore import QObject, QThread, Qt, Signal
    from PySide6.QtGui import QImage, QMouseEvent, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QSplitter,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ModuleNotFoundError as exc:
    if exc.name == "PySide6":
        raise ModuleNotFoundError(
            "PySide6 is required for the GUI app. Install with: pip install -r requirements-gui.txt"
        ) from exc
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SurgicalSAM2") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SurgicalSAM2"))

from gui_live_tracking.config import LiveTrackingConfig
from gui_live_tracking.path_utils import validate_config
from gui_live_tracking.source_factory import create_frame_source
from gui_live_tracking.worker import TrackingWorker
from sam2.build_sam import build_sam2_camera_predictor


class PreviewLoader(QObject):
    completed = Signal(object, str)

    def __init__(self, config: LiveTrackingConfig, timeout_sec: float):
        super().__init__()
        self.config = config
        self.timeout_sec = timeout_sec

    def run(self) -> None:
        source = None
        try:
            source = create_frame_source(self.config)
            source.start()
            sample = source.get_sample(timeout_sec=self.timeout_sec)
            if sample is None:
                self.completed.emit(None, "")
            else:
                self.completed.emit(sample, "")
        except Exception:
            self.completed.emit(None, traceback.format_exc())
        finally:
            if source is not None:
                source.stop()


class VideoLabel(QLabel):
    clicked = Signal(int, int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(960, 540)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #111; color: #ddd;")
        self.setText("Load first frame, add prompts, then Start")

        self._img_w = 1
        self._img_h = 1

    def set_image_size(self, w: int, h: int) -> None:
        self._img_w = max(1, w)
        self._img_h = max(1, h)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.pixmap() is None:
            return

        x = int(event.position().x() * self._img_w / max(1, self.width()))
        y = int(event.position().y() * self._img_h / max(1, self.height()))

        if event.button() == Qt.LeftButton:
            self.clicked.emit(x, y, 1)
        elif event.button() == Qt.RightButton:
            self.clicked.emit(x, y, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("dVRK Live Tracking App")
        self.resize(1600, 900)

        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[TrackingWorker] = None
        self.preview_thread: Optional[QThread] = None
        self.preview_worker: Optional[PreviewLoader] = None

        self.preview_rgb = None
        self.preview_base_rgb = None
        self.prompt_points: List[Tuple[int, int]] = []
        self.prompt_labels: List[int] = []
        self.init_predictor = None
        self.init_mask_logits = None
        self._paused = False
        self._relabel_mode_active = False
        self._paused = False
        self._relabel_mode_active = False

        self._build_ui()
        self._set_defaults()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)

        file_box = QGroupBox("Paths")
        file_form = QFormLayout(file_box)

        self.video_path = QLineEdit()
        self.joint_angles_path = QLineEdit()
        self.cam_calib_path = QLineEdit()
        self.handeye_path = QLineEdit()
        self.lnd_path = QLineEdit()

        self.video_path_row = self._path_row(self.video_path, is_video=True)
        self.joint_angles_path_row = self._path_row(self.joint_angles_path)
        file_form.addRow("Video", self.video_path_row)
        file_form.addRow("Joint angles yaml", self.joint_angles_path_row)
        file_form.addRow("Camera calibration yaml", self._path_row(self.cam_calib_path))
        file_form.addRow("Handeye yaml", self._path_row(self.handeye_path))
        file_form.addRow("LND json", self._path_row(self.lnd_path))

        settings_box = QGroupBox("Tracking Settings")
        settings_form = QFormLayout(settings_box)

        input_box = QGroupBox("Input Source")
        input_form = QFormLayout(input_box)

        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItem("Offline", "offline")
        self.input_mode_combo.addItem("Mock live", "mock_live")
        self.input_mode_combo.addItem("ROS 2", "ros2")

        self.mock_rate_spin = QDoubleSpinBox()
        self.mock_rate_spin.setRange(0.1, 240.0)
        self.mock_rate_spin.setDecimals(1)
        self.mock_rate_spin.setValue(30.0)
        self.mock_loop_checkbox = QCheckBox("Loop")

        self.ros_image_topic = QLineEdit("/stereo/left/rectified_downscaled_image")
        self.ros_joint_topic = QLineEdit("/PSM1/measured_js")
        self.ros_jaw_topic = QLineEdit("/PSM1/jaw/measured_js")
        self.ros_sync_queue_size = QSpinBox()
        self.ros_sync_queue_size.setRange(1, 100)
        self.ros_sync_queue_size.setValue(5)
        self.ros_sync_slop = QDoubleSpinBox()
        self.ros_sync_slop.setRange(0.0, 1.0)
        self.ros_sync_slop.setDecimals(3)
        self.ros_sync_slop.setSingleStep(0.005)
        self.ros_sync_slop.setValue(0.015)
        self.sample_timeout_spin = QDoubleSpinBox()
        self.sample_timeout_spin.setRange(0.05, 10.0)
        self.sample_timeout_spin.setDecimals(2)
        self.sample_timeout_spin.setValue(0.5)
        self.ros_frame_id = QLineEdit("camera_left_optical_frame")
        self.ros_child_frame_id = QLineEdit("PSM1_joint4_tracked")

        self._input_mode_rows = []
        input_form.addRow("Mode", self.input_mode_combo)
        self._add_mode_row(input_form, "Replay rate", self.mock_rate_spin, {"mock_live"})
        self._add_mode_row(input_form, "Loop", self.mock_loop_checkbox, {"mock_live"})
        self._add_mode_row(input_form, "Image topic", self.ros_image_topic, {"ros2"})
        self._add_mode_row(input_form, "Arm joint topic", self.ros_joint_topic, {"ros2"})
        self._add_mode_row(input_form, "Jaw topic", self.ros_jaw_topic, {"ros2"})
        self._add_mode_row(input_form, "Sync queue size", self.ros_sync_queue_size, {"ros2"})
        self._add_mode_row(input_form, "Sync slop", self.ros_sync_slop, {"ros2"})
        self._add_mode_row(input_form, "Sample timeout", self.sample_timeout_spin, {"ros2"})
        self._add_mode_row(input_form, "ROS frame ID", self.ros_frame_id, {"ros2"})
        self._add_mode_row(input_form, "ROS child frame ID", self.ros_child_frame_id, {"ros2"})

        self.renderer_combo = QComboBox()
        self.renderer_combo.addItems(["nvdiffrast", "pytorch3d"])

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["CMA-ES", "XNES", "Gradient"])

        self.iters_spin = QSpinBox()
        self.iters_spin.setRange(1, 30)
        self.iters_spin.setValue(3)

        self.downscale_combo = QComboBox()
        self.downscale_combo.addItems(["1", "2", "4"])
        self.downscale_combo.setCurrentText("2")

        self.low_res_mesh_checkbox = QCheckBox("Use low-res mesh")
        self.low_res_mesh_checkbox.setChecked(True)

        self.point_loss_checkbox = QCheckBox("Use point loss")
        self.point_loss_checkbox.setChecked(True)

        self.lumped_checkbox = QCheckBox("Use lumped error init")
        self.lumped_checkbox.setChecked(False)
        self.turbo_handeye_btn = QPushButton("TuRBO Hand-Eye Init")
        self.turbo_handeye_btn.setCheckable(True)
        self.turbo_handeye_btn.setToolTip(
            "On Start, estimate the first-frame pose with TuRBO and use it to correct the hand-eye transform for this run."
        )

        settings_form.addRow("Renderer", self.renderer_combo)
        settings_form.addRow("Optimizer", self.optimizer_combo)
        settings_form.addRow("Downscale factor", self.downscale_combo)
        settings_form.addRow("Low-res mesh", self.low_res_mesh_checkbox)
        settings_form.addRow("Point loss", self.point_loss_checkbox)
        settings_form.addRow("Iterations/frame", self.iters_spin)
        settings_form.addRow("Lumped error", self.lumped_checkbox)
        settings_form.addRow("First frame", self.turbo_handeye_btn)

        prompt_box = QGroupBox("Prompting")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.addWidget(QLabel("On video: Left click = FG, Right click = BG"))

        buttons_row = QHBoxLayout()
        self.load_frame_btn = QPushButton("Load Initialization Frame")
        self.clear_prompts_btn = QPushButton("Clear Prompts")
        buttons_row.addWidget(self.load_frame_btn)
        buttons_row.addWidget(self.clear_prompts_btn)
        prompt_layout.addLayout(buttons_row)

        run_row = QHBoxLayout()
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop (Pause)")
        self.stop_btn.setEnabled(False)
        run_row.addWidget(self.start_btn)
        run_row.addWidget(self.stop_btn)

        resume_row = QHBoxLayout()
        self.continue_btn = QPushButton("Continue")
        self.reinit_continue_btn = QPushButton("Re-init")
        self.continue_btn.setEnabled(False)
        self.reinit_continue_btn.setEnabled(False)
        resume_row.addWidget(self.continue_btn)
        resume_row.addWidget(self.reinit_continue_btn)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(180)
        self.status_text.document().setMaximumBlockCount(300)

        left_layout.addWidget(file_box)
        left_layout.addWidget(input_box)
        left_layout.addWidget(settings_box)
        left_layout.addWidget(prompt_box)
        left_layout.addLayout(run_row)
        left_layout.addLayout(resume_row)
        left_layout.addWidget(self.status_text)
        left_layout.addStretch(1)

        self.video_label = VideoLabel()

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.video_label)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([500, 1100])

        self.video_label.clicked.connect(self._on_video_clicked)
        self.load_frame_btn.clicked.connect(self._load_first_frame)
        self.clear_prompts_btn.clicked.connect(self._clear_prompts)
        self.start_btn.clicked.connect(self._start_tracking)
        self.stop_btn.clicked.connect(self._stop_tracking)
        self.continue_btn.clicked.connect(self._continue_tracking)
        self.reinit_continue_btn.clicked.connect(self._activate_relabel_mode)

        self.iters_spin.valueChanged.connect(self._runtime_update)
        self.lumped_checkbox.stateChanged.connect(self._runtime_update)
        self.input_mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        self._update_mode_controls()

    def _add_mode_row(self, form: QFormLayout, label_text: str, widget: QWidget, modes: set[str]) -> None:
        label = QLabel(label_text)
        form.addRow(label, widget)
        self._input_mode_rows.append((label, widget, modes))

    def _set_defaults(self) -> None:
        self.video_path.setText("data/custom/bag1/left.mp4")
        self.joint_angles_path.setText("data/custom/bag1/joint_angles.yaml")
        self.cam_calib_path.setText("data/custom/camera_calibration.yaml")
        self.handeye_path.setText("data/custom/handeye.yaml")
        self.lnd_path.setText("data/custom/LND.json")

    def _path_row(self, line_edit: QLineEdit, is_video: bool = False) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(line_edit)

        btn = QPushButton("...")
        btn.setFixedWidth(40)

        def browse() -> None:
            if is_video:
                path, _ = QFileDialog.getOpenFileName(self, "Select video", str(REPO_ROOT), "Videos (*.mp4 *.avi *.mov)")
            else:
                path, _ = QFileDialog.getOpenFileName(self, "Select file", str(REPO_ROOT), "All files (*.*)")
            if path:
                rel = os.path.relpath(path, REPO_ROOT)
                line_edit.setText(rel)

        btn.clicked.connect(browse)
        row_layout.addWidget(btn)
        return row

    def _log(self, text: str) -> None:
        self.status_text.append(text)
        if text in {"Waiting for ROS 2 samples...", "ROS 2 sample received."}:
            self.statusBar().showMessage(text)

    def _error_summary(self, error: str) -> str:
        lines = [line.strip() for line in error.splitlines() if line.strip()]
        return lines[-1] if lines else "Unknown error"

    def _resolve_path(self, text: str) -> Path:
        p = Path(text)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p

    def _build_config(self) -> LiveTrackingConfig:
        return LiveTrackingConfig(
            video_path=self._resolve_path(self.video_path.text()),
            joint_angles_path=self._resolve_path(self.joint_angles_path.text()),
            camera_calibration_path=self._resolve_path(self.cam_calib_path.text()),
            handeye_path=self._resolve_path(self.handeye_path.text()),
            lnd_json_path=self._resolve_path(self.lnd_path.text()),
            machine_label="PSM1",
            input_mode=self.input_mode_combo.currentData(),
            mock_rate_hz=self.mock_rate_spin.value(),
            mock_loop=self.mock_loop_checkbox.isChecked(),
            sample_timeout_sec=self.sample_timeout_spin.value(),
            ros_image_topic=self.ros_image_topic.text().strip(),
            ros_joint_topic=self.ros_joint_topic.text().strip(),
            ros_jaw_topic=self.ros_jaw_topic.text().strip(),
            ros_sync_queue_size=self.ros_sync_queue_size.value(),
            ros_sync_slop_sec=self.ros_sync_slop.value(),
            ros_frame_id=self.ros_frame_id.text().strip(),
            ros_child_frame_id=self.ros_child_frame_id.text().strip(),
            renderer=self.renderer_combo.currentText(),
            searcher=self.optimizer_combo.currentText(),
            downscale_factor=int(self.downscale_combo.currentText()),
            use_low_res_mesh=self.low_res_mesh_checkbox.isChecked(),
            use_pts_loss=self.point_loss_checkbox.isChecked(),
            online_iters=self.iters_spin.value(),
            use_lumped_error_init=self.lumped_checkbox.isChecked(),
            use_turbo_handeye_init=self.turbo_handeye_btn.isChecked(),
        )

    def _update_mode_controls(self) -> None:
        mode = self.input_mode_combo.currentData()
        for label, widget, modes in self._input_mode_rows:
            visible = mode in modes
            label.setVisible(visible)
            widget.setVisible(visible)
            widget.setEnabled(visible)

        file_inputs_enabled = mode in {"offline", "mock_live"}
        self.video_path_row.setEnabled(file_inputs_enabled)
        self.joint_angles_path_row.setEnabled(file_inputs_enabled)

    def _render_preview(self) -> None:
        if self.preview_base_rgb is None:
            return

        frame = self.preview_base_rgb.copy()
        if self.init_mask_logits is not None:
            import cv2

            mask = (self.init_mask_logits.squeeze() > 0).detach().float().cpu().numpy()
            mask_u8 = (mask * 255).astype(np.uint8)
            color = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            frame = cv2.addWeighted(frame, 0.7, color, 0.3, 0)

        for (x, y), lb in zip(self.prompt_points, self.prompt_labels):
            color = (0, 255, 0) if lb == 1 else (255, 0, 0)
            # RGB frame
            import cv2

            cv2.circle(frame, (x, y), 5, color, -1)

        h, w, _ = frame.shape
        self.video_label.set_image_size(w, h)
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _load_first_frame(self) -> None:
        cfg = self._build_config()
        errors = validate_config(cfg)
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return

        if cfg.input_mode == "ros2":
            self._load_ros_initialization_frame(cfg)
            return

        source = create_frame_source(cfg)
        try:
            source.start()
            sample = source.get_sample(timeout_sec=cfg.sample_timeout_sec)
        except Exception as exc:
            QMessageBox.warning(self, "Source error", str(exc))
            return
        finally:
            source.stop()

        if sample is None:
            QMessageBox.warning(self, "Source error", "Could not read an initialization sample.")
            return

        self._use_initialization_sample(sample)

    def _load_ros_initialization_frame(self, cfg: LiveTrackingConfig) -> None:
        if self.preview_thread is not None:
            self._log("Already waiting for a ROS 2 initialization sample.")
            return

        self._log("Waiting for ROS 2 initialization sample...")
        self.statusBar().showMessage("Waiting for ROS 2 initialization sample...")
        self.load_frame_btn.setEnabled(False)

        self.preview_thread = QThread(self)
        self.preview_worker = PreviewLoader(cfg, timeout_sec=5.0)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.completed.connect(self._on_ros_preview_completed)
        self.preview_worker.completed.connect(self.preview_thread.quit)
        self.preview_thread.finished.connect(self._teardown_preview_worker)
        self.preview_thread.start()

    def _on_ros_preview_completed(self, sample, error: str) -> None:
        self.load_frame_btn.setEnabled(True)
        if error:
            self._log(f"ROS 2 initialization sample failed:\n{error}")
            QMessageBox.warning(
                self,
                "ROS 2 source error",
                f"{self._error_summary(error)}\n\nSee the progress log for the file and line number.",
            )
            return

        if sample is None:
            QMessageBox.warning(
                self,
                "No ROS 2 sample",
                "No sample received from ROS 2.\n\n"
                "Confirm that:\n"
                "1. The ROS 2 publisher is running.\n"
                "2. The Conda environment was activated before sourcing /opt/ros/humble/setup.bash.\n"
                "3. The configured image, arm joint, and jaw topics are being published.\n"
                "4. The topic QoS and synchronization slop are compatible.",
            )
            self._log("Timed out waiting for ROS 2 initialization sample.")
            return

        self._log("ROS 2 initialization sample received.")
        self.statusBar().showMessage("ROS 2 initialization sample received.")
        self._use_initialization_sample(sample)

    def _teardown_preview_worker(self) -> None:
        if self.preview_thread is not None:
            self.preview_thread.deleteLater()
        if self.preview_worker is not None:
            self.preview_worker.deleteLater()
        self.preview_thread = None
        self.preview_worker = None

    def _use_initialization_sample(self, sample) -> None:
        import cv2

        frame = cv2.cvtColor(sample.frame_bgr, cv2.COLOR_BGR2RGB)
        self.preview_rgb = frame.copy()
        self.preview_base_rgb = frame.copy()
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.init_mask_logits = None

        # Build an initialization predictor once so prompt clicks show live mask updates.
        self.init_predictor = build_sam2_camera_predictor(
            "./configs/sam2.1/sam2.1_hiera_s.yaml",
            "./SurgicalSAM2/checkpoints/sam2.1_hiera_s_endo18.pth",
            vos_optimized=True,
        )
        self.init_predictor.load_first_frame(cv2.cvtColor(self.preview_base_rgb, cv2.COLOR_RGB2BGR))

        self._render_preview()
        self._log("Loaded initialization frame. Add prompts with left/right click; mask updates live.")

    def _clear_prompts(self) -> None:
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.init_mask_logits = None
        if self.init_predictor is not None and self.preview_base_rgb is not None:
            import cv2

            self.init_predictor.load_first_frame(cv2.cvtColor(self.preview_base_rgb, cv2.COLOR_RGB2BGR))
        self._render_preview()
        self._log("Prompt points cleared.")

    def _on_video_clicked(self, x: int, y: int, label: int) -> None:
        if self.preview_rgb is None or self.init_predictor is None:
            return

        if self._paused and not self._relabel_mode_active:
            self._log("Paused: click 'Re-label Current Frame' first to edit prompts.")
            return

        self.prompt_points.append((x, y))
        self.prompt_labels.append(label)

        pts = np.array(self.prompt_points, dtype=np.float32)
        lbs = np.array(self.prompt_labels, dtype=np.int64)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.init_predictor.add_new_points(
                frame_idx=0,
                obj_id=0,
                points=pts,
                labels=lbs,
            )

        if isinstance(out, tuple):
            if len(out) >= 3:
                self.init_mask_logits = out[2]
            elif len(out) >= 2:
                self.init_mask_logits = out[1]
            else:
                self.init_mask_logits = None
        else:
            self.init_mask_logits = out

        if isinstance(self.init_mask_logits, (list, tuple)) and len(self.init_mask_logits) > 0:
            self.init_mask_logits = self.init_mask_logits[0]

        self._render_preview()
        lb_txt = "FG" if label == 1 else "BG"
        self._log(f"Added {lb_txt} prompt at ({x}, {y})")

    def _start_tracking(self) -> None:
        cfg = self._build_config()
        errors = validate_config(cfg)
        if errors:
            QMessageBox.warning(self, "Invalid configuration", "\n".join(errors))
            return

        if len(self.prompt_points) == 0 or 1 not in self.prompt_labels:
            QMessageBox.warning(self, "Prompts required", "Add at least one foreground prompt before starting.")
            return

        self.worker_thread = QThread(self)
        self.worker = TrackingWorker(cfg, self.prompt_points.copy(), self.prompt_labels.copy())
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.paused.connect(self._on_paused)
        self.worker.status.connect(self._log)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.continue_btn.setEnabled(False)
        self.reinit_continue_btn.setEnabled(False)
        self.renderer_combo.setEnabled(False)
        self.input_mode_combo.setEnabled(False)
        self.turbo_handeye_btn.setEnabled(False)

        self.worker_thread.start()
        self._paused = False
        self._relabel_mode_active = False
        self._log("Tracking started.")

    def _stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.request_pause()
            self._log("Pause requested...")

    def _continue_tracking(self) -> None:
        if self.worker is None:
            return

        if self._relabel_mode_active:
            if len(self.prompt_points) == 0 or 1 not in self.prompt_labels:
                QMessageBox.warning(self, "Prompts required", "Add at least one foreground prompt before continuing.")
                return
            self.worker.resume_with_reinit(self.prompt_points.copy(), self.prompt_labels.copy())
            self._log("Continuing with re-labeling from current frame.")
        else:
            self.worker.resume()
            self._log("Continuing without changes.")

        self.continue_btn.setEnabled(False)
        self.reinit_continue_btn.setEnabled(False)
        self._paused = False
        self._relabel_mode_active = False

    def _activate_relabel_mode(self) -> None:
        if self.worker is None:
            return

        if not self._paused:
            self._log("Re-label is only available while paused.")
            return

        # Enter relabel mode: wipe old prompts and mask visualization.
        self._relabel_mode_active = True
        self.prompt_points.clear()
        self.prompt_labels.clear()
        self.init_mask_logits = None

        # Show current paused frame as a fresh initialization canvas.
        if self.preview_rgb is not None:
            self.preview_base_rgb = self.preview_rgb.copy()

        if self.init_predictor is not None and self.preview_base_rgb is not None:
            import cv2

            self.init_predictor.load_first_frame(cv2.cvtColor(self.preview_base_rgb, cv2.COLOR_RGB2BGR))

        self._render_preview()
        self._log("Re-init collection active: previous prompts and segmentation overlay cleared. Add prompts, then click Continue.")

    def _runtime_update(self) -> None:
        if self.worker is not None:
            self.worker.update_runtime(
                online_iters=self.iters_spin.value(),
                use_lumped_error=self.lumped_checkbox.isChecked(),
            )

    def _on_frame_ready(self, frame_rgb) -> None:
        h, w, _ = frame_rgb.shape
        self.video_label.set_image_size(w, h)
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _on_paused(self, frame_rgb, frame_idx: int) -> None:
        # Keep current tracking visualization as-is until user explicitly enters relabel mode.
        self.preview_rgb = frame_rgb.copy()
        self._paused = True
        self._relabel_mode_active = False

        self.continue_btn.setEnabled(True)
        self.reinit_continue_btn.setEnabled(True)
        self._log(
            f"Paused at frame {frame_idx}. Continue keeps state unchanged. "
            "Or click Re-init, relabel this frame, then Continue."
        )

    def _on_metrics(self, fps: float, loss: float, frame_idx: int) -> None:
        self.statusBar().showMessage(f"Frame {frame_idx} | FPS {fps:.2f} | Loss {loss:.4f}")

    def _on_failed(self, error: str) -> None:
        self._log(f"Tracking failed:\n{error}")
        QMessageBox.critical(
            self,
            "Tracking failed",
            f"{self._error_summary(error)}\n\nSee the progress log for the file and line number.",
        )
        self._teardown_worker()

    def _on_finished(self) -> None:
        self._log("Tracking finished.")
        self._teardown_worker()

    def _teardown_worker(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.continue_btn.setEnabled(False)
        self.reinit_continue_btn.setEnabled(False)
        self.renderer_combo.setEnabled(True)
        self.input_mode_combo.setEnabled(True)
        self.turbo_handeye_btn.setEnabled(True)
        self._update_mode_controls()
        self._paused = False
        self._relabel_mode_active = False

        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.worker = None
        self.worker_thread = None

    def _shutdown_worker(self) -> None:
        """Gracefully stop worker/thread when closing the app."""
        if self.preview_thread is not None:
            self.preview_thread.quit()
            self.preview_thread.wait()
            self.preview_thread = None
            self.preview_worker = None

        if self.worker is not None:
            self.worker.request_stop()

        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.worker = None
        self.worker_thread = None

    def closeEvent(self, event):
        self._shutdown_worker()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
