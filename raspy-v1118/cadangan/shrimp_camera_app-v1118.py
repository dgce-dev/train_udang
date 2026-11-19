#!/usr/bin/env python3
"""PySide6 camera dashboard for shrimp capture and measurement.

Run in fullscreen kiosk mode (default):
    python shrimp_camera_app.py

Run in a resizable development window:
    python shrimp_camera_app.py --windowed

The interface is sized for a 1024x600 touch display. Shrimp segmentation and
Feret measurements are provided by the Hailo runtime and ``yolov11_udang.hef``.

how to run in a virtual environment with dependencies installed:
./venv-udang/bin/python shrimp_camera_app-v1118.py --conf 0.30
"""

from __future__ import annotations

import argparse
import errno
import json
import re
import sys
from pathlib import Path
from typing import Any, Protocol, cast, final, override

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup,
                               QComboBox, QFrame, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

try:
    import serial
except ImportError:
    serial = None


class LightSerialPort(Protocol):
    is_open: bool
    in_waiting: int

    def write(self, data: object) -> int | None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 600
LIGHT_SERIAL_DEVICE = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIGHT_SERIAL_BAUD = 115200
CALIBRATION_FILE = Path(__file__).with_name("calibration1.json")
DEFAULT_HEF = Path(__file__).with_name("yolov11_udang.hef")
DEFAULT_MM_PER_PIXEL = 65.0 / 170.0


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Shrimp camera and Hailo Feret measurement app.")
    parser.add_argument("--hef", default=str(DEFAULT_HEF), help="Segmentation HEF file")
    parser.add_argument(
        "--interface", choices=("pcie", "ethernet"), default="pcie",
        help="Hailo device interface (default: pcie)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square network input size")
    parser.add_argument("--conf", type=float, default=0.51, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--classes", type=int, nargs="+", default=[3], help="Class IDs to measure")
    parser.add_argument("--boxes-output", default=None, help="Detection output stream name")
    parser.add_argument("--masks-output", default=None, help="Mask output stream name")
    scale_options = parser.add_mutually_exclusive_group()
    scale_options.add_argument(
        "--mm-per-pixel", type=float, default=DEFAULT_MM_PER_PIXEL,
        help="Millimeters per pixel (default calibration: 65 mm / 170 px)",
    )
    scale_options.add_argument(
        "--pixels", dest="mm_per_pixel", action="store_const", const=None,
        default=argparse.SUPPRESS, help="Display unscaled Feret lengths in pixels",
    )
    parser.add_argument("--windowed", action="store_true", help="Show a resizable window instead of fullscreen")
    settings, qt_args = parser.parse_known_args(argv)
    if settings.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if not 0 <= settings.conf <= 1:
        parser.error("--conf must be between 0 and 1")
    if settings.mm_per_pixel is not None and settings.mm_per_pixel <= 0:
        parser.error("--mm-per-pixel must be positive")
    return settings, qt_args


APP_STYLE = """
QWidget {
    background-color: #f3f6f8;
    color: #17212b;
    font-family: "DejaVu Sans";
    font-size: 15px;
}
QFrame#header, QFrame#actionBar, QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d4dde4;
    border-radius: 12px;
}
QFrame#header {
    border-radius: 10px;
}
QLabel#appTitle {
    color: #14212b;
    font-size: 22px;
    font-weight: 700;
}
QLabel#eyebrow {
    color: #667788;
    font-size: 12px;
    font-weight: 700;
}
QLabel#cameraView {
    background-color: #17212b;
    border: 1px solid #c3ced7;
    border-radius: 12px;
    color: #d6e0e7;
    font-size: 18px;
}
QLabel#metricValue {
    color: #13232d;
    font-size: 38px;
    font-weight: 700;
}
QLabel#metricUnit, QLabel#secondaryText {
    color: #667788;
}
QTableWidget#lengthTable {
    background-color: #ffffff;
    alternate-background-color: #f6f8fa;
    border: 1px solid #d4dde4;
    border-radius: 6px;
    gridline-color: #e2e8ed;
    selection-background-color: #e8efff;
    selection-color: #17212b;
}
QTableWidget#lengthTable QHeaderView::section {
    background-color: #e7edf1;
    border: none;
    border-bottom: 1px solid #c8d2da;
    color: #435466;
    font-size: 12px;
    font-weight: 700;
    padding: 6px;
}
QLabel#statusBadge {
    background-color: #e7edf1;
    border-radius: 12px;
    color: #435466;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 10px;
}
QGroupBox {
    font-size: 14px;
    font-weight: 700;
    margin-top: 12px;
    padding-top: 16px;
}
QGroupBox#resultsPanel {
    padding-top: 10px;
}
QGroupBox::title {
    color: #526475;
    left: 14px;
    padding: 0 5px;
}
QComboBox, QPushButton {
    min-height: 40px;
    background-color: #f8fafb;
    border: 1px solid #bdc9d2;
    border-radius: 9px;
    color: #1d2b36;
    padding: 0 14px;
}
QComboBox:hover, QPushButton:hover {
    background-color: #eef3ff;
    border-color: #356ae6;
}
QComboBox:disabled, QPushButton:disabled {
    color: #98a5b0;
    background-color: #edf1f3;
    border-color: #d8e0e5;
}
QPushButton:pressed, QPushButton:checked {
    background-color: #2f5fcc;
    border-color: #254da5;
    color: white;
}
QPushButton#refreshButton {
    min-width: 44px;
    max-width: 44px;
    padding: 0;
}
QPushButton#captureButton {
    background-color: #e8eef2;
    border-color: #afbec9;
    min-height: 50px;
    font-size: 17px;
    font-weight: 700;
}
QPushButton#captureButton:hover {
    background-color: #dce6eb;
    border-color: #718493;
}
QPushButton#measureButton {
    background-color: #356ae6;
    border-color: #254da5;
    color: #061511;
    min-height: 50px;
    font-size: 17px;
    font-weight: 800;
}
QPushButton#measureButton:hover {
    background-color: #4a7cf0;
}
QPushButton#quitButton {
    background-color: #fff0f1;
    border-color: #dc9da5;
    color: #9b2939;
    min-width: 44px;
    max-width: 44px;
    min-height: 40px;
    padding: 0;
    font-size: 22px;
    font-size: 16px;
    font-weight: 700;
}
QPushButton#quitButton:hover {
    background-color: #b83e4f;
    border-color: #a52f40;
    color: white;
}
QSlider::groove:horizontal {
    background: #d8e1e6;
    border-radius: 4px;
    height: 8px;
}
QSlider::sub-page:horizontal {
    background: #356ae6;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 2px solid #356ae6;
    border-radius: 10px;
    height: 20px;
    margin: -7px 0;
    width: 20px;
}
QSlider:disabled::groove:horizontal {
    background: #e1e6ea;
}
QSlider:disabled::sub-page:horizontal {
    background: #c7d0d7;
}
QSlider:disabled::handle:horizontal {
    background: #c7d0d7;
    border-color: #aebbc4;
}
"""


@final
class CameraThread(QThread):
    """Read camera frames away from the GUI thread."""

    frame_ready: Signal = Signal(QImage)
    camera_error: Signal = Signal(str)

    def __init__(self, device: str | int, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.device: str | int = device

    @override
    def run(self) -> None:
        if cv2 is None:
            self.camera_error.emit("OpenCV is not installed")
            return

        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            self.camera_error.emit(f"Unable to open {self.device}")
            capture.release()
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Calibration is optional so the camera app remains usable before a
        # calibration file has been generated.  The remap is prepared once
        # per camera start rather than recalculated for every frame.
        undistort_map: tuple[object, object] | None = None
        try:
            with CALIBRATION_FILE.open() as calibration_file:
                calibration = json.load(calibration_file)
            camera_matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
            distortion = np.asarray(calibration["dist_coeffs"], dtype=np.float64)
            calibration_width, calibration_height = calibration["image_size_wh"]
            if calibration_width <= 0 or calibration_height <= 0:
                raise ValueError("Calibration image dimensions must be positive")

            # The frame is normalized to CAMERA_WIDTH x CAMERA_HEIGHT below.
            # Scale the intrinsics when calibration was captured at another
            # resolution so the principal point and focal lengths stay valid.
            scale_x = CAMERA_WIDTH / calibration_width
            scale_y = CAMERA_HEIGHT / calibration_height
            camera_matrix = camera_matrix.astype("float64", copy=True)
            camera_matrix[0, :] *= scale_x
            camera_matrix[1, :] *= scale_y
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                distortion,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
                0,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
            )
            undistort_map = cv2.initUndistortRectifyMap(
                camera_matrix,
                distortion,
                None,
                new_camera_matrix,
                (CAMERA_WIDTH, CAMERA_HEIGHT),
                cv2.CV_16SC2,
            )
        except (
            OSError, KeyError, TypeError, ValueError, IndexError, ArithmeticError, cv2.error
        ):
            # No calibration (or an incompatible file) should not prevent the
            # live camera view from starting.
            undistort_map = None

        try:
            failures = 0
            while not self.isInterruptionRequested():
                ok, frame = capture.read()
                if not ok:
                    failures += 1
                    if failures >= 20:
                        self.camera_error.emit("Camera stopped returning frames")
                        break
                    self.msleep(25)
                    continue

                failures = 0
                # Normalize the EasyCap feed to 640x480 without stretching. Most
                # devices already return 4:3; unexpected shapes are center-cropped.
                frame_height, frame_width = frame.shape[:2]
                target_ratio = CAMERA_WIDTH / CAMERA_HEIGHT
                source_ratio = frame_width / frame_height
                if source_ratio > target_ratio:
                    crop_width = round(frame_height * target_ratio)
                    crop_x = (frame_width - crop_width) // 2
                    frame = frame[:, crop_x : crop_x + crop_width]
                elif source_ratio < target_ratio:
                    crop_height = round(frame_width / target_ratio)
                    crop_y = (frame_height - crop_height) // 2
                    frame = frame[crop_y : crop_y + crop_height, :]
                if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                    frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)
                if undistort_map is not None:
                    frame = cv2.remap(
                        frame,
                        undistort_map[0],
                        undistort_map[1],
                        cv2.INTER_LINEAR,
                    )
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = rgb_frame.shape
                image = QImage(
                    rgb_frame.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
                self.frame_ready.emit(image)
        except Exception as exc:
            if not self.isInterruptionRequested():
                self.camera_error.emit(f"Camera processing failed: {exc}")
        finally:
            capture.release()


@final
class HailoMeasurementWorker(QObject):
    """Own the Hailo device and run segmentation away from the GUI thread."""

    backend_ready: Signal = Signal(bool, str)
    measurement_ready: Signal = Signal(object, object, str)

    def __init__(self, settings: argparse.Namespace) -> None:
        super().__init__()
        self.settings = settings
        self.segmenter: Any = None
        self.segmenter_active = False
        self._choose_outputs: Any = None
        self._restore_mask: Any = None
        self._restore_box: Any = None
        self._longest_mask_points: Any = None
        self._annotate: Any = None

    @Slot()
    def initialize(self) -> None:
        try:
            if cv2 is None:
                raise RuntimeError("OpenCV is not installed")
            hef_path = Path(self.settings.hef).expanduser()
            if not hef_path.is_absolute():
                hef_path = Path(__file__).resolve().parent / hef_path
            if not hef_path.is_file():
                raise FileNotFoundError(f"Hailo model not found: {hef_path}")

            from hailo_feret_measure import (HailoSegmenter, annotate,
                                             choose_outputs,
                                             longest_mask_points, restore_box,
                                             restore_mask)

            self._choose_outputs = choose_outputs
            self._restore_mask = restore_mask
            self._restore_box = restore_box
            self._longest_mask_points = longest_mask_points
            self._annotate = annotate
            self.segmenter = HailoSegmenter(
                hef_path,
                self.settings.imgsz,
                self.settings.boxes_output,
                self.settings.masks_output,
                self.settings.interface,
            )
            self.segmenter.__enter__()
            self.segmenter_active = True
        except (Exception, SystemExit) as exc:
            message = str(exc).strip() or type(exc).__name__
            self.backend_ready.emit(False, message)
            return

        self.backend_ready.emit(True, "Hailo segmentation ready")

    @Slot(object)
    def measure(self, frame: object) -> None:
        if not self.segmenter_active or cv2 is None:
            self.measurement_ready.emit([], None, "Hailo measurement backend is unavailable")
            return

        try:
            image = np.asarray(frame)
            confidence_threshold = self.settings.conf
            outputs, letterbox_state = self.segmenter.infer(image)
            boxes, masks = self._choose_outputs(
                outputs,
                self.settings.boxes_output,
                self.settings.masks_output,
                self.settings.imgsz,
                confidence_threshold,
                self.settings.iou,
                self.settings.classes,
            )
            scale, pad_x, pad_y = letterbox_state
            measurements: list[dict[str, object]] = []
            for index, row in enumerate(np.atleast_2d(boxes)):
                if len(row) < 6:
                    continue
                confidence, class_id = float(row[4]), int(row[5])
                if (
                    confidence < confidence_threshold
                    or class_id not in self.settings.classes
                    or index >= len(masks)
                ):
                    continue
                mask = self._restore_mask(
                    masks[index], image.shape, self.settings.imgsz, scale, pad_x, pad_y
                ) >= self.settings.mask_threshold
                length, endpoint_a, endpoint_b = self._longest_mask_points(mask)
                measurements.append({
                    "instance": index + 1,
                    "class_id": class_id,
                    "confidence": round(confidence, 4),
                    "length_px": round(length, 2),
                    "endpoint_a_xy": endpoint_a,
                    "endpoint_b_xy": endpoint_b,
                    "mask_area_px": int(mask.sum()),
                    "box_xyxy": self._restore_box(
                        row, image.shape, self.settings.imgsz, scale, pad_x, pad_y
                    ),
                })
            annotated = self._annotate(image, measurements)
            self.measurement_ready.emit(measurements, annotated, "")
        except (Exception, SystemExit) as exc:
            message = str(exc).strip() or type(exc).__name__
            self.measurement_ready.emit([], None, message)

    @Slot()
    def shutdown(self) -> None:
        try:
            if self.segmenter_active:
                self.segmenter.__exit__(None, None, None)
        finally:
            self.segmenter_active = False
            QThread.currentThread().quit()


@final
class VideoView(QLabel):
    """A video label that preserves the frame aspect ratio when resized."""

    def __init__(self) -> None:
        super().__init__("Select a camera to begin")
        self.setObjectName("cameraView")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 315)
        size_policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(True)
        self.setSizePolicy(size_policy)
        self._image: QImage | None = None

    @override
    def sizeHint(self) -> QSize:
        return QSize(CAMERA_WIDTH, CAMERA_HEIGHT)

    @override
    def minimumSizeHint(self) -> QSize:
        return QSize(420, 315)

    @override
    def heightForWidth(self, width: int) -> int:
        return round(width * CAMERA_HEIGHT / CAMERA_WIDTH)

    def show_image(self, image: QImage) -> None:
        self._image = image
        self.setText("")
        self._update_pixmap()

    def show_message(self, message: str) -> None:
        self._image = None
        self.clear()
        self.setText(message)

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # The parent layout can give this view extra horizontal space. Limit
        # the widget itself to the 4:3 frame size so its border never extends
        # beyond the camera image.
        target_width = max(
            self.minimumWidth(),
            round(self.height() * CAMERA_WIDTH / CAMERA_HEIGHT),
        )
        if self.maximumWidth() != target_width or self.minimumWidth() != target_width:
            self.setMinimumWidth(target_width)
            self.setMaximumWidth(target_width)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._image is None or self.width() <= 0 or self.height() <= 0:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


@final
class ShrimpCameraWindow(QMainWindow):
    """Single-page camera and shrimp measurement dashboard."""

    light_mode_changed: Signal = Signal(str)
    light_intensity_changed: Signal = Signal(int)
    measurement_requested: Signal = Signal(object)
    measurement_shutdown: Signal = Signal()

    def __init__(self, settings: argparse.Namespace | None = None) -> None:
        super().__init__()
        if settings is None:
            settings, _ = parse_args([])
        self.settings = settings
        self.setWindowTitle("Shrimp Measurement Station")
        self.setMinimumSize(800, 480)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.camera_thread: CameraThread | None = None
        self._pending_camera_device: str | int | None = None
        self.latest_frame: QImage | None = None
        self.captured_frame: QImage | None = None
        self.is_frozen: bool = False
        self.serial_connection: LightSerialPort | None = None
        self.measurement_backend_ready = False
        self.measurement_busy = False
        self._measurement_is_live = False
        self._closing = False
        self._allow_close = False

        self.light_command_timer = QTimer(self)
        self.light_command_timer.setSingleShot(True)
        self.light_command_timer.setInterval(75)
        self.light_command_timer.timeout.connect(self._send_manual_intensity)

        self.live_measurement_timer = QTimer(self)
        self.live_measurement_timer.setInterval(1000)
        self.live_measurement_timer.timeout.connect(self._measure_live_frame)

        self.serial_retry_timer = QTimer(self)
        self.serial_retry_timer.setInterval(2000)
        self.serial_retry_timer.timeout.connect(self._check_light_controller)
        self.serial_retry_timer.start()

        # Create shared widgets here so static type checkers can see that every
        # instance attribute exists before the layout helper methods use it.
        self.video_view = VideoView()
        self.light_controls: QGroupBox | None = None
        self.camera_combo = QComboBox()
        self.auto_button = QPushButton("Auto")
        self.manual_button = QPushButton("Manual")
        self.light_slider = QSlider(Qt.Orientation.Horizontal)
        self.light_value = QLabel("50%")
        self.status_badge = QLabel("READY")
        self.status_message = QLabel("Waiting for a capture")
        self.count_value = QLabel("—")
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_value = QLabel(f"{self.settings.conf:.3f}")
        self.length_table = QTableWidget(4, 2)
        self.camera_status = QLabel("● Camera offline")
        self.light_controller_status = QLabel("● Light controller offline")
        self.reconnect_light_button = QPushButton("Reconnect light")
        self.capture_button = QPushButton("CAPTURE")
        self.measure_button = QPushButton("MEASURE")
        self.quit_button = QPushButton("QUIT")

        self._build_ui()
        self.measurement_thread = QThread(self)
        self.measurement_worker = HailoMeasurementWorker(self.settings)
        self.measurement_worker.moveToThread(self.measurement_thread)
        self.measurement_thread.started.connect(self.measurement_worker.initialize)
        self.measurement_requested.connect(self.measurement_worker.measure)
        self.measurement_shutdown.connect(self.measurement_worker.shutdown)
        self.measurement_worker.backend_ready.connect(self._measurement_backend_state)
        self.measurement_worker.measurement_ready.connect(self._measurement_completed)
        self.measurement_thread.finished.connect(self._finish_close_if_ready)
        self.measurement_thread.start()
        self.set_light_mode("manual")
        self.refresh_cameras()
        self.connect_light_controller()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(8)
        self.setCentralWidget(root)

        root_layout.addWidget(self._build_header())

        content_layout = QHBoxLayout()
        content_layout.setSpacing(8)

        left_column = QVBoxLayout()
        left_column.setSpacing(8)
        left_column.addWidget(self.video_view, 0, Qt.AlignmentFlag.AlignHCenter)
        self.light_controls = self._build_light_controls()
        left_column.addWidget(self.light_controls, 0)

        content_layout.addLayout(left_column, 7)
        content_layout.addWidget(self._build_results_panel(), 3)
        root_layout.addLayout(content_layout, 1)
        root_layout.addWidget(self._build_action_bar())
        QTimer.singleShot(0, self._sync_camera_controls_width)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 6, 10, 6)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        eyebrow = QLabel("UNDERWATER VISION SYSTEM")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Shrimp Measurement")
        title.setObjectName("appTitle")
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        layout.addLayout(titles)
        layout.addStretch()

        camera_label = QLabel("Camera device")
        camera_label.setObjectName("secondaryText")
        layout.addWidget(camera_label)

        self.camera_combo.setMinimumWidth(100)
        self.camera_combo.setMaximumWidth(200)
        self.camera_combo.currentIndexChanged.connect(self._camera_selection_changed)
        layout.addWidget(self.camera_combo)

        refresh_button = QPushButton("↻")
        refresh_button.setObjectName("refreshButton")
        refresh_button.setToolTip("Refresh camera list")
        refresh_button.clicked.connect(self.refresh_cameras)
        layout.addWidget(refresh_button)

        self.reconnect_light_button .setObjectName("reconnectButton")
        self.reconnect_light_button.setToolTip("Reconnect the light controller")
        self.reconnect_light_button.clicked.connect(self.connect_light_controller)
        layout.addWidget(self.reconnect_light_button)

        self.quit_button.setObjectName("quitButton")
        self.quit_button.setText("✕")
        self.quit_button.setToolTip("Quit application")
        self.quit_button.setMinimumWidth(44)
        self.quit_button.setMaximumWidth(44)
        self.quit_button.clicked.connect(self.close)
        layout.addWidget(self.quit_button)
        return header

    def _build_light_controls(self) -> QGroupBox:
        group = QGroupBox("Light controls")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(12, 14, 12, 8)
        layout.setSpacing(8)

        mode_label = QLabel("Mode")
        mode_label.setObjectName("secondaryText")
        layout.addWidget(mode_label)

        for button in (self.auto_button, self.manual_button):
            button.setCheckable(True)
            button.setMinimumWidth(82)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.auto_button)
        mode_group.addButton(self.manual_button)
        self.manual_button.setChecked(True)
        self.auto_button.toggled.connect(self._auto_mode_toggled)
        self.manual_button.toggled.connect(self._manual_mode_toggled)
        layout.addWidget(self.auto_button)
        layout.addWidget(self.manual_button)

        layout.addSpacing(6)
        intensity_label = QLabel("☀")
        intensity_label.setObjectName("secondaryText")
        layout.addWidget(intensity_label)

        self.light_slider.setRange(0, 100)
        self.light_slider.setValue(50)
        self.light_slider.setEnabled(True)
        self.light_slider.valueChanged.connect(self._light_value_changed)
        layout.addWidget(self.light_slider, 1)

        self.light_value.setMinimumWidth(52)
        self.light_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.light_value)
        return group

    def _build_results_panel(self) -> QGroupBox:
        group = QGroupBox("Measurement results")
        group.setObjectName("resultsPanel")
        group.setMinimumWidth(240)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 8)
        layout.setSpacing(4)

        status_row = QHBoxLayout()
        status_title = QLabel("STATUS")
        status_title.setObjectName("eyebrow")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_row.addWidget(status_title)
        status_row.addStretch()
        status_row.addWidget(self.status_badge)
        layout.addLayout(status_row)

        self.status_message.setObjectName("secondaryText")
        self.status_message.setWordWrap(True)
        layout.addWidget(self.status_message)
        layout.addSpacing(4)

        confidence_row = QHBoxLayout()
        confidence_title = QLabel("CONFIDENCE")
        confidence_title.setObjectName("eyebrow")
        self.confidence_value.setMinimumWidth(44)
        self.confidence_value.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        confidence_row.addWidget(confidence_title)
        confidence_row.addStretch()
        confidence_row.addWidget(self.confidence_value)
        layout.addLayout(confidence_row)
        self.confidence_slider.setRange(0, 1000)
        self.confidence_slider.setSingleStep(1)
        self.confidence_slider.setPageStep(50)
        self.confidence_slider.setValue(round(self.settings.conf * 1000))
        self.confidence_slider.setToolTip(
            "Set the minimum detection confidence; changes apply to the next measurement"
        )
        self.confidence_slider.valueChanged.connect(self._confidence_value_changed)
        layout.addWidget(self.confidence_slider)

        count_title = QLabel("SHRIMP COUNT")
        count_title.setObjectName("eyebrow")
        layout.addWidget(count_title)
        self.count_value.setObjectName("metricValue")
        layout.addWidget(self.count_value)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #d4dde4;")
        layout.addWidget(divider)
        layout.addSpacing(2)

        length_title = QLabel("SHRIMP LENGTHS")
        length_title.setObjectName("eyebrow")
        layout.addWidget(length_title)

        self.length_table.setObjectName("lengthTable")
        length_unit = "mm" if self.settings.mm_per_pixel is not None else "px"
        self.length_table.setHorizontalHeaderLabels(["Shrimp", f"Length ({length_unit})"])
        self.length_table.verticalHeader().setVisible(False)
        self.length_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.length_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.length_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.length_table.setAlternatingRowColors(True)
        self.length_table.setShowGrid(True)
        self.length_table.verticalHeader().setDefaultSectionSize(30)
        four_row_height = (
            self.length_table.horizontalHeader().sizeHint().height()
            + (4 * self.length_table.verticalHeader().defaultSectionSize())
            + (2 * self.length_table.frameWidth())
            + 12
        )
        self.length_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.length_table.setMaximumHeight(four_row_height)
        layout.addWidget(self.length_table, 1)

        # placeholder = QLabel("Measurement pipeline placeholder")
        # placeholder.setObjectName("secondaryText")
        # placeholder.setWordWrap(True)
        # layout.addWidget(placeholder)
        # layout.addStretch()

        return group

    def _build_action_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("actionBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        connection_status = QVBoxLayout()
        connection_status.setSpacing(2)
        for status in (self.light_controller_status, self.camera_status):
            status.setObjectName("secondaryText")
            status.setWordWrap(True)
            connection_status.addWidget(status)
        layout.addLayout(connection_status, 1)

        self.capture_button.setObjectName("captureButton")
        self.capture_button.setMinimumWidth(145)
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.toggle_capture)
        layout.addWidget(self.capture_button)

        self.measure_button.setObjectName("measureButton")
        self.measure_button.setMinimumWidth(215)
        self.measure_button.setEnabled(False)
        self.measure_button.clicked.connect(self.measure_current_frame)
        layout.addWidget(self.measure_button)

        return bar

    @staticmethod
    def available_cameras() -> list[tuple[str, str | int]]:
        """Return friendly names and OpenCV identifiers for video devices."""
        if sys.platform.startswith("linux"):
            def device_number(path: Path) -> int:
                match = re.search(r"\d+$", path.name)
                return int(match.group()) if match is not None else 999

            devices = sorted(Path("/dev").glob("video*"), key=device_number)
            device_cameras: list[tuple[str, str | int]] = []
            for device in devices:
                name_file = Path("/sys/class/video4linux") / device.name / "name"
                try:
                    name = name_file.read_text(encoding="utf-8").strip()
                except OSError:
                    name = "Video device"
                device_cameras.append((f"{name} ({device})", str(device)))
            return device_cameras

        if cv2 is None:
            return []
        index_cameras: list[tuple[str, str | int]] = []
        for index in range(6):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                index_cameras.append((f"Camera {index}", index))
            capture.release()
        return index_cameras

    def refresh_cameras(self) -> None:
        selected_device = self.camera_combo.currentData()
        self.stop_camera()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        cameras = self.available_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera detected", None)
            self.camera_combo.setEnabled(False)
            self.video_view.show_message("No camera detected\nConnect a USB camera and press refresh")
            self._set_camera_state(False, "Camera offline")
        else:
            self.camera_combo.setEnabled(True)
            selected_index = 0
            for index, (name, device) in enumerate(cameras):
                self.camera_combo.addItem(name, device)
                if device == selected_device:
                    selected_index = index
            self.camera_combo.setCurrentIndex(selected_index)

        self.camera_combo.blockSignals(False)
        if cameras:
            self.start_camera(self.camera_combo.currentData())

    def _camera_selection_changed(self, index: int) -> None:
        if index >= 0 and self.camera_combo.currentData() is not None:
            self.start_camera(self.camera_combo.currentData())

    def start_camera(self, device: str | int) -> None:
        self.latest_frame = None
        self.captured_frame = None
        self.is_frozen = False
        self.capture_button.setText("CAPTURE")
        self.video_view.show_message("Connecting to camera…")
        self._set_camera_state(False, "Connecting…")

        self._pending_camera_device = device
        if self.camera_thread is not None:
            self.camera_thread.requestInterruption()
            if self.camera_thread.isRunning():
                return
            self.camera_thread.deleteLater()
            self.camera_thread = None
        self._start_pending_camera()

    def _start_pending_camera(self) -> None:
        if self._closing or self._pending_camera_device is None:
            return
        device = self._pending_camera_device
        self._pending_camera_device = None
        self.camera_thread = CameraThread(device, self)
        self.camera_thread.frame_ready.connect(self._receive_frame)
        self.camera_thread.camera_error.connect(self._camera_error)
        self.camera_thread.finished.connect(self._camera_thread_finished)
        self.camera_thread.start()

    def stop_camera(self) -> None:
        self._pending_camera_device = None
        if self.camera_thread is not None:
            self.camera_thread.requestInterruption()
            if not self.camera_thread.isRunning():
                self.camera_thread.deleteLater()
                self.camera_thread = None

    @Slot()
    def _camera_thread_finished(self) -> None:
        finished_thread = self.sender()
        if finished_thread is self.camera_thread:
            self.camera_thread.deleteLater()
            self.camera_thread = None
        if self._closing:
            self._finish_close_if_ready()
        else:
            self._start_pending_camera()

    def _receive_frame(self, image: QImage) -> None:
        self.latest_frame = image
        if not self.is_frozen:
            self.video_view.show_image(image)
        self._set_camera_state(True, "Camera OK")
        self.capture_button.setEnabled(not self.measurement_busy)
        self.measure_button.setEnabled(self.measurement_backend_ready and not self.measurement_busy)

    def _camera_error(self, message: str) -> None:
        self.video_view.show_message(message)
        self._set_camera_state(False, "Camera error")
        self.capture_button.setEnabled(False)
        self.measure_button.setEnabled(False)

    def _set_camera_state(self, online: bool, message: str) -> None:
        color = "#356ae6" if online else "#e07171";
        self.camera_status.setText(f"● {message}")
        self.camera_status.setStyleSheet(f"color: {color};")

    def toggle_capture(self) -> None:
        if self.latest_frame is None:
            return
        if self.is_frozen:
            self.is_frozen = False
            self.captured_frame = None
            self.capture_button.setText("CAPTURE")
            self.status_badge.setText("READY")
            self.status_badge.setStyleSheet("")
            self.status_message.setText("Live view resumed")
            return

        self.captured_frame = self.latest_frame.copy()
        self.is_frozen = True
        self.video_view.show_image(self.captured_frame)
        self.capture_button.setText("RESUME LIVE")
        self.status_badge.setText("CAPTURED")
        self.status_badge.setStyleSheet("")
        self.status_message.setText("Frame captured and ready to measure")

    def measure_current_frame(self) -> None:
        """Freeze the selected frame and run Hailo inference in the worker thread."""
        if self.measurement_busy or not self.measurement_backend_ready:
            return
        frame = self.captured_frame if self.is_frozen and self.captured_frame is not None else self.latest_frame
        if frame is None:
            return

        self.captured_frame = frame.copy()
        self.is_frozen = True
        self.video_view.show_image(self.captured_frame)
        self.capture_button.setText("RESUME LIVE")
        self._submit_measurement(self.captured_frame, live=False)

    def _measure_live_frame(self) -> None:
        if (
            self.measurement_busy
            or not self.measurement_backend_ready
            or self.is_frozen
            or self.latest_frame is None
        ):
            return
        self._submit_measurement(self.latest_frame.copy(), live=True)

    def _submit_measurement(self, frame: QImage, *, live: bool) -> None:
        self._measurement_is_live = live
        if live:
            self.status_badge.setText("LIVE")
            self.status_badge.setStyleSheet(
                "background-color: #fff1c9; color: #765600; border-radius: 12px;"
                " font-size: 12px; font-weight: 700; padding: 5px 10px;"
            )
            self.status_message.setText("Checking the latest camera frame…")
        else:
            self.count_value.setText("—")
            self.length_table.setRowCount(0)
            self.status_badge.setText("MEASURING")
            self.status_badge.setStyleSheet(
                "background-color: #fff1c9; color: #765600; border-radius: 12px;"
                " font-size: 12px; font-weight: 700; padding: 5px 10px;"
            )
            self.status_message.setText("Running Hailo segmentation and Feret measurement…")
        self.measurement_busy = True
        self.capture_button.setEnabled(False)
        self.measure_button.setEnabled(False)
        try:
            frame_bgr = self._qimage_to_bgr(frame)
        except (RuntimeError, ValueError) as exc:
            self._measurement_failed(str(exc))
            return
        self.measurement_requested.emit(frame_bgr)

    @staticmethod
    def _qimage_to_bgr(image: QImage) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed")
        rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.frombuffer(
            rgb_image.constBits(), dtype=np.uint8, count=rgb_image.sizeInBytes()
        ).reshape((rgb_image.height(), rgb_image.bytesPerLine()))
        pixels = pixels[:, : rgb_image.width() * 3].reshape(
            (rgb_image.height(), rgb_image.width(), 3)
        )
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_qimage(image: np.ndarray) -> QImage:
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        return QImage(
            rgb_image.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()

    def _measurement_backend_state(self, ready: bool, message: str) -> None:
        self.measurement_backend_ready = ready
        if ready:
            self.live_measurement_timer.start()
            self.status_badge.setText("READY")
            self.status_badge.setStyleSheet("")
            self.status_message.setText(
                "Live measurement active; point the camera at the shrimp."
            )
            self.measure_button.setEnabled(self.latest_frame is not None and not self.measurement_busy)
        else:
            self.live_measurement_timer.stop()
            self.status_badge.setText("MODEL ERROR")
            self.status_badge.setStyleSheet(
                "background-color: #fbe4e7; color: #a42f3e; border-radius: 12px;"
                " font-size: 12px; font-weight: 700; padding: 5px 10px;"
            )
            self.status_message.setText(f"Hailo measurement is unavailable: {message}")
            self.measure_button.setEnabled(False)

    def _measurement_completed(self, result: object, annotated: object, error: str) -> None:
        live_measurement = self._measurement_is_live
        self._measurement_is_live = False
        self.measurement_busy = False
        self.capture_button.setEnabled(self.latest_frame is not None)
        self.measure_button.setEnabled(self.measurement_backend_ready and self.latest_frame is not None)
        if error:
            self._measurement_failed(error)
            return

        measurements = cast(list[dict[str, object]], result)
        if not live_measurement and isinstance(annotated, np.ndarray):
            self.captured_frame = self._bgr_to_qimage(annotated)
            self.video_view.show_image(self.captured_frame)

        self.count_value.setText(str(len(measurements)))
        self.length_table.setRowCount(len(measurements))
        for row, item in enumerate(measurements):
            length_px = float(cast(float, item["length_px"]))
            if self.settings.mm_per_pixel is None:
                length_text = f"{length_px:.1f}"
            else:
                length_text = f"{length_px * self.settings.mm_per_pixel:.1f}"
            self.length_table.setItem(row, 0, QTableWidgetItem(f"Shrimp {row + 1}"))
            self.length_table.setItem(row, 1, QTableWidgetItem(length_text))

        if not measurements:
            self.status_badge.setText("LIVE" if live_measurement else "NO SHRIMP")
            self.status_badge.setStyleSheet(
                "background-color: #fbe4e7; color: #a42f3e; border-radius: 12px;"
                " font-size: 12px; font-weight: 700; padding: 5px 10px;"
            )
            if live_measurement:
                self.status_message.setText(
                    "Live update: no shrimp detected. Check the frame, lighting, and confidence threshold."
                )
            else:
                self.status_message.setText(
                    "No shrimp detected. Check the frame, lighting, and model class settings."
                )
            return

        self.status_badge.setText("LIVE" if live_measurement else "MEASURED")
        self.status_badge.setStyleSheet(
            "background-color: #e3f3e8; color: #28613a; border-radius: 12px;"
            " font-size: 12px; font-weight: 700; padding: 5px 10px;"
        )
        if live_measurement:
            self.status_message.setText(f"Live update: measured {len(measurements)} shrimp.")
        else:
            self.status_message.setText(f"Measured {len(measurements)} shrimp.")

    def _measurement_failed(self, message: str) -> None:
        self._measurement_is_live = False
        self.measurement_busy = False
        self.capture_button.setEnabled(self.latest_frame is not None)
        self.measure_button.setEnabled(self.measurement_backend_ready and self.latest_frame is not None)
        self.status_badge.setText("MEASURE ERROR")
        self.status_badge.setStyleSheet(
            "background-color: #fbe4e7; color: #a42f3e; border-radius: 12px;"
            " font-size: 12px; font-weight: 700; padding: 5px 10px;"
        )
        self.status_message.setText(f"Measurement failed: {message}")

    def _auto_mode_toggled(self, checked: bool) -> None:
        if checked:
            self.set_light_mode("auto")

    def _manual_mode_toggled(self, checked: bool) -> None:
        if checked:
            self.set_light_mode("manual")

    def connect_light_controller(self) -> None:
        """Open the CP2102 serial device and restore the selected light mode."""
        if self.serial_connection is not None and self.serial_connection.is_open:
            return
        if serial is None:
            self._set_light_controller_state(False, "Install pyserial for light control")
            return

        try:
            self.serial_connection = cast(
                LightSerialPort,
                cast(
                    object,
                    serial.Serial(
                        LIGHT_SERIAL_DEVICE,
                        LIGHT_SERIAL_BAUD,
                        timeout=0,
                        write_timeout=0.5,
                    ),
                ),
            )
        except OSError as exc:
            self.serial_connection = None
            if exc.errno == errno.EBUSY:
                message = "Light controller busy (close serial monitor)"
            elif exc.errno in (errno.ENOENT, errno.ENODEV):
                message = "Light controller not found"
            elif exc.errno == errno.EACCES:
                message = "Light controller permission denied"
            else:
                message = "Light controller offline"
            self._set_light_controller_state(False, message)
            return

        self._set_light_controller_state(True, "Light control OK")
        if self.manual_button.isChecked():
            self._write_light_command(f"M{self.light_slider.value():03d}")
        else:
            self._write_light_command("A")

    def _set_light_controller_state(self, online: bool, message: str) -> None:
        color = "#356ae6" if online else "#c04756"
        self.light_controller_status.setText(f"● {message}")
        self.light_controller_status.setStyleSheet(f"color: {color};")

    def _check_light_controller(self) -> None:
        """Detect a serial device removed after it was successfully opened."""
        if self.serial_connection is None:
            self.connect_light_controller()
            return

        try:
            if not self.serial_connection.is_open:
                raise OSError(errno.ENODEV, "Light controller port is closed")
            _ = self.serial_connection.in_waiting
        except OSError:
            self._mark_light_controller_disconnected()

    def _mark_light_controller_disconnected(self) -> None:
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except OSError:
                pass
        self.serial_connection = None
        self._set_light_controller_state(False, "Light controller disconnected")

    def _write_light_command(self, command: str) -> None:
        if self.serial_connection is None or not self.serial_connection.is_open:
            self.connect_light_controller()
            return
        try:
            self.serial_connection.write(command.encode("ascii"))
            self.serial_connection.flush()
        except OSError:
            self._mark_light_controller_disconnected()

    def set_light_mode(self, mode: str) -> None:
        is_manual = mode == "manual"
        self.light_slider.setEnabled(is_manual)
        self.light_command_timer.stop()
        if is_manual:
            self._write_light_command(f"M{self.light_slider.value():03d}")
        else:
            self._write_light_command("A")
        self.light_mode_changed.emit(mode)

    def _light_value_changed(self, value: int) -> None:
        self.light_value.setText(f"{value}%")
        if self.manual_button.isChecked():
            self.light_command_timer.start()
        self.light_intensity_changed.emit(value)

    def _confidence_value_changed(self, value: int) -> None:
        self.settings.conf = value / 1000
        self.confidence_value.setText(f"{self.settings.conf:.3f}")

    def _send_manual_intensity(self) -> None:
        if self.manual_button.isChecked():
            self._write_light_command(f"M{self.light_slider.value():03d}")

    def _sync_camera_controls_width(self) -> None:
        if self.light_controls is not None:
            self.light_controls.setFixedWidth(self.video_view.width())

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_camera_controls_width)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return

        self._closing = True
        self.stop_camera()
        self.live_measurement_timer.stop()
        self.light_command_timer.stop()
        self.serial_retry_timer.stop()
        self.status_message.setText("Waiting for camera and measurement worker to stop…")
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except OSError:
                pass
            self.serial_connection = None
        if self.measurement_thread.isRunning():
            self.measurement_shutdown.emit()
        self._finish_close_if_ready()

    @Slot()
    def _finish_close_if_ready(self) -> None:
        camera_stopped = self.camera_thread is None or not self.camera_thread.isRunning()
        measurement_stopped = not self.measurement_thread.isRunning()
        if self._closing and camera_stopped and measurement_stopped:
            self._allow_close = True
            QTimer.singleShot(0, self.close)


def main() -> int:
    settings, qt_args = parse_args()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Shrimp Measurement Station")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = ShrimpCameraWindow(settings)
    if settings.windowed:
        window.show()
    else:
        window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
