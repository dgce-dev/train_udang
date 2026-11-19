#!/usr/bin/env python3
"""PySide6 camera dashboard for shrimp capture and measurement.

Run in fullscreen kiosk mode (default):
    python shrimp_camera_app.py

Run in a resizable development window:
    python shrimp_camera_app.py --windowed

The interface is sized for a 1024x600 touch display. The measurement action is
intentionally a placeholder. Connect the existing
vision pipeline in ``ShrimpCameraWindow.measure_current_frame`` when it is
ready.
"""

from __future__ import annotations

import errno
import json
import re
import sys
from pathlib import Path
from typing import Protocol, cast, final, override

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

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


from PySide6.QtCore import QObject, QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 600
LIGHT_SERIAL_DEVICE = "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0"
LIGHT_SERIAL_BAUD = 115200
CALIBRATION_FILE = Path(__file__).with_name("calibration1.json")


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
        except (OSError, KeyError, TypeError, ValueError, cv2.error):
            # No calibration (or an incompatible file) should not prevent the
            # live camera view from starting.
            undistort_map = None

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

        capture.release()


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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Shrimp Measurement Station")
        self.setMinimumSize(800, 480)
        self.resize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.camera_thread: CameraThread | None = None
        self.latest_frame: QImage | None = None
        self.captured_frame: QImage | None = None
        self.is_frozen: bool = False
        self.serial_connection: LightSerialPort | None = None

        self.light_command_timer = QTimer(self)
        self.light_command_timer.setSingleShot(True)
        self.light_command_timer.setInterval(75)
        self.light_command_timer.timeout.connect(self._send_manual_intensity)

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
        self.length_table = QTableWidget(4, 2)
        self.camera_status = QLabel("● Camera offline")
        self.light_controller_status = QLabel("● Light controller offline")
        self.reconnect_light_button = QPushButton("Reconnect light")
        self.capture_button = QPushButton("CAPTURE")
        self.measure_button = QPushButton("MEASURE")
        self.quit_button = QPushButton("QUIT")

        self._build_ui()
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

        self.reconnect_light_button.setObjectName("reconnectButton")
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
        self.length_table.setHorizontalHeaderLabels(["Shrimp", "Length (mm)"])
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
        self.length_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
            cameras: list[tuple[str, str | int]] = []
            for device in devices:
                name_file = Path("/sys/class/video4linux") / device.name / "name"
                try:
                    name = name_file.read_text(encoding="utf-8").strip()
                except OSError:
                    name = "Video device"
                cameras.append((f"{name} ({device})", str(device)))
            return cameras

        if cv2 is None:
            return []
        cameras: list[tuple[str, str | int]] = []
        for index in range(6):
            capture = cv2.VideoCapture(index)
            if capture.isOpened():
                cameras.append((f"Camera {index}", index))
            capture.release()
        return cameras

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
        self.stop_camera()
        self.latest_frame = None
        self.captured_frame = None
        self.is_frozen = False
        self.capture_button.setText("CAPTURE")
        self.video_view.show_message("Connecting to camera…")
        self._set_camera_state(False, "Connecting…")

        self.camera_thread = CameraThread(device, self)
        self.camera_thread.frame_ready.connect(self._receive_frame)
        self.camera_thread.camera_error.connect(self._camera_error)
        self.camera_thread.start()

    def stop_camera(self) -> None:
        if self.camera_thread is not None:
            self.camera_thread.requestInterruption()
            if not self.camera_thread.wait(1500):
                self.camera_thread.terminate()
                self.camera_thread.wait()
            self.camera_thread.deleteLater()
            self.camera_thread = None

    def _receive_frame(self, image: QImage) -> None:
        self.latest_frame = image
        if not self.is_frozen:
            self.video_view.show_image(image)
        self._set_camera_state(True, "Camera OK")
        self.capture_button.setEnabled(True)
        self.measure_button.setEnabled(True)

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
        """Capture the current frame and provide a hook for CV inference."""
        if self.latest_frame is None:
            return
        self.captured_frame = self.latest_frame.copy()
        self.is_frozen = True
        self.video_view.show_image(self.captured_frame)
        self.capture_button.setText("RESUME LIVE")

        # TODO: Pass self.captured_frame to the shrimp detection/measurement
        # pipeline and update count_value, length_table, and the result status.
        self.count_value.setText("—")
        self.length_table.setRowCount(4)
        self.length_table.clearContents()
        self.status_badge.setText("PENDING")
        pending_style = (
            "background-color: #fff1c9; color: #765600; border-radius: 12px;"
            " font-size: 12px; font-weight: 700; padding: 5px 10px;"
        )
        self.status_badge.setStyleSheet(pending_style)
        self.status_message.setText("Frame captured. Measurement processing is not connected yet.")

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

    def _send_manual_intensity(self) -> None:
        if self.manual_button.isChecked():
            self._write_light_command(f"M{self.light_slider.value():03d}")

    @override
    def _sync_camera_controls_width(self) -> None:
        if self.light_controls is not None:
            self.light_controls.setFixedWidth(self.video_view.width())

    @override
    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self._sync_camera_controls_width)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_camera()
        self.light_command_timer.stop()
        self.serial_retry_timer.stop()
        if self.serial_connection is not None:
            try:
                self.serial_connection.close()
            except OSError:
                pass
            self.serial_connection = None
        event.accept()


def main() -> int:
    windowed = "--windowed" in sys.argv
    qt_args = [argument for argument in sys.argv if argument != "--windowed"]
    app = QApplication(qt_args)
    app.setApplicationName("Shrimp Measurement Station")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = ShrimpCameraWindow()
    if windowed:
        window.show()
    else:
        window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
