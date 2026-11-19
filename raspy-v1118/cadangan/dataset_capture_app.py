#!/usr/bin/env python3
"""Capture shrimp images for later dataset annotation.

Run in a resizable window (default):
    python dataset_capture_app.py

Run fullscreen:
    python dataset_capture_app.py --fullscreen

Images are saved as numbered JPEGs in a timestamped folder under
``dataset_captures/`` by default. The images are not labeled automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from shrimp_camera_app import CameraThread, ShrimpCameraWindow, VideoView

ROOT = Path(__file__).resolve().parent


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Capture images for a shrimp dataset.")
    parser.add_argument("--output", type=Path, default=None, help="Image output folder")
    parser.add_argument("--fullscreen", action="store_true", help="Show fullscreen instead of windowed")
    return parser.parse_known_args(argv)


class DatasetCaptureWindow(QMainWindow):
    """Camera preview and one-button dataset image capture."""

    def __init__(self, output_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Shrimp Dataset Capture")
        self.setMinimumSize(800, 480)
        self.resize(1024, 600)

        self.camera_thread: CameraThread | None = None
        self.latest_frame: QImage | None = None
        self.output_dir = output_dir or (
            ROOT / "dataset_captures" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
        )
        self.next_index = 1
        self.saved_count = 0

        self.video_view = VideoView()
        self.camera_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh")
        self.folder_button = QPushButton("Choose folder")
        self.folder_label = QLabel()
        self.count_label = QLabel("0")
        self.status_label = QLabel("Select a camera to begin")
        self.capture_button = QPushButton("CAPTURE IMAGE [SPACE]")

        self._build_ui()
        self.capture_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        self.capture_shortcut.setAutoRepeat(False)
        self.capture_shortcut.activated.connect(self.capture_image)
        self._set_output_directory(self.output_dir)
        self.refresh_cameras()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QWidget { background: #f3f6f8; color: #17212b; font-family: "DejaVu Sans"; font-size: 15px; }
            QFrame#topBar, QFrame#capturePanel { background: white; border: 1px solid #d4dde4; border-radius: 12px; }
            QLabel#title { font-size: 21px; font-weight: 700; }
            QLabel#count { font-size: 42px; font-weight: 700; color: #2458bf; }
            QLabel#cameraView { background: #17212b; border: 1px solid #c3ced7; border-radius: 12px; }
            QPushButton, QComboBox { min-height: 42px; border: 1px solid #bdc9d2; border-radius: 9px; padding: 0 12px; background: #f8fafb; }
            QPushButton:hover { background: #eef3ff; border-color: #356ae6; }
            QPushButton#captureButton { min-height: 72px; background: #2f5fcc; color: white; font-size: 21px; font-weight: 700; }
            QPushButton#captureButton:disabled { background: #aebbc4; color: #edf1f3; }
        """)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        self.setCentralWidget(root)

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 8, 12, 8)
        title = QLabel("SHRIMP DATASET CAPTURE")
        title.setObjectName("title")
        top_layout.addWidget(title)
        top_layout.addStretch(1)
        top_layout.addWidget(QLabel("Camera"))
        self.camera_combo.setMinimumWidth(220)
        self.camera_combo.currentIndexChanged.connect(self._camera_selection_changed)
        top_layout.addWidget(self.camera_combo)
        self.refresh_button.clicked.connect(self.refresh_cameras)
        top_layout.addWidget(self.refresh_button)
        layout.addWidget(top_bar)

        content = QHBoxLayout()
        content.setSpacing(10)
        self.video_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content.addWidget(self.video_view, 7)

        panel = QFrame()
        panel.setObjectName("capturePanel")
        panel.setMinimumWidth(250)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(16, 16, 16, 16)
        panel_layout.setSpacing(10)

        folder_title = QLabel("SAVE TO")
        folder_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #667788;")
        panel_layout.addWidget(folder_title)
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        panel_layout.addWidget(self.folder_label)
        self.folder_button.clicked.connect(self.choose_output_directory)
        panel_layout.addWidget(self.folder_button)

        count_title = QLabel("IMAGES CAPTURED")
        count_title.setStyleSheet("font-size: 12px; font-weight: 700; color: #667788;")
        panel_layout.addWidget(count_title)
        self.count_label.setObjectName("count")
        panel_layout.addWidget(self.count_label)

        self.capture_button.setObjectName("captureButton")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self.capture_image)
        panel_layout.addWidget(self.capture_button)

        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #526475;")
        panel_layout.addWidget(self.status_label)
        note = QLabel("Images are saved without labels. Annotate them before training.")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 12px; color: #667788;")
        panel_layout.addStretch(1)
        panel_layout.addWidget(note)
        content.addWidget(panel, 3)
        layout.addLayout(content, 1)

    def _set_output_directory(self, path: Path) -> None:
        self.output_dir = path.expanduser().resolve()
        self.folder_label.setText(str(self.output_dir))
        self._refresh_count()

    def _refresh_count(self) -> None:
        files = list(self.output_dir.glob("capture_*.jpg")) if self.output_dir.is_dir() else []
        self.saved_count = len(files)
        indices = []
        for image_path in files:
            match = re.fullmatch(r"capture_(\d+)\.jpg", image_path.name)
            if match:
                indices.append(int(match.group(1)))
        self.next_index = max(indices, default=0) + 1
        self.count_label.setText(str(self.saved_count))

    def choose_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose dataset image folder",
            str(self.output_dir),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self._set_output_directory(Path(selected))
            self.status_label.setText("Output folder changed")

    def refresh_cameras(self) -> None:
        selected_device = self.camera_combo.currentData()
        self.stop_camera()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cameras = ShrimpCameraWindow.available_cameras()
        if not cameras:
            self.camera_combo.addItem("No camera detected", None)
            self.camera_combo.setEnabled(False)
            self.video_view.show_message("No camera detected\nConnect a camera and press refresh")
            self.status_label.setText("Camera offline")
            self.capture_button.setEnabled(False)
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
        device = self.camera_combo.itemData(index)
        if index >= 0 and device is not None:
            self.start_camera(device)

    def start_camera(self, device: str | int) -> None:
        self.stop_camera()
        self.latest_frame = None
        self.capture_button.setEnabled(False)
        self.video_view.show_message("Connecting to camera…")
        self.status_label.setText("Connecting to camera…")
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
        self.video_view.show_image(image)
        self.capture_button.setEnabled(True)
        self.status_label.setText("Camera ready — capture an image when the shrimp is in frame")

    def _camera_error(self, message: str) -> None:
        self.video_view.show_message(message)
        self.capture_button.setEnabled(False)
        self.status_label.setText(message)

    @staticmethod
    def _qimage_to_bgr(image: QImage) -> np.ndarray:
        rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
        pixels = np.frombuffer(
            rgb_image.constBits(), dtype=np.uint8, count=rgb_image.sizeInBytes()
        ).reshape((rgb_image.height(), rgb_image.bytesPerLine()))
        pixels = pixels[:, : rgb_image.width() * 3].reshape(
            (rgb_image.height(), rgb_image.width(), 3)
        )
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)

    def capture_image(self) -> None:
        if self.latest_frame is None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            image = self._qimage_to_bgr(self.latest_frame)
            while True:
                filename = f"capture_{self.next_index:06d}.jpg"
                output_path = self.output_dir / filename
                if not output_path.exists():
                    break
                self.next_index += 1
            saved = cv2.imwrite(
                str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not saved:
                raise OSError(f"Could not write image: {output_path}")
        except (OSError, cv2.error, ValueError) as exc:
            self.status_label.setText(f"Save failed: {exc}")
            return

        self.next_index += 1
        self.saved_count += 1
        self.count_label.setText(str(self.saved_count))
        self.status_label.setText(f"Saved {filename}")


    def closeEvent(self, event: QCloseEvent) -> None:
        self.stop_camera()
        event.accept()


def main() -> int:
    settings, qt_args = parse_args()
    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Shrimp Dataset Capture")
    app.setStyle("Fusion")
    window = DatasetCaptureWindow(settings.output)
    if settings.fullscreen:
        window.showFullScreen()
    else:
        window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
