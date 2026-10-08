#!/usr/bin/env python3
"""Point Cloud Studio BIN/PLY/PCD point-cloud viewer. Run with the system Python 3."""
import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from cloud_io import LoadCancelled, load_cloud, write_ply, write_pcd
from auto_circle import DetectionCancelled, detect_circles
from measurements import validate_measurement_points
from surface_detection import SurfaceDetectionCancelled, detect_planes


CLOUD_EXTENSIONS = {".bin", ".ply", ".pcd"}
CLOUD_FILTER = "포인트 클라우드 (*.bin *.BIN *.ply *.PLY *.pcd *.PCD);;BIN (*.bin *.BIN);;PLY (*.ply *.PLY);;PCD (*.pcd *.PCD)"

APP_NAME = "Point Cloud Studio"
DEFAULT_ROOT = Path(__file__).resolve().parent / "data"
STYLE = """
QMainWindow, QWidget { background: #101722; color: #dce5f1; font-size: 12px; }
QLabel#brand { color: #ffffff; font-size: 23px; font-weight: 700; }
QLabel#eyebrow { color: #55d8cd; font-size: 11px; font-weight: 700; }
QLabel#muted { color: #8799b0; }
QLabel#section { color: #9fb2c9; font-size: 11px; font-weight: 700; }
QLabel#filename { color: #f2f7fc; font-size: 17px; font-weight: 600; }
QLabel#number { color: #69dfd2; font-size: 22px; font-weight: 700; }
QFrame#panel { background: #151f2e; border: 1px solid #263448; border-radius: 8px; }
QPushButton { background: #223148; border: 1px solid #35465e; border-radius: 5px; padding: 7px 10px; }
QPushButton:hover { background: #2c405c; border-color: #55bfb9; }
QPushButton:pressed, QPushButton:checked { background: #185551; color: #8ef6e5; border-color: #47bbaf; }
QPushButton#primary { background: #4ad4c1; color: #092c30; font-weight: 700; border: none; }
QPushButton:disabled { color: #526477; background: #182332; border-color: #263448; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #0c1420; border: 1px solid #2e4058; border-radius: 5px; padding: 6px; }
QComboBox QAbstractItemView { background: #182537; selection-background-color: #245553; }
QTreeWidget { background: #101a28; border: 1px solid #263448; border-radius: 6px; outline: 0; }
QTreeWidget::item { padding: 6px 2px; }
QTreeWidget::item:selected { background: #1d4b4b; color: #acfff0; }
QTreeWidget::item:hover { background: #23344b; }
QHeaderView::section { background: #1c293b; color: #91a6c0; border: none; padding: 6px; }
QCheckBox { spacing: 7px; padding: 3px 0; }
QSlider::groove:horizontal { background: #2b3b51; height: 4px; border-radius: 2px; }
QSlider::handle:horizontal { background: #61d9cb; width: 12px; margin: -5px 0; border-radius: 6px; }
QProgressBar { border: 1px solid #2b4055; border-radius: 4px; text-align: center; height: 16px; }
QProgressBar::chunk { background: #20756d; }
QStatusBar { background: #0c131e; color: #93a9c1; }
QToolTip { background: #22344d; color: #ecf4ff; border: 1px solid #52657a; padding: 5px; }
QSplitter::handle { background: #101722; }
QScrollArea { border: none; }
QScrollBar:vertical { background: #111b29; width: 8px; }
QScrollBar::handle:vertical { background: #3b506a; min-height: 30px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #111b29; height: 8px; }
QScrollBar::handle:horizontal { background: #3b506a; min-width: 30px; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


def label(text, name=None):
    widget = QtWidgets.QLabel(text)
    if name:
        widget.setObjectName(name)
    return widget


class Loader(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    loaded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path, limit, exclude_origin):
        super().__init__()
        self.path, self.limit, self.exclude_origin = path, limit, exclude_origin

    def run(self):
        try:
            cloud = load_cloud(self.path, self.limit, self.exclude_origin,
                               self.progress.emit, self.isInterruptionRequested)
            if not self.isInterruptionRequested():
                self.loaded.emit(cloud)
        except LoadCancelled:
            pass
        except Exception as exc:
            self.failed.emit(str(exc))


class CircleDetector(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    detected = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, xyz, generation, options):
        super().__init__()
        self.xyz, self.generation, self.options = xyz, generation, options

    def run(self):
        try:
            result = detect_circles(self.xyz, progress=self.progress.emit,
                                    cancelled=self.isInterruptionRequested, **self.options)
            if not self.isInterruptionRequested():
                self.detected.emit((self.generation, result))
        except DetectionCancelled:
            pass
        except Exception as exc:
            self.failed.emit((self.generation, str(exc)))


class PlaneDetector(QtCore.QThread):
    progress = QtCore.pyqtSignal(int)
    detected = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, xyz, generation, options):
        super().__init__()
        self.xyz, self.generation, self.options = xyz, generation, options

    def run(self):
        try:
            results = detect_planes(self.xyz, progress=self.progress.emit,
                                     cancelled=self.isInterruptionRequested, **self.options)
            if not self.isInterruptionRequested():
                self.detected.emit((self.generation, results))
        except SurfaceDetectionCancelled:
            pass
        except Exception as exc:
            self.failed.emit((self.generation, str(exc)))


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, root=DEFAULT_ROOT, initial=None):
        super().__init__()
        self.root = Path(root)
        self.cloud = None
        self.worker = None
        self.detector = None
        self.auto_generation = 0
        self.auto_results = []
        self.surface_detector = None
        self.surface_generation = 0
        self.surface_results = []
        self.selected_plane = None
        self.display_xyz = np.empty((0, 3), np.float32)
        self.display_rgb = np.empty((0, 3), np.uint8)
        self.measure_points = []
        self.measure_actors = []
        self.measure_mode = "navigate"
        self.measure_kind = "length"
        self.circle_measurement = None
        self.measure_complete = False
        self.marker_radius = 0.01
        self.keep_camera = False
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self.setMinimumSize(1100, 720)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_scene()
        self.scan_files()
        if initial:
            target = Path(initial)
        else:
            target = self.files[0] if self.files else None
        if target:
            QtCore.QTimer.singleShot(150, lambda: self.open_cloud(target))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(20, 16, 20, 12)
        outer.setSpacing(14)
        self.setCentralWidget(central)
        top = QtWidgets.QHBoxLayout()
        title = QtWidgets.QVBoxLayout()
        title.addWidget(label("3D POINT CLOUD VIEWER", "eyebrow"))
        title.addWidget(label(APP_NAME, "brand"))
        top.addLayout(title)
        top.addStretch()
        top.addWidget(label("BIN / PLY / PCD 포인트 클라우드 뷰어", "muted"))
        self.open_button = QtWidgets.QPushButton("포인트 클라우드 열기…")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.choose_file)
        top.addWidget(self.open_button)
        outer.addLayout(top)
        splitter = QtWidgets.QSplitter()
        outer.addWidget(splitter, 1)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 10, 0)
        ll.addWidget(label("DATASET  /  파일 탐색", "section"))
        self.root_label = label(str(self.root), "muted")
        self.root_label.setWordWrap(True)
        self.root_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        ll.addWidget(self.root_label)
        self.folder_button = QtWidgets.QPushButton("폴더 변경…")
        self.folder_button.clicked.connect(self.choose_folder)
        ll.addWidget(self.folder_button)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("파일명 / 폴더 검색")
        self.search.textChanged.connect(self.filter_files)
        ll.addWidget(self.search)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["파일", "MB"])
        self.tree.setColumnWidth(0, 192)
        self.tree.setIndentation(13)
        self.tree.itemClicked.connect(self.tree_clicked)
        ll.addWidget(self.tree, 1)
        self.file_count = label("", "muted")
        ll.addWidget(self.file_count)
        hint = label("파일 클릭으로 열기\nBIN / PLY / PCD 파일을 창으로 끌어와도 됩니다.", "muted")
        hint.setWordWrap(True)
        ll.addWidget(hint)
        splitter.addWidget(left)

        middle = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(middle)
        ml.setContentsMargins(0, 0, 0, 0)
        self.filename = label("포인트 클라우드를 선택하세요", "filename")
        self.filename.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        ml.addWidget(self.filename)
        self.subtitle = label("XYZ + RGB  ·  원본 좌표 단위", "muted")
        ml.addWidget(self.subtitle)
        views = QtWidgets.QHBoxLayout()
        for text, direction in [("입체", "iso"), ("위 XY", "top"), ("정면 XZ", "front"), ("측면 YZ", "side")]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _, d=direction: self.set_view(d))
            views.addWidget(button)
        fit = QtWidgets.QPushButton("화면 맞춤  F")
        fit.clicked.connect(self.fit_camera)
        views.addWidget(fit)
        ml.addLayout(views)
        measurement_tools = QtWidgets.QHBoxLayout()
        self.measure_group = QtWidgets.QButtonGroup(self)
        self.measure_group.setExclusive(True)
        self.measure_buttons = {}
        for text, mode in [("탐색", "navigate"), ("길이 (2점)", "length")]:
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            button.setChecked(mode == self.measure_mode)
            button.clicked.connect(lambda _, m=mode: self.set_measure_mode(m))
            self.measure_group.addButton(button)
            self.measure_buttons[mode] = button
            measurement_tools.addWidget(button)
        self.surface_button = QtWidgets.QPushButton("평면 찾기")
        self.surface_button.clicked.connect(self.start_surface_detection)
        self.surface_button.setToolTip("현재 표본·Z 범위에서 주요 평면들을 자동 검출합니다.")
        measurement_tools.addWidget(self.surface_button)
        self.auto_button = QtWidgets.QPushButton("자동 원 검출")
        self.auto_button.clicked.connect(self.start_auto_detection)
        self.auto_button.setToolTip("현재 표본·Z 범위에서 평면에 가까운 원형 윤곽을 검출합니다.")
        measurement_tools.addWidget(self.auto_button)
        self.clear_measure_button = QtWidgets.QPushButton("측정 초기화  Esc")
        self.clear_measure_button.clicked.connect(lambda: self.clear_measurement())
        measurement_tools.addWidget(self.clear_measure_button)
        ml.addLayout(measurement_tools)
        self.vtk_widget = QVTKRenderWindowInteractor()
        self.vtk_widget.setMinimumSize(400, 350)
        ml.addWidget(self.vtk_widget, 1)
        self.measure_label = label("탐색 · Shift+클릭으로 길이 2점 선택 / 위 버튼으로 측정 도구 선택", "muted")
        self.measure_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.measure_label.setWordWrap(True)
        self.measure_label.setMinimumHeight(44)
        ml.addWidget(self.measure_label)
        self.measure_coords = label("선택점 없음 · 원본 좌표 단위 (단위 미상) · 현재 표시 표본점 기준", "muted")
        self.measure_coords.setWordWrap(True)
        self.measure_coords.setMinimumHeight(54)
        self.measure_coords.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        ml.addWidget(self.measure_coords)
        ml.addWidget(label("왼쪽 드래그: 회전    ·    휠: 확대/축소    ·    가운데 또는 Shift+드래그: 이동", "muted"))
        splitter.addWidget(middle)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(12, 0, 0, 0)
        rl.setSpacing(9)
        rl.addWidget(label("SURFACE  /  주요 평면 찾기", "section"))
        self.surface_tolerance = QtWidgets.QDoubleSpinBox()
        self.surface_tolerance.setDecimals(6)
        self.surface_tolerance.setRange(0, 1e12)
        self.surface_tolerance.setPrefix("허용 거리  ")
        self.surface_tolerance.setSpecialValueText("허용 거리  자동")
        self.surface_tolerance.setKeyboardTracking(False)
        self.surface_tolerance.setToolTip("원본 좌표 단위. 0=분석 범위의 0.2%. 변경 후 다시 검출하세요.")
        self.surface_min_percent = QtWidgets.QDoubleSpinBox()
        self.surface_min_percent.setRange(1, 100)
        self.surface_min_percent.setDecimals(1)
        self.surface_min_percent.setValue(5)
        self.surface_min_percent.setPrefix("최소 점 비율  ")
        self.surface_min_percent.setSuffix(" %")
        self.surface_min_percent.setKeyboardTracking(False)
        self.surface_max_planes = QtWidgets.QSpinBox()
        self.surface_max_planes.setRange(1, 10)
        self.surface_max_planes.setValue(5)
        self.surface_max_planes.setPrefix("최대 평면 수  ")
        self.surface_max_planes.setKeyboardTracking(False)
        for spin in (self.surface_tolerance, self.surface_min_percent, self.surface_max_planes):
            rl.addWidget(spin)
        self.surface_candidates = QtWidgets.QComboBox()
        self.surface_candidates.setPlaceholderText("검출된 평면 후보")
        self.surface_candidates.currentIndexChanged.connect(self.show_surface_candidate)
        rl.addWidget(self.surface_candidates)
        self.surface_progress = QtWidgets.QProgressBar()
        self.surface_progress.hide()
        rl.addWidget(self.surface_progress)
        self.surface_cancel = QtWidgets.QPushButton("평면 검출 취소")
        self.surface_cancel.clicked.connect(self.cancel_surface_detection)
        self.surface_cancel.hide()
        rl.addWidget(self.surface_cancel)
        surface_hint = label("평면에 가까운 점을 강조합니다.\n법선·RMS·잔차 폭은 추정값입니다.", "muted")
        surface_hint.setWordWrap(True)
        rl.addWidget(surface_hint)
        rl.addSpacing(10)
        rl.addWidget(label("CIRCLE  /  자동 원 검출", "section"))
        self.auto_tolerance = QtWidgets.QDoubleSpinBox()
        self.auto_min_diameter = QtWidgets.QDoubleSpinBox()
        self.auto_max_diameter = QtWidgets.QDoubleSpinBox()
        for spin, prefix in [(self.auto_tolerance, "허용 오차  "),
                             (self.auto_min_diameter, "최소 지름  "),
                             (self.auto_max_diameter, "최대 지름  ")]:
            spin.setDecimals(6)
            spin.setRange(0, 1e12)
            spin.setPrefix(prefix)
            spin.setSpecialValueText(prefix + "자동")
            spin.setKeyboardTracking(False)
            spin.setToolTip("원본 좌표 단위 (단위 미상). 0이면 자동 설정. 변경 후 다시 검출하세요.")
            rl.addWidget(spin)
        self.auto_candidates = QtWidgets.QComboBox()
        self.auto_candidates.setPlaceholderText("검출된 원 후보")
        self.auto_candidates.currentIndexChanged.connect(self.show_auto_candidate)
        rl.addWidget(self.auto_candidates)
        self.auto_progress = QtWidgets.QProgressBar()
        self.auto_progress.hide()
        rl.addWidget(self.auto_progress)
        self.auto_cancel = QtWidgets.QPushButton("검출 취소")
        self.auto_cancel.clicked.connect(self.cancel_auto_detection)
        self.auto_cancel.hide()
        rl.addWidget(self.auto_cancel)
        auto_hint = label("표시점의 원형 윤곽을 추정합니다.\n최소 호 범위 120° · 결과를 확인하세요.", "muted")
        auto_hint.setWordWrap(True)
        rl.addWidget(auto_hint)
        rl.addSpacing(10)
        rl.addWidget(label("DISPLAY  /  표시 설정", "section"))
        rl.addWidget(label("최대 표시 점 수"))
        self.budget = QtWidgets.QComboBox()
        for text, n in [("25만 점 · 빠르게", 250_000), ("100만 점 · 기본", 1_000_000),
                        ("200만 점 · 정밀", 2_000_000), ("500만 점", 5_000_000), ("전체 유효점", 0)]:
            self.budget.addItem(text, n)
        self.budget.setCurrentIndex(1)
        self.budget.currentIndexChanged.connect(self.reload_cloud)
        rl.addWidget(self.budget)
        self.exclude = QtWidgets.QCheckBox("(0, 0, 0) 좌표 제외")
        self.exclude.setChecked(True)
        self.exclude.setToolTip("미측정으로 추정되는 원점 좌표를 숨깁니다. 해제하면 원점도 포함합니다.")
        self.exclude.toggled.connect(self.reload_cloud)
        rl.addWidget(self.exclude)
        rl.addWidget(label("색상 기준"))
        self.color_mode = QtWidgets.QComboBox()
        self.color_mode.addItems(["Z 높이", "X 좌표", "Y 좌표", "원본 RGB", "단색"])
        self.color_mode.currentIndexChanged.connect(self.update_colors)
        rl.addWidget(self.color_mode)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(label("점 크기"))
        self.size_label = label("2 px", "muted")
        size_row.addStretch()
        size_row.addWidget(self.size_label)
        rl.addLayout(size_row)
        self.point_size = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size.setRange(1, 8)
        self.point_size.setValue(2)
        self.point_size.valueChanged.connect(self.change_point_size)
        rl.addWidget(self.point_size)
        self.ortho = QtWidgets.QCheckBox("평행 투영")
        self.ortho.setChecked(True)
        self.ortho.toggled.connect(self.change_projection)
        rl.addWidget(self.ortho)
        self.outline_check = QtWidgets.QCheckBox("경계 상자")
        self.outline_check.setChecked(True)
        self.outline_check.toggled.connect(self.change_outline)
        rl.addWidget(self.outline_check)
        rl.addSpacing(10)
        rl.addWidget(label("SECTION  /  Z 범위", "section"))
        self.zmin, self.zmax = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for spin in (self.zmin, self.zmax):
            spin.setDecimals(4)
            spin.setRange(-1e12, 1e12)
            spin.setKeyboardTracking(False)
            spin.setEnabled(False)
        self.zmin.setPrefix("최소  ")
        self.zmax.setPrefix("최대  ")
        self.clip_timer = QtCore.QTimer(self)
        self.clip_timer.setSingleShot(True)
        self.clip_timer.timeout.connect(self.apply_clip)
        self.zmin.valueChanged.connect(lambda: self.clip_timer.start(150))
        self.zmax.valueChanged.connect(lambda: self.clip_timer.start(150))
        rl.addWidget(self.zmin)
        rl.addWidget(self.zmax)
        reset_clip = QtWidgets.QPushButton("전체 Z 범위 복원")
        reset_clip.clicked.connect(self.reset_clip)
        rl.addWidget(reset_clip)
        rl.addSpacing(10)
        rl.addWidget(label("POINTS  /  데이터 정보", "section"))
        self.shown_label = label("—", "number")
        rl.addWidget(self.shown_label)
        rl.addWidget(label("현재 화면의 점 수", "muted"))
        self.stats = label("파일을 열면 정보가 표시됩니다.", "muted")
        self.stats.setWordWrap(True)
        self.stats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        rl.addWidget(self.stats)
        self.bounds_label = label("", "muted")
        self.bounds_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        rl.addWidget(self.bounds_label)
        rl.addStretch()
        self.export_button = QtWidgets.QPushButton("표시 점 → PLY / PCD 저장…")
        self.export_button.clicked.connect(self.export_cloud)
        self.export_button.setEnabled(False)
        self.png_button = QtWidgets.QPushButton("현재 뷰 → PNG 저장…")
        self.png_button.clicked.connect(self.save_png)
        right_scroll.setWidget(right)
        right_column = QtWidgets.QWidget()
        rc = QtWidgets.QVBoxLayout(right_column)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.addWidget(right_scroll, 1)
        rc.addWidget(self.export_button)
        rc.addWidget(self.png_button)
        splitter.addWidget(right_column)
        splitter.setSizes([290, 870, 255])
        splitter.setStretchFactor(1, 1)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(230)
        self.progress.hide()
        self.cancel_button = QtWidgets.QPushButton("읽기 취소")
        self.cancel_button.clicked.connect(self.cancel_load)
        self.cancel_button.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.statusBar().showMessage("준비됨 · 원본 좌표 단위는 파일에 명시되어 있지 않습니다.")
        self.shortcuts = []
        for key, callback in [("F", self.fit_camera), ("Ctrl+O", self.choose_file), ("Escape", self.clear_measurement)]:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def _build_scene(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.035, 0.065, 0.105)
        self.renderer.SetBackground2(0.085, 0.125, 0.19)
        self.renderer.GradientBackgroundOn()
        rw = self.vtk_widget.GetRenderWindow()
        rw.SetMultiSamples(0)
        rw.SetNumberOfLayers(2)
        rw.AddRenderer(self.renderer)
        # Keep coplanar circles and selected points legible over dense surfaces.
        # Shared camera preserves their true 3D positions during navigation.
        self.measure_renderer = vtk.vtkRenderer()
        self.measure_renderer.SetLayer(1)
        self.measure_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        self.measure_renderer.InteractiveOff()
        rw.AddRenderer(self.measure_renderer)
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetPointSize(2)
        self.actor.GetProperty().LightingOff()
        self.renderer.AddActor(self.actor)
        self.outline = vtk.vtkOutlineSource()
        om = vtk.vtkPolyDataMapper()
        om.SetInputConnection(self.outline.GetOutputPort())
        self.outline_actor = vtk.vtkActor()
        self.outline_actor.SetMapper(om)
        self.outline_actor.GetProperty().SetColor(0.25, 0.36, 0.48)
        self.outline_actor.SetVisibility(False)
        self.renderer.AddActor(self.outline_actor)
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.Build()
        stops = np.array([[64, 44, 128], [35, 124, 169], [42, 179, 158], [166, 218, 93], [255, 222, 93]]) / 255
        for i in range(256):
            t = i / 255 * (len(stops) - 1)
            j = min(int(t), len(stops) - 2)
            rgb = stops[j] * (1 - (t - j)) + stops[j + 1] * (t - j)
            self.lut.SetTableValue(i, *rgb, 1)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(self.lut)
        self.scalar_bar.SetNumberOfLabels(5)
        self.scalar_bar.SetLabelFormat("%.3g")
        self.scalar_bar.SetWidth(0.09)
        self.scalar_bar.SetHeight(0.4)
        self.scalar_bar.SetPosition(0.88, 0.10)
        self.scalar_bar.SetUnconstrainedFontSize(True)
        self.scalar_bar.SetMaximumHeightInPixels(220)
        self.scalar_bar.SetMaximumWidthInPixels(85)
        self.scalar_bar.SetTitleRatio(0.13)
        for prop in [self.scalar_bar.GetTitleTextProperty(), self.scalar_bar.GetLabelTextProperty()]:
            prop.SetColor(0.75, 0.83, 0.91)
            prop.SetFontFamilyToArial()
            prop.SetFontSize(12)
            prop.ItalicOff()
            prop.ShadowOff()
        self.scalar_bar.VisibilityOff()
        self.renderer.AddActor2D(self.scalar_bar)
        self.interactor = rw.GetInteractor()
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self._left_down)
        self.style.AddObserver("LeftButtonReleaseEvent", self._left_up)
        self.style.AddObserver("MouseMoveEvent", self._mouse_move)
        self.style.AddObserver("CharEvent", lambda *_: None)  # Disable VTK's hidden w/s/q shortcuts.
        axes = vtk.vtkAxesActor()
        self.orientation = vtk.vtkOrientationMarkerWidget()
        self.orientation.SetOrientationMarker(axes)
        self.orientation.SetInteractor(self.interactor)
        self.orientation.SetViewport(0, 0, 0.17, 0.22)
        self.orientation.SetEnabled(1)
        self.orientation.InteractiveOff()
        self.renderer.GetActiveCamera().ParallelProjectionOn()
        self.vtk_widget.Initialize()
        self._press = None
        self.polydata = None

    def scan_files(self):
        self.tree.clear()
        self.files = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in CLOUD_EXTENSIONS) if self.root.is_dir() else []
        groups = {}
        total = 0
        for path in self.files:
            group_name = str(path.parent.relative_to(self.root))
            if group_name not in groups:
                group = QtWidgets.QTreeWidgetItem(self.tree, [group_name if group_name != "." else self.root.name])
                group.setFlags(group.flags() & ~QtCore.Qt.ItemIsSelectable)
                groups[group_name] = group
            size = path.stat().st_size
            total += size
            item = QtWidgets.QTreeWidgetItem(groups[group_name], [path.name, f"{size / 1e6:.0f}"])
            item.setData(0, QtCore.Qt.UserRole, str(path))
            item.setToolTip(0, str(path))
        self.tree.expandAll()
        self.root_label.setText(str(self.root))
        self.file_count.setText(f"{len(self.files)}개 파일  ·  {total / 1e9:.2f} GB")
        self.filter_files(self.search.text())

    def filter_files(self, text):
        query = text.casefold()
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            visible = 0
            for j in range(group.childCount()):
                item = group.child(j)
                match = query in (group.text(0) + "/" + item.text(0)).casefold()
                item.setHidden(not match)
                visible += match
            group.setHidden(not visible)

    def tree_clicked(self, item, _):
        path = item.data(0, QtCore.Qt.UserRole)
        if path:
            self.open_cloud(Path(path))

    def choose_file(self):
        if self.worker:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "포인트 클라우드 열기", str(self.root), CLOUD_FILTER)
        if path:
            self.open_cloud(Path(path))

    def choose_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "포인트 클라우드 폴더 선택", str(self.root))
        if path:
            self.root = Path(path)
            self.scan_files()

    def open_cloud(self, path, preserve=False):
        if self.worker:
            self.statusBar().showMessage("파일을 읽는 중입니다. 완료 후 선택하거나 읽기를 취소하세요.")
            return
        self._invalidate_auto()
        self._invalidate_surface()
        self.keep_camera = preserve
        self._loaded_this_time = False
        self._load_error = None
        self.load_start = time.monotonic()
        self.statusBar().showMessage(f"읽는 중: {path.name}")
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_button.show()
        for widget in (self.tree, self.open_button, self.folder_button, self.budget, self.exclude):
            widget.setEnabled(False)
        self.worker = Loader(path, self.budget.currentData(), self.exclude.isChecked())
        self.worker.progress.connect(self.progress.setValue)
        self.worker.loaded.connect(self._loaded)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def cancel_load(self):
        if self.worker:
            self.worker.requestInterruption()
            self.statusBar().showMessage("읽기 취소 중…")

    def _loaded(self, cloud):
        self._loaded_this_time = True
        self.loaded_options = (self.worker.limit, self.worker.exclude_origin)
        self.cloud = cloud
        self.filename.setText(cloud.path.name)
        self.filename.setToolTip(str(cloud.path))
        color_info = "XYZ + RGB" if cloud.has_rgb else "XYZ · RGB 없음 (기본색)"
        self.subtitle.setText(f"{cloud.path.suffix[1:].upper()} · {cloud.header.width:,} × {cloud.header.height:,} · {color_info} · {cloud.header.size / 1e6:.1f} MB")
        self.stats.setText(
            f"원본 레코드  {cloud.header.count:,}\n"
            f"유효점  {cloud.valid_count:,}\n"
            f"원점 좌표  {cloud.zero_count:,}\n"
            f"비유한 좌표  {cloud.nonfinite_count:,}\n"
            f"표시용 표본  {len(cloud.xyz):,}"
        )
        self.bounds_label.setText("전체 유효점 범위\n" + "\n".join(
            f"{axis}  {cloud.bounds[0, i]:.4f} … {cloud.bounds[1, i]:.4f}" for i, axis in enumerate("XYZ")
        ) + "\n단위: 원본 좌표 단위")
        self.clear_measurement(render=False)
        self._set_clip_values()
        self.apply_clip()
        if not self.keep_camera:
            self.set_view("iso")
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for j in range(group.childCount()):
                item = group.child(j)
                item.setSelected(Path(item.data(0, QtCore.Qt.UserRole)).resolve() == cloud.path)
        self.statusBar().showMessage(f"{cloud.path.name} · {cloud.valid_count:,} 유효점 · {time.monotonic() - self.load_start:.1f}초")

    def _failed(self, message):
        self._load_error = message
        QtWidgets.QMessageBox.warning(self, "파일 읽기 실패", message)

    def _finished(self):
        old = self.worker
        self.worker = None
        old.deleteLater()
        for widget in (self.tree, self.open_button, self.folder_button, self.budget, self.exclude):
            widget.setEnabled(True)
        self.progress.hide()
        self.cancel_button.hide()
        if not self._loaded_this_time:
            if self.cloud:
                limit, exclude_origin = self.loaded_options
                self.budget.blockSignals(True)
                self.exclude.blockSignals(True)
                self.budget.setCurrentIndex(self.budget.findData(limit))
                self.exclude.setChecked(exclude_origin)
                self.budget.blockSignals(False)
                self.exclude.blockSignals(False)
            self.statusBar().showMessage("읽기 실패" if self._load_error else "읽기를 취소했습니다.")

    def reload_cloud(self, *_):
        if self.cloud and not self.worker:
            self.open_cloud(self.cloud.path, preserve=True)

    def _set_clip_values(self):
        for spin, val in [(self.zmin, self.cloud.bounds[0, 2]), (self.zmax, self.cloud.bounds[1, 2])]:
            spin.blockSignals(True)
            spin.setValue(float(val))
            spin.setSingleStep(max(float(np.ptp(self.cloud.bounds[:, 2])) / 100, 0.001))
            spin.setEnabled(True)
            spin.blockSignals(False)

    def reset_clip(self):
        if self.cloud:
            self._set_clip_values()
            self.apply_clip()

    def apply_clip(self):
        if self.cloud is None:
            return
        if self.zmin.value() > self.zmax.value():
            self.statusBar().showMessage("Z 최소값은 최대값보다 작거나 같아야 합니다.")
            return
        self.clear_measurement(render=False)
        # Half a displayed decimal unit keeps boundary points after spinbox rounding.
        z = self.cloud.xyz[:, 2]
        mask = (z >= self.zmin.value() - 0.000051) & (z <= self.zmax.value() + 0.000051)
        self.display_xyz = np.ascontiguousarray(self.cloud.xyz[mask])
        self.display_rgb = np.ascontiguousarray(self.cloud.rgb[mask])
        n = len(self.display_xyz)
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(self.display_xyz, deep=True))
        vertices = vtk.vtkCellArray()
        vertices.SetData(numpy_to_vtkIdTypeArray(np.arange(n + 1, dtype=np.int64), deep=True),
                         numpy_to_vtkIdTypeArray(np.arange(n, dtype=np.int64), deep=True))
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(points)
        self.polydata.SetVerts(vertices)
        self.mapper.SetInputData(self.polydata)
        if n:
            lo, hi = self.display_xyz.min(axis=0), self.display_xyz.max(axis=0)
            self.outline.SetBounds(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            self.marker_radius = max(float(np.linalg.norm(hi.astype(np.float64) - lo)) * 0.003, 1e-12)
        self.outline_actor.SetVisibility(bool(n) and self.outline_check.isChecked())
        self.shown_label.setText(f"{n:,}")
        self.export_button.setEnabled(bool(n))
        self.update_colors()

    def update_colors(self, *_):
        if self.cloud is None or self.polydata is None:
            return
        mode = self.color_mode.currentIndex()
        self.mapper.ScalarVisibilityOn()
        if mode < 3:
            axis = [2, 0, 1][mode]
            values = numpy_to_vtk(np.ascontiguousarray(self.display_xyz[:, axis]), deep=True)
            values.SetName("XYZ"[axis])
            self.polydata.GetPointData().SetScalars(values)
            lo, hi = self.cloud.bounds[:, axis]
            if hi <= lo:
                hi = lo + 1
            self.lut.SetTableRange(float(lo), float(hi))
            self.mapper.SetLookupTable(self.lut)
            self.mapper.SetScalarRange(float(lo), float(hi))
            self.mapper.SetColorModeToMapScalars()
            self.scalar_bar.SetTitle("XYZ"[axis])
            self.scalar_bar.SetVisibility(bool(len(self.display_xyz)))
        elif mode == 3:
            self.polydata.GetPointData().SetScalars(numpy_to_vtk(self.display_rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR))
            self.mapper.SetColorModeToDirectScalars()
            self.scalar_bar.VisibilityOff()
        else:
            self.mapper.ScalarVisibilityOff()
            self.actor.GetProperty().SetColor(0.43, 0.91, 0.84)
            self.scalar_bar.VisibilityOff()
        self.polydata.Modified()
        self.render()

    def change_point_size(self, value):
        self.size_label.setText(f"{value} px")
        self.actor.GetProperty().SetPointSize(value)
        self.render()

    def change_projection(self, enabled):
        self.renderer.GetActiveCamera().SetParallelProjection(enabled)
        self.fit_camera()

    def change_outline(self, enabled):
        self.outline_actor.SetVisibility(enabled and bool(len(self.display_xyz)))
        self.render()

    def set_view(self, direction):
        if not len(self.display_xyz):
            return
        lo, hi = self.display_xyz.min(axis=0), self.display_xyz.max(axis=0)
        center = (lo + hi) / 2
        distance = max(float(np.linalg.norm(hi - lo)), 1)
        vector, up = {"iso": ((1, -1.4, 1.3), (0, 0, 1)),
                      "top": ((0, 0, 1), (0, 1, 0)),
                      "front": ((0, -1, 0), (0, 0, 1)),
                      "side": ((1, 0, 0), (0, 0, 1))}[direction]
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(*(center + np.array(vector) * distance))
        camera.SetViewUp(*up)
        self.fit_camera()

    def fit_camera(self):
        if self.polydata and len(self.display_xyz):
            self.renderer.ResetCamera(self.polydata.GetBounds())
            self.renderer.GetActiveCamera().Zoom(1.08)
            self.renderer.ResetCameraClippingRange()
            self.render()

    def render(self):
        if self.measure_actors and self.polydata is not None and len(self.display_xyz):
            cloud_bounds = np.array(self.polydata.GetBounds()).reshape(3, 2)
            measure_bounds = np.array(self.measure_renderer.ComputeVisiblePropBounds()).reshape(3, 2)
            if np.isfinite(measure_bounds).all() and (measure_bounds[:, 0] <= measure_bounds[:, 1]).all():
                bounds = np.column_stack((np.minimum(cloud_bounds[:, 0], measure_bounds[:, 0]),
                                          np.maximum(cloud_bounds[:, 1], measure_bounds[:, 1])))
                self.renderer.ResetCameraClippingRange(bounds.ravel())
        self.vtk_widget.GetRenderWindow().Render()

    def _left_down(self, *_):
        kind = "length" if self.interactor.GetShiftKey() else self.measure_mode
        if self.interactor.GetControlKey() or self.interactor.GetAltKey():
            kind = "navigate"
        self._press = (self.interactor.GetEventPosition(), kind)
        self._dragged = False
        self.style.OnLeftButtonDown()

    def _mouse_move(self, *_):
        if self._press and np.linalg.norm(np.array(self.interactor.GetEventPosition()) - self._press[0]) >= 4:
            self._dragged = True
        self.style.OnMouseMove()

    def _left_up(self, *_):
        self.style.OnLeftButtonUp()
        pos = self.interactor.GetEventPosition()
        if (self._press and self._press[1] != "navigate" and not self._dragged
                and np.linalg.norm(np.array(pos) - self._press[0]) < 4):
            self.pick_point(*pos, kind=self._press[1])
        self._press = None

    def set_measure_mode(self, mode):
        if mode not in self.measure_buttons:
            raise ValueError("Unknown measurement mode")
        self.measure_mode = mode
        self.measure_buttons[mode].setChecked(True)
        self.vtk_widget.setCursor(QtCore.Qt.ArrowCursor if mode == "navigate" else QtCore.Qt.CrossCursor)
        self.clear_measurement()

    def _measurement_hint(self):
        if self.measure_mode == "length":
            return "길이 · 두 점을 클릭하세요. 드래그로 회전할 수 있습니다."
        return "탐색 · Shift+클릭으로 길이 2점 선택 / 위 버튼으로 측정 도구 선택"

    def _invalidate_auto(self):
        self.auto_generation += 1
        if self.detector:
            self.detector.requestInterruption()
        self.auto_results = []
        self.auto_candidates.blockSignals(True)
        self.auto_candidates.clear()
        self.auto_candidates.blockSignals(False)

    def cancel_auto_detection(self):
        self._invalidate_auto()
        self.statusBar().showMessage("자동 원 검출을 취소했습니다.")
        self.measure_label.setText(self._measurement_hint())

    def start_auto_detection(self):
        if self.worker or self.detector or self.surface_detector:
            self.statusBar().showMessage("진행 중인 읽기/검출이 끝난 후 다시 시도하세요.")
            return
        if len(self.display_xyz) < 30:
            self.statusBar().showMessage("자동 검출에는 표시된 점이 최소 30개 필요합니다.")
            return
        self.clear_measurement()
        self.measure_mode = "navigate"
        self.measure_buttons["navigate"].setChecked(True)
        self.vtk_widget.setCursor(QtCore.Qt.ArrowCursor)
        self.measure_label.setText("자동 원 검출 중 · 현재 표본/Z 범위의 원형 윤곽을 찾습니다. Esc 또는 검출 취소로 중지")
        options = dict(tolerance=self.auto_tolerance.value(),
                       min_diameter=self.auto_min_diameter.value(),
                       max_diameter=self.auto_max_diameter.value())
        self.detector = CircleDetector(self.display_xyz, self.auto_generation, options)
        self.detector.progress.connect(self.auto_progress.setValue)
        self.detector.detected.connect(self._auto_detected)
        self.detector.failed.connect(self._auto_failed)
        self.detector.finished.connect(self._auto_finished)
        self.auto_progress.setValue(0)
        self.auto_progress.show()
        self.auto_cancel.show()
        self.auto_button.setEnabled(False)
        for spin in (self.auto_tolerance, self.auto_min_diameter, self.auto_max_diameter):
            spin.setEnabled(False)
        self.detector.start()

    def _auto_detected(self, packet):
        generation, results = packet
        if generation != self.auto_generation:
            return
        self.auto_results = results
        self.auto_candidates.blockSignals(True)
        self.auto_candidates.clear()
        for i, result in enumerate(results):
            self.auto_candidates.addItem(f"후보 {i + 1} · 지름 {result.circle.diameter:.6g} · 호 {result.arc_degrees:.0f}°")
        self.auto_candidates.setCurrentIndex(0)
        self.auto_candidates.blockSignals(False)
        self.show_auto_candidate(0)

    def _auto_failed(self, packet):
        generation, message = packet
        if generation == self.auto_generation:
            self.measure_label.setText("자동 검출: " + message)
            self.statusBar().showMessage(message)

    def _auto_finished(self):
        old = self.detector
        self.detector = None
        old.deleteLater()
        self.auto_progress.hide()
        self.auto_cancel.hide()
        self.auto_button.setEnabled(True)
        for spin in (self.auto_tolerance, self.auto_min_diameter, self.auto_max_diameter):
            spin.setEnabled(True)

    def show_auto_candidate(self, index):
        if not 0 <= index < len(self.auto_results):
            return
        result = self.auto_results[index]
        self.clear_measurement(render=False, cancel_auto=False)
        self.measure_kind = "auto"
        self.measure_complete = True
        self.circle_measurement = result.circle
        self._draw_circle(result.circle, result.support[0],
                          f"Auto circle {index + 1} (estimate)\nRMS = {result.rms:.6g}  Arc = {result.arc_degrees:.1f} deg")
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(result.support, deep=True))
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        glyphs = vtk.vtkVertexGlyphFilter()
        glyphs.SetInputData(data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphs.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(4)
        actor.GetProperty().SetColor(1, 0.5, 0.25)
        actor.GetProperty().LightingOff()
        self._add_measure_actor(actor)
        circle = result.circle
        self.measure_label.setText(
            f"자동 후보 {index + 1} · 지름 {circle.diameter:.8g} · 반지름 {circle.radius:.8g}\n"
            f"중심 O {self._format_xyz(circle.center)} · 원본 좌표 단위 (단위 미상)")
        self.measure_coords.setText(
            f"3D RMS 잔차 {result.rms:.6g} · 호 범위 {result.arc_degrees:.1f}° · 허용 오차 {result.tolerance:.6g}\n"
            f"분석 {result.sample_count:,}점 → 윤곽 {result.boundary_count:,}점 / 지지점 {len(result.support):,}개 (주황색)\n"
            "원형 윤곽의 추정값입니다. 표본·Z 범위·노이즈의 영향을 받습니다.")
        self.statusBar().showMessage(f"자동 원 후보 {len(self.auto_results)}개 · 오른쪽 목록에서 후보 선택 가능")
        self.render()

    def pick_point(self, x, y, kind=None):
        if not len(self.display_xyz):
            self.statusBar().showMessage("표시 중인 점이 없습니다. 파일과 Z 범위를 확인하세요.")
            return
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.006)
        picker.PickFromListOn()
        picker.AddPickList(self.actor)
        # Measurement overlays can extend the camera clipping range. Use only
        # cloud bounds for the pick ray so a repeated click selects the same point.
        camera = self.renderer.GetActiveCamera()
        clipping = camera.GetClippingRange()
        self.renderer.ResetCameraClippingRange(self.polydata.GetBounds())
        try:
            picker.Pick(x, y, 0, self.renderer)
        finally:
            camera.SetClippingRange(*clipping)
        idx = picker.GetPointId()
        if idx < 0:
            self.statusBar().showMessage("점을 선택하지 못했습니다. 확대 후 표시된 점 위를 클릭하세요. (탐색 모드: Shift+클릭)")
            return
        self.add_measurement_point(self.display_xyz[idx], kind or "length")

    def _invalidate_surface(self):
        self.surface_generation += 1
        if self.surface_detector:
            self.surface_detector.requestInterruption()
        self.surface_results = []
        self.surface_candidates.blockSignals(True)
        self.surface_candidates.clear()
        self.surface_candidates.blockSignals(False)

    def cancel_surface_detection(self):
        self._invalidate_surface()
        self.measure_label.setText(self._measurement_hint())
        self.statusBar().showMessage("평면 검출을 취소했습니다.")

    def start_surface_detection(self):
        if self.worker or self.detector or self.surface_detector:
            self.statusBar().showMessage("진행 중인 읽기/검출이 끝난 후 다시 시도하세요.")
            return
        if len(self.display_xyz) < 30:
            self.statusBar().showMessage("평면 검출에는 표시된 점이 최소 30개 필요합니다.")
            return
        self.clear_measurement()
        self.measure_mode = "navigate"
        self.measure_buttons["navigate"].setChecked(True)
        self.vtk_widget.setCursor(QtCore.Qt.ArrowCursor)
        self.measure_label.setText("주요 평면 검출 중 · 현재 표본/Z 범위에서 찾습니다. Esc 또는 평면 검출 취소로 중지")
        options = dict(tolerance=self.surface_tolerance.value(),
                       min_fraction=self.surface_min_percent.value() / 100,
                       max_planes=self.surface_max_planes.value())
        self.surface_detector = PlaneDetector(self.display_xyz, self.surface_generation, options)
        self.surface_detector.progress.connect(self.surface_progress.setValue)
        self.surface_detector.detected.connect(self._surface_detected)
        self.surface_detector.failed.connect(self._surface_failed)
        self.surface_detector.finished.connect(self._surface_finished)
        self.surface_progress.setValue(0)
        self.surface_progress.show()
        self.surface_cancel.show()
        self.surface_button.setEnabled(False)
        for spin in (self.surface_tolerance, self.surface_min_percent, self.surface_max_planes):
            spin.setEnabled(False)
        self.surface_detector.start()

    def _surface_detected(self, packet):
        generation, results = packet
        if generation != self.surface_generation:
            return
        self.surface_results = results
        self.surface_candidates.blockSignals(True)
        self.surface_candidates.clear()
        for i, result in enumerate(results):
            self.surface_candidates.addItem(f"평면 {i + 1} · {len(result.indices):,}점 ({result.fraction:.1%})")
        self.surface_candidates.setCurrentIndex(0)
        self.surface_candidates.blockSignals(False)
        self.show_surface_candidate(0)

    def _surface_failed(self, packet):
        generation, message = packet
        if generation == self.surface_generation:
            self.measure_label.setText("평면 검출: " + message)
            self.statusBar().showMessage(message)

    def _surface_finished(self):
        old = self.surface_detector
        self.surface_detector = None
        old.deleteLater()
        self.surface_progress.hide()
        self.surface_cancel.hide()
        self.surface_button.setEnabled(True)
        for spin in (self.surface_tolerance, self.surface_min_percent, self.surface_max_planes):
            spin.setEnabled(True)

    def show_surface_candidate(self, index):
        if not 0 <= index < len(self.surface_results):
            return
        result = self.surface_results[index]
        self.clear_measurement(render=False, cancel_surface=False)
        self.selected_plane = result
        self.measure_kind = "surface"
        self.measure_complete = True
        # Membership/statistics cover all displayed points; bound highlight cost.
        ids = result.indices
        if len(ids) > 200_000:
            ids = ids[np.linspace(0, len(ids) - 1, 200_000).astype(int)]
        self._add_point_highlight(self.display_xyz[ids], (0.15, 1.0, 0.85))
        lo, hi = result.bounds_uv
        uv = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]], lo])
        corners = result.center + uv @ result.basis.T
        self._add_polyline(corners, (1, 0.8, 0.25))
        self._add_marker(result.center, "C", (1, 0.8, 0.25))
        arrow_length = float(np.linalg.norm(hi - lo)) * 0.18
        tip = result.center + arrow_length * result.normal
        shoulder = tip - arrow_length * 0.22 * result.normal
        width = arrow_length * 0.09 * result.basis[:, 0]
        self._add_polyline(np.array([result.center, tip]), (1, 0.5, 0.85))
        self._add_polyline(np.array([shoulder + width, tip, shoulder - width]), (1, 0.5, 0.85))
        self._add_measurement_text(
            f"Plane {index + 1} (estimate) | {len(result.indices):,} points\n"
            f"RMS = {result.rms:.6g}   Max = {result.max_deviation:.6g}\n"
            f"n = {self._format_xyz(result.normal)}\nSource units (unknown)")
        self.measure_label.setText(
            f"평면 {index + 1} · {len(result.indices):,}점 ({result.fraction:.1%}) · RMS {result.rms:.6g}\n"
            f"기준점 C {self._format_xyz(result.center)}")
        self.measure_coords.setText(
            f"법선 n {self._format_xyz(result.normal)}\n"
            f"최대 편차 {result.max_deviation:.5g} · 잔차 폭 {result.residual_span:.5g} · 허용 거리 {result.tolerance:.5g}\n"
            "원본 단위 (미상) · 노란 선: 투영 범위 / 분홍 화살표: 법선")
        self.statusBar().showMessage(
            f"주요 평면 {len(self.surface_results)}개 · 분석 {result.sample_count:,}점 / 분류 {result.finite_count:,}점 · 후보를 선택하세요.")
        self.render()

    def add_measurement_point(self, point, kind):
        """Commit a picked point only after validation, preserving earlier points on error."""
        if kind != "length":
            raise ValueError("Unknown measurement tool")
        point = np.asarray(point, dtype=np.float64).copy()
        starting = self.measure_complete or kind != self.measure_kind
        candidates = ([] if starting else self.measure_points) + [point]
        try:
            validate_measurement_points(candidates)
        except ValueError as exc:
            message = f"선택 거부: {exc} (기존 선택 유지 · Esc로 초기화)"
            self.measure_label.setText(message)
            self.statusBar().showMessage(message)
            return False
        if self.detector or self.auto_results:
            self._invalidate_auto()
        if self.surface_detector or self.surface_results:
            self._invalidate_surface()
        if starting:
            self.clear_measurement(render=False)
        self.measure_kind = kind
        self.measure_points.append(point)
        point_colors = [(1.0, 0.45, 0.25), (1.0, 0.8, 0.3)]
        color = point_colors[len(self.measure_points) - 1]
        self._add_marker(point, f"P{len(self.measure_points)}", color)
        self.measure_coords.setText("\n".join(
            f"P{i + 1}  {self._format_xyz(p)}" for i, p in enumerate(self.measure_points)
        ))
        units = "원본 좌표 단위 (단위 미상) · 표시 표본점 기준"
        if kind == "length" and len(candidates) == 2:
            a, b = self.measure_points
            delta = b - a
            distance = float(np.linalg.norm(delta))
            self._add_polyline(np.array([a, b]), (1, 0.75, 0.4))
            self._add_measurement_text(f"Length = {distance:.8g}\nSource units (unknown)")
            self.measure_label.setText(
                f"길이 {distance:.8g} · ΔX {delta[0]:.8g}  ΔY {delta[1]:.8g}  ΔZ {delta[2]:.8g}\n{units} · 다음 클릭은 새 측정")
            self.measure_complete = True
        else:
            gesture = "클릭" if self.measure_mode == "length" else "Shift+클릭"
            self.measure_label.setText(f"길이 · 1/2점 선택 · 다음 점을 {gesture}하세요.\n{units}")
        self.statusBar().showMessage("측정 완료 · 다음 점 선택으로 새 측정 / 측정 초기화 또는 Esc" if self.measure_complete
                                     else f"P{len(candidates)} 선택됨 · 다음 점을 선택하세요.")
        self.render()
        return True

    @staticmethod
    def _format_xyz(point):
        return "(" + ", ".join(f"{x:.8g}" for x in point) + ")"

    def _add_measure_actor(self, actor):
        # Annotations must never intercept point picking.
        actor.PickableOff()
        self.measure_renderer.AddViewProp(actor)
        self.measure_actors.append(actor)

    def _add_point_highlight(self, xyz, color):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(xyz), deep=True))
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        glyphs = vtk.vtkVertexGlyphFilter()
        glyphs.SetInputData(data)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyphs.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(4)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().LightingOff()
        self._add_measure_actor(actor)

    def _add_marker(self, point, name, color):
        sphere = vtk.vtkSphereSource()
        sphere.SetOutputPointsPrecision(vtk.vtkAlgorithm.DOUBLE_PRECISION)
        sphere.SetCenter(*point)
        sphere.SetRadius(self.marker_radius)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().LightingOff()
        self._add_measure_actor(actor)
        caption = vtk.vtkBillboardTextActor3D()
        caption.SetInput(name)
        caption.SetPosition(*point)
        caption.SetDisplayOffset(10, 10)
        caption.GetTextProperty().SetFontSize(16)
        caption.GetTextProperty().SetColor(*color)
        caption.GetTextProperty().BoldOn()
        caption.GetTextProperty().SetBackgroundColor(0.025, 0.05, 0.09)
        caption.GetTextProperty().SetBackgroundOpacity(0.7)
        self._add_measure_actor(caption)

    def _add_polyline(self, xyz, color):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(xyz, dtype=np.float64), deep=True))
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(xyz))
        for i in range(len(xyz)):
            lines.InsertCellPoint(i)
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(3)
        actor.GetProperty().LightingOff()
        self._add_measure_actor(actor)

    def _add_measurement_text(self, text):
        overlay = vtk.vtkTextActor()
        overlay.SetInput(text)
        overlay.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        overlay.SetPosition(0.025, 0.97)
        prop = overlay.GetTextProperty()
        prop.SetVerticalJustificationToTop()
        prop.SetFontSize(15)
        prop.SetColor(0.85, 1.0, 0.96)
        prop.SetBackgroundColor(0.025, 0.05, 0.09)
        prop.SetBackgroundOpacity(0.8)
        self._add_measure_actor(overlay)

    def _draw_circle(self, circle, start_point, title="Automatic circle (estimate)"):
        radial = start_point - circle.center
        radial = radial - np.dot(radial, circle.normal) * circle.normal
        radial /= np.linalg.norm(radial)
        tangent = np.cross(circle.normal, radial)
        angles = np.linspace(0, 2 * np.pi, 361)
        xyz = circle.center + circle.radius * (
            np.cos(angles)[:, None] * radial + np.sin(angles)[:, None] * tangent)
        self._add_polyline(xyz, (0.3, 1.0, 0.85))
        self._add_marker(circle.center, "O", (0.3, 1.0, 0.85))
        self._add_polyline(np.array([circle.center + circle.radius * radial,
                                    circle.center - circle.radius * radial]), (1, 0.8, 0.3))
        self._add_measurement_text(
            f"{title}\nD = {circle.diameter:.8g}   R = {circle.radius:.8g}\n"
            f"O = {self._format_xyz(circle.center)}\nSource units (unknown)")

    def clear_measurement(self, render=True, cancel_auto=True, cancel_surface=True):
        if cancel_auto:
            self._invalidate_auto()
        if cancel_surface:
            self._invalidate_surface()
        for actor in self.measure_actors:
            self.measure_renderer.RemoveViewProp(actor)
        self.measure_points.clear()
        self.measure_actors.clear()
        self.circle_measurement = None
        self.selected_plane = None
        self.measure_complete = False
        self.measure_kind = "length"
        self.measure_label.setText(self._measurement_hint())
        self.measure_coords.setText("선택점 없음 · 원본 좌표 단위 (단위 미상) · 현재 표시 표본점 기준")
        if render:
            self.render()

    def export_ply(self):
        self.export_cloud(default_format="ply")

    def export_cloud(self, _checked=False, default_format=None):
        if not len(self.display_xyz):
            return
        preferred = default_format or ("pcd" if self.cloud.path.suffix.lower() == ".pcd" else "ply")
        filters = "PCD (*.pcd);;PLY (*.ply)" if preferred == "pcd" else "PLY (*.ply);;PCD (*.pcd)"
        path, chosen = QtWidgets.QFileDialog.getSaveFileName(
            self, "표시 중인 점 저장 (XYZ/RGB)",
            str(Path.cwd() / (self.cloud.path.stem + "_display." + preferred)), filters)
        if not path:
            return
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pcd" if chosen.startswith("PCD") else ".ply")
        if path.suffix.lower() not in (".ply", ".pcd"):
            QtWidgets.QMessageBox.warning(self, "저장 형식", ".ply 또는 .pcd 확장자로 저장해 주세요.")
            return
        try:
            writer = write_pcd if path.suffix.lower() == ".pcd" else write_ply
            writer(path, self.display_xyz, self.display_rgb)
            self.statusBar().showMessage(f"{len(self.display_xyz):,}점 저장 완료 (현재 표본·Z 범위, XYZ/RGB): {path}")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "저장 실패", str(exc))

    def save_png(self, path=None):
        if not isinstance(path, (str, Path)):
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "현재 뷰 저장", str(Path.cwd() / "point_cloud.png"), "PNG (*.png)")
        if not path:
            return
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".png")
        if path.suffix.lower() != ".png":
            QtWidgets.QMessageBox.warning(self, "저장 형식", ".png 확장자로 저장해 주세요.")
            return
        self.render()
        image = vtk.vtkWindowToImageFilter()
        image.SetInput(self.vtk_widget.GetRenderWindow())
        image.ReadFrontBufferOff()
        image.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(os.fsencode(path))
        writer.SetInputConnection(image.GetOutputPort())
        writer.Write()
        if writer.GetErrorCode():
            QtWidgets.QMessageBox.warning(self, "저장 실패", "PNG 파일을 저장하지 못했습니다.")
        else:
            self.statusBar().showMessage(f"PNG 저장 완료: {path}")

    def dragEnterEvent(self, event):
        if not self.worker and event.mimeData().hasUrls() and any(Path(u.toLocalFile()).suffix.lower() in CLOUD_EXTENSIONS for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in CLOUD_EXTENSIONS:
                self.open_cloud(path)
                event.acceptProposedAction()
                break

    def closeEvent(self, event):
        if self.surface_detector:
            self.surface_detector.requestInterruption()
            self.surface_detector.wait()
        if self.detector:
            self.detector.requestInterruption()
            self.detector.wait()
        if self.worker:
            self.worker.requestInterruption()
            self.worker.wait()
        self.orientation.SetEnabled(0)
        self.vtk_widget.Finalize()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} BIN / PLY / PCD 포인트 클라우드 뷰어")
    parser.add_argument("file", nargs="?", help="처음 열 BIN / PLY / PCD 파일")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="포인트 클라우드 검색 폴더 (기본: 앱 폴더 내 data)")
    args = parser.parse_args()
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        parser.error("그래픽 데스크톱 세션에서 실행하세요 (DISPLAY / WAYLAND_DISPLAY 없음).")
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setFont(QtGui.QFont("Noto Sans CJK KR", 9))
    app.setStyleSheet(STYLE)
    window = Viewer(args.root, args.file)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
