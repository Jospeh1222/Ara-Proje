import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QFrame, QProgressBar,
    QScrollArea, QToolTip, QSizePolicy,
)
from PyQt6.QtCore import (
    Qt, QUrl, QSize, QRectF, QTimer, QPointF, QThread, pyqtSignal, QEvent,
)
from PyQt6.QtGui import (
    QPalette, QColor, QPainter, QPainterPath, QPen, QBrush, QIcon, QPixmap,
    QConicalGradient, QFont, QCursor,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

import feature_extractor as fx
import prediction as pred


COLORS = {
    "bg_primary":     "#1C1C1E",
    "bg_secondary":   "#2C2C2E",
    "bg_tertiary":    "#3A3A3C",
    "text_primary":   "#FFFFFF",
    "text_secondary": "#A0A0A0",
    "text_tertiary":  "#6E6E73",
    "accent":         "#C8102E",
    "accent_hover":   "#E01F3D",
    "accent_disabled":"#5A1419",
    "border":         "#3A3A3C",
    "vinyl_black":    "#0A0A0A",
    "vinyl_groove":   "#1F1F1F",
    "vinyl_label":    "#C8102E",
    "progress_bg":    "#3A3A3C",
    "card_inner":     "#3A3A3C",
    "featured_bg":    "#2C2C2E",
    "featured_border":"#C8102E",
}

FONT_FAMILY = '-apple-system, "SF Pro Display", "Segoe UI", Inter, system-ui, sans-serif'

QSS_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {COLORS['bg_primary']};
    color: {COLORS['text_primary']};
    font-family: {FONT_FAMILY};
}}

QLabel {{
    background-color: transparent;
    color: {COLORS['text_primary']};
}}

QFrame#card {{
    background-color: {COLORS['bg_secondary']};
    border-radius: 18px;
    border: 1px solid {COLORS['border']};
}}

QFrame#expCardFeatured {{
    background-color: {COLORS['featured_bg']};
    border-radius: 14px;
    border: 1px solid {COLORS['featured_border']};
}}

QFrame#expCardCollapsible {{
    background-color: {COLORS['card_inner']};
    border-radius: 12px;
    border: none;
}}

QWidget#expCardHeader {{
    background-color: transparent;
}}
QWidget#expCardHeader:hover {{
    background-color: rgba(255,255,255,0.04);
    border-radius: 8px;
}}

QLabel#title {{
    color: {COLORS['text_primary']};
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
}}

QLabel#subtitle {{
    color: {COLORS['text_secondary']};
    font-size: 13px;
    font-weight: 400;
}}

QLabel#songTitle {{
    color: {COLORS['text_primary']};
    font-size: 16px;
    font-weight: 600;
}}

QLabel#songMeta {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
}}

QLabel#timeLabel {{
    color: {COLORS['text_secondary']};
    font-size: 11px;
}}

QLabel#emptyState {{
    color: {COLORS['text_tertiary']};
    font-size: 14px;
    font-style: italic;
}}

QLabel#placeholderText {{
    color: {COLORS['text_tertiary']};
    font-size: 13px;
}}

QLabel#progressText {{
    color: {COLORS['text_secondary']};
    font-size: 12px;
}}

QLabel#statusText {{
    color: {COLORS['text_primary']};
    font-size: 14px;
    font-weight: 500;
}}

QLabel#expLabel {{
    color: {COLORS['text_primary']};
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.2px;
}}

QLabel#expLabelFeatured {{
    color: {COLORS['accent']};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}}

QLabel#expTitle {{
    color: {COLORS['text_primary']};
    font-size: 17px;
    font-weight: 700;
}}

QLabel#modelName {{
    color: {COLORS['text_secondary']};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}

QLabel#unavailableText {{
    color: {COLORS['text_tertiary']};
    font-size: 12px;
    font-style: italic;
}}

QLabel#chevron {{
    color: {COLORS['text_secondary']};
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#primaryButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
QPushButton#primaryButton:disabled {{
    background-color: {COLORS['accent_disabled']};
    color: {COLORS['text_tertiary']};
}}

QPushButton#secondaryButtonText {{
    background-color: transparent;
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
}}
QPushButton#secondaryButtonText:hover {{
    background-color: {COLORS['bg_tertiary']};
}}
QPushButton#secondaryButtonText:disabled {{
    color: {COLORS['text_tertiary']};
    border-color: {COLORS['bg_tertiary']};
}}

QPushButton#playButton {{
    background-color: {COLORS['accent']};
    color: white;
    border: none;
    border-radius: 22px;
    min-width: 76px;
    max-width: 76px;
    min-height: 76px;
    max-height: 76px;
}}
QPushButton#playButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
QPushButton#playButton:disabled {{
    background-color: {COLORS['accent_disabled']};
}}

QPushButton#secondaryButton {{
    background-color: transparent;
    border: none;
    border-radius: 18px;
    min-width: 56px;
    max-width: 56px;
    min-height: 56px;
    max-height: 56px;
}}
QPushButton#secondaryButton:hover {{
    background-color: {COLORS['bg_tertiary']};
}}

QSlider {{
    background-color: transparent;
}}
QSlider::groove:horizontal {{
    background: {COLORS['bg_tertiary']};
    height: 5px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {COLORS['text_primary']};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS['accent']};
}}
QSlider::sub-page:horizontal {{
    background: {COLORS['accent']};
    height: 5px;
    border-radius: 2px;
}}

QProgressBar {{
    background-color: {COLORS['progress_bg']};
    border: none;
    border-radius: 2px;
    height: 4px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {COLORS['accent']};
    border-radius: 2px;
}}

QScrollArea {{
    background-color: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['bg_tertiary']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QToolTip {{
    background-color: {COLORS['bg_secondary']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12px;
}}
"""


def format_time(ms: int) -> str:
    if ms <= 0:
        return "0:00"
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"


#**************************************************************************
#MAKAM ROW - Spotify tarzi bar + text
#***************************************************************

class MakamRow(QWidget):

    SIZE_PRESETS = {
        'big':    {'height': 52, 'font_size': 22, 'weight': 700,
                   'left_pad': 16, 'bar_alpha_max': 90, 'radius': 10},
        'medium': {'height': 40, 'font_size': 16, 'weight': 700,
                   'left_pad': 12, 'bar_alpha_max': 80, 'radius': 7},
        'small':  {'height': 26, 'font_size': 12, 'weight': 500,
                   'left_pad': 12, 'bar_alpha_max': 45, 'radius': 5},
    }

    def __init__(self, label: str, prob: float, mode: str = 'medium', parent=None):
        super().__init__(parent)
        self.label_text = label
        self.prob = max(0.0, min(1.0, prob))
        self.mode = mode

        cfg = self.SIZE_PRESETS[mode]
        self.setFixedHeight(cfg['height'])
        self._cfg = cfg

        layout = QHBoxLayout(self)
        layout.setContentsMargins(cfg['left_pad'], 0, cfg['left_pad'], 0)
        layout.setSpacing(8)

        self.label_widget = QLabel(label)
        if mode == 'small':
            self.label_widget.setStyleSheet(
                f"color: {COLORS['text_secondary']}; "
                f"font-size: {cfg['font_size']}px; "
                f"font-weight: {cfg['weight']};"
            )
        else:
            self.label_widget.setStyleSheet(
                f"color: {COLORS['text_primary']}; "
                f"font-size: {cfg['font_size']}px; "
                f"font-weight: {cfg['weight']};"
            )
        layout.addWidget(self.label_widget)
        layout.addStretch()

        self.prob_widget = QLabel(f"%{prob * 100:.0f}")
        if mode == 'small':
            self.prob_widget.setStyleSheet(
                f"color: {COLORS['text_secondary']}; "
                f"font-size: {cfg['font_size']}px; "
                f"font-weight: {cfg['weight']};"
            )
        else:
            self.prob_widget.setStyleSheet(
                f"color: {COLORS['accent']}; "
                f"font-size: {cfg['font_size']}px; "
                f"font-weight: {cfg['weight']};"
            )
        layout.addWidget(self.prob_widget)


    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        bar_width = int(rect.width() * self.prob)
        if bar_width < 4:
            bar_width = 4

        bar_color = QColor(COLORS['accent'])
        bar_color.setAlpha(self._cfg['bar_alpha_max'])

        p.setBrush(QBrush(bar_color))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(
            QRectF(0, 0, bar_width, rect.height()),
            self._cfg['radius'], self._cfg['radius']
        )


#***************************************************************************
#INFO ICON WIDGET
#****************************************************************

class InfoIcon(QLabel):

    def __init__(self, tooltip_text: str, parent=None):
        super().__init__("i", parent)
        self.setFixedSize(20, 20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.tooltip_text = tooltip_text
        self.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            background-color: {COLORS['bg_tertiary']};
            border-radius: 10px;
            font-size: 12px;
            font-weight: 700;
            font-style: italic;
        """)
        self.setMouseTracking(True)

    def enterEvent(self, event):
        global_pos = self.mapToGlobal(self.rect().bottomRight())
        QToolTip.showText(global_pos, self.tooltip_text, self)
        self.setStyleSheet(f"""
            color: {COLORS['text_primary']};
            background-color: {COLORS['accent']};
            border-radius: 10px;
            font-size: 12px;
            font-weight: 700;
            font-style: italic;
        """)
        super().enterEvent(event)

    def leaveEvent(self, event):
        QToolTip.hideText()
        self.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            background-color: {COLORS['bg_tertiary']};
            border-radius: 10px;
            font-size: 12px;
            font-weight: 700;
            font-style: italic;
        """)
        super().leaveEvent(event)


#************************************************************************
#FEATURE EXTRACTION + PREDICTION THREADS
#***************************************************************

class FeatureExtractionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    status_updated = pyqtSignal(str)
    finished_success = pyqtSignal(list)
    finished_error = pyqtSignal(str)

    def __init__(self, audio_path: str, parent=None):
        super().__init__(parent)
        self.audio_path = audio_path

    def run(self):
        try:
            self.status_updated.emit("Ses yükleniyor...")
            try:
                y, sr = fx.load_audio(self.audio_path)
            except Exception as e:
                self.finished_error.emit(f"Ses yüklenemedi: {e}")
                return

            duration_sec = len(y) / sr
            n_segments = fx.count_segments(duration_sec)

            if n_segments == 0:
                self.finished_error.emit(
                    f"Şarkı çok kısa (en az {fx.SEGMENT_SEC:.0f} saniye olmalı)"
                )
                return

            self.status_updated.emit(f"{n_segments} segment çıkarılıyor")

            segment_images = []
            for idx, seg_y in fx.split_audio_to_segments(y, sr):
                rgb = fx.extract_segment_features(seg_y, sr)
                segment_images.append(rgb)
                self.progress_updated.emit(idx + 1, n_segments)

            self.finished_success.emit(segment_images)
        except Exception as e:
            self.finished_error.emit(f"Hata: {e}")


class PredictionThread(QThread):
    progress_updated = pyqtSignal(int, int)
    status_updated = pyqtSignal(str)
    finished_success = pyqtSignal(list)
    finished_error = pyqtSignal(str)

    def __init__(self, segment_images, registry, parent=None):
        super().__init__(parent)
        self.segment_images = segment_images
        self.registry = registry

    def run(self):
        try:
            results = []
            total = len(self.registry)

            for i, entry in enumerate(self.registry):
                self.progress_updated.emit(i, total)
                self.status_updated.emit(f"Tahmin: {entry.exp_label} ({entry.model_type})")

                if not entry.is_loaded:
                    entry.load()

                if not entry.is_available:
                    results.append((entry, None))
                    continue

                try:
                    result = entry.predict(self.segment_images)
                    results.append((entry, result))
                except Exception as e:
                    entry.error_message = str(e)
                    entry.is_available = False
                    results.append((entry, None))

            self.progress_updated.emit(total, total)
            self.finished_success.emit(results)
        except Exception as e:
            self.finished_error.emit(f"Tahmin hatası: {e}")


#*****************************************************************************
#EXPERIMENT CARD
#********************************************************************

class ExperimentCard(QFrame):

    def __init__(self, exp_short: str, exp_label: str, results: list,
                 is_featured: bool, class_list: list, parent=None):
        super().__init__(parent)

        self.is_featured = is_featured
        self.expanded = is_featured
        self.exp_label = exp_label
        self.class_list = class_list

        self.setObjectName("expCardFeatured" if is_featured else "expCardCollapsible")
        #Grid'de kartlar genisleyebilsin
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        #HEADER
        self.header = QWidget()
        self.header.setObjectName("expCardHeader")
        if not is_featured:
            self.header.setCursor(Qt.CursorShape.PointingHandCursor)

        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(10)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(2)

        if is_featured:
            badge = QLabel("◆ ANA TAHMİN")
            badge.setObjectName("expLabelFeatured")
            title_box.addWidget(badge)

            title_lbl = QLabel(exp_label)
            title_lbl.setObjectName("expTitle")
            title_box.addWidget(title_lbl)
        else:
            title_lbl = QLabel(exp_label)
            title_lbl.setObjectName("expLabel")
            title_box.addWidget(title_lbl)

        header_layout.addLayout(title_box)
        header_layout.addStretch()

        info_icon = InfoIcon(self._build_class_tooltip())
        header_layout.addWidget(info_icon)

        if not is_featured:
            self.chevron = QLabel("▾")
            self.chevron.setObjectName("chevron")
            header_layout.addWidget(self.chevron)
            self.header.mousePressEvent = self._on_header_click

        outer.addWidget(self.header)

        #BODY
        self.body = QWidget()
        body_layout = QVBoxLayout(self.body)
        body_layout.setContentsMargins(16, 0, 16, 14)
        body_layout.setSpacing(0)

        for i, (entry, result) in enumerate(results):
            if i > 0:
                spacer = QWidget()
                spacer.setFixedHeight(10)
                body_layout.addWidget(spacer)

                divider = QFrame()
                divider.setFixedHeight(1)
                divider.setStyleSheet(
                    f"background-color: {COLORS['border']}; border: none;"
                )
                body_layout.addWidget(divider)

                spacer2 = QWidget()
                spacer2.setFixedHeight(10)
                body_layout.addWidget(spacer2)

            self._add_model_section(body_layout, entry, result, is_featured)

        outer.addWidget(self.body)

        if not is_featured:
            self.body.setVisible(False)


    def _build_class_tooltip(self) -> str:
        if not self.class_list:
            return f"{self.exp_label} - sınıf listesi yüklenemedi"
        lines = [f"{self.exp_label}", f"{len(self.class_list)} sınıf:", ""]
        for cls in self.class_list:
            lines.append(f"  •  {cls}")
        return "\n".join(lines)


    def _add_model_section(self, layout, entry, result, is_featured: bool):
        name_label = QLabel(entry.display_name)
        name_label.setObjectName("modelName")
        layout.addWidget(name_label)

        layout.addSpacing(6)

        if result is None:
            err = entry.error_message or "Yüklenemedi"
            msg = QLabel(f"⚠ {err[:80]}")
            msg.setObjectName("unavailableText")
            msg.setWordWrap(True)
            layout.addWidget(msg)
            return

        top1 = result['top1']
        top1_mode = 'big' if is_featured else 'medium'
        top1_row = MakamRow(top1['label'], top1['prob'], mode=top1_mode)
        layout.addWidget(top1_row)

        if len(result['top3']) > 1:
            layout.addSpacing(3)
            for entry_top in result['top3'][1:]:
                run_row = MakamRow(entry_top['label'], entry_top['prob'], mode='small')
                layout.addWidget(run_row)


    def _on_header_click(self, event):
        self.expanded = not self.expanded
        self.body.setVisible(self.expanded)
        if hasattr(self, 'chevron'):
            self.chevron.setText("▴" if self.expanded else "▾")


#*******************************************************
#VINYL WIDGET
#***************************************************************************

class VinylWidget(QWidget):
    def __init__(self, size: int = 240, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self._rotation = 0.0
        self._has_track = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_spinning(self, spinning: bool):
        if spinning:
            if not self._timer.isActive():
                self._timer.start(50)
        else:
            self._timer.stop()

    def set_has_track(self, has: bool):
        self._has_track = has
        self.update()

    def _tick(self):
        self._rotation = (self._rotation + 1.8) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        s = self._size
        cx, cy = s / 2, s / 2
        outer_r = s / 2 - 4
        label_r = outer_r * 0.32
        hole_r = outer_r * 0.04

        for i in range(3):
            p.setBrush(QBrush(QColor(0, 0, 0, 30 - i * 10)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(cx, cy + i + 1), outer_r, outer_r)

        p.setBrush(QBrush(QColor(COLORS['vinyl_black'])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), outer_r, outer_r)

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(COLORS['vinyl_groove']), 1))
        for i in range(8):
            r = outer_r * (0.42 + i * 0.07)
            p.drawEllipse(QPointF(cx, cy), r, r)

        p.save()
        p.translate(cx, cy)
        p.rotate(self._rotation)
        gradient = QConicalGradient(0, 0, 0)
        gradient.setColorAt(0.0,   QColor(255, 255, 255, 0))
        gradient.setColorAt(0.15,  QColor(255, 255, 255, 35))
        gradient.setColorAt(0.25,  QColor(255, 255, 255, 0))
        gradient.setColorAt(0.5,   QColor(255, 255, 255, 0))
        gradient.setColorAt(0.65,  QColor(255, 255, 255, 35))
        gradient.setColorAt(0.75,  QColor(255, 255, 255, 0))
        gradient.setColorAt(1.0,   QColor(255, 255, 255, 0))
        p.setBrush(QBrush(gradient))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(0, 0), outer_r - 2, outer_r - 2)
        p.restore()

        p.save()
        p.translate(cx, cy)
        p.rotate(self._rotation)
        p.setBrush(QBrush(QColor(COLORS['vinyl_label'])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(0, 0), label_r, label_r)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 50), 1))
        p.drawEllipse(QPointF(0, 0), label_r * 0.7, label_r * 0.7)
        p.setPen(QPen(QColor(255, 255, 255, 35), 0.8))
        for off in [-label_r * 0.25, -label_r * 0.1, label_r * 0.1, label_r * 0.25]:
            p.drawLine(QPointF(-label_r * 0.55, off), QPointF(label_r * 0.55, off))

        if not self._has_track:
            p.setBrush(QBrush(QColor(255, 255, 255, 180)))
            p.setPen(Qt.PenStyle.NoPen)
            head_r = label_r * 0.18
            p.drawEllipse(QPointF(-label_r * 0.1, label_r * 0.15), head_r, head_r)
            p.drawRoundedRect(
                QRectF(label_r * 0.05, -label_r * 0.3, label_r * 0.05, label_r * 0.45),
                1, 1
            )
        p.restore()

        p.setBrush(QBrush(QColor(COLORS['bg_primary'])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(cx, cy), hole_r, hole_r)
        p.end()


#*******************************************************************************
#IKONLAR
#***************************************************************

def _make_icon(size: int, color: str, draw_fn) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(QColor(color)))
    painter.setPen(Qt.PenStyle.NoPen)
    draw_fn(painter, size)
    painter.end()
    return QIcon(pixmap)


def play_icon(size: int = 24, color: str = "white") -> QIcon:
    def draw(p, s):
        path = QPainterPath()
        m = s * 0.28
        path.moveTo(m, m); path.lineTo(m, s - m); path.lineTo(s - m, s / 2); path.closeSubpath()
        p.drawPath(path)
    return _make_icon(size, color, draw)


def pause_icon(size: int = 24, color: str = "white") -> QIcon:
    def draw(p, s):
        bw, bh, gap = s * 0.16, s * 0.5, s * 0.14
        y = (s - bh) / 2
        x1 = s / 2 - gap / 2 - bw
        x2 = s / 2 + gap / 2
        p.drawRoundedRect(QRectF(x1, y, bw, bh), 2, 2)
        p.drawRoundedRect(QRectF(x2, y, bw, bh), 2, 2)
    return _make_icon(size, color, draw)


def skip_back_icon(size: int = 24, color: str = "white") -> QIcon:
    def draw(p, s):
        m = s * 0.28
        bw = s * 0.07
        p.drawRoundedRect(QRectF(m, m, bw, s - 2 * m), 1, 1)
        mx = s / 2
        for x1, x2 in [(mx, m + bw + 2), (s - m, mx)]:
            path = QPainterPath()
            path.moveTo(x1, m); path.lineTo(x2, s / 2); path.lineTo(x1, s - m); path.closeSubpath()
            p.drawPath(path)
    return _make_icon(size, color, draw)


def skip_forward_icon(size: int = 24, color: str = "white") -> QIcon:
    def draw(p, s):
        m = s * 0.28
        bw = s * 0.07
        p.drawRoundedRect(QRectF(s - m - bw, m, bw, s - 2 * m), 1, 1)
        mx = s / 2
        for x1, x2 in [(m, mx), (mx, s - m - bw - 2)]:
            path = QPainterPath()
            path.moveTo(x1, m); path.lineTo(x2, s / 2); path.lineTo(x1, s - m); path.closeSubpath()
            p.drawPath(path)
    return _make_icon(size, color, draw)


def volume_icon(size: int = 20, color: str = "#A0A0A0") -> QIcon:
    def draw(p, s):
        path = QPainterPath()
        rw, rh = s * 0.15, s * 0.35
        rx, ry = s * 0.18, (s - rh) / 2
        path.addRect(QRectF(rx, ry, rw, rh))
        trap_right = s * 0.55
        path.moveTo(rx + rw, s * 0.25)
        path.lineTo(trap_right, s * 0.15)
        path.lineTo(trap_right, s * 0.85)
        path.lineTo(rx + rw, s * 0.75)
        path.closeSubpath()
        p.drawPath(path)
        p.setPen(QPen(QColor(color), s * 0.06, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for sz in [s * 0.18, s * 0.32]:
            p.drawArc(int(s * 0.6 - (sz - s * 0.18) / 2), int((s - sz) / 2),
                     int(sz), int(sz), -45 * 16, 90 * 16)
    return _make_icon(size, color, draw)


#********************************************************************


class MakamGUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Türk Sanat Müziği — Makam Sınıflandırma")
        #DAHA BUYUK PENCERE - sag panelde 2 kart yan yana sigsin
        self.setMinimumSize(1200, 740)
        self.resize(1400, 820)

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.7)

        self.slider_being_dragged = False
        self.current_file_path = None
        self.segment_features = None
        self.extraction_thread = None
        self.prediction_thread = None

        self.registry = pred.build_model_registry(Path("experiments"))

        self._build_ui()
        self._connect_signals()


    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        left_panel = self._build_player_panel()
        #ESIT GENISLIK - sol/sag 1:1
        main_layout.addWidget(left_panel, stretch=1)

        right_panel = self._build_prediction_panel()
        main_layout.addWidget(right_panel, stretch=1)


    def _build_player_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(0)

        title = QLabel("Müzik Çalar")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Bir MP3 veya WAV dosyası seçin")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        vinyl_row = QHBoxLayout()
        vinyl_row.addStretch()
        self.vinyl = VinylWidget(size=240)
        vinyl_row.addWidget(self.vinyl)
        vinyl_row.addStretch()
        layout.addLayout(vinyl_row)

        layout.addSpacing(20)

        self.song_title_label = QLabel("Henüz dosya seçilmedi")
        self.song_title_label.setObjectName("emptyState")
        self.song_title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.song_title_label)

        self.song_meta_label = QLabel("")
        self.song_meta_label.setObjectName("songMeta")
        self.song_meta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.song_meta_label)

        layout.addStretch()

        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setEnabled(False)
        layout.addWidget(self.progress_slider)

        layout.addSpacing(4)

        time_row = QHBoxLayout()
        time_row.setContentsMargins(0, 0, 0, 0)
        self.current_time_label = QLabel("0:00")
        self.current_time_label.setObjectName("timeLabel")
        self.total_time_label = QLabel("0:00")
        self.total_time_label.setObjectName("timeLabel")
        self.total_time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        time_row.addWidget(self.current_time_label)
        time_row.addStretch()
        time_row.addWidget(self.total_time_label)
        layout.addLayout(time_row)

        layout.addSpacing(20)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(20)
        controls_row.addStretch()

        self.back_button = QPushButton()
        self.back_button.setObjectName("secondaryButton")
        self.back_button.setIcon(skip_back_icon(28, COLORS['text_primary']))
        self.back_button.setIconSize(QSize(28, 28))
        self.back_button.setEnabled(False)
        controls_row.addWidget(self.back_button)

        self.play_button = QPushButton()
        self.play_button.setObjectName("playButton")
        self.play_button.setIcon(play_icon(36, "white"))
        self.play_button.setIconSize(QSize(36, 36))
        self.play_button.setEnabled(False)
        controls_row.addWidget(self.play_button)

        self.forward_button = QPushButton()
        self.forward_button.setObjectName("secondaryButton")
        self.forward_button.setIcon(skip_forward_icon(28, COLORS['text_primary']))
        self.forward_button.setIconSize(QSize(28, 28))
        self.forward_button.setEnabled(False)
        controls_row.addWidget(self.forward_button)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        layout.addSpacing(20)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        vol_icon_label = QLabel()
        vol_icon_label.setPixmap(volume_icon(20, COLORS['text_secondary']).pixmap(20, 20))
        bottom_row.addWidget(vol_icon_label)

        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        self.volume_slider.setMaximumWidth(160)
        bottom_row.addWidget(self.volume_slider)

        bottom_row.addStretch()

        self.select_button = QPushButton("Dosya Seç")
        self.select_button.setObjectName("secondaryButtonText")
        bottom_row.addWidget(self.select_button)

        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        bottom_row.addWidget(self.predict_button)

        layout.addLayout(bottom_row)

        return panel


    def _build_prediction_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")

        outer_layout = QVBoxLayout(panel)
        outer_layout.setContentsMargins(28, 28, 28, 28)
        outer_layout.setSpacing(0)

        title = QLabel("Makam Tahmini")
        title.setObjectName("title")
        outer_layout.addWidget(title)

        subtitle = QLabel("4 deney • Ana tahmin + detaylı sonuçlar")
        subtitle.setObjectName("subtitle")
        outer_layout.addWidget(subtitle)

        outer_layout.addSpacing(20)

        self.status_label = QLabel("Müzik yüklenmedi")
        self.status_label.setObjectName("placeholderText")
        outer_layout.addWidget(self.status_label)

        outer_layout.addSpacing(8)

        self.extraction_progress = QProgressBar()
        self.extraction_progress.setRange(0, 100)
        self.extraction_progress.setValue(0)
        self.extraction_progress.setTextVisible(False)
        self.extraction_progress.setFixedHeight(4)
        self.extraction_progress.setVisible(False)
        outer_layout.addWidget(self.extraction_progress)

        outer_layout.addSpacing(4)

        self.progress_text = QLabel("")
        self.progress_text.setObjectName("progressText")
        self.progress_text.setVisible(False)
        outer_layout.addWidget(self.progress_text)

        outer_layout.addSpacing(16)

        self.results_scroll = QScrollArea()
        self.results_scroll.setWidgetResizable(True)
        self.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(0, 0, 4, 0)
        self.results_layout.setSpacing(10)

        self._show_placeholder()

        self.results_scroll.setWidget(self.results_container)
        outer_layout.addWidget(self.results_scroll, stretch=1)

        return panel


    def _connect_signals(self):
        self.select_button.clicked.connect(self._open_file)
        self.predict_button.clicked.connect(self._start_extraction)
        self.play_button.clicked.connect(self._toggle_play)
        self.back_button.clicked.connect(lambda: self._seek_relative(-10000))
        self.forward_button.clicked.connect(lambda: self._seek_relative(10000))

        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self._on_slider_released)
        self.volume_slider.valueChanged.connect(
            lambda v: self.audio_output.setVolume(v / 100.0)
        )

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._on_state_changed)


    def _open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Müzik Dosyası Seç", "",
            "Ses Dosyaları (*.mp3 *.wav *.m4a *.flac *.ogg);;Tüm Dosyalar (*)"
        )
        if not file_path:
            return

        self.current_file_path = file_path
        self.segment_features = None
        self.player.setSource(QUrl.fromLocalFile(file_path))

        path_obj = Path(file_path)
        file_name = path_obj.stem
        file_size_mb = path_obj.stat().st_size / (1024 * 1024)
        file_ext = path_obj.suffix.lstrip(".").upper()

        display_name = file_name if len(file_name) <= 56 else file_name[:53] + "..."

        self.song_title_label.setText(display_name)
        self.song_title_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 16px; "
            f"font-weight: 600; font-style: normal;"
        )
        self.song_meta_label.setText(f"{file_ext}  •  {file_size_mb:.1f} MB")

        self.play_button.setEnabled(True)
        self.back_button.setEnabled(True)
        self.forward_button.setEnabled(True)
        self.progress_slider.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.vinyl.set_has_track(True)

        self.status_label.setText("Tahmin için hazır")
        self.extraction_progress.setVisible(False)
        self.progress_text.setVisible(False)

        self._clear_results()
        self._show_placeholder()


    def _start_extraction(self):
        if not self.current_file_path:
            return

        self.predict_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.extraction_progress.setVisible(True)
        self.extraction_progress.setValue(0)
        self.progress_text.setVisible(True)
        self.progress_text.setText("Başlatılıyor...")
        self.status_label.setText("Özellikler çıkarılıyor")

        self._clear_results()
        self._show_placeholder("Özellikler hesaplanıyor...")

        self.extraction_thread = FeatureExtractionThread(self.current_file_path)
        self.extraction_thread.progress_updated.connect(self._on_extraction_progress)
        self.extraction_thread.status_updated.connect(self._on_extraction_status)
        self.extraction_thread.finished_success.connect(self._on_extraction_done)
        self.extraction_thread.finished_error.connect(self._on_extraction_error)
        self.extraction_thread.start()


    def _on_extraction_progress(self, current: int, total: int):
        percent = int((current / total) * 100) if total > 0 else 0
        self.extraction_progress.setValue(percent)
        self.progress_text.setText(f"Segment {current}/{total}")


    def _on_extraction_status(self, msg: str):
        self.status_label.setText(msg)


    def _on_extraction_done(self, segments: list):
        self.segment_features = segments
        self.status_label.setText(f"{len(segments)} segment hazır, modeller çalışıyor")
        self.extraction_progress.setValue(0)
        self.progress_text.setText("Modeller yükleniyor")

        self.prediction_thread = PredictionThread(segments, self.registry)
        self.prediction_thread.progress_updated.connect(self._on_prediction_progress)
        self.prediction_thread.status_updated.connect(self._on_prediction_status)
        self.prediction_thread.finished_success.connect(self._on_prediction_done)
        self.prediction_thread.finished_error.connect(self._on_prediction_error)
        self.prediction_thread.start()


    def _on_extraction_error(self, error_msg: str):
        self.extraction_progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.status_label.setText(f"Hata: {error_msg}")
        self.predict_button.setEnabled(True)
        self.select_button.setEnabled(True)


    def _on_prediction_progress(self, current: int, total: int):
        percent = int((current / total) * 100) if total > 0 else 0
        self.extraction_progress.setValue(percent)
        self.progress_text.setText(f"Model {current}/{total}")


    def _on_prediction_status(self, msg: str):
        self.status_label.setText(msg)


    def _on_prediction_done(self, results: list):
        self.extraction_progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.status_label.setText("Tahmin tamamlandı")
        self.predict_button.setEnabled(True)
        self.select_button.setEnabled(True)

        self._clear_results()

        #Sonuclari deney bazinda grupla
        groups = {}
        order = []

        for entry, result in results:
            if entry.exp_short not in groups:
                groups[entry.exp_short] = {
                    'label': entry.exp_label,
                    'is_featured': entry.is_featured,
                    'entries': [],
                    'class_list': entry.get_class_list(),
                }
                order.append(entry.exp_short)
            groups[entry.exp_short]['entries'].append((entry, result))

        featured_first = [s for s in order if groups[s]['is_featured']]
        others = [s for s in order if not groups[s]['is_featured']]

        #FEATURED (exp4) - ust tarafta tam genislik
        if featured_first:
            for exp_short in featured_first:
                g = groups[exp_short]
                card = ExperimentCard(
                    exp_short=exp_short, exp_label=g['label'],
                    results=g['entries'], is_featured=True,
                    class_list=g['class_list'],
                )
                self.results_layout.addWidget(card)

        #DETAYLI (exp1/2/3) - 2 kolonlu grid
        if others:
            self.results_layout.addSpacing(8)
            divider_label = QLabel("DETAYLI SONUÇLAR")
            divider_label.setStyleSheet(
                f"color: {COLORS['text_tertiary']}; "
                f"font-size: 10px; font-weight: 600; letter-spacing: 0.8px;"
            )
            self.results_layout.addWidget(divider_label)
            self.results_layout.addSpacing(2)

            #Grid container
            grid_container = QWidget()
            grid_layout = QGridLayout(grid_container)
            grid_layout.setContentsMargins(0, 0, 0, 0)
            grid_layout.setSpacing(10)
            grid_layout.setColumnStretch(0, 1)
            grid_layout.setColumnStretch(1, 1)

            for i, exp_short in enumerate(others):
                g = groups[exp_short]
                card = ExperimentCard(
                    exp_short=exp_short, exp_label=g['label'],
                    results=g['entries'], is_featured=False,
                    class_list=g['class_list'],
                )
                row = i // 2
                col = i % 2
                #Yukseklikler farkli olabilir, ustten hizala
                grid_layout.addWidget(card, row, col, Qt.AlignmentFlag.AlignTop)

            self.results_layout.addWidget(grid_container)

        self.results_layout.addStretch()


    def _on_prediction_error(self, error_msg: str):
        self.extraction_progress.setVisible(False)
        self.progress_text.setVisible(False)
        self.status_label.setText(f"Tahmin hatası: {error_msg}")
        self.predict_button.setEnabled(True)
        self.select_button.setEnabled(True)


    def _clear_results(self):
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()


    def _show_placeholder(self, text: str = None):
        if text is None:
            text = (
                "Bir müzik dosyası seçip\n'Tahmin Et' butonuna basın.\n\n"
                "Sonuçlar:\n"
                "•  Ana tahmin: 20 sınıflı genel CNN\n"
                "•  Detaylı: 3 deney, her biri ResNet + CNN"
            )
        lbl = QLabel(text)
        lbl.setObjectName("placeholderText")
        lbl.setWordWrap(True)
        lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results_layout.addWidget(lbl)
        self.results_layout.addStretch()


    def _toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()


    def _seek_relative(self, delta_ms: int):
        new_pos = self.player.position() + delta_ms
        new_pos = max(0, min(new_pos, self.player.duration()))
        self.player.setPosition(new_pos)


    def _on_slider_pressed(self):
        self.slider_being_dragged = True


    def _on_slider_released(self):
        self.slider_being_dragged = False
        self.player.setPosition(self.progress_slider.value())


    def _on_position_changed(self, position: int):
        if not self.slider_being_dragged:
            self.progress_slider.setValue(position)
        self.current_time_label.setText(format_time(position))


    def _on_duration_changed(self, duration: int):
        self.progress_slider.setRange(0, duration)
        self.total_time_label.setText(format_time(duration))


    def _on_state_changed(self, state):
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(pause_icon(36, "white"))
            self.vinyl.set_spinning(True)
        else:
            self.play_button.setIcon(play_icon(36, "white"))
            self.vinyl.set_spinning(False)


def main():
    app = QApplication(sys.argv)

    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS['bg_primary']))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS['bg_secondary']))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS['bg_secondary']))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS['text_primary']))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS['accent']))
    app.setPalette(palette)

    app.setStyleSheet(QSS_STYLESHEET)

    window = MakamGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()