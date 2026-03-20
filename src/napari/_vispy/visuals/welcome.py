from __future__ import annotations

import logging
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import qtpy
from qtpy.QtCore import QPointF
from qtpy.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter
from vispy.scene.node import Node
from vispy.scene.visuals import Polygon
from vispy.util.svg import Document
from vispy.visuals.transforms import STTransform

from napari._app_model import get_app_model
from napari._vispy.visuals.image import Image
from napari.resources import get_icon_path
from napari.settings import get_settings
from napari.utils.action_manager import action_manager
from napari.utils.interactions import Shortcut

if TYPE_CHECKING:
    from vispy.visuals.text.text import FontManager

    from napari.utils.color import ColorValue

vispy_logger = logging.getLogger('vispy')


def _load_logo() -> np.ndarray:
    # load logo (disabling logging for some svg reading warnings)
    old_level = vispy_logger.level
    vispy_logger.setLevel(logging.ERROR)
    coords = Document(get_icon_path('logo_silhouette')).paths[0].vertices[0][0]
    vispy_logger.setLevel(old_level)
    # drop z: causes issues with polygon agg mode
    coords = coords[:, :2]
    # center
    coords -= (np.max(coords, axis=0) + np.min(coords, axis=0)) / 2
    return coords


def _qimage_to_array(
    image: QImage,
) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
    if image.format() != QImage.Format_ARGB32:
        image = image.convertToFormat(QImage.Format_ARGB32)
    bits = image.constBits()
    height, width, channels = image.height(), image.width(), 4

    if qtpy.API_NAME.startswith('PySide'):
        array = np.array(bits).reshape(height, width, channels)
    else:
        bits.setsize(height * width * channels)
        array = np.frombuffer(bits, np.uint8).reshape(height, width, channels)

    return array[:, :, [2, 1, 0, 3]]


class Welcome(Node):
    _font_size = 12
    _header_line_height = 1.75
    _body_line_height = 1.15
    _text_padding = 12.0

    def __init__(self, font_manager: FontManager, face: str) -> None:
        self.logo_coords = _load_logo()
        super().__init__()
        self._face = face
        self._text_color = np.ones(4, dtype=float)
        self._header_text = ''
        self._shortcut_keybindings_text = ''
        self._shortcut_descriptions_text = ''
        self._tip_text = ''
        self._text_scale = 1.0
        self._device_pixel_ratio = 1.0

        # make logo smaller and move it up (magic number)
        self.logo_coords /= 4
        self.logo_coords[:, 1] -= 130

        self.logo = Polygon(
            self.logo_coords, border_method='agg', border_width=2, parent=self
        )
        self.logo.transform = STTransform()

        self._font = QFont(self._face)
        self._font.setPixelSize(self._font_size)
        self._font_metrics = QFontMetricsF(self._font)
        self.font_height = self._font_metrics.height()

        self.text_image = Image(
            np.zeros((1, 1, 4), dtype=np.uint8),
            parent=self,
            texture_format='auto',
        )
        self.text_image.transform = STTransform()
        self._text_bounds = (-0.5, -0.5, 0.5, 0.5)

    def set_color(self, color: ColorValue) -> None:
        self.logo.color = color
        self.logo.border_color = color
        self._text_color = np.asarray(color, dtype=float)
        self._update_text_image()

    def set_version(self, version: str) -> None:
        self._header_text = (
            f'napari {version}\n'
            'Drag file(s) here to open, or use the shortcuts below:'
        )
        self._update_text_image()

    def set_shortcuts(self, commands: tuple[str, ...]) -> None:
        shortcuts = {}
        for command_id in commands:
            shortcut, command = self._command_shortcut_and_description(
                command_id
            )
            if shortcut is not None and command is not None:
                shortcuts[shortcut] = command

        # TODO: use template strings in the future
        self._shortcut_keybindings_text = '\n'.join(shortcuts.keys())
        self._shortcut_descriptions_text = '\n'.join(shortcuts.values())
        self._update_text_image()

    def set_tip(self, tip: str) -> None:
        # TODO: this should use template strings in the future
        for match in re.finditer(r'{(.*?)}', tip):
            command_id = match.group(1)
            shortcut, _ = self._command_shortcut_and_description(command_id)
            # this can be none at launch (not yet initialized), will be updated after
            if shortcut is None:
                # maybe it was just a direct keybinding given
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    shortcut = Shortcut(command_id).platform
            if shortcut:
                tip = re.sub(match.group(), str(shortcut), tip)

        # wrap tip so it's not clipped
        self._tip_text = 'Did you know?\n' + '\n'.join(
            textwrap.wrap(tip, break_on_hyphens=False)
        )
        self._update_text_image()

    def _current_device_pixel_ratio(self) -> float:
        if self.canvas is not None and self.canvas.native is not None:
            native = self.canvas.native
            if hasattr(native, 'devicePixelRatioF'):
                return max(float(native.devicePixelRatioF()), 1.0)
            if hasattr(native, 'devicePixelRatio'):
                return max(float(native.devicePixelRatio()), 1.0)
        return 1.0

    def _update_font_metrics(self) -> None:
        pixel_size = max(
            round(
                self._font_size * self._text_scale * self._device_pixel_ratio
            ),
            1,
        )
        self._font = QFont(self._face)
        self._font.setPixelSize(pixel_size)
        self._font_metrics = QFontMetricsF(self._font)

    def _update_raster_settings(
        self, *, text_scale: float | None = None
    ) -> bool:
        if text_scale is None:
            text_scale = self._text_scale

        device_pixel_ratio = self._current_device_pixel_ratio()
        if np.isclose(text_scale, self._text_scale) and np.isclose(
            device_pixel_ratio, self._device_pixel_ratio
        ):
            return False

        self._text_scale = text_scale
        self._device_pixel_ratio = device_pixel_ratio
        self._update_font_metrics()
        return True

    def _text_blocks(self) -> tuple[dict[str, Any], ...]:
        text_scale = self._text_scale
        return (
            {
                'text': self._header_text,
                'x': 0.0,
                'y': -10.0 * text_scale,
                'line_height': self._header_line_height,
                'anchor_x': 'center',
            },
            {
                'text': self._shortcut_keybindings_text,
                'x': -80.0 * text_scale,
                'y': 2.75 * self.font_height * text_scale,
                'line_height': self._body_line_height,
                'anchor_x': 'right',
            },
            {
                'text': self._shortcut_descriptions_text,
                'x': -60.0 * text_scale,
                'y': 2.75 * self.font_height * text_scale,
                'line_height': self._body_line_height,
                'anchor_x': 'left',
            },
            {
                'text': self._tip_text,
                'x': 0.0,
                'y': 7.5 * self.font_height * text_scale,
                'line_height': self._body_line_height,
                'anchor_x': 'center',
            },
        )

    def _text_size(
        self, text: str, *, line_height: float
    ) -> tuple[float, float]:
        if not text:
            return 0.0, 0.0

        lines = text.splitlines()
        width = max(
            self._font_metrics.horizontalAdvance(line) for line in lines
        )
        height = self._font_metrics.height() * len(lines)
        height += (
            self._font_metrics.height()
            * (line_height - 1)
            * max(len(lines) - 1, 0)
        )
        return (
            width / self._device_pixel_ratio,
            height / self._device_pixel_ratio,
        )

    def _block_bounds(
        self, block: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        width, height = self._text_size(
            block['text'], line_height=block['line_height']
        )
        x = block['x']
        y = block['y']
        anchor_x = block['anchor_x']

        if anchor_x == 'center':
            left = x - width / 2
            right = x + width / 2
        elif anchor_x == 'right':
            left = x - width
            right = x
        else:
            left = x
            right = x + width

        top = y - height
        bottom = y
        return left, top, right, bottom

    def _draw_block(
        self,
        painter: QPainter,
        block: dict[str, Any],
        *,
        left: float,
        top: float,
    ) -> None:
        if not block['text']:
            return

        anchor_x = block['anchor_x']
        line_height = block['line_height']
        x = block['x']
        y = block['y']
        _width, height = self._text_size(
            block['text'], line_height=line_height
        )
        lines = block['text'].splitlines()
        current_y = (
            y - height + self._font_metrics.ascent() / self._device_pixel_ratio
        )
        line_step = (
            self._font_metrics.height() / self._device_pixel_ratio
        ) * line_height

        for line in lines:
            line_width = (
                self._font_metrics.horizontalAdvance(line)
                / self._device_pixel_ratio
            )
            if anchor_x == 'center':
                line_x = x - line_width / 2
            elif anchor_x == 'right':
                line_x = x - line_width
            else:
                line_x = x

            painter.drawText(
                QPointF(
                    (line_x - left) * self._device_pixel_ratio,
                    (current_y - top) * self._device_pixel_ratio,
                ),
                line,
            )
            current_y += line_step

    def _update_text_image(self) -> None:
        blocks = tuple(block for block in self._text_blocks() if block['text'])
        if not blocks:
            self._text_bounds = (-0.5, -0.5, 0.5, 0.5)
            self.text_image.set_data(np.zeros((1, 1, 4), dtype=np.uint8))
            return

        bounds = [self._block_bounds(block) for block in blocks]
        left = min(bound[0] for bound in bounds) - self._text_padding
        top = min(bound[1] for bound in bounds) - self._text_padding
        right = max(bound[2] for bound in bounds) + self._text_padding
        bottom = max(bound[3] for bound in bounds) + self._text_padding

        width = max(int(np.ceil((right - left) * self._device_pixel_ratio)), 1)
        height = max(
            int(np.ceil((bottom - top) * self._device_pixel_ratio)), 1
        )
        image = QImage(width, height, QImage.Format_ARGB32)
        image.fill(0)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self._font)
        painter.setPen(
            QColor.fromRgbF(
                float(self._text_color[0]),
                float(self._text_color[1]),
                float(self._text_color[2]),
                float(self._text_color[3]),
            )
        )
        for block in blocks:
            self._draw_block(painter, block, left=left, top=top)
        painter.end()

        self._text_bounds = (left, top, right, bottom)
        self.text_image.set_data(_qimage_to_array(image))

    @staticmethod
    def _command_shortcut_and_description(
        command_id: str,
    ) -> tuple[str | None, str | None]:
        app = get_app_model()
        all_shortcuts = get_settings().shortcuts.shortcuts
        keybinding = app.keybindings.get_keybinding(command_id)

        shortcut = command = None
        if keybinding is not None:
            shortcut = Shortcut(keybinding.keybinding).platform
            command = app.commands[command_id].title
        else:
            # might be an action_manager action
            keybinding = all_shortcuts.get(command_id, [None])[0]
            if keybinding is not None:
                shortcut = Shortcut(keybinding).platform
                command = action_manager._actions[command_id].description
            else:
                shortcut = command = None

        return shortcut, command

    def set_scale_and_position(self, x: float, y: float) -> None:
        trans = (x / 2, y / 2, 0, 0)
        # we don't want the logo to be affected by dpi ratio which is included in
        # font_height, so we scale it separately
        logo_scale = min(x, y) * 0.002  # magic number
        self.logo.transform.translate = trans
        self.logo.transform.scale = (logo_scale, logo_scale, 0, 0)

        text_scale = min(x, y) / self.font_height * 0.04  # magic number
        if self._update_raster_settings(text_scale=text_scale):
            self._update_text_image()

        left, top, _, _ = self._text_bounds
        self.text_image.transform.translate = (
            x / 2 + left,
            y / 2 + top,
            0,
            0,
        )
        self.text_image.transform.scale = (
            1 / self._device_pixel_ratio,
            1 / self._device_pixel_ratio,
            1,
            1,
        )

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
