from __future__ import annotations

import logging
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple

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
    """Convert a QImage to an RGBA uint8 numpy array."""
    if image.format() != QImage.Format_ARGB32:
        image = image.convertToFormat(QImage.Format_ARGB32)
    bits = image.constBits()
    h, w, c = image.height(), image.width(), 4

    if qtpy.API_NAME.startswith('PySide'):
        array = np.array(bits).reshape(h, w, c)
    else:
        bits.setsize(h * w * c)
        array = np.frombuffer(bits, np.uint8).reshape(h, w, c)

    # ARGB32 → RGBA
    return array[:, :, [2, 1, 0, 3]]


class _TextBlock(NamedTuple):
    """A positioned block of text with alignment."""

    text: str
    x: float
    y: float
    line_height: float
    anchor_x: str  # 'left', 'center', or 'right'


def _aligned_x(x: float, width: float, anchor: str) -> float:
    """Return left-edge x coordinate given anchor alignment."""
    if anchor == 'center':
        return x - width / 2
    if anchor == 'right':
        return x - width
    return x


class Welcome(Node):
    _font_size = 12
    _header_line_height = 1.75
    _body_line_height = 1.15
    _text_padding = 12.0

    def __init__(self, face: str, **_kwargs: Any) -> None:
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

    def _text_blocks(self) -> tuple[_TextBlock, ...]:
        s = self._text_scale
        fh = self.font_height
        body_lh = self._body_line_height
        return (
            _TextBlock(
                text=self._header_text,
                x=0.0,
                y=-10.0 * s,
                line_height=self._header_line_height,
                anchor_x='center',
            ),
            _TextBlock(
                text=self._shortcut_keybindings_text,
                x=-80.0 * s,
                y=2.75 * fh * s,
                line_height=body_lh,
                anchor_x='right',
            ),
            _TextBlock(
                text=self._shortcut_descriptions_text,
                x=-60.0 * s,
                y=2.75 * fh * s,
                line_height=body_lh,
                anchor_x='left',
            ),
            _TextBlock(
                text=self._tip_text,
                x=0.0,
                y=7.5 * fh * s,
                line_height=body_lh,
                anchor_x='center',
            ),
        )

    def _text_size(
        self, text: str, *, line_height: float
    ) -> tuple[float, float]:
        if not text:
            return 0.0, 0.0
        lines = text.splitlines()
        dpr = self._device_pixel_ratio
        fm = self._font_metrics
        width = max(fm.horizontalAdvance(line) for line in lines) / dpr
        n = len(lines)
        # total line heights + extra inter-line spacing
        height = fm.height() * (n + (line_height - 1) * max(n - 1, 0)) / dpr
        return width, height

    def _block_bounds(
        self, block: _TextBlock
    ) -> tuple[float, float, float, float]:
        width, height = self._text_size(
            block.text, line_height=block.line_height
        )
        left = _aligned_x(block.x, width, block.anchor_x)
        return left, block.y - height, left + width, block.y

    def _draw_block(
        self,
        painter: QPainter,
        block: _TextBlock,
        *,
        block_top: float,
        img_left: float,
        img_top: float,
    ) -> None:
        if not block.text:
            return

        dpr = self._device_pixel_ratio
        fm = self._font_metrics
        current_y = block_top + fm.ascent() / dpr
        line_step = (fm.height() / dpr) * block.line_height

        for line in block.text.splitlines():
            line_width = fm.horizontalAdvance(line) / dpr
            line_x = _aligned_x(block.x, line_width, block.anchor_x)
            painter.drawText(
                QPointF(
                    (line_x - img_left) * dpr,
                    (current_y - img_top) * dpr,
                ),
                line,
            )
            current_y += line_step

    def _update_text_image(self) -> None:
        blocks = [b for b in self._text_blocks() if b.text]
        if not blocks:
            self._text_bounds = (-0.5, -0.5, 0.5, 0.5)
            self.text_image.set_data(np.zeros((1, 1, 4), dtype=np.uint8))
            return

        block_bounds = [self._block_bounds(b) for b in blocks]
        pad = self._text_padding
        img_left = min(b[0] for b in block_bounds) - pad
        img_top = min(b[1] for b in block_bounds) - pad
        img_right = max(b[2] for b in block_bounds) + pad
        img_bottom = max(b[3] for b in block_bounds) + pad

        dpr = self._device_pixel_ratio
        px_w = max(int(np.ceil((img_right - img_left) * dpr)), 1)
        px_h = max(int(np.ceil((img_bottom - img_top) * dpr)), 1)
        image = QImage(px_w, px_h, QImage.Format_ARGB32)
        image.fill(0)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self._font)
        painter.setPen(QColor.fromRgbF(*map(float, self._text_color)))
        for block, (_, block_top, _, _) in zip(
            blocks, block_bounds, strict=False
        ):
            self._draw_block(
                painter,
                block,
                block_top=block_top,
                img_left=img_left,
                img_top=img_top,
            )
        painter.end()

        self._text_bounds = (img_left, img_top, img_right, img_bottom)
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
        dpr = self._device_pixel_ratio
        self.text_image.transform.translate = (x / 2 + left, y / 2 + top, 0, 0)
        self.text_image.transform.scale = (1 / dpr, 1 / dpr, 1, 1)

    def set_gl_state(self, *args: Any, **kwargs: Any) -> None:
        for node in self.children:
            node.set_gl_state(*args, **kwargs)
