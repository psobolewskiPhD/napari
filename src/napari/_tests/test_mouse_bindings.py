import os
from unittest.mock import Mock

import numpy as np
import pytest
from vispy.app.canvas import MouseEvent

from napari._tests.utils import skip_on_win_ci
from napari.layers import Image
from napari.layers.base._base_constants import InteractionBoxHandle
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)


@skip_on_win_ci
def test_viewer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to the viewer"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    canvas = viewer.window._qt_viewer.canvas

    if os.getenv('CI'):
        viewer.show()

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @viewer.mouse_drag_callbacks.append
    def drag_callback(v, event):
        assert viewer == v

        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @viewer.mouse_move_callbacks.append
    def move_callback(v, event):
        assert viewer == v
        # on move
        mock_move.method()

    # Simulate press only
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    canvas._scene_canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_called_once()
    mock_move.reset_mock()

    # Simulate press, drag, release
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_move(
        pos=(0, 0),
        modifiers=(),
        button=0,
        press_event=MouseEvent(pos=(0, 0), type='mouse_press'),
    )
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    qtbot.wait(10)
    mock_press.method.assert_called_once()
    mock_drag.method.assert_called_once()
    mock_release.method.assert_called_once()
    mock_move.method.assert_not_called()


@skip_on_win_ci
def test_layer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to a layer that is selected"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    canvas = viewer.window._qt_viewer.canvas

    if os.getenv('CI'):
        viewer.show()

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.add(layer)

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @layer.mouse_drag_callbacks.append
    def drag_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @layer.mouse_move_callbacks.append
    def move_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_move.method()

    # Simulate press only
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_called_once()
    mock_press.reset_mock()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_called_once()
    mock_release.reset_mock()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    canvas._scene_canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_called_once()
    mock_move.reset_mock()

    # Simulate press, drag, release
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_move(
        pos=(0, 0),
        modifiers=(),
        button=0,
        press_event=MouseEvent(pos=(0, 0), type='mouse_press'),
    )
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    qtbot.wait(10)
    mock_press.method.assert_called_once()
    mock_drag.method.assert_called_once()
    mock_release.method.assert_called_once()
    mock_move.method.assert_not_called()


@skip_on_win_ci
def test_unselected_layer_mouse_bindings(qtbot, make_napari_viewer):
    """Test adding mouse bindings to a layer that is not selected"""
    np.random.seed(0)
    viewer = make_napari_viewer()
    canvas = viewer.window._qt_viewer.canvas

    if os.getenv('CI'):
        viewer.show()

    layer = viewer.add_image(np.random.random((10, 20)))
    viewer.layers.selection.remove(layer)

    mock_press = Mock()
    mock_drag = Mock()
    mock_release = Mock()
    mock_move = Mock()

    @layer.mouse_drag_callbacks.append
    def drag_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_press.method()

        yield

        # on move
        while event.type == 'mouse_move':
            mock_drag.method()
            yield

        # on release
        mock_release.method()

    @layer.mouse_move_callbacks.append
    def move_callback(_layer, event):
        assert layer == _layer
        # on press
        mock_move.method()

    # Simulate press only
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate release only
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate move with no press
    canvas._scene_canvas.events.mouse_move(pos=(0, 0), modifiers=())
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()

    # Simulate press, drag, release
    canvas._scene_canvas.events.mouse_press(pos=(0, 0), modifiers=(), button=0)
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_move(
        pos=(0, 0),
        modifiers=(),
        button=0,
        press_event=MouseEvent(pos=(0, 0), type='mouse_press'),
    )
    qtbot.wait(10)
    canvas._scene_canvas.events.mouse_release(
        pos=(0, 0), modifiers=(), button=0
    )
    qtbot.wait(10)
    mock_press.method.assert_not_called()
    mock_drag.method.assert_not_called()
    mock_release.method.assert_not_called()
    mock_move.method.assert_not_called()


@pytest.mark.parametrize(
    ('position', 'dims_displayed', 'nearby_handle'),
    [
        # Postion inside the transform box space so the inside value should be set as selected
        ([0, 3], [0, 1], InteractionBoxHandle.INSIDE),
        ([0, 3, 3], [1, 2], InteractionBoxHandle.INSIDE),
        # Postion outside the transform box space so no handle should be set as selected
        ([0, 11], [0, 1], None),
        ([0, 11, 11], [1, 2], None),
        # When 3 dimensions are being displayed no `highlight_box_handles` logic should be run
        ([0, 3, 3], [0, 1, 2], None),
    ],
)
def test_highlight_box_handles(position, dims_displayed, nearby_handle):
    # `np.zeros`, not `np.empty`: these tests assert on box handles and the
    # affine, so the pixel values are irrelevant - but `Image` computes contrast
    # limits from them, and uninitialised memory can contain a signaling-NaN bit
    # pattern. Any arithmetic on one raises the IEEE invalid flag, which
    # `filterwarnings` turns into a failure. See `test_transform_box` below.
    layer = Image(np.zeros((10, 10)))
    event = Mock(
        position=position, dims_displayed=dims_displayed, modifiers=[None]
    )
    highlight_box_handles(
        layer,
        event,
    )
    # mouse event should be detected over the expected handle
    assert layer._overlays['transform_box'].selected_handle == nearby_handle


def test_transform_box():
    # `np.zeros`, not `np.empty` - see above. `np.empty` returns uninitialised
    # memory, so this test's input was whatever bytes the allocator handed back.
    # On one CI run those included a signaling NaN, and `Image.__init__` derived
    # contrast limits large enough to take `scale_down`, so
    # `CoercedContrastLimits.coerce_data`'s `data * self.scale` raised
    # `RuntimeWarning: invalid value encountered in multiply`. A quiet NaN or an
    # inf would not have: for multiplication, IEEE 754 raises *invalid* only for
    # `0 * inf` or a signaling-NaN operand, and the run's scale was finite and
    # non-zero.
    layer = Image(np.zeros((10, 10)))
    event = Mock(position=[0, 3], dims_displayed=[0, 1], modifiers=[None])
    next(transform_with_box(layer, event))
    # no interaction has been done so affine should be the same as the initial
    assert layer.affine == layer._initial_affine
