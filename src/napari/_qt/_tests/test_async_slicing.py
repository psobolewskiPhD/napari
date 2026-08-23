# The tests in this module for the new style of async slicing in napari:
# https://napari.org/dev/naps/4-async-slicing.html
import logging
import threading
from functools import partial

import numpy as np
import pytest
from vispy.visuals import VolumeVisual

from napari import Viewer
from napari._tests.utils import LockableData
from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.vectors import (
    VispyVectorsLayer,
    generate_vector_meshes_2D,
)
from napari.layers import Image, Layer, Points, Vectors
from napari.utils.events import Event


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def _enable_async(_fresh_settings, make_napari_viewer):
    """
    This fixture depends on _fresh_settings and make_napari_viewer
    to enforce proper order of fixture execution.
    """
    from napari import settings

    settings.get_settings().experimental.async_ = True


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_current_step_change(
    make_napari_viewer, qtbot, rng
):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[2, :, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_out_of_bounds_layer_loaded(make_napari_viewer, qtbot):
    """Check that images that are out of bounds when slicing appear loaded.

    See https://github.com/napari/napari/issues/7070.
    """
    viewer = make_napari_viewer()
    l0 = viewer.add_image(np.random.random((5, 5, 5)))
    l1 = viewer.add_image(np.random.random((5, 5, 5)), translate=(5, 0, 0))
    assert viewer.dims.nsteps == (10, 5, 5)

    def layer_loaded(ly):
        return ly.loaded

    for i in range(viewer.dims.nsteps[0]):
        viewer.dims.current_step = (i, 0, 0)
        qtbot.waitUntil(partial(layer_loaded, l0), timeout=500)
        qtbot.waitUntil(partial(layer_loaded, l1), timeout=500)


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_order_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 5, 7))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.order != (1, 0, 2)

    viewer.dims.order = (1, 0, 2)

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[:, 2, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_on_ndisplay_change(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)
    assert viewer.dims.ndisplay != 3

    viewer.dims.ndisplay = 3

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data)


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_multiscale_image_on_pan(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)

    # Force the image to be at the lower resolution, with #7870 this behavior
    # changed so that initial zoom was high, resulting in data_level=0.
    # Likely due to better sequencing of async slicing.
    viewer.scene.camera.zoom = 0.1
    viewer.window._qt_viewer.canvas.on_draw(None)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 1
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 3, 4]])

    # Simulate panning to the left by changing the corner pixels in the last
    # dimension, which corresponds to x/columns, then triggering a reload.
    image.corner_pixels = np.array([[0, 0, 0], [0, 3, 2]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[1][0, 0:4, 0:3])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_multiscale_image_on_zoom(qtbot, make_napari_viewer, rng):
    viewer = make_napari_viewer()
    data = [rng.random((4, 8, 10)), rng.random((2, 4, 5))]
    image = Image(data)
    vispy_image = setup_viewer_for_async_slicing(viewer, image)

    # Force the image to be at the lower resolution, with #7870 this behavior
    # changed so that initial zoom was high, resulting in data_level=0.
    # Likely due to better sequencing of async slicing.
    viewer.scene.camera.zoom = 0.1
    viewer.window._qt_viewer.canvas.on_draw(None)

    # Check that we're initially slicing the middle of the first dimension
    # over the whole of lowest resolution image.
    assert viewer.dims.not_displayed == (0,)
    assert viewer.dims.current_step[0] == 1
    assert image._data_level == 1
    np.testing.assert_equal(image.corner_pixels, [[0, 0, 0], [0, 3, 4]])

    # Simulate zooming into the middle of the higher resolution image.
    image._data_level = 0
    image.corner_pixels = np.array([[0, 2, 3], [0, 5, 6]])
    image.events.reload(Event('reload', layer=image))

    wait_until_vispy_image_data_equal(qtbot, vispy_image, data[0][1, 2:6, 3:7])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_points_on_current_step_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.array(
        [
            [0, 2, 3],
            [1, 3, 4],
            [2, 4, 5],
            [3, 5, 6],
            [4, 6, 7],
        ]
    )
    points = Points(data)
    vispy_points = setup_viewer_for_async_slicing(viewer, points)
    assert viewer.dims.current_step != (3, 0, 0)

    viewer.dims.current_step = (3, 0, 0)

    wait_until_vispy_points_data_equal(qtbot, vispy_points, np.array([[5, 6]]))


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_points_on_point_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    # Define data so that slicing at 1.6 in the first dimension should match the
    # second point, but won't if that index is prematurely rounded as for other
    # layers.
    data = np.array(
        [
            [0, 2, 3],
            [1.4, 3, 4],
            [2.4, 4, 5],
            [3.4, 5, 6],
            [4, 6, 7],
        ]
    )
    points = Points(data)
    vispy_points = setup_viewer_for_async_slicing(viewer, points)
    assert viewer.dims.point != (1.6, 0, 0)

    viewer.dims.point = (1.6, 0, 0)

    wait_until_vispy_points_data_equal(qtbot, vispy_points, np.array([[3, 4]]))


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_image_loaded(make_napari_viewer, qtbot, rng):
    viewer = make_napari_viewer()
    data = rng.random((3, 4, 5))
    lockable_data = LockableData(data)
    layer = Image(lockable_data, multiscale=False)
    vispy_layer = setup_viewer_for_async_slicing(viewer, layer)

    assert layer.loaded
    assert viewer.dims.current_step != (2, 0, 0)

    with lockable_data.lock:
        viewer.dims.current_step = (2, 0, 0)
        assert not layer.loaded

    qtbot.waitUntil(lambda: layer.loaded)

    np.testing.assert_allclose(vispy_layer.node._data, data[2, :, :])


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_vectors_on_current_step_change(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.array(
        [
            [[0, 2, 3], [1, 2, 2]],
            [[2, 4, 5], [0, -3, 3]],
            [[4, 6, 7], [3, 0, -2]],
        ]
    )
    vectors = Vectors(data)
    vispy_vectors = setup_viewer_for_async_slicing(viewer, vectors)
    assert viewer.dims.current_step != (2, 0, 0)

    viewer.dims.current_step = (2, 0, 0)

    wait_until_vispy_vectors_data_equal(
        qtbot, vispy_vectors, np.array([[[2, 4, 5], [0, -3, 3]]])
    )


@pytest.mark.usefixtures('_enable_async')
def test_async_slice_two_layers_shutdown(make_napari_viewer):
    """See https://github.com/napari/napari/issues/6685"""
    viewer = make_napari_viewer()
    # To reproduce the issue, we need two points layers where the second has
    # some non-zero coordinates.
    viewer.add_points()
    points = viewer.add_points()
    points.add([[1, 2]])

    viewer.close()


# The tests below cover `QtViewer`'s slice-ready queue directly, rather than
# through a real slicing round-trip. The queue exists because
# `_layer_slicer.events.ready` is emitted from the slicing thread and the
# handler must run on the main thread; the tests above prove the happy path
# end-to-end, while these pin the three behaviours that only show up when
# something goes wrong or when the viewer is being torn down.


def _pending_ready_event() -> Event:
    """A slice-ready event with no responses, as a stand-in for a real one."""
    return Event('ready', value={})


@pytest.mark.usefixtures('_enable_async')
def test_screenshot_shows_the_slice_it_waited_for(make_napari_viewer):
    """`screenshot()` must not capture the frame before the pending slice.

    Regression test for napari/napari#8033. `_screenshot` called
    `wait_until_idle`, which waits for the slicing *futures* - but the response
    is handed to the main thread through
    `_queue_slice_ready`/`_process_slice_ready_events`, so a finished future can
    still have its result sitting in the queue, unapplied. The screenshot then
    showed the previous slice.

    Asserted on pixel content rather than by diffing two screenshots: a second
    capture can be taken at a different canvas size if the window manager
    resizes in between, which fails for reasons unrelated to slicing.
    """
    viewer = make_napari_viewer(show=True)
    # Two slices that cannot be confused: one black, one white.
    data = np.zeros((2, 32, 32), dtype=np.uint8)
    data[1] = 255
    viewer.add_image(data, contrast_limits=[0, 255])
    viewer.dims.current_step = (0, 0, 0)

    def centre_value(screenshot: np.ndarray) -> float:
        h, w = screenshot.shape[:2]
        centre = screenshot[h // 2 - 2 : h // 2 + 2, w // 2 - 2 : w // 2 + 2]
        return centre[..., :3].mean()

    # Establish that the black slice is what is on screen to begin with, so a
    # stale capture below is a real staleness and not just a dark canvas.
    assert centre_value(viewer.screenshot(canvas_only=True, flash=False)) < 64

    viewer.dims.current_step = (1, 0, 0)
    # No event pumping here on purpose: this is exactly the user-facing call
    # from the issue, and it must flush what it needs by itself.
    assert (
        centre_value(viewer.screenshot(canvas_only=True, flash=False)) > 192
    ), 'screenshot captured the previous slice'


def test_slice_ready_hops_to_the_main_thread(make_napari_viewer, qtbot):
    """`_queue_slice_ready` may run on the slicing thread; the handler may not.

    This is the whole point of the queue-and-signal pair, and it is the one
    part that cannot fail loudly on its own: if the hop silently did nothing,
    slices would simply never be applied.
    """
    viewer = make_napari_viewer()
    qt_viewer = viewer.window._qt_viewer

    handled_on: list[int] = []
    qt_viewer._on_slice_ready = lambda event: handled_on.append(
        threading.get_ident()
    )

    main_thread = threading.get_ident()
    emitted_on: list[int] = []

    def emit_from_worker() -> None:
        emitted_on.append(threading.get_ident())
        qt_viewer._queue_slice_ready(_pending_ready_event())

    worker = threading.Thread(target=emit_from_worker)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), 'emitting from a worker thread blocked'

    # Nothing may have run yet: the connection is queued, so the handler waits
    # for the main thread to return to its event loop.
    assert handled_on == []

    qtbot.waitUntil(lambda: len(handled_on) == 1)
    assert worker.ident != main_thread
    assert emitted_on == [worker.ident]
    assert handled_on == [main_thread]


def test_slice_ready_error_propagates_on_the_normal_path(make_napari_viewer):
    """A failing slice-ready handler must not be swallowed outside teardown.

    Connecting to the emitter directly used to re-raise (napari's
    `EventEmitter` defaults to `ignore_callback_errors=False`), so draining
    must too - otherwise a real slicing bug becomes a log line no test fails
    on.
    """
    viewer = make_napari_viewer()
    qt_viewer = viewer.window._qt_viewer

    def boom(event):
        raise RuntimeError('slice handling failed')

    qt_viewer._on_slice_ready = boom
    qt_viewer._slice_ready_events.put(_pending_ready_event())

    with pytest.raises(RuntimeError, match='slice handling failed'):
        qt_viewer._process_slice_ready_events()


def test_slice_ready_errors_are_suppressed_during_teardown(
    make_napari_viewer, caplog
):
    """Teardown wants the opposite: drain everything, raise nothing.

    An exception escaping here would skip the rest of `Viewer.close`, and
    skipped Qt teardown is itself a source of segfaults. The queue must still
    end up empty, and the failure must still be reported somewhere.
    """
    viewer = make_napari_viewer()
    qt_viewer = viewer.window._qt_viewer

    seen = []

    def boom(event):
        seen.append(event)
        raise RuntimeError('slice handling failed')

    qt_viewer._on_slice_ready = boom
    qt_viewer._slice_ready_events.put(_pending_ready_event())
    qt_viewer._slice_ready_events.put(_pending_ready_event())

    with caplog.at_level(logging.ERROR, logger='napari'):
        qt_viewer._drain_slice_ready_events(suppress_errors=True)

    # A failing event must not abort the drain of the ones behind it.
    assert len(seen) == 2
    assert qt_viewer._slice_ready_events.empty()
    assert 'slice handling failed' in caplog.text


def test_viewer_close_drains_pending_slice_ready(make_napari_viewer):
    """`Viewer.close` flushes the queue *while the layers are still valid*.

    `_layer_slicer.shutdown()` waits for in-flight computations, but the
    queued hop to the main thread can still be pending, and `close` goes on to
    `layers.clear()`. Note that asserting only "the event was drained" would
    not catch a missing drain here: `QtViewer.closeEvent` drains too, and
    `close()` reaches it via `self.window.close()` - just several steps *after*
    clearing the layers. So assert what the layer list looked like at the
    moment the event was handled, which is the thing that actually differs.
    """
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((4, 5)))
    qt_viewer = viewer.window._qt_viewer

    layers_when_handled = []
    qt_viewer._on_slice_ready = lambda event: layers_when_handled.append(
        len(viewer.layers)
    )
    qt_viewer._slice_ready_events.put(_pending_ready_event())

    viewer.close()

    assert layers_when_handled == [1], (
        'the pending slice-ready event was handled after the layers were '
        f'cleared (layer counts seen: {layers_when_handled})'
    )
    assert qt_viewer._slice_ready_events.empty()


def setup_viewer_for_async_slicing(
    viewer: Viewer,
    layer: Layer,
) -> VispyBaseLayer:
    # Initially force synchronous slicing so any slicing caused
    # by adding the layer finishes before any other slicing starts.
    with viewer._layer_slicer.force_sync():
        # Add the layer and get the corresponding vispy layer.
        layer = viewer.add_layer(layer)
        vispy_layer = viewer.window._qt_viewer.layer_to_visual[layer]

    return vispy_layer


def wait_until_vispy_image_data_equal(
    qtbot, vispy_layer: VispyImageLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_image_data_equal() -> None:
        node = vispy_layer.node
        data = (
            node._last_data if isinstance(node, VolumeVisual) else node._data
        )
        # Vispy node data may have been post-processed (e.g. through a colormap),
        # so check that values are close rather than exactly equal.
        np.testing.assert_allclose(data, expected_data)

    qtbot.waitUntil(assert_vispy_image_data_equal)


def wait_until_vispy_points_data_equal(
    qtbot, vispy_layer: VispyPointsLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_points_data_equal() -> None:
        positions = vispy_layer.node._subvisuals[0]._data['a_position']
        # Flip the coordinates because vispy uses xy instead of rc ordering.
        # Also only take the number of dimensions expected since vispy points
        # are always 3D even when displaying 2D slices.
        data = positions[:, -expected_data.shape[1] :: -1]
        np.testing.assert_array_equal(data, expected_data)

    qtbot.waitUntil(assert_vispy_points_data_equal)


def wait_until_vispy_vectors_data_equal(
    qtbot, vispy_layer: VispyVectorsLayer, expected_data: np.ndarray
) -> None:
    def assert_vispy_vectors_data_equal() -> None:
        displayed = expected_data[..., -2:]
        exp_vertices, exp_faces = generate_vector_meshes_2D(
            displayed, 1, 1, 'triangle'
        )
        meshdata = vispy_layer.node._meshdata
        vertices = meshdata.get_vertices()
        faces = meshdata.get_faces()
        # invert for vispy
        np.testing.assert_array_equal(vertices, exp_vertices[..., ::-1])
        np.testing.assert_array_equal(faces, exp_faces)

    qtbot.waitUntil(assert_vispy_vectors_data_equal)
