import sys

import numpy as np
import pytest


def test_camera(make_napari_viewer):
    """Test vispy camera creation in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    np.testing.assert_almost_equal(viewer.scene.camera.angles, (0, 0, 0))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (0, 5.0, 5.0))
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_vispy_camera_update_from_model(make_napari_viewer):
    """Test vispy camera update from model in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    # Update camera center and zoom
    viewer.scene.camera.center = (11, 12)
    viewer.scene.camera.zoom = 4

    np.testing.assert_almost_equal(viewer.scene.camera.angles, (0, 0, 0))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (0, 11, 12))
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 4)
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_camera_model_update_from_vispy(make_napari_viewer):
    """Test camera model updates from vispy in 2D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Test default values camera values are used and vispy camera has been
    # updated
    assert viewer.dims.ndisplay == 2

    vispy_camera.on_draw(None)  # required for proper initialization
    # Update vispy camera center and zoom
    vispy_camera.center = (11, 12)
    assert vispy_camera.center == (0, 11, 12)
    vispy_camera.zoom = 4
    vispy_camera.on_draw(None)

    np.testing.assert_almost_equal(viewer.scene.camera.angles, (0, 0, 0))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (0, 11, 12))
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 4)
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_3D_camera(make_napari_viewer):
    """Test vispy camera creation in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    # Test camera values have updated
    np.testing.assert_almost_equal(viewer.scene.camera.angles, (0, 0, 0))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (5.0, 5.0, 5.0))
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_vispy_camera_update_from_model_3D(make_napari_viewer):
    """Test vispy camera update from model in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    # Update camera angles, center, and zoom
    viewer.scene.camera.angles = (24, 12, -19)
    viewer.scene.camera.center = (11, 12, 15)
    viewer.scene.camera.zoom = 4

    np.testing.assert_almost_equal(viewer.scene.camera.angles, (24, 12, -19))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (11, 12, 15))
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 4)
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_camera_model_update_from_vispy_3D(make_napari_viewer):
    """Test camera model updates from vispy in 3D."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    viewer.dims.ndisplay = 3

    vispy_camera.on_draw(None)  # required for proper initialization
    # Update vispy camera angles, center, and zoom
    viewer.scene.camera.angles = (24, 12, -19)
    vispy_camera.center = (11, 12, 15)
    vispy_camera.zoom = 4
    vispy_camera.on_draw(None)

    np.testing.assert_almost_equal(viewer.scene.camera.angles, (24, 12, -19))
    np.testing.assert_almost_equal(viewer.scene.camera.center, (11, 12, 15))
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 4)
    np.testing.assert_almost_equal(
        viewer.scene.camera.angles, vispy_camera.angles
    )
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


def test_synced_camera_ndisplay_vispy_sync(make_napari_viewer):
    """Test that synced camera center/zoom persist and sync with vispy across ndisplay switches."""
    viewer = make_napari_viewer()
    vispy_camera = viewer.window._qt_viewer.canvas.camera

    np.random.seed(0)
    data = np.random.random((11, 11, 11))
    viewer.add_image(data)

    # Customize 2D view
    viewer.scene.camera.zoom = 2.5
    viewer.scene.camera.center = (0, 3, 7)

    # Switch to 3D — center/zoom persist (synced default)
    viewer.dims.ndisplay = 3
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 2.5)
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)

    # Switch back to 2D — same
    viewer.dims.ndisplay = 2
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, 2.5)
    np.testing.assert_almost_equal(
        viewer.scene.camera.center, vispy_camera.center
    )
    np.testing.assert_almost_equal(viewer.scene.camera.zoom, vispy_camera.zoom)


@pytest.mark.skipif(
    sys.platform == 'win32', reason='This new test is flaky on windows'
)
def test_camera_orientation_2d(make_napari_viewer, qtbot):
    """Test that flipping orientation of the camera flips displayed image."""
    viewer = make_napari_viewer(show=True)
    data = np.arange(16).reshape((4, 4))
    _ = viewer.add_image(data, interpolation2d='linear')

    # in the default axis orientation of (down, right), the values in a
    # screenshot should continually increase as you go down in the image.
    # We take only the first channel in the RGBA array for simplicity, since
    # this is a grayscale image.
    qtbot.wait(50)
    sshot0 = viewer.window.export_figure(scale=10, flash=False)[..., 0]
    # check that the values are monotonically increasing down:
    avg_row_intensity_grad0 = np.diff(np.mean(sshot0, axis=1))
    assert np.all(avg_row_intensity_grad0 >= 0)

    # same but to the right
    avg_col_intensity_grad0 = np.diff(np.mean(sshot0, axis=0))
    assert np.all(avg_col_intensity_grad0 >= 0)

    # now we reverse the orientation of the vertical axis, and check that the
    # row gradient has changed direction but not the col gradient
    viewer.scene.camera.orientation2d = ('up', 'right')
    qtbot.wait(50)
    sshot1 = viewer.window.export_figure(scale=10, flash=False)[..., 0]
    avg_row_intensity_grad1 = np.diff(np.mean(sshot1, axis=1))
    assert np.all(avg_row_intensity_grad1 <= 0)  # note inverted sign
    avg_col_intensity_grad1 = np.diff(np.mean(sshot1, axis=0))
    assert np.all(avg_col_intensity_grad1 >= 0)

    # finally, reverse orientation of horizontal axis, check that col gradient
    # has now also changed direction
    viewer.scene.camera.orientation2d = ('up', 'left')
    qtbot.wait(50)
    sshot2 = viewer.window.export_figure(scale=10, flash=False)[..., 0]
    avg_row_intensity_grad2 = np.diff(np.mean(sshot2, axis=1))
    assert np.all(avg_row_intensity_grad2 <= 0)  # note inverted sign
    avg_col_intensity_grad2 = np.diff(np.mean(sshot2, axis=0))
    assert np.all(avg_col_intensity_grad2 <= 0)


@pytest.mark.skipif(
    sys.platform == 'win32', reason='This new test is flaky on windows'
)
def test_camera_orientation_3d(make_napari_viewer, qtbot):
    """Test that flipping camera orientation in 3D flips volume as expected."""
    viewer = make_napari_viewer(show=True)
    viewer.dims.ndisplay = 3
    gradient_z = np.arange(16).reshape((16, 1, 1))
    image = np.ones((16, 16))
    image_3d = gradient_z * image
    _ = viewer.add_image(image_3d)

    # We have a 3D image with brightness *increasing with z* (the 0th axis).
    # Therefore, when using a perspective projection, if we point z *away* from
    # us, the bright part of the image will be small, whereas if we point z
    # *towards* us, it will be large. We test the dimension flip by comparing
    # the overall brightness of the image when pointing z in different
    # directions

    viewer.scene.camera.perspective = 60

    captured = {}

    def _pair_captured_at_one_size() -> bool:
        viewer.scene.camera.orientation = ('away', 'down', 'right')
        captured['away'] = viewer.screenshot(canvas_only=True, flash=False)[
            ..., 0
        ]
        viewer.scene.camera.orientation = ('towards', 'down', 'right')
        captured['towards'] = viewer.screenshot(canvas_only=True, flash=False)[
            ..., 0
        ]
        return captured['away'].shape == captured['towards'].shape

    # These are whole-canvas means, so they scale with how much black
    # background surrounds the volume - which makes them comparable only if
    # both frames were rendered into the same canvas. The window manager can
    # resize the canvas at any moment, and on CI it does so late and by a lot
    # (see the 376px mismatch behind `_capture_pair_agrees` in
    # `_qt/_tests/test_viewer_qt_integration.py::test_screenshot`). A resize
    # landing between these two captures inverts the comparison outright:
    # measured locally, away@(800, 1000)=170.83 against towards@(1100,
    # 1400)=104.46, which is the shape of the CI failure this replaced
    # (towards=77.16 < away=105.76). Capture the pair and retry if the canvas
    # moved underneath us, rather than trying to predict when it will stop.
    qtbot.waitUntil(_pair_captured_at_one_size, timeout=5000)
    sshot_away = captured['away']
    sshot_towards = captured['towards']

    assert np.mean(sshot_towards) > np.mean(sshot_away)
