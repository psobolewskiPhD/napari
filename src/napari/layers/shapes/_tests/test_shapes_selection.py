import numpy as np

from napari.components import ViewerModel


def test_preserve_selection_toggling_3d():
    viewer = ViewerModel()
    # Create a shapes layer with a rectangle
    shapes = [np.array([[10, 10], [10, 20], [20, 20], [20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Select the shape
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Toggle 3D mode
    viewer.dims.ndisplay = 3
    # In some cases we might need to wait for events or call _set_view_slice
    # but ViewerModel usually handles this.

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_preserve_selection_scrolling():
    viewer = ViewerModel()
    # Create a shapes layer with rectangles on different slices
    shapes = [
        np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]]),
        np.array([[2, 10, 10], [2, 10, 20], [2, 20, 20], [2, 20, 10]]),
    ]
    layer = viewer.add_shapes(shapes)

    # Select the shape on slice 0
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Scroll to next slice
    viewer.dims.set_point(0, 1)

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_preserve_selection_order_change():
    viewer = ViewerModel()
    # Create a shapes layer
    shapes = [np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Select the shape
    layer.selected_data = {0}
    assert layer.selected_data == {0}

    # Change display order
    viewer.dims.order = (0, 1, 2)

    # Selection should be preserved
    assert layer.selected_data == {0}


def test_hover_highlight_cleared_on_scrolling():
    viewer = ViewerModel()
    # Add a dummy 3D image to set the dimensions properly
    viewer.add_image(np.zeros((5, 30, 30)))
    # Create a shapes layer with a rectangle only on slice 0
    shapes = [np.array([[0, 10, 10], [0, 10, 20], [0, 20, 20], [0, 20, 10]])]
    layer = viewer.add_shapes(shapes)

    # Set mode to SELECT
    layer.mode = 'select'

    # Mock hover over shape 0 on slice 0
    layer._value = (0, None)

    # Scroll to slice 1
    viewer.dims.set_point(0, 1)

    # _value should be reset
    assert layer._value == (None, None)

    # Outline highlight should NOT be visible on slice 1
    outline_vertices, _ = layer._outline_shapes()
    assert outline_vertices is None or outline_vertices.shape[0] == 0
