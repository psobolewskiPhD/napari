import numpy as np

from napari.layers import Image
from napari.layers._scalar_field.scalar_field import ScalarFieldBase
from napari.utils._test_utils import (
    validate_all_params_in_docstring,
    validate_docstring_parent_class_consistency,
    validate_kwargs_sorted,
)


def test_docstring():
    validate_all_params_in_docstring(ScalarFieldBase)
    validate_kwargs_sorted(ScalarFieldBase)
    validate_docstring_parent_class_consistency(
        ScalarFieldBase, skip=('data', 'ndim', 'multiscale')
    )


def test_multiscale_thumbnail_level_prematerialized():
    """Thumbnail level should be pre-materialized as numpy for 2D multiscale."""
    rng = np.random.default_rng(0)
    data = [rng.random((128, 128)), rng.random((64, 64))]
    layer = Image(data, multiscale=True)
    assert isinstance(layer._slicing_state._thumbnail_level_data, np.ndarray)
    np.testing.assert_array_equal(
        layer._slicing_state._thumbnail_level_data,
        data[layer._thumbnail_level],
    )


def test_3d_multiscale_thumbnail_not_prematerialized():
    """3D multiscale images should NOT pre-materialize: the full volume load
    is deferred until the user switches to ndisplay=3."""
    data = [np.random.random((16, 128, 128)), np.random.random((8, 64, 64))]
    layer = Image(data, multiscale=True)
    assert layer._slicing_state._thumbnail_level_data is None


def test_single_scale_no_prematerialization():
    """Single-scale images should not pre-materialize thumbnail data."""
    layer = Image(np.random.random((32, 32)))
    assert layer._slicing_state._thumbnail_level_data is None


def test_multiscale_thumbnail_uses_prematerialized_data():
    """Slice request should receive the pre-materialized thumbnail data."""
    from napari.components import Dims

    data = [np.random.random((64, 64)), np.random.random((32, 32))]
    layer = Image(data, multiscale=True)
    dims = Dims(
        ndim=2,
        ndisplay=2,
        range=tuple((0, s - 1, 1) for s in data[0].shape),
    )
    request = layer._slicing_state._make_slice_request(dims)
    assert request.thumbnail_level_data is not None
    np.testing.assert_array_equal(
        request.thumbnail_level_data, data[layer._thumbnail_level]
    )


def test_thumbnail_level_data_refreshed_on_data_replacement():
    """Replacing layer.data must update both _thumbnail_level and
    _thumbnail_level_data so slices never read from the old array."""
    rng = np.random.default_rng(42)
    data1 = [rng.random((64, 64)), rng.random((32, 32))]
    data2 = [rng.random((90, 90)), rng.random((45, 45)), rng.random((22, 22))]

    layer = Image(data1, multiscale=True)
    assert layer._thumbnail_level == 1
    np.testing.assert_array_equal(
        layer._slicing_state._thumbnail_level_data, data1[1]
    )

    layer.data = data2
    assert layer._thumbnail_level == 2  # new last level
    np.testing.assert_array_equal(
        layer._slicing_state._thumbnail_level_data, data2[2]
    )


def test_thumbnail_level_data_uses_new_data_ndim_2d_to_3d():
    """Replacing 2D data with 3D should clear the prematerialized cache."""
    from napari.layers._multiscale_data import MultiScaleData

    rng = np.random.default_rng(7)
    data2d = [rng.random((64, 64)), rng.random((32, 32))]
    data3d = [rng.random((16, 64, 64)), rng.random((8, 32, 32))]

    layer = Image(data2d, multiscale=True)
    assert isinstance(layer._slicing_state._thumbnail_level_data, np.ndarray)

    layer._data = MultiScaleData(data3d)
    layer._reset_thumbnail_level_data()
    assert layer._slicing_state._thumbnail_level_data is None


def test_thumbnail_level_data_uses_new_data_ndim_3d_to_2d():
    """Replacing 3D data with 2D should prematerialize the new thumbnail."""
    from napari.layers._multiscale_data import MultiScaleData

    rng = np.random.default_rng(8)
    data3d = [rng.random((16, 64, 64)), rng.random((8, 32, 32))]
    data2d = [rng.random((64, 64)), rng.random((32, 32))]

    layer = Image(data3d, multiscale=True)
    assert layer._slicing_state._thumbnail_level_data is None

    layer._data = MultiScaleData(data2d)
    layer._reset_thumbnail_level_data()
    np.testing.assert_array_equal(
        layer._slicing_state._thumbnail_level_data,
        data2d[layer._thumbnail_level],
    )


def test_rgb_2d_multiscale_prematerialized():
    """RGB (H, W, 3) multiscale images have spatial ndim=2 and should be
    pre-materialized, with the full channel axis included."""
    rng = np.random.default_rng(99)
    data = [rng.random((128, 128, 3)), rng.random((64, 64, 3))]
    layer = Image(data, multiscale=True, rgb=True)

    assert layer.ndim == 2  # spatial dims only
    tld = layer._slicing_state._thumbnail_level_data
    assert isinstance(tld, np.ndarray)
    assert tld.shape == (64, 64, 3)
    np.testing.assert_array_equal(tld, data[1])
