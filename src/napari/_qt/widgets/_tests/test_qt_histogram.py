import threading

import numpy as np
import pytest

from napari._qt.qthreading import create_worker
from napari._qt.widgets.qt_histogram import QtHistogramWidget
from napari._qt.widgets.qt_histogram_content import QtHistogramContentWidget
from napari._qt.widgets.qt_histogram_settings import QtHistogramSettingsWidget
from napari.components.histogram import HistogramModel
from napari.layers import Image
from napari.settings import get_settings
from napari.utils.theme import get_theme


def test_qt_histogram_settings_mode_sync(qtbot):
    """Settings widget mode combobox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert widget.mode_combobox.currentText() == 'canvas'
    assert model.mode == 'canvas'

    # UI → model: changing combobox updates model
    widget.mode_combobox.setCurrentText('full')
    assert model.mode == 'full'

    # Model → UI: changing model updates combobox
    model.mode = 'canvas'
    assert widget.mode_combobox.currentText() == 'canvas'

    widget.cleanup()


def test_qt_histogram_settings_log_scale_sync(qtbot):
    """Settings widget log scale checkbox should sync bidirectionally with model."""
    layer = Image(np.random.rand(10, 10))
    model = layer.histogram
    widget = QtHistogramSettingsWidget(model)
    qtbot.addWidget(widget)

    # Default state
    assert not widget.log_scale_checkbox.isChecked()
    assert not model.log_scale

    # UI → model: toggling checkbox updates model
    widget.log_scale_checkbox.setChecked(True)
    assert model.log_scale

    # Model → UI: changing model updates checkbox
    model.log_scale = False
    assert not widget.log_scale_checkbox.isChecked()

    widget.cleanup()


def test_qt_histogram_content_composition_and_cleanup(qtbot):
    """Content widget should create histogram + settings children and clean up."""
    layer = Image(np.random.rand(10, 10))
    content = QtHistogramContentWidget(layer)
    qtbot.addWidget(content)

    # Both child widgets exist
    assert content.histogram_widget is not None
    assert content.settings_widget is not None
    assert content.settings_widget.mode_combobox is not None
    assert content.settings_widget.log_scale_checkbox is not None

    # Settings controls are functional
    content.settings_widget.mode_combobox.setCurrentText('full')
    assert layer.histogram.mode == 'full'
    content.settings_widget.log_scale_checkbox.setChecked(True)
    assert layer.histogram.log_scale

    # Cleanup does not crash
    content.cleanup()


def test_qt_histogram_widget_updates_theme(qtbot):
    settings = get_settings()
    old_theme = settings.appearance.theme
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    layer.histogram.enabled = True
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    layer.histogram.compute()

    try:
        settings.appearance.theme = 'light'
        light_theme = get_theme('light')

        qtbot.waitUntil(
            lambda: np.allclose(
                widget.canvas.bgcolor.rgba[:3],
                np.array(light_theme.canvas.as_rgb_tuple()) / 255,
            )
        )

        assert widget.histogram_visual._lut_color == (
            *(
                np.array(light_theme.highlight.as_rgb_tuple(), dtype=float)
                / 255
            ),
            0.95,
        )
        assert widget.histogram_visual._axes_color == (
            *(np.array(light_theme.text.as_rgb_tuple(), dtype=float) / 255),
            0.7,
        )
    finally:
        settings.appearance.theme = old_theme
        widget.cleanup()


def test_qt_histogram_widget_updates_from_settings_theme(
    make_napari_viewer, qtbot
):
    """Histogram widget responds to theme changes via settings (canonical source)."""
    settings = get_settings()
    old_theme = settings.appearance.theme
    viewer = make_napari_viewer()
    layer = viewer.add_image(
        np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    )
    controls = viewer.window._qt_viewer.controls.widgets[layer]
    controls._histogram_control.ensure_content()
    widget = controls._histogram_control.histogram_widget
    assert widget is not None

    qtbot.addWidget(widget)
    layer.histogram.enabled = True
    layer.histogram.compute()

    try:
        settings.appearance.theme = 'light'
        light_theme = get_theme('light')

        qtbot.waitUntil(
            lambda: np.allclose(
                widget.canvas.bgcolor.rgba[:3],
                np.array(light_theme.canvas.as_rgb_tuple()) / 255,
            )
        )

        assert widget.histogram_visual._lut_color == (
            *(
                np.array(light_theme.highlight.as_rgb_tuple(), dtype=float)
                / 255
            ),
            0.95,
        )
    finally:
        settings.appearance.theme = old_theme


def test_qt_histogram_async_compute_with_dask(qtbot):
    """Histogram compute on chunked dask data should work via create_worker."""
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    model = layer.histogram
    model.mode = 'full'
    model.max_samples = 50000

    done = [False]
    result = [None]

    def _work():
        list(model.compute())
        return model._bin_edges, model._counts

    def _on_done(bins_counts):
        result[0] = bins_counts
        done[0] = True

    worker = create_worker(_work)
    worker.returned.connect(_on_done)
    worker.start()

    qtbot.waitUntil(lambda: done[0], timeout=10000)

    assert result[0] is not None
    bins, counts = result[0]
    assert len(bins) == 257
    assert len(counts) == 256
    assert counts.sum() > 0


def test_qt_histogram_sequential_async_with_param_change(qtbot):
    """Sequential async computes with different parameters should each produce correct results.

    This tests the typical pattern where one async compute
    finishes before the next is triggered by a parameter change.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    model = layer.histogram
    model.mode = 'full'
    model.max_samples = 50000

    done = [False]
    result = [None]

    def _work():
        list(model.compute())
        return model._bin_edges, model._counts

    def _on_done(bins_counts):
        result[0] = bins_counts
        done[0] = True

    # Start first worker and wait for it to complete
    worker1 = create_worker(_work)
    worker1.returned.connect(_on_done)
    worker1.start()
    qtbot.waitUntil(lambda: done[0], timeout=30000)
    assert result[0] is not None
    bins1, counts1 = result[0]
    assert len(bins1) == 257
    assert counts1.sum() > 0

    # Change a parameter and run a second async compute
    done[0] = False
    result[0] = None
    model.bins = 128

    worker2 = create_worker(_work)
    worker2.returned.connect(_on_done)
    worker2.start()
    qtbot.waitUntil(lambda: done[0], timeout=30000)
    assert result[0] is not None
    bins2, counts2 = result[0]
    assert len(bins2) == 129  # bins=128 → 129 bin edges
    assert counts2.sum() > 0


def test_qt_histogram_teardown_during_async_compute(qtbot):
    """Closing a histogram widget while an async compute is in flight
    should not crash.  This guards against the race where a background
    worker's ``finished`` signal fires after ``cleanup()`` has already
    disconnected psygnal events and destroyed the canvas.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'
    layer.histogram.max_samples = 50000

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)
    layer.histogram.enabled = True

    # Trigger async compute by calling the internal path used in production
    widget._ensure_histogram_computed()
    # Capture the worker before cleanup (cleanup sets _compute_worker to None)
    worker = widget._compute_worker

    # Immediately clean up — simulates viewer close while worker is running
    widget.cleanup()

    # No crash = success

    # Wait for the worker to finish so the thread pool drains before
    # the conftest's _dangling_qthread_pool fixture checks for leaks.
    if worker is not None:
        from qtpy.QtCore import QThreadPool

        qtbot.waitUntil(
            lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
            timeout=10000,
        )


def test_qt_histogram_widget_ensure_computed_worker_cancel(qtbot):
    """Calling _ensure_histogram_computed while a worker is already running
    should cancel the previous worker."""
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'
    layer.histogram.max_samples = 50000

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)
    layer.histogram.enabled = True

    # First call starts a worker
    widget._ensure_histogram_computed()
    first_worker = widget._compute_worker
    assert first_worker is not None

    # Second call should cancel the first and start a new one
    widget._ensure_histogram_computed()
    if widget._compute_worker is not None:
        # If a new worker was created (the first had not yet finished),
        # it should be a different object. If the first already finished,
        # _compute_worker may be None — that's OK too.
        assert widget._compute_worker is not first_worker

    widget.cleanup()

    # Drain thread pool
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


def test_qt_histogram_widget_ensure_content_disconnect(qtbot):
    """Calling cleanup on QtHistogramContentWidget should disconnect events."""
    layer = Image(np.random.rand(10, 10))
    content = QtHistogramContentWidget(layer)
    qtbot.addWidget(content)

    layer.histogram.enabled = True
    layer.histogram.compute()

    # Before cleanup, changing log_scale triggers recompute
    assert content.histogram_widget._updating is not None  # widget is alive

    content.cleanup()
    # After cleanup, changing log_scale should not crash
    layer.histogram.log_scale = True


def test_histogram_visual_set_data_clear_path(qtbot):
    """Calling set_data with no bins/counts should clear the visual."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # First set some data to get a non-empty state
    layer.histogram.enabled = True
    list(layer.histogram.compute())
    visual.set_data(
        bin_edges=layer.histogram._bin_edges,
        counts=layer.histogram.counts,
        gamma=1.0,
        clims=(0.25, 0.75),
        data_range=(0, 1),
    )

    # Now call set_data with None to trigger _clear path
    visual.set_data()
    # After clear, gamma should be reset to 1.0
    assert visual._gamma == 1.0
    assert visual._clims is None
    assert visual._data_range is None

    widget.cleanup()


def test_histogram_visual_update_lut_line_clims_equal(qtbot):
    """LUT line should handle equal clim values gracefully."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual
    layer.histogram.enabled = True
    list(layer.histogram.compute())

    # Call with clims where min == max
    visual.set_data(
        bin_edges=layer.histogram._bin_edges,
        counts=layer.histogram.counts,
        gamma=1.0,
        clims=(0.5, 0.5),  # equal clims
        data_range=(0, 1),
    )
    # Should not crash; uses the else branch in _update_lut_line
    assert visual._clims == (0.5, 0.5)

    widget.cleanup()


def test_histogram_visual_destroy(qtbot):
    """Calling destroy on the histogram visual should clean up sub-visuals."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # destroy should not crash
    visual.destroy()

    widget.cleanup()


def test_histogram_visual_update_bars_empty(qtbot):
    """_update_bars with fewer than 2 bins should call _set_empty_data."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # Call _update_bars directly with a single bin (len(bins) < 2)
    visual._update_bars(np.array([0.0]), np.array([5.0]))
    # Should not crash; calls _set_empty_data internally
    # After _set_empty_data, the bars mesh should have 3 dummy vertices
    assert visual._bars.mesh_data.get_vertices() is not None

    widget.cleanup()


def test_histogram_visual_update_bars_zero_range(qtbot):
    """_update_bars should handle zero bin range (all bins identical)."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    visual = widget.histogram_visual

    # All bins have the same value → bin_range == 0 → should use bin_range = 1
    bins = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    counts = np.array([10.0, 5.0], dtype=np.float32)
    visual._update_bars(bins, counts)
    # Should not crash; with 2 bins, should produce 8 vertices (4 per bar)
    vertices = visual._bars.mesh_data.get_vertices()
    assert vertices is not None
    assert len(vertices) == 8, (
        'zero-range bars should produce 8 vertices for 2 bins'
    )

    widget.cleanup()


def test_qt_histogram_layer_bar_color(qtbot):
    """_layer_bar_color should return a 4-tuple based on the layer's colormap."""
    layer = Image(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8))
    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    # Default colormap (gray) → bar color should be a 4-element tuple
    color = widget._layer_bar_color()
    assert len(color) == 4
    assert all(0 <= c <= 1 for c in color)

    # With a reversed colormap, the bar color should still be non-zero
    # (the method uses map([0.8]) to avoid black-on-black for gray_r)
    layer.colormap = 'gray_r'
    color_r = widget._layer_bar_color()
    assert len(color_r) == 4
    # Even on a reversed colormap, the 0.8 position is near-white, so at
    # least one channel should be > 0.5.
    assert any(c > 0.5 for c in color_r), (
        f'gray_r bar color should be light, got {color_r}'
    )

    widget.cleanup()


def test_qt_histogram_mode_switch_uses_async_for_chunked_data(
    qtbot, monkeypatch
):
    """Switching to full mode on chunked data should use async compute,
    not block the main thread on synchronous chunk I/O.

    Regression test for the issue where setting mode='full' on a
    dask- or zarr-backed Image layer would synchronously iterate
    chunks in HistogramModel._mark_dirty()/compute(), blocking the
    viewer while each chunk was loaded over I/O (e.g. remote zarr).

    The fix: _mark_dirty() skips compute() for chunked+full data,
    and QtHistogramWidget._on_model_mode_change() triggers the
    GeneratorWorker-based async path instead.
    """
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((500, 500), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.enabled = True

    # Record which thread each chunk read happens on. The regression this
    # test guards is *chunk I/O blocking the main thread*, so record that
    # directly rather than inferring it from `_dirty`: the async worker
    # clears `_dirty` from its own thread, so a fast worker can clear it
    # before the main thread looks, and the old assertion then reported a
    # synchronous compute when the compute had in fact correctly gone async.
    main_thread = threading.get_ident()
    chunk_load_threads: list[int] = []
    real_load_chunk = HistogramModel._load_chunk

    def recording_load_chunk(data, flat_idx):
        chunk_load_threads.append(threading.get_ident())
        return real_load_chunk(data, flat_idx)

    monkeypatch.setattr(
        HistogramModel, '_load_chunk', staticmethod(recording_load_chunk)
    )

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    # Initial state: model is clean from the canvas-mode compute that
    # ran during __init__, which does not touch chunks.
    assert not layer.histogram._dirty
    assert chunk_load_threads == []

    # Switch to full mode.  In the buggy code this would trigger
    # _mark_dirty() → compute() → synchronous chunk iteration.
    layer.histogram.mode = 'full'

    # The regression: _mark_dirty() iterating compute() inline would have
    # loaded chunks on this thread before the assignment returned.
    assert main_thread not in chunk_load_threads, (
        '_mark_dirty() called compute() synchronously on mode switch with '
        'chunked data — this would block the main thread on chunk I/O'
    )

    # The widget should have started an async worker via
    # _on_model_mode_change() → _ensure_histogram_computed(). `_compute_worker`
    # is only ever written on the main thread (set here, cleared from the
    # queued `finished` handler), and we have not re-entered the event loop,
    # so this cannot race with the worker.
    assert widget._compute_worker is not None

    qtbot.waitUntil(
        lambda: not layer.histogram._dirty,
        timeout=30000,
    )

    # Non-vacuous: chunks really were read, and never on the main thread.
    assert chunk_load_threads
    assert main_thread not in chunk_load_threads

    # Verify valid histogram results from the async path
    assert len(layer.histogram._bin_edges) == 257
    assert len(layer.histogram.counts) == 256
    assert layer.histogram.counts.sum() > 0

    widget.cleanup()


def _full_data_counts(base, hist, clims_range):
    """Ground-truth full-data histogram for comparison with the model."""
    ground_truth, _ = np.histogram(
        base.ravel(),
        bins=hist.bins,
        range=tuple(float(v) for v in clims_range),
    )
    return ground_truth.astype(np.int64)


def test_two_views_share_single_worker_and_both_animate(qtbot):
    """Two histogram views on one layer (inline + popup) share a single
    compute worker, yet *both* animate progressively — regardless of which
    view owns the worker.

    Regression test for the popup staying blank while the inline animated:
    the two views used to run competing workers over the shared model,
    corrupting the progressive accumulation, and only the worker-owning view
    rendered chunk-by-chunk.
    """
    dask = pytest.importorskip('dask.array')
    # Deterministic data with all chunks sampled (size < max_samples), so the
    # final accumulation is exactly a full-data np.histogram.
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    layer = Image(dask.from_array(base, chunks=(32, 32)))  # 64 chunks
    layer.histogram.mode = 'full'

    # Whichever view reaches _ensure_histogram_computed first owns the single
    # worker; the other must still animate from partial_computed broadcasts.
    view_a = QtHistogramWidget(layer)
    view_b = QtHistogramWidget(layer)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    # Count how many times each view's visual is actually redrawn.
    draws = {'a': 0, 'b': 0}
    orig_a = view_a.histogram_visual.set_data
    orig_b = view_b.histogram_visual.set_data

    def count_a(*args, **kwargs):
        draws['a'] += 1
        return orig_a(*args, **kwargs)

    def count_b(*args, **kwargs):
        draws['b'] += 1
        return orig_b(*args, **kwargs)

    view_a.histogram_visual.set_data = count_a
    view_b.histogram_visual.set_data = count_b

    hist = layer.histogram
    layer.histogram.enabled = True  # triggers the shared async compute

    qtbot.waitUntil(
        lambda: not hist._compute_scheduled and not hist._dirty,
        timeout=15000,
    )

    # Neither view was left blank. That is the deterministic core of the
    # regression this guards - the popup showed nothing at all while the
    # inline view rendered.
    #
    # Deliberately not `draws[...] > 1`, nor `draws['a'] == draws['b']`.
    # Both are races between the worker producing chunks and the main thread
    # dispatching the queued `yielded` signals, and the draw count is not
    # merely late but *lossy*: `_on_partial_histogram` returns early when
    # `_histogram._dirty` is False, and finishing the compute clears
    # `_dirty`, so a backlog of partials is discarded rather than delayed.
    # Measured by blocking the main thread for 1.5s while the worker runs,
    # draws goes from {'a': 64, 'b': 64} to {'a': 1, 'b': 1} - and with the
    # two views starting at different times, to {'a': 1, 'b': 2}. That is
    # what failed on macOS CI, and no timeout recovers a dropped signal; a
    # `qtbot.wait(50)` and then a `waitUntil` both tried and both failed.
    #
    # The progressive animation and the broadcast-to-every-view behaviour are
    # covered deterministically by
    # `test_partial_histogram_broadcasts_to_all_views` below, which drives the
    # partial path instead of racing it.
    assert draws['a'] >= 1, f'view_a never rendered: {draws}'
    assert draws['b'] >= 1, f'view_b never rendered: {draws}'
    # Single-worker invariant held to completion — nothing left dangling.
    assert not hist._compute_scheduled
    assert view_a._compute_worker is None
    assert view_b._compute_worker is None
    # The result is the full accumulation, not a single-chunk fragment.
    assert np.array_equal(
        hist._counts.astype(np.int64),
        _full_data_counts(base, hist, layer.contrast_limits_range),
    )

    view_a.cleanup()
    view_b.cleanup()
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


def test_partial_histogram_broadcasts_to_all_views(qtbot):
    """A partial result from one view's worker redraws *every* view.

    This is the progressive-animation half of
    `test_two_views_share_single_worker_and_both_animate`, driven directly
    rather than raced. Going through the real worker makes the draw count a
    function of how promptly the main thread services the queued `yielded`
    signals - and `_on_partial_histogram` drops partials once
    `_histogram._dirty` is cleared by the compute finishing, so a starved
    main thread loses them outright instead of merely arriving late.
    """
    dask = pytest.importorskip('dask.array')
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    layer = Image(dask.from_array(base, chunks=(32, 32)))
    layer.histogram.mode = 'full'

    view_a = QtHistogramWidget(layer)
    view_b = QtHistogramWidget(layer)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    draws = {'a': 0, 'b': 0}
    for key, view in (('a', view_a), ('b', view_b)):
        orig = view.histogram_visual.set_data

        def count(*args, _key=key, _orig=orig, **kwargs):
            draws[_key] += 1
            return _orig(*args, **kwargs)

        view.histogram_visual.set_data = count

    hist = layer.histogram
    # `_on_partial_histogram` is a no-op unless a compute is outstanding
    hist._dirty = True

    n_partials = 3
    for i in range(1, n_partials + 1):
        counts = np.full(hist.bins, i, dtype=np.int64)
        bin_edges = np.linspace(0, 1, hist.bins + 1)
        view_a._on_partial_histogram((bin_edges, counts))

    assert draws == {'a': n_partials, 'b': n_partials}, (
        'each partial from one view must redraw both views once'
    )

    view_a.cleanup()
    view_b.cleanup()


def test_closing_owning_view_mid_compute_hands_off_to_survivor(
    qtbot, monkeypatch
):
    """Closing the view that owns the in-flight worker nudges a surviving
    view to finish the compute, instead of stranding it with partial data.

    Covers closing the contrast-limits popup mid-load while the inline
    histogram remains open.
    """
    dask = pytest.importorskip('dask.array')
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    layer = Image(dask.from_array(base, chunks=(32, 32)))
    layer.histogram.mode = 'full'

    # Park the worker inside its first chunk load so the compute is
    # *guaranteed* unfinished when the owning view closes.
    #
    # "Still computing" is not a state the main thread can assert, only one it
    # can enforce: `_compute_chunked_progressive` clears `_dirty` from the
    # worker thread, and these 64 chunks of 1024 elements are ~1ms of numpy
    # (which drops the GIL), so losing the GIL for a single switch interval
    # after `worker.start()` is enough for the whole compute to land first.
    # Asserting `_dirty` unguarded raced against that and failed on CI.
    #
    # Blocking in `_load_chunk` also exercises the mid-chunk branch that
    # `QtHistogramWidget.cleanup` documents but nothing else reaches: the
    # generator is stopped while stuck on a chunk read rather than between
    # iterations.
    entered_chunk_load = threading.Event()
    release_chunk_load = threading.Event()
    real_load_chunk = HistogramModel._load_chunk

    def gated_load_chunk(data, flat_idx):
        entered_chunk_load.set()
        # A timeout rather than a bare wait, so a regression that never
        # releases the worker fails the test instead of hanging the suite.
        release_chunk_load.wait(timeout=30)
        return real_load_chunk(data, flat_idx)

    monkeypatch.setattr(
        HistogramModel, '_load_chunk', staticmethod(gated_load_chunk)
    )

    view_a = QtHistogramWidget(layer)
    view_b = QtHistogramWidget(layer)
    qtbot.addWidget(view_a)
    qtbot.addWidget(view_b)

    hist = layer.histogram
    layer.histogram.enabled = True  # starts the shared worker synchronously

    # Identify the owning view and close it before the compute finishes.
    assert entered_chunk_load.wait(timeout=10), (
        'worker never reached _load_chunk'
    )
    assert hist._compute_scheduled
    if view_a._compute_worker is not None:
        owner, survivor = view_a, view_b
    else:
        owner, survivor = view_b, view_a
    assert hist._dirty  # compute is parked in _load_chunk, so cannot be done
    owner.cleanup()
    release_chunk_load.set()

    # The survivor must take over and finish the compute correctly.
    qtbot.waitUntil(
        lambda: not hist._compute_scheduled and not hist._dirty,
        timeout=15000,
    )
    assert survivor._compute_worker is None
    assert np.array_equal(
        hist._counts.astype(np.int64),
        _full_data_counts(base, hist, layer.contrast_limits_range),
    )

    survivor.cleanup()
    from qtpy.QtCore import QThreadPool

    qtbot.waitUntil(
        lambda: QThreadPool.globalInstance().activeThreadCount() == 0,
        timeout=10000,
    )


def test_persistent_chunk_load_error_does_not_retry_forever(
    qtbot, monkeypatch
):
    """Persistent chunk-load error must not spawn infinite retry workers."""
    dask = pytest.importorskip('dask.array')
    data = dask.random.random((200, 200), chunks=(50, 50))
    layer = Image(data)
    layer.histogram.mode = 'full'

    widget = QtHistogramWidget(layer)
    qtbot.addWidget(widget)

    load_calls = {'n': 0}

    def boom(*args, **kwargs):
        load_calls['n'] += 1
        raise OSError('simulated remote chunk read failure')

    monkeypatch.setattr(HistogramModel, '_load_chunk', staticmethod(boom))

    # Enabling triggers the shared async compute for chunked full-mode data.
    layer.histogram.enabled = True

    # Wait for the single worker to run, fail, and release the compute slot.
    qtbot.waitUntil(
        lambda: (
            not layer.histogram._compute_scheduled
            and widget._compute_worker is None
        ),
        timeout=15000,
    )

    # Give the event loop room to spin: a retry loop would keep spawning
    # workers (and calling _load_chunk) during this window.
    qtbot.wait(300)

    # Exactly one worker ran — the persistent failure was not retried.
    assert load_calls['n'] == 1
    # The model correctly stayed dirty (no valid result was produced) ...
    assert layer.histogram._dirty
    # ... and the compute slot was released, not leaked.
    assert not layer.histogram._compute_scheduled
    assert widget._compute_worker is None

    widget.cleanup()
