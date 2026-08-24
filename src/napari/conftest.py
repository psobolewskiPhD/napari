"""

Notes for using the plugin-related fixtures here:

1. The `npe2pm_` fixture is always used, and it mocks the global npe2 plugin
   manager instance with a discovery-deficient plugin manager.  No plugins should be
   discovered in tests without explicit registration.
2. wherever the builtins need to be tested, the `builtins` fixture should be explicitly
   added to the test.  (it's a DynamicPlugin that registers our builtins.yaml with the
   global mock npe2 plugin manager)
3. wherever *additional* plugins or contributions need to be added, use the `tmp_plugin`
   fixture, and add additional contributions _within_ the test (not in the fixture):
    ```python
    def test_something(tmp_plugin):
        @tmp_plugin.contribute.reader(filname_patterns=["*.ext"])
        def f(path): ...

        # the plugin name can be accessed at:
        tmp_plugin.name
    ```
4. If you need a _second_ mock plugin, use `tmp_plugin.spawn(register=True)` to create
   another one.
   ```python
   new_plugin = tmp_plugin.spawn(register=True)

   @new_plugin.contribute.reader(filename_patterns=["*.tiff"])
   def get_reader(path):
       ...
   ```
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import timedelta
from functools import partial
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from unittest.mock import MagicMock
from weakref import WeakKeyDictionary

import dask.threaded
import numpy as np
import pytest
from _pytest.pathlib import bestrelpath
from IPython.core.history import HistoryManager
from npe2 import PackageMetadata
from packaging.version import parse as parse_version
from pytest_pretty import CustomTerminalReporter

from napari.components import LayerList
from napari.layers import Image, Labels, Points, Shapes, Vectors
from napari.utils.misc import ROOT_DIR

if TYPE_CHECKING:
    from collections.abc import Container

    from npe2._pytest_plugin import TestPluginManager
    from pytestqt.qtbot import QtBot

    from napari._qt.qt_viewer import QtViewer
    from napari.components import ViewerModel


# touch ~/.Xauthority for Xlib support, must happen before importing pyautogui
if os.getenv('CI') and sys.platform.startswith('linux'):
    xauth = Path('~/.Xauthority').expanduser()
    if not xauth.exists():
        xauth.touch()


@pytest.fixture
def layer_data_and_types():
    """Fixture that provides some layers and filenames

    Returns
    -------
    tuple
        ``layers, layer_data, layer_types, filenames``

        - layers: some image and points layers
        - layer_data: same as above but in LayerData form
        - layer_types: list of strings with type of layer
        - filenames: the expected filenames with extensions for the layers.
    """
    layers = [
        Image(np.random.rand(20, 20), name='ex_img'),
        Image(np.random.rand(20, 20)),
        Points(np.random.rand(20, 2), name='ex_pts'),
        Points(
            np.random.rand(20, 2), properties={'values': np.random.rand(20)}
        ),
    ]
    extensions = ['.tif', '.tif', '.csv', '.csv']
    layer_data = [layer.as_layer_data_tuple() for layer in layers]
    layer_types = [layer._type_string for layer in layers]
    filenames = [
        layer.name + e for layer, e in zip(layers, extensions, strict=False)
    ]
    return layers, layer_data, layer_types, filenames


@pytest.fixture
def surface_data() -> tuple[
    np.ndarray[tuple[int, Literal[2]], np.dtype[np.float32]],
    np.ndarray[tuple[int], np.dtype[np.int32]],
    np.ndarray[tuple[int], np.dtype[np.float32]],
]:
    data = np.array([[0, 0], [0, 20], [10, 0], [10, 10]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    values = np.linspace(0, 1, len(data), dtype=np.float32)
    return (data, faces, values)


class TrackDataDict(TypedDict):
    data: np.ndarray[tuple[int, Literal[4]], np.dtype[np.float32]]
    properties: dict[Literal['track_id', 'time', 'speed'], list]


@pytest.fixture
def tracks_data() -> TrackDataDict:
    data = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 20], [1, 0, 10, 0], [1, 1, 10, 10]],
        dtype=np.float32,
    )
    properties: dict[Literal['track_id', 'time', 'speed'], list[int]] = {
        'track_id': [0, 0, 1, 1],
        'time': [0, 1, 0, 1],
        'speed': [50, 30, 20, 10],
    }
    return {'data': data, 'properties': properties}


@pytest.fixture
def vectors_data() -> np.ndarray[
    tuple[int, Literal[2], Literal[2]], np.dtype[np.float32]
]:
    return np.array([[[0, 0], [0, 20]], [[10, 0], [10, 10]]], dtype=np.float32)


@pytest.fixture(
    params=[
        'image',
        'labels',
        'points',
        'shapes',
        'shapes-rectangles',
        'vectors',
    ]
)
def layer(request):
    """Parameterized fixture that supplies a layer for testing.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The pytest request object

    Returns
    -------
    napari.layers.Layer
        The desired napari Layer.
    """
    np.random.seed(0)
    if request.param == 'image':
        data = np.random.rand(20, 20)
        return Image(data)
    if request.param == 'labels':
        data = np.random.randint(10, size=(20, 20))
        return Labels(data)
    if request.param == 'points':
        data = np.random.rand(20, 2)
        return Points(data)
    if request.param == 'shapes':
        data = [
            np.random.rand(2, 2),
            np.random.rand(2, 2),
            np.random.rand(6, 2),
            np.random.rand(6, 2),
            np.random.rand(2, 2),
        ]
        shape_type = ['ellipse', 'line', 'path', 'polygon', 'rectangle']
        return Shapes(data, shape_type=shape_type)
    if request.param == 'shapes-rectangles':
        data = np.random.rand(7, 4, 2)
        return Shapes(data)
    if request.param == 'vectors':
        data = np.random.rand(20, 2, 2)
        return Vectors(data)

    return None


@pytest.fixture
def layers():
    """Fixture that supplies a layers list for testing.

    Returns
    -------
    napari.components.LayerList
        The desired napari LayerList.
    """
    np.random.seed(0)
    list_of_layers = [
        Image(np.random.rand(20, 20)),
        Labels(np.random.randint(10, size=(20, 2))),
        Points(np.random.rand(20, 2)),
        Shapes(np.random.rand(10, 2, 2)),
        Vectors(np.random.rand(10, 2, 2)),
    ]
    return LayerList(list_of_layers)


@pytest.fixture(autouse=True)
def _skip_examples(request):
    """Skip examples test if ."""
    if request.node.get_closest_marker(
        'examples'
    ) and request.config.getoption('--skip_examples'):
        pytest.skip('running with --skip_examples')


# _PYTEST_RAISE=1 will prevent pytest from handling exceptions.
# Use with a debugger that's set to break on "unhandled exceptions".
# https://github.com/pytest-dev/pytest/issues/7409
if os.getenv('_PYTEST_RAISE', '0') != '0':

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


@pytest.fixture(autouse=True)
def _fresh_settings(monkeypatch):
    """This fixture ensures that default settings are used for every test.

    and ensures that changes to settings in a test are reverted, and never
    saved to disk.
    """
    from napari import settings
    from napari.settings import NapariSettings
    from napari.settings._experimental import ExperimentalSettings
    from napari.utils.triangulation_backend import TriangulationBackend

    # prevent the developer's config file from being used if it exists
    cp = NapariSettings.model_fields['config_path']
    monkeypatch.setattr(cp, 'default', None)

    monkeypatch.setattr(
        ExperimentalSettings.model_fields['triangulation_backend'],
        'default',
        TriangulationBackend.fastest_available,
    )

    # calling save() with no config path is normally an error
    # here we just have save() return if called without a valid path
    NapariSettings.__original_save__ = NapariSettings.save

    def _mock_save(self, path=None, **dict_kwargs):
        if not (path or self.config_path):
            return
        NapariSettings.__original_save__(self, path, **dict_kwargs)

    monkeypatch.setattr(NapariSettings, 'save', _mock_save)

    settings._SETTINGS = None
    # this makes sure that we start with fresh settings for every test.
    return


def _reset_dask_threadpool() -> None:
    """Shut down `dask.threaded.default_pool` and clear it, if there is one."""
    pool = dask.threaded.default_pool
    if isinstance(pool, ThreadPool):
        pool.close()
        pool.join()
    elif pool is not None:
        pool.shutdown()
    dask.threaded.default_pool = None


@pytest.fixture(autouse=True)
def _auto_shutdown_dask_threadworkers():
    """
    This automatically shutdown dask thread workers.

    We don't assert the number of threads in unchanged as other things
    modify the number of threads.
    """
    # This fixture's own `finally` below always resets the pool, so a leftover
    # one here means a previous test leaked it. Reset before failing so that
    # the leak does not cascade into failures across the rest of an xdist
    # worker's tests.
    if dask.threaded.default_pool is not None:
        _reset_dask_threadpool()
        pytest.fail(
            'dask.threaded.default_pool was not None at test start '
            '(a previous test likely leaked it); it has been reset.'
        )

    try:
        yield
    finally:
        _reset_dask_threadpool()


@pytest.fixture(autouse=True)
def _auto_shutdown_zarr_iothread():
    """
    This automatically shuts down zarr's background IO thread and executor.

    Like dask's threadpool above, zarr lazily creates a long-lived background
    thread (and threadpool executor) for async I/O on first use, and only
    tears it down at process exit via `atexit`. Left dangling between tests,
    delayed work on that thread can end up touching objects that a later
    test has already torn down.
    """
    try:
        yield
    finally:
        # Nothing to do unless zarr's sync machinery was actually imported:
        # the loop lives inside that module and is created lazily, so a test
        # that never touched zarr cannot have one. Checking `sys.modules`
        # rather than importing keeps every unrelated test's teardown from
        # pulling zarr in.
        if 'zarr.core.sync' in sys.modules:
            # `zarr.core.sync` is zarr's internal module, not public API, so
            # pin the behaviour we rely on rather than let a zarr refactor
            # turn every test's teardown into an ImportError: `loop` is a
            # one-element list holding the lazily created event loop, and
            # `cleanup_resources()` shuts it (and the executor) down and sets
            # `loop[0]` back to None. Verified against zarr 3.x; `napari`
            # requires zarr>=3.0.8.
            try:
                from zarr.core.sync import cleanup_resources, loop
            except ImportError:  # pragma: no cover - zarr internals moved
                warnings.warn(
                    'Could not reach zarr.core.sync to shut down its '
                    'background IO thread between tests; zarr internals have '
                    'moved and this fixture needs updating.',
                    stacklevel=2,
                )
            else:
                if loop[0] is not None:
                    cleanup_resources()


# this is not the proper way to configure IPython, but it's an easy one.
# This will prevent IPython to try to write history on its sql file and do
# everything in memory.
# 1) it saves a thread and
# 2) it can prevent issues with slow or read-only file systems in CI.
HistoryManager.enabled = False


@pytest.fixture
def napari_svg_name():
    """the plugin name changes with npe2 to `napari-svg` from `svg`."""
    from importlib.metadata import version

    if parse_version(version('napari-svg')) < parse_version('0.1.6'):
        return 'svg'

    return 'napari-svg'


@pytest.fixture(autouse=True)
def npe2pm_(npe2pm, monkeypatch):
    """Autouse npe2 & npe1 mock plugin managers with no registered plugins."""
    return npe2pm


@pytest.fixture
def mock_pm(npe2pm: TestPluginManager, manifest_path: str):
    from napari.plugins import _initialize_plugins

    _initialize_plugins.cache_clear()
    mock_reg = MagicMock()
    npe2pm._command_registry = mock_reg
    with npe2pm.tmp_plugin(manifest=manifest_path):
        yield npe2pm


@pytest.fixture(autouse=True)
def plugin_settings_(plugin_settings):
    """Autouse `plugin_settings` so `get_plugin_settings` is fresh for each test.

    Without this, whichever test happens to call `get_plugin_settings`
    first (e.g. by constructing a `PreferencesDialog`) would populate and
    freeze `_PLUGIN_SETTINGS` for the rest of the session, against the
    real user config directory.
    """
    return plugin_settings


@pytest.fixture
def builtins(npe2pm_: TestPluginManager):
    with npe2pm_.tmp_plugin(package='napari') as plugin:
        yield plugin


@pytest.fixture
def tmp_plugin(npe2pm_: TestPluginManager):
    with npe2pm_.tmp_plugin() as plugin:
        plugin.manifest.package_metadata = PackageMetadata(
            version='0.1.0', name='test'
        )
        plugin.manifest.display_name = 'Temp Plugin'
        yield plugin


@pytest.fixture
def manifest_path() -> str:
    path_to = (
        Path(__file__)
        .parent.joinpath('plugins', '_tests', '_sample_manifest.yaml')
        .resolve()
    )
    assert path_to.exists(), f'Manifest path {path_to} does not exist.'
    return str(path_to)


@pytest.fixture
def viewer_model() -> ViewerModel:
    from napari.components import ViewerModel

    return ViewerModel()


@pytest.fixture
def qt_viewer_(
    qtbot: QtBot, viewer_model: ViewerModel, monkeypatch: pytest.MonkeyPatch
) -> QtViewer:
    from napari._qt.qt_viewer import QtViewer

    viewer = QtViewer(viewer_model)

    original_controls = viewer.__class__.controls.fget  # type: ignore[attr-defined]
    original_layers = viewer.__class__.layers.fget  # type: ignore[attr-defined]
    original_layer_buttons = viewer.__class__.layerButtons.fget  # type: ignore[attr-defined]
    original_viewer_buttons = viewer.__class__.viewerButtons.fget  # type: ignore[attr-defined]
    original_dock_layer_list = viewer.__class__.dockLayerList.fget  # type: ignore[attr-defined]
    original_dock_layer_controls = viewer.__class__.dockLayerControls.fget  # type: ignore[attr-defined]
    original_dock_console = viewer.__class__.dockConsole.fget  # type: ignore[attr-defined]
    original_dock_performance = viewer.__class__.dockPerformance.fget  # type: ignore[attr-defined]

    def hide_widget(widget):
        widget.hide()

    def hide_and_clear_qt_viewer(viewer: QtViewer):
        viewer._instances.clear()
        viewer.hide()

    def patched_controls(self):
        if self._controls is None:
            self._controls = original_controls(self)
            qtbot.addWidget(self._controls, before_close_func=hide_widget)
        return self._controls

    def patched_layers(self):
        if self._layers is None:
            self._layers = original_layers(self)
            qtbot.addWidget(self._layers, before_close_func=hide_widget)
        return self._layers

    def patched_layer_buttons(self):
        if self._layersButtons is None:
            self._layersButtons = original_layer_buttons(self)
            qtbot.addWidget(self._layersButtons, before_close_func=hide_widget)
        return self._layersButtons

    def patched_viewer_buttons(self):
        if self._viewerButtons is None:
            self._viewerButtons = original_viewer_buttons(self)
            qtbot.addWidget(self._viewerButtons, before_close_func=hide_widget)
        return self._viewerButtons

    def patched_dock_layer_list(self):
        if self._dockLayerList is None:
            self._dockLayerList = original_dock_layer_list(self)
            qtbot.addWidget(self._dockLayerList, before_close_func=hide_widget)
        return self._dockLayerList

    def patched_dock_layer_controls(self):
        if self._dockLayerControls is None:
            self._dockLayerControls = original_dock_layer_controls(self)
            qtbot.addWidget(
                self._dockLayerControls, before_close_func=hide_widget
            )
        return self._dockLayerControls

    def patched_dock_console(self):
        if self._dockConsole is None:
            self._dockConsole = original_dock_console(self)
            qtbot.addWidget(self._dockConsole, before_close_func=hide_widget)
        return self._dockConsole

    def patched_dock_performance(self):
        if self._dockPerformance is None:
            self._dockPerformance = original_dock_performance(self)
            qtbot.addWidget(
                self._dockPerformance, before_close_func=hide_widget
            )
        return self._dockPerformance

    monkeypatch.setattr(
        viewer.__class__, 'controls', property(patched_controls)
    )
    monkeypatch.setattr(viewer.__class__, 'layers', property(patched_layers))
    monkeypatch.setattr(
        viewer.__class__, 'layerButtons', property(patched_layer_buttons)
    )
    monkeypatch.setattr(
        viewer.__class__, 'viewerButtons', property(patched_viewer_buttons)
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockLayerList', property(patched_dock_layer_list)
    )
    monkeypatch.setattr(
        viewer.__class__,
        'dockLayerControls',
        property(patched_dock_layer_controls),
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockConsole', property(patched_dock_console)
    )
    monkeypatch.setattr(
        viewer.__class__, 'dockPerformance', property(patched_dock_performance)
    )

    qtbot.addWidget(viewer, before_close_func=hide_and_clear_qt_viewer)
    return viewer


@pytest.fixture
def qt_viewer(
    qt_viewer_: QtViewer, request: pytest.FixtureRequest
) -> QtViewer:
    """We created `qt_viewer_` fixture to allow modifying qt_viewer
    if module-level-specific modifications are necessary.
    For example, in `test_qt_viewer.py`.
    """
    if 'show_qt_viewer' in request.keywords:
        qt_viewer_.show()
    return qt_viewer_


@pytest.fixture
def mock_qt_method(monkeypatch):
    """Since PySide6 6.10, the tests deterministically segfault when mocking
    methods of Qt objects using `unittest.mock.Mock` (or `MagicMock`) directly.

    This fixture provides a workaround for this by wrapping the mock in a function.
    Should be used as a replacement of `monkeypatch.setattr` and `mock.patch`

    FUTURE NOTE: Similar to `mock_qt_method_ctx `, it might be worth
     adding `qtbot.addWidget` when the first argument is a `QObject` in the future.
    Currently, this fixture is only used in tests where that look not necessary.
    """

    def _mock_fun(obj: str | object, method: str | None = None):
        mock = MagicMock()

        def _mocked_method(_self, *args, **kwargs):
            return mock(*args, **kwargs)

        if method is None:
            monkeypatch.setattr(obj, _mocked_method)
        else:
            monkeypatch.setattr(obj, method, _mocked_method)
        return mock

    return _mock_fun


@pytest.fixture
def mock_qt_method_ctx(monkeypatch, qtbot):
    """Since PySide6 6.10, the tests deterministically segfault when mocking
    methods of Qt objects using `unittest.mock.Mock` (or `MagicMock`) directly.

    This fixture provides a workaround for this by wrapping the mock in a function.
    Should be used as a replacement of `monkeypatch.context` and `object.patch`

    When the mocking is performed before creating the Qt object, the
    mocking function will get access to the created `object` using the first
    argument of the mocked method and will check if the object has no parent.
    In such case, the created `QWidget` will be added to `qtbot` using
    `qtbot.add_widget`.
    """
    from qtpy.QtWidgets import QWidget

    @contextmanager
    def _mock_fun(obj: str | object, method: str | None = None):
        mock = MagicMock()

        def _mocked_method(*args, **kwargs):
            if (
                len(args) > 0
                and isinstance(args[0], QWidget)
                and args[0].parent() is None
            ):
                qtbot.add_widget(args[0])
                args = args[1:]

            return mock(*args, **kwargs)

        with monkeypatch.context() as m:
            if method is None:
                m.setattr(obj, _mocked_method)
            else:
                m.setattr(obj, method, _mocked_method)
            yield mock

    return _mock_fun


@pytest.fixture(autouse=True)
def _clear_cached_action_injection():
    """Automatically clear cached property `Action.injected`.

    Allows action manager actions to be injected using current provider/processors
    and dependencies. See #7219 for details.
    To be removed after ActionManager deprecation.
    """
    from napari.utils.action_manager import action_manager

    for action in action_manager._actions.values():
        if 'injected' in action.__dict__:
            del action.__dict__['injected']


def _event_check(instance):
    def _prepare_check(name, no_event_):
        def check(instance, no_event=no_event_):
            if name in no_event:
                assert not hasattr(instance.events, name), (
                    f'event {name} defined'
                )
            else:
                assert hasattr(instance.events, name), (
                    f'event {name} not defined'
                )

        return check

    no_event_set = set()
    if isinstance(instance, tuple):
        no_event_set = instance[1]
        instance = instance[0]

    for name, value in instance.__class__.__dict__.items():
        if isinstance(value, property) and name[0] != '_':
            yield _prepare_check(name, no_event_set), instance, name


def pytest_generate_tests(metafunc):
    """Generate separate test for each test toc check if all events are defined."""
    if 'event_define_check' in metafunc.fixturenames:
        res = []
        ids = []

        for obj in metafunc.cls.get_objects():
            for check, instance, name in _event_check(obj):
                res.append((check, instance))
                ids.append(f'{name}-{instance}')

        metafunc.parametrize('event_define_check,obj', res, ids=ids)


def pytest_collection_modifyitems(session, config, items):
    test_subset = os.environ.get('NAPARI_TEST_SUBSET')

    test_order_prefix = [
        os.path.join('napari', 'utils'),
        os.path.join('napari', 'layers'),
        os.path.join('napari', 'components'),
        os.path.join('napari', 'settings'),
        os.path.join('napari', 'plugins'),
        os.path.join('napari', '_vispy'),
        os.path.join('napari', '_qt'),
        os.path.join('napari', 'qt'),
        os.path.join('napari', '_tests'),
        os.path.join('napari', '_tests', 'test_examples.py'),
    ]
    test_order = [[] for _ in test_order_prefix]
    test_order.append([])  # for not matching tests
    for item in items:
        if test_subset:
            if test_subset.lower() == 'qt' and 'qapp' not in item.fixturenames:
                # Skip non Qt tests
                continue
            if (
                test_subset.lower() == 'headless'
                and 'qapp' in item.fixturenames
            ):
                # Skip Qt tests
                continue

        index = -1
        for i, prefix in enumerate(test_order_prefix):
            if prefix in str(item.fspath):
                index = i
        test_order[index].append(item)
    items[:] = list(chain(*test_order))


@pytest.fixture(autouse=True)
def _disable_notification_dismiss_timer(monkeypatch):
    """
    This fixture disables starting timer for closing notification
    by setting the value of `NapariQtNotification.DISMISS_AFTER` to 0.

    As Qt timer is realised by thread and keep reference to the object,
    without increase of reference counter object could be garbage collected and
    cause segmentation fault error when Qt (C++) code try to access it without
    checking if Python object exists.

    This fixture is used in all tests because it is possible to call Qt code
    from non Qt test by connection of `NapariQtNotification.show_notification` to
    `NotificationManager` global instance.
    """

    with suppress(ImportError):
        from napari._qt.dialogs.qt_notification import NapariQtNotification

        monkeypatch.setattr(NapariQtNotification, 'DISMISS_AFTER', 0)
        monkeypatch.setattr(NapariQtNotification, 'FADE_IN_RATE', 0)
        monkeypatch.setattr(NapariQtNotification, 'FADE_OUT_RATE', 0)

        # disable slide in animation
        monkeypatch.setattr(NapariQtNotification, 'slide_in', lambda x: None)


@pytest.fixture(autouse=True)
def _prevent_thread(request, monkeypatch):
    if 'allow_animation_thread' in request.keywords:
        return
    if 'qt_dims' in request.fixturenames or 'ref_view' in request.fixturenames:
        return

    if 'qtbot' not in request.fixturenames:
        return

    from napari._qt.widgets.qt_dims_slider import AnimationThread

    def fake_start(self):
        raise RuntimeError(
            'QtDims animation thread should not be started outside of tests '
            "without using the 'qt_dims' fixture."
        )

    monkeypatch.setattr(AnimationThread, 'start', fake_start)


@pytest.fixture
def single_threaded_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    yield executor
    executor.shutdown()


def _get_calling_stack():  # pragma: no cover
    stack = []
    for i in range(2, sys.getrecursionlimit()):
        try:
            frame = sys._getframe(i)
        except ValueError:
            break
        stack.append(f'{frame.f_code.co_filename}:{frame.f_lineno}')
    return '\n'.join(stack)


def _get_calling_place(depth=1):  # pragma: no cover
    if not hasattr(sys, '_getframe'):
        return ''
    frame = sys._getframe(1 + depth)
    result = f'{frame.f_code.co_filename}:{frame.f_lineno}'
    if not frame.f_code.co_filename.startswith(ROOT_DIR):
        with suppress(ValueError):
            while not frame.f_code.co_filename.startswith(ROOT_DIR):
                frame = frame.f_back
                if frame is None:
                    break
            else:
                result += f' called from\n{frame.f_code.co_filename}:{frame.f_lineno}'
    return result


@pytest.fixture
def _dangling_qthreads(monkeypatch, qtbot, request):
    from qtpy.QtCore import QThread

    base_start = QThread.start
    thread_dict = WeakKeyDictionary()
    request.node.stash[_PENDING_THREADS_KEY] = thread_dict
    base_constructor = QThread.__init__

    def run_with_trace(self):  # pragma: no cover
        """
        QThread.run but adding execution to sys.settrace when measuring coverage.

        See https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
        and `init_with_trace`. When running QThreads during testing, we monkeypatch
        the QThread constructor and run methods with traceable equivalents.
        """
        if 'coverage' in sys.modules:
            # https://github.com/nedbat/coveragepy/issues/686#issuecomment-634932753
            sys.settrace(threading._trace_hook)
        self._base_run()

    def init_with_trace(self, *args, **kwargs):
        """Constructor for QThread adding tracing for coverage measurements.

        Functions running in QThreads don't get measured by coverage.py, see
        https://github.com/nedbat/coveragepy/issues/686. Therefore, we will
        monkeypatch the constructor to add to the thread to `sys.settrace` when
        we call `run` and `coverage` is in `sys.modules`.
        """
        base_constructor(self, *args, **kwargs)
        self._base_run = self.run
        self.run = partial(run_with_trace, self)

    # dict of threads that have been started but not yet terminated

    if 'disable_qthread_start' in request.keywords:

        def start_with_save_reference(self, priority=QThread.InheritPriority):
            """Dummy function to prevent thread starts."""

    else:

        def start_with_save_reference(self, priority=QThread.InheritPriority):
            """Thread start function with logs to detect hanging threads.

            Saves a weak reference to the thread and detects hanging threads,
            as well as where the threads were started.
            """
            thread_dict[self] = _get_calling_place()
            base_start(self, priority)

    monkeypatch.setattr(QThread, 'start', start_with_save_reference)
    monkeypatch.setattr(QThread, '__init__', init_with_trace)

    yield

    dangling_threads_li = []

    for thread, calling in thread_dict.items():
        try:
            if thread.isRunning():
                dangling_threads_li.append((thread, calling))
        except RuntimeError as e:
            if (
                'wrapped C/C++ object of type' not in e.args[0]
                and 'Internal C++ object' not in e.args[0]
            ):
                raise

    for thread, _ in dangling_threads_li:
        with suppress(RuntimeError):
            thread.quit()
            qtbot.waitUntil(thread.isFinished, timeout=2000)

    dangling_places = [calling for _, calling in dangling_threads_li]
    # Threads the `pytest_runtest_teardown` hookimpl had to stop before
    # pytest-qt's event pump could deliver a queued `deleteLater()` to their
    # parent widget are no longer running, so the loop above cannot see them.
    # They were still left running by this test, and destroying a running
    # QThread is a fatal abort, so report them as leaks all the same.
    dangling_places += [
        f'{calling} (still running before pytest-qt pumped the event loop, so '
        'it was force-stopped to keep a queued deleteLater() from destroying '
        'it mid-run)'
        for calling in request.node.stash.get(_FORCE_STOPPED_THREADS_KEY, [])
    ]

    long_desc = (
        'If you see this error, it means that a QThread was started in a test '
        'but not terminated. This can cause segfaults in the test suite. '
        'Please use the `qtbot` fixture to wait for the thread to finish. '
        'If you think that the thread is obsolete for this test, you can '
        'use the `@pytest.mark.disable_qthread_start` mark or  `monkeypatch` '
        'fixture to patch the `start` method of the '
        'QThread class to do nothing.\n'
    )

    if len(dangling_places) > 1:
        long_desc += ' The QThreads were started in:\n'
    else:
        long_desc += ' The QThread was started in:\n'

    assert not dangling_places, long_desc + '\n'.join(dangling_places)


@pytest.fixture
def _dangling_qthread_pool(monkeypatch, request):
    from qtpy.QtCore import QThreadPool

    base_start = QThreadPool.start
    threadpool_dict = WeakKeyDictionary()
    # dict of threadpools that have been used to run QRunnables

    if 'disable_qthread_pool_start' in request.keywords:

        def my_start(self, runnable, priority=0):
            """dummy function to prevent thread start"""

    else:

        def my_start(self, runnable, priority=0):
            if self not in threadpool_dict:
                threadpool_dict[self] = []
            threadpool_dict[self].append(_get_calling_place())
            base_start(self, runnable, priority)

    monkeypatch.setattr(QThreadPool, 'start', my_start)
    yield

    dangling_threads_pools = []

    for thread_pool, calling in threadpool_dict.items():
        thread_pool.clear()
        thread_pool.waitForDone(20)
        if thread_pool.activeThreadCount():
            dangling_threads_pools.append((thread_pool, calling))

    for thread_pool, _ in dangling_threads_pools:
        with suppress(RuntimeError):
            thread_pool.clear()
            thread_pool.waitForDone(2000)

    long_desc = (
        'If you see this error, it means that a QThreadPool was used to run '
        'a QRunnable in a test but not terminated. This can cause segfaults '
        'in the test suite. Please use the `qtbot` fixture to wait for the '
        'thread to finish. If you think that the thread is obsolete for this '
        'use the `@pytest.mark.disable_qthread_pool_start` mark or  `monkeypatch` '
        'fixture to patch the `start` '
        'method of the QThreadPool class to do nothing.\n'
    )
    if len(dangling_threads_pools) > 1:
        long_desc += ' The QThreadPools were used in:\n'
    else:
        long_desc += ' The QThreadPool was used in:\n'

    assert not dangling_threads_pools, long_desc + '\n'.join(
        '; '.join(x[1]) for x in dangling_threads_pools
    )


# Shared between `_dangling_qtimers`/`_dangling_qthreads` and the
# `pytest_runtest_teardown` hookimpl below: those fixtures' own finalizers run
# too late to prevent the crashes they detect. `pytest-qt`'s own
# `pytest_runtest_teardown` hookimpl pumps the Qt event loop (to flush pending
# events between tests) and, because it has no `tryfirst`, it runs *before*
# our fixture finalizers under pluggy's LIFO ordering. Two ways that bites:
#
# - A `QTimer.singleShot` from the finished test can still be pending (e.g.
#   `superqt`'s `WorkerBase.start()` defers submitting a `QRunnable` to
#   `QThreadPool` by 1ms when called from a running event loop). The pump can
#   fire it *after* the test's own widgets/objects have already been torn
#   down, crashing the whole process (`QRunnable::warnNullCallable` -> abort).
# - A still-running `QThread` (e.g. `AnimationThread` in `qt_dims_slider.py`)
#   can be a child of a widget with a queued `deleteLater()`. The pump
#   delivers that deferred deletion, and Qt fatally aborts if a `QThread` is
#   destroyed while still running.
#
# Both are prevented by acting in a `tryfirst` hookimpl, which guarantees we
# run before any non-tryfirst teardown hookimpl, regardless of fixture
# teardown order.
#
# Acting that early has a catch: the fixtures decide what leaked by looking at
# `isActive()`/`isRunning()`, and stopping something makes it look clean. Left
# alone, this hookimpl would silently disable the very checks it exists to
# protect. So it records what it had to force-stop, keyed per resource, and the
# fixtures fold those records back into their own reports - see
# `_FORCE_STOPPED_*_KEY` below.
#: `(started_timers, single_shot_timers)` from `_dangling_qtimers`. Both
#: halves matter: a timer left running by `QTimer.start()` can fire into a
#: torn-down test during pytest-qt's event pump exactly as a `singleShot` can.
_PENDING_TIMERS_KEY: pytest.StashKey[tuple] = pytest.StashKey()
_PENDING_THREADS_KEY: pytest.StashKey[WeakKeyDictionary] = pytest.StashKey()

# Calling places of resources this hookimpl force-stopped, so that
# `_dangling_qtimers`/`_dangling_qthreads` can still report them as leaks.
_FORCE_STOPPED_TIMERS_KEY: pytest.StashKey[list] = pytest.StashKey()
_FORCE_STOPPED_THREADS_KEY: pytest.StashKey[list] = pytest.StashKey()


def _stop_thread_early(thread) -> None:
    """Best-effort graceful stop, used before any event-loop pump can risk
    destroying `thread` while it's still running (see comment above)."""
    from qtpy.QtCore import QThread

    stop = getattr(thread, '_stop', None)
    if stop is None and type(thread).terminate is not QThread.terminate:
        # an overridden `terminate()` (e.g. StatusChecker's) is a graceful
        # stop request; the base QThread.terminate() is a dangerous forced
        # kill, so it's deliberately never called here.
        stop = thread.terminate
    if stop is not None:
        with contextlib.suppress(RuntimeError):
            stop()
    with contextlib.suppress(RuntimeError):
        thread.quit()
    with contextlib.suppress(RuntimeError):
        thread.wait(2000)


def _is_live(qt_object, predicate_name: str) -> bool:
    """Call `isActive()`/`isRunning()`, treating a dead C++ object as not live."""
    try:
        return bool(getattr(qt_object, predicate_name)())
    except RuntimeError:
        return False


def force_stop_pending_qt_resources(item):
    """Stop timers/threads this test left running, before anything pumps events.

    Called from the `tryfirst` `pytest_runtest_teardown` hookimpls at the
    bottom of this file and in `napari_builtins/conftest.py`. A conftest only
    applies to items below its own directory, so `napari_builtins` needs its
    own hook rather than inheriting this one - see `pytest_runtest_setup`
    below for why that does not double-register.
    """
    stopped_timers = item.stash.setdefault(_FORCE_STOPPED_TIMERS_KEY, [])
    started_timers, single_shot_timers = item.stash.get(
        _PENDING_TIMERS_KEY, ({}, [])
    )
    # `list()` the WeakKeyDictionary: stopping a timer can drop the last
    # reference to another, and mutating it mid-iteration would raise.
    for timer, calling in chain(
        list(started_timers.items()), single_shot_timers
    ):
        if _is_live(timer, 'isActive'):
            # record before stopping: a resource we then fail to stop matters
            # more, not less.
            stopped_timers.append(calling)
            with contextlib.suppress(RuntimeError):
                timer.stop()

    stopped_threads = item.stash.setdefault(_FORCE_STOPPED_THREADS_KEY, [])
    for thread, calling in list(
        item.stash.get(_PENDING_THREADS_KEY, {}).items()
    ):
        if _is_live(thread, 'isRunning'):
            stopped_threads.append(calling)
            _stop_thread_early(thread)


@pytest.fixture
def _dangling_qtimers(monkeypatch, request):
    from qtpy.QtCore import QTimer

    base_start = QTimer.start
    timer_dkt = WeakKeyDictionary()
    single_shot_list = []
    request.node.stash[_PENDING_TIMERS_KEY] = (timer_dkt, single_shot_list)

    if 'disable_qtimer_start' in request.keywords:
        from pytestqt.qt_compat import qt_api

        def my_start(self, msec=None):
            """dummy function to prevent timer start"""

        _single_shot = my_start

        class OldTimer(QTimer):
            def start(self, time=None):
                if time is not None:
                    base_start(self, time)
                else:
                    base_start(self)

        monkeypatch.setattr(qt_api.QtCore, 'QTimer', OldTimer)
        # This monkeypatch is require to keep `qtbot.waitUntil` working

    else:

        def my_start(self, msec=None):
            calling_place = _get_calling_place()
            if 'superqt' in calling_place and 'throttler' in calling_place:
                calling_place += f' - {_get_calling_place(2)}'
            timer_dkt[self] = calling_place
            if msec is not None:
                base_start(self, msec)
            else:
                base_start(self)

        def single_shot(msec, reciver, method=None):
            t = QTimer()
            t.setSingleShot(True)
            if method is None:
                t.timeout.connect(reciver)
            else:
                t.timeout.connect(getattr(reciver, method))
            calling_place = _get_calling_place(2)
            if 'superqt' in calling_place and 'throttler' in calling_place:
                calling_place += _get_calling_stack()
            single_shot_list.append((t, _get_calling_place(2)))
            base_start(t, msec)

        def _single_shot(self, *args):
            if isinstance(self, QTimer):
                single_shot(*args)
            else:
                single_shot(self, *args)

    monkeypatch.setattr(QTimer, 'start', my_start)
    monkeypatch.setattr(QTimer, 'singleShot', _single_shot)

    yield

    dangling_timers = []

    for timer, calling in chain(timer_dkt.items(), single_shot_list):
        # `_is_live` rather than a bare `isActive()`: a timer whose C++ object
        # has already been deleted raises RuntimeError from that call, and a
        # deleted timer is by definition not still active. Same guard
        # `_dangling_qanimations` needed for its `state()` read.
        if _is_live(timer, 'isActive'):
            dangling_timers.append((timer, calling))

    for timer, _ in dangling_timers:
        with suppress(RuntimeError):
            timer.stop()

    dangling_places = [calling for _, calling in dangling_timers]
    # Timers the `pytest_runtest_teardown` hookimpl above had to stop before
    # pytest-qt's event pump could fire them are no longer active, so the loop
    # above cannot see them - but they were still left running by this test,
    # which is exactly the hazard. Report them too, tagged so they can be told
    # apart from a timer that was still active at fixture-teardown time.
    dangling_places += [
        f'{calling} (still active before pytest-qt pumped the event loop, so '
        'it was force-stopped to keep it from firing into a torn-down test)'
        for calling in request.node.stash.get(_FORCE_STOPPED_TIMERS_KEY, [])
    ]

    long_desc = (
        'If you see this error, it means that a QTimer was started but not stopped. '
        'This can cause tests to fail, and can also cause segfaults. '
        'If this test does not require a QTimer to pass you could monkeypatch it out. '
        'If it does require a QTimer, you should stop or wait for it to finish before test ends. '
    )
    if len(dangling_places) > 1:
        long_desc += 'The QTimers were started in:\n'
    else:
        long_desc += 'The QTimer was started in:\n'

    def _check_throttle_info(path):
        if 'superqt' in path and 'throttler' in path:
            return (
                path
                + " it's possible that there was a problem with unfinished work by a "
                'qthrottler; to solve this, you can either try to wait (such as with '
                '`qtbot.wait`) or disable throttling with the disable_throttling fixture'
            )
        return path

    assert not dangling_places, long_desc + '\n'.join(
        _check_throttle_info(path) for path in dangling_places
    )


def _throttle_mock(self):
    self.triggered.emit()


def _flush_mock(self):
    """There are no waiting events."""


@pytest.fixture
def _disable_throttling(monkeypatch):
    """Disable qthrottler from superqt.

    This is sometimes necessary to avoid flaky failures in tests
    due to dangling qt timers.
    """
    # if this monkeypath fails then you should update path to GenericSignalThrottler
    monkeypatch.setattr(
        'superqt.utils._throttler.GenericSignalThrottler.throttle',
        _throttle_mock,
    )
    monkeypatch.setattr(
        'superqt.utils._throttler.GenericSignalThrottler.flush', _flush_mock
    )


@pytest.fixture
def _dangling_qanimations(monkeypatch, request):
    from qtpy.QtCore import QPropertyAnimation

    base_start = QPropertyAnimation.start
    animation_dkt = WeakKeyDictionary()

    if 'disable_qanimation_start' in request.keywords:

        def my_start(self):
            """dummy function to prevent thread start"""

    else:

        def my_start(self):
            animation_dkt[self] = _get_calling_place()
            base_start(self)

    monkeypatch.setattr(QPropertyAnimation, 'start', my_start)
    yield

    dangling_animations = []

    for animation, calling in animation_dkt.items():
        # Guard the state read, not just the `stop()` below: a flash animation
        # that already finished has had its C++ object deleted by
        # `remove_flash_animation`, and `state()` then raises RuntimeError.
        # A deleted animation is by definition not still running. This only
        # surfaced once `napari_builtins`' Qt tests started being checked -
        # five of its `test_features_table` tests errored here, none of them
        # actually leaking anything.
        with suppress(RuntimeError):
            if animation.state() == QPropertyAnimation.Running:
                dangling_animations.append((animation, calling))

    for animation, _ in dangling_animations:
        with suppress(RuntimeError):
            animation.stop()

    long_desc = (
        'If you see this error, it means that a QPropertyAnimation was started but not stopped. '
        'This can cause tests to fail, and can also cause segfaults. '
        'If this test does not require a QPropertyAnimation to pass you could monkeypatch it out. '
        'If it does require a QPropertyAnimation, you should stop or wait for it to finish before test ends. '
    )
    if len(dangling_animations) > 1:
        long_desc += ' The QPropertyAnimations were started in:\n'
    else:
        long_desc += ' The QPropertyAnimation was started in:\n'
    assert not dangling_animations, long_desc + '\n'.join(
        x[1] for x in dangling_animations
    )


with contextlib.suppress(ImportError):
    # in headless test suite we don't have Qt bindings
    # So we cannot inherit from QtBot and declare the fixture

    from pytestqt.qtbot import QtBot
    from qtpy import PYQT5
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QApplication

    class QtBotWithOnCloseRenaming(QtBot):
        """Modified QtBot that renames widgets when closing them in tests.

        After a test ends that uses QtBot, all instantiated widgets added to
        the bot have their name changed to 'handled_widget'. This allows us to
        detect leaking widgets at the end of a test run, and avoid the
        segmentation faults that often result from such leaks. [1]_

        See Also
        --------
        `_find_dangling_widgets`: fixture that finds all widgets that have not
        been renamed to 'handled_widget'.

        References
        ----------
        .. [1] https://czaki.github.io/blog/2024/09/16/preventing-segfaults-in-test-suite-that-has-qt-tests/
        """

        def addWidget(self, widget, *, before_close_func=None):
            if widget.objectName() == '':
                # object does not have a name, so we can set it
                widget.setObjectName('handled_widget')
                before_close_func_ = before_close_func
            elif before_close_func is None:
                # there is no custom teardown function,
                # so we provide one that will set object name

                def before_close_func_(w):
                    w.setObjectName('handled_widget')
            else:
                # user provided custom teardown function,
                # so we need to wrap it to set object name

                def before_close_func_(w):
                    before_close_func(w)
                    w.setObjectName('handled_widget')

            super().addWidget(widget, before_close_func=before_close_func_)

    @pytest.fixture
    def qtbot(qapp, request):  # pragma: no cover
        """Fixture to create a QtBotWithOnCloseRenaming instance for testing.

        Make sure to call addWidget for each top-level widget you create to
        ensure that they are properly closed after the test ends.

        The `qapp` fixture is used to ensure that the QApplication is created
        before, so we need it, even without using it directly in this fixture.
        """
        return QtBotWithOnCloseRenaming(request)

    @pytest.fixture(scope='session')
    def qapp_cls():
        """The qapp fixture uses the qapp_cls fixture to select
        the class to use for create the QApplication instance.

        As qapp fixture is using more complex logic, we decided
        not to override it but overwrite the fixture used by it.

        We need to set attributte before the QApplication is created.
        """
        if PYQT5:
            # As Qt6 autodetect High dpi scaling, we need to
            # enable it only on Qt5 bindings.
            # https://doc.qt.io/qtforpython-6/faq/porting_from2.html#class-function-deprecations
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        return QApplication

    @pytest.fixture(autouse=True)
    def disable_get_log_level_value(monkeypatch):
        """Enforce to not set logging to logging.NOTSET,
        that crashes current tests
        """

        import logging

        monkeypatch.setattr(
            'napari._qt.widgets.qt_logger.get_log_level_value',
            lambda x: logging.WARNING,
        )


#: Widget classes the dangling-widget check must not fail on. Both are GL
#: canvas widgets that show up as parentless top-levels without any napari
#: code having constructed them; see the comment at the use site.
_GL_WIDGET_EXEMPTIONS = frozenset({'CanvasBackendDesktop', 'QOpenGLWidget'})


def _short_repr(obj: object, limit: int = 200) -> str:
    """repr() an arbitrary gc referrer without risking a crash/huge dump."""
    try:
        text = repr(obj)
    except Exception as e:  # noqa: BLE001
        return f'<{type(obj)!r}, repr failed: {e!r}>'
    return f'{type(obj)!r}: {text[:limit]}' + (
        '...' if len(text) > limit else ''
    )


def _keys_holding(container: object, target: object) -> list:
    """Names of the slots in `container` that refer to `target`, if any.

    Only dicts, lists and tuples: those cover the containers that actually
    show up here (a mock's `call_args`, a frame's locals, a registry) and are
    cheap and safe to walk. Anything raising is simply not described.
    """
    try:
        if isinstance(container, dict):
            return [k for k, v in container.items() if v is target]
        if isinstance(container, (list, tuple)):
            return [i for i, v in enumerate(container) if v is target]
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        pass
    return []


def _describe_dangling_widgets(request, creation_places) -> str:
    """Return a description of leaked top-level widgets, or '' if there are none.

    Every widget reference lives in *this* function's frame, and this function
    has returned by the time `_find_dangling_widgets` raises - so none of them
    can be reachable from the resulting exception's traceback. That matters:
    the exception is retained for the whole session (pytest_pretty's
    CustomTerminalReporter holds failure reports to build its final summary
    table), so a frame in its traceback holding `QApplication.topLevelWidgets()`
    pins every Qt top-level widget that existed at the first failure. Those
    then show up as "dangling" in every later test in the worker, whether or
    not that test leaked anything - see the CI failure that motivated this,
    where four unrelated tests all reported the same _QtMainWindow.

    Keep widget references out of `_find_dangling_widgets` itself rather than
    clearing locals by hand there; a returned frame cannot be re-entered, so
    this stays correct as the code changes.
    """
    from qtpy.QtWidgets import QApplication

    from napari._qt.qt_main_window import _QtMainWindow

    top_level_widgets = QApplication.topLevelWidgets()
    # `make_napari_viewer` stashes a `WeakSet[Viewer]` here; the default covers
    # tests that never asked for a viewer. Only ever tested for membership, so
    # `Container` is all this needs to promise.
    viewer_weak_set: Container[Any] = getattr(
        request.node, '_viewer_weak_set', set()
    )

    problematic_widgets = []

    for widget in top_level_widgets:
        if widget.parent() is not None:
            continue
        if (
            isinstance(widget, _QtMainWindow)
            and widget._qt_viewer.viewer in viewer_weak_set
        ):
            continue

        if widget.__class__.__module__.startswith('qtconsole'):
            continue

        if widget.objectName() == 'handled_widget':
            continue

        if widget.__class__.__name__ in _GL_WIDGET_EXEMPTIONS:
            # TODO: we don't understand why `CanvasBackendDesktop` leaks in
            #  napari/_tests/test_sys_info.py, so we make an exception
            #  here and we don't raise when this class leaks.
            #
            # `QOpenGLWidget` is the same shape one class further down -
            # `CanvasBackendDesktop` subclasses it on PyQt5 - and was added on
            # this evidence: it appeared once, as a bare parentless
            # `PyQt5.QtWidgets.QOpenGLWidget`, against an unrelated test
            # (`test_build_qmodel_menu`) in a re-run of a commit whose three
            # previous runs of the same job were green. Crucially it had **no
            # recorded creation place**, and `_WIDGET_CREATION_PLACES` covers
            # every widget any Python constructor builds during a test - so no
            # napari code built it, it came from the C++ side. PyQt6 puts
            # `QOpenGLWidget` in `QtOpenGLWidgets`, so only the PyQt5 jobs can
            # see this at all.
            #
            # Not reproducible locally (Linux + software GL); two hypotheses
            # were tested and disproved - sip does not re-wrap a subclass as
            # its base after the wrapper is dropped, and Qt5 creates no hidden
            # global-share QOpenGLWidget on macOS. Left as an exemption rather
            # than chased, because a widget with no Python constructor is by
            # definition not a napari leak.
            continue

        problematic_widgets.append(widget)

    if not problematic_widgets:
        return ''

    lines = []
    for widget in problematic_widgets:
        lines.append(
            f'Widget: {widget} of type {type(widget)} with name {widget.objectName()}'
        )
        lines.append(
            f'  created at: {creation_places.get(widget, "<unknown - built before any tracked test, or by a class whose __init__ is not patched>")}'
        )
        for ref in gc.get_referrers(widget):
            if (
                ref is problematic_widgets
                or ref is top_level_widgets
                # this function's own frame shows up while it is running
                # (`widget` is one of its locals) - not signal
                or type(ref).__name__ == 'frame'
            ):
                continue
            lines.append(f'  referrer: {_short_repr(ref)}')
            # A container's truncated repr often cuts off exactly the part
            # that matters - which slot holds the widget. Name it. Seen on
            # min_req, where a leaked FeaturesTable's only referrer was a
            # dict whose repr was cut off before reaching the widget, leaving
            # the holder unidentifiable from the log alone.
            lines.extend(
                f'    held at key: {key!r}'
                for key in _keys_holding(ref, widget)
            )

    for widget in problematic_widgets:
        widget.setObjectName('handled_widget')

    return '\n'.join(lines)


#: Classes whose `__init__` is patched to record where a widget was built.
#:
#: `QWidget` alone is enough on PyQt and not on PySide, so patch the bases
#: explicitly rather than rely on either. PyQt resolves a subclass's
#: `__init__` through the MRO, so one patch on `QWidget` catches everything;
#: Shiboken gives each class its own `__init__`, which ends the chain.
#: Measured, patching only `QWidget`:
#:
#:   PyQt6 6.10.0    misses nothing
#:   PySide6 6.10.3  misses QMainWindow, QDialog, QDockWidget, QLabel,
#:                   QPushButton, and any subclass of those
#:
#: `_QtMainWindow` subclasses `QMainWindow`, so on PySide6 the tracker was
#: blind to the widget it is most often asked about.
_TRACKED_WIDGET_BASES = (
    'QWidget',
    'QMainWindow',
    'QDialog',
    'QDockWidget',
    'QMenu',
)


#: Creation places, keyed weakly by widget and **shared across the session**.
#:
#: Deliberately not per-test. The widget this check reports is very often one
#: an *earlier* test leaked - a failing test's retained traceback pins its
#: widgets, and the next test is the one that notices - so a per-test record
#: can only ever answer "constructed before this fixture was active", which is
#: true and useless. Seen exactly that way on CI: a leaked `_QtMainWindow`
#: reported against `test_restart`, built by the `test_screenshot_to_clipboard`
#: that failed immediately before it. Weak keys, so nothing is retained.
_WIDGET_CREATION_PLACES: WeakKeyDictionary = WeakKeyDictionary()


@pytest.fixture
def _find_dangling_widgets(request, qtbot, monkeypatch):
    # `gc.get_referrers()` only walks one level, so a widget kept alive by an
    # indirect referrer (e.g. captured in a mock's call args) can show up with
    # no apparent referrer at all. Record where every widget was constructed
    # as a fallback, mirroring how
    # `_dangling_qthreads`/`_dangling_qtimers`/`_dangling_qthread_pool`
    # already track their resource's calling place.
    from qtpy import QtWidgets

    creation_places = _WIDGET_CREATION_PLACES

    def _tracking_init(klass, base_init):
        def init_with_tracking(self, *args, **kwargs):
            base_init(self, *args, **kwargs)
            # Only the first (outermost) patched `__init__` to run records,
            # so the place named is the one that built the concrete widget
            # rather than a base class reached on the way down.
            if self not in creation_places:
                creation_places[self] = _get_calling_place()

        return init_with_tracking

    installed: list = []
    for name in _TRACKED_WIDGET_BASES:
        klass = getattr(QtWidgets, name)
        current = klass.__init__
        # Skip a class that already inherits a patch installed above, rather
        # than wrapping the wrapper: the inner one would then record the outer
        # one's frame in `conftest.py` instead of the line that built the
        # widget. On PyQt every one of these resolves to
        # `sip.simplewrapper.__init__`, so patching `QWidget` covers the whole
        # hierarchy and the rest are skipped; on PySide each class has its own
        # `__init__`, so each is patched exactly once.
        if any(current is wrapper for wrapper in installed):
            continue
        wrapper = _tracking_init(klass, current)
        installed.append(wrapper)
        monkeypatch.setattr(klass, '__init__', wrapper)

    yield

    # No widget may be referenced from this frame: see
    # `_describe_dangling_widgets` for why.
    text = _describe_dangling_widgets(request, creation_places)
    if text:
        raise RuntimeError(f'Found dangling widgets:\n{text}')


@pytest.fixture(autouse=True)
def _fix_magic_name(monkeypatch, request):
    """Fix napari.utils.naming.magic_name to handle test as as internal napari module."""

    from napari.utils import naming

    monkeypatch.setitem(
        naming.magic_name.__kwdefaults__,
        'path_prefix',
        (naming.ROOT_DIR, str(request.fspath)),
    )


@pytest.fixture(autouse=True)
def _reset_colormaps(monkeypatch):
    from napari.utils.colormaps import colormap_utils

    prev = dict(colormap_utils.AVAILABLE_COLORMAPS)
    yield
    colormap_utils.AVAILABLE_COLORMAPS.clear()
    colormap_utils.AVAILABLE_COLORMAPS.update(prev)


def apply_leak_detection_fixtures(item):
    """Add Qt leak detection fixtures *only* in tests using the qapp fixture.

    Called from the `pytest_runtest_setup` hookimpls at the bottom of this
    file and in `napari_builtins/conftest.py`, so it covers both trees - see
    `pytest_runtest_setup` below for why there are two.

    Because we have headless test suite that does not include Qt, we cannot
    simply use `@pytest.fixture(autouse=True)` on all our fixtures for
    detecting leaking Qt objects.

    Instead, here we detect whether the `qapp` fixture is being used, detecting
    tests that use Qt and need to be checked for Qt objects leaks.

    A note to maintainers: tests *may* attempt to use Qt classes but not use
    the `qapp` fixture. This is BAD, and may cause Qt failures to be reported
    far away from the problematic code or test. If you find any tests
    instantiating Qt objects but not using qapp or qtbot, please submit a PR
    adding the qtbot fixture and adding any top-level Qt widgets with::

        qtbot.addWidget(widget_instance)

    """

    if 'qapp' in item.fixturenames:
        # here we do autouse for dangling fixtures only if qapp is used
        if 'qtbot' not in item.fixturenames:
            # for proper waiting for threads to finish
            item.fixturenames.append('qtbot')
        item.fixturenames.extend(
            [
                '_find_dangling_widgets',
                '_dangling_qthread_pool',
                '_dangling_qanimations',
                '_dangling_qthreads',
                '_dangling_qtimers',
            ]
        )


def pytest_runtest_setup(item):
    """Register the leak detectors for items under ``src/napari``.

    Deliberately a thin delegation rather than the logic itself: a conftest
    only applies to items below its own directory, so ``napari_builtins`` has
    an identical pair of hooks in its own conftest, delegating to the same
    helpers. The trees are disjoint, so pluggy never registers two
    implementations for one item - which it would if a shared parent conftest
    also defined them.
    """
    apply_leak_detection_fixtures(item)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """See `pytest_runtest_setup` above; `tryfirst` is load-bearing.

    It must run before pytest-qt's own teardown hookimpl pumps the event loop.
    """
    force_stop_pending_qt_resources(item)


class NapariTerminalReporter(CustomTerminalReporter):
    """
    This ia s custom terminal reporter to how long it takes to finish given part of tests.
    It prints time each time when test from different file is started.

    It is created to be able to see if timeout is caused by long time execution, or it is just hanging.
    """

    currentfspath: Path | None

    def write_fspath_result(self, nodeid: str, res, **markup: bool) -> None:
        if getattr(self, '_start_time', None) is None:
            self._start_time = perf_counter()
        fspath = self.config.rootpath / nodeid.split('::')[0]
        if self.currentfspath is None or fspath != self.currentfspath:
            if self.currentfspath is not None and self._show_progress_info:
                self._write_progress_information_filling_space()
                if os.environ.get('CI', False):
                    self.write(
                        f' [{timedelta(seconds=int(perf_counter() - self._start_time))}]'
                    )
            self.currentfspath = fspath
            relfspath = bestrelpath(self.startpath, fspath)
            self._tw.line()
            self.write(relfspath + ' ')
        self.write(res, flush=True, **markup)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    # Get the standard terminal reporter plugin and replace it with our
    standard_reporter = config.pluginmanager.getplugin('terminalreporter')
    custom_reporter = NapariTerminalReporter(config, sys.stdout)
    if standard_reporter._session is not None:
        custom_reporter._session = standard_reporter._session
    config.pluginmanager.unregister(standard_reporter)
    config.pluginmanager.register(custom_reporter, 'terminalreporter')
