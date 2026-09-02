import os
import sys
import warnings
from collections import defaultdict
from unittest.mock import Mock

import pytest
from qtpy.QtWidgets import QAction, QShortcut

from napari._qt.qt_event_loop import (
    _ipython_has_eventloop,
    get_qapp,
    run,
    set_app_id,
)


@pytest.mark.skipif(os.name != 'nt', reason='Windows specific')
def test_windows_grouping_overwrite(qapp):
    import ctypes

    def get_app_id():
        mem = ctypes.POINTER(ctypes.c_wchar)()
        ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID(
            ctypes.byref(mem)
        )
        res = ctypes.wstring_at(mem)
        ctypes.windll.Ole32.CoTaskMemFree(mem)
        return res

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('test_text')

    assert get_app_id() == 'test_text'
    set_app_id('custom_string')
    assert get_app_id() == 'custom_string'
    set_app_id('')  # app id can't be an empty string
    assert get_app_id() == 'custom_string'
    set_app_id(' ')
    assert get_app_id() == ' '


def test_run_outside_ipython(make_napari_viewer, qapp, monkeypatch):
    """Test that we don't incorrectly give ipython the event loop."""
    # `IPython.get_ipython()` is a process-global singleton shell, and an
    # earlier test in this worker process (e.g. one using the embedded Qt
    # console) can leave one behind. Simulate the plain, non-IPython
    # environment this test is about.
    monkeypatch.setitem(sys.modules, 'IPython', None)

    # These three are *preconditions*, not assertions about napari, and they
    # cannot be made otherwise - which is worth stating, because "this
    # assertion can never fail, so make it falsifiable" is the attractive
    # wrong move here. `_ipython_has_eventloop()` is True exactly when a
    # shell exists whose `active_eventloop` is 'qt', and `get_qapp()`
    # deliberately *sets* that via `_try_enable_ipython_gui` whenever a shell
    # exists. So in any environment where these could fail, napari makes them
    # fail by design; leaving the shell in place and only clearing
    # `active_eventloop` makes the second one fail as soon as a viewer is
    # created. What this test actually asserts about napari is the
    # `exec_` call below, and the True branch of that decision is covered by
    # `test_run_does_not_steal_the_loop_from_ipython`.
    assert not _ipython_has_eventloop()
    v1 = make_napari_viewer()
    assert not _ipython_has_eventloop()
    v2 = make_napari_viewer()
    assert not _ipython_has_eventloop()

    with monkeypatch.context() as m:
        mock_exec = Mock()
        m.setattr(qapp, 'exec_', mock_exec)
        run()
        mock_exec.assert_called_once()

    v1.close()
    v2.close()


def test_run_does_not_steal_the_loop_from_ipython(qapp, monkeypatch):
    """`run()` must return without starting a loop when `%gui qt` is active.

    The complement of `test_run_outside_ipython`, and the only place
    `_ipython_has_eventloop()` returning True changes what napari does - so
    the only place that branch can be tested. A stub shell rather than a real
    one: `InteractiveShell.enable_gui` raises `NotImplementedError` on the
    base class, and nothing here needs a working IPython.
    """

    class _Shell:
        active_eventloop = 'qt'

    class _IPython:
        @staticmethod
        def get_ipython():
            return _Shell()

    monkeypatch.setitem(sys.modules, 'IPython', _IPython)
    assert _ipython_has_eventloop()

    mock_exec = Mock()
    monkeypatch.setattr(qapp, 'exec_', mock_exec)
    run()
    mock_exec.assert_not_called()


def test_wayland_warning_on_preexisting_app(qapp, monkeypatch):
    """Warn when a pre-existing QApplication is on Wayland with Nvidia."""
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(qapp, 'platformName', lambda: 'wayland')
    monkeypatch.setattr(
        'napari._qt.qt_event_loop._nvidia_driver_loaded', lambda: True
    )
    with pytest.warns(UserWarning, match='Wayland startup workaround'):
        get_qapp()


@pytest.mark.parametrize(
    ('platform', 'platform_name', 'nvidia'),
    [
        ('linux', 'xcb', True),  # not on Wayland
        ('darwin', 'cocoa', True),  # not Linux
        ('linux', 'wayland', False),  # Wayland but no Nvidia driver
    ],
)
def test_no_wayland_warning(
    qapp, monkeypatch, platform, platform_name, nvidia
):
    """No warning unless on Linux+Wayland with the Nvidia driver loaded."""
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(qapp, 'platformName', lambda: platform_name)
    monkeypatch.setattr(
        'napari._qt.qt_event_loop._nvidia_driver_loaded', lambda: nvidia
    )
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('always')
        get_qapp()
    assert not any(
        'Wayland startup workaround' in str(w.message) for w in records
    )


def test_shortcut_collision(qtbot, make_napari_viewer):
    viewer = make_napari_viewer()
    defined_shortcuts = defaultdict(list)
    problematic_shortcuts = []
    shortcuts = viewer.window._qt_window.findChildren(QShortcut)
    for shortcut in shortcuts:
        key = shortcut.key().toString()
        if key == 'Ctrl+M':
            # menubar toggle support
            # https://github.com/napari/napari/pull/3204
            continue
        if key and key in defined_shortcuts:
            problematic_shortcuts.append(key)
        defined_shortcuts[key].append(key)

    actions = viewer.window._qt_window.findChildren(QAction)
    for action in actions:
        key = action.shortcut().toString()
        if key and key in defined_shortcuts:
            problematic_shortcuts.append(key)
        defined_shortcuts[key].append(key)
    assert not problematic_shortcuts
    # due to throttled mouse_move, a timer is started by the viewer, so we
    # need to wait for it to be done
    qtbot.wait(10)
