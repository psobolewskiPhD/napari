from unittest.mock import Mock, patch

import pytest
from magicgui import magic_factory, magicgui
from magicgui.widgets import Container
from npe2 import DynamicPlugin
from qtpy.QtWidgets import QWidget

import napari
from napari._app_model import get_app_model
from napari._qt._qplugins._qnpe2 import _get_widget_viewer_param
from napari._qt.qt_main_window import _instantiate_dock_widget
from napari.utils._proxies import PublicOnlyProxy
from napari.viewer import Viewer


class ErrorWidget:
    pass


class QWidget_example(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()


class QWidget_string_annnot(QWidget):
    def __init__(self, test: 'napari.viewer.Viewer'):
        super().__init__()  # pragma: no cover


class Container_example(Container):
    def __init__(self, test: Viewer):
        super().__init__()


@magic_factory
def magic_widget_example():
    """Example magic factory widget."""


def callable_example():
    @magicgui
    def magic_widget_example():
        """Example magic factory widget."""

    return magic_widget_example


class Widg2(QWidget):
    def __init__(self, napari_viewer) -> None:
        self.viewer = napari_viewer
        super().__init__()


class Widg3(QWidget):
    def __init__(self, v: Viewer) -> None:
        self.viewer = v
        super().__init__()

    def fail(self):
        """private attr not allowed"""
        self.viewer.window._qt_window


def magicfunc(viewer: 'napari.Viewer'):
    return viewer


dwidget_args = {
    'single_class': QWidget_example,
    'class_tuple': (QWidget_example, {'area': 'right'}),
    'tuple_list': [(QWidget_example, {'area': 'right'}), (Widg2, {})],
    'tuple_list2': [(QWidget_example, {'area': 'right'}), Widg2],
    'bad_class': 1,
    'bad_tuple1': (QWidget_example, 1),
    'bad_double_tuple': ((QWidget_example, {}), (Widg2, {})),
}


def test_inject_viewer_proxy(make_napari_viewer):
    """Test that the injected viewer is a public-only proxy"""
    viewer = make_napari_viewer()
    wdg = _instantiate_dock_widget(Widg3, viewer)
    assert isinstance(wdg.viewer, PublicOnlyProxy)

    # simulate access from outside napari
    with (
        patch('napari.utils.misc.ROOT_DIR', new='/some/other/package'),
        pytest.warns(FutureWarning),
    ):
        wdg.fail()


@pytest.mark.parametrize(
    ('widget_callable', 'param'),
    [
        (QWidget_example, 'napari_viewer'),
        (QWidget_string_annnot, 'test'),
        (Container_example, 'test'),
    ],
)
def test_get_widget_viewer_param(widget_callable, param):
    """Test `_get_widget_viewer_param` returns correct parameter name."""
    out = _get_widget_viewer_param(widget_callable, 'widget_name')
    assert out == param


def test_get_widget_viewer_param_error():
    """Test incorrect subclass raises error in `_get_widget_viewer_param`."""
    with pytest.raises(TypeError) as e:
        _get_widget_viewer_param(ErrorWidget, 'widget_name')
    assert "'widget_name' must be `QtWidgets.QWidget`" in str(e)


def test_plugin_widget_window_menu(make_napari_viewer, qtbot, tmp_plugin):
    """Plugin widgets are listed in the Window menu with a visibility checkmark.

    Opening a plugin widget adds its toggle action to the Window menu, separated
    from napari's own widgets by a separator. The checkmark tracks visibility,
    and closing the widget removes the action again.
    """
    tmp_plugin.contribute.widget(display_name='Widget')(QWidget_example)

    app = get_app_model()
    viewer = make_napari_viewer(show=True)
    (widget_contrib,) = tmp_plugin.manifest.contributions.widgets
    app.commands.execute_command(widget_contrib.command)

    menu = viewer.window.window_menu
    full_name = 'Widget (Temp Plugin)'
    dock_widget = viewer.window._wrapped_dock_widgets[full_name]
    action = dock_widget.toggleViewAction()
    qtbot.waitUntil(dock_widget.isVisible)

    # action present in the Window menu, separated from napari's own widgets
    assert action in menu.actions()
    separators = [a for a in menu.actions() if a.isSeparator()]
    assert separators
    assert menu.actions().index(separators[-1]) < menu.actions().index(action)

    # checkmark reflects visibility
    assert action.isChecked()
    dock_widget.hide()
    qtbot.waitUntil(lambda: not dock_widget.isVisible())
    assert not action.isChecked()
    dock_widget.show()
    qtbot.waitUntil(dock_widget.isVisible)
    assert action.isChecked()

    # closing the widget removes the action from the menu
    dock_widget.destroyOnClose()
    assert full_name not in viewer.window._wrapped_dock_widgets
    assert action not in menu.actions()


def test_plugin_widget_window_menu_separator(make_napari_viewer, tmp_plugin):
    """The window menu separator is removed once all plugin widgets are closed."""
    tmp_plugin.contribute.widget(display_name='Widget A')(QWidget_example)
    tmp_plugin.contribute.widget(display_name='Widget B')(Widg2)

    app = get_app_model()
    viewer = make_napari_viewer(show=True)
    for widget_contrib in tmp_plugin.manifest.contributions.widgets:
        app.commands.execute_command(widget_contrib.command)

    menu = viewer.window.window_menu
    docks = [
        viewer.window._wrapped_dock_widgets[name]
        for name in ('Widget A (Temp Plugin)', 'Widget B (Temp Plugin)')
    ]
    actions = [dock.toggleViewAction() for dock in docks]

    # the separator stays while a widget remains, and is removed with the last
    docks[0].destroyOnClose()
    assert actions[0] not in menu.actions()
    assert actions[1] in menu.actions()
    assert [a for a in menu.actions() if a.isSeparator()]
    docks[1].destroyOnClose()
    assert actions[1] not in menu.actions()
    assert not [a for a in menu.actions() if a.isSeparator()]


def test_widget_hide_destroy(make_napari_viewer, qtbot):
    """Test that widget hide and destroy works."""
    viewer = make_napari_viewer()
    viewer.window.add_dock_widget(QWidget_example(viewer), name='test')
    dock_widget = viewer.window._wrapped_dock_widgets['test']

    # Check widget persists after hide
    widget = dock_widget.widget()
    dock_widget.title.hide_button.click()
    assert widget
    # Check that widget removed from `_dock_widgets` dict and parent
    # `QtViewerDockWidget` is `None` when closed
    dock_widget.destroyOnClose()
    assert 'test' not in viewer.window._wrapped_dock_widgets
    assert widget.parent() is None
    widget.deleteLater()
    widget.close()
    qtbot.wait(50)


@pytest.mark.parametrize(
    'Widget',
    [
        QWidget_example,
        Container_example,
        magic_widget_example,
        callable_example,
    ],
)
def test_widget_types_supported(
    make_napari_viewer,
    tmp_plugin: DynamicPlugin,
    Widget,
):
    """Test all supported widget types correctly instantiated and call processor.

    The 4 parametrized `Widget`s represent the varing widget constructors and
    signatures that we want to support.
    """
    # Using the decorator as a function on the parametrized `Widget`
    # This allows `Widget` to be callable object that, when called, returns an
    # instance of a widget
    tmp_plugin.contribute.widget(display_name='Widget')(Widget)

    app = get_app_model()
    viewer = make_napari_viewer()

    # `side_effect` required so widget is added to window and then
    # cleaned up, preventing widget leaks
    viewer.window.add_dock_widget = Mock(
        side_effect=viewer.window.add_dock_widget
    )
    (widget_contrib,) = tmp_plugin.manifest.contributions.widgets
    app.commands.execute_command(widget_contrib.command)
    viewer.window.add_dock_widget.assert_called_once()
