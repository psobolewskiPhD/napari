from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from pretend import stub

from napari._qt.perf import qt_performance
from napari._tests.utils import skip_local_popups, skip_on_win_ci

if TYPE_CHECKING:
    from pathlib import Path

# NOTE:
# for some reason, running this test fails in a subprocess with a segfault
# if you don't show the viewer...
PERFMON_SCRIPT = """
import napari
from qtpy.QtCore import QTimer

v = napari.Viewer()
v.add_points()
QTimer.singleShot(100, napari._qt.qt_event_loop.quit_app)
napari.run()
"""

CONFIG = {
    'trace_qt_events': True,
    'trace_file_on_start': '',
    'trace_callables': ['chunk_loader'],
    'callable_lists': {
        'chunk_loader': [
            'napari.components.experimental.chunk._loader.ChunkLoader.load_request',
            'napari.components.experimental.chunk._loader.ChunkLoader._on_done',
        ]
    },
}


@pytest.fixture
def perf_config(tmp_path: Path):
    trace_path = tmp_path / 'trace.json'
    config_path = tmp_path / 'perfmon.json'
    CONFIG['trace_file_on_start'] = str(trace_path)
    config_path.write_text(json.dumps(CONFIG))

    return stub(path=config_path, trace_path=trace_path)


@pytest.fixture
def perfmon_script(tmp_path):
    script = PERFMON_SCRIPT
    if 'coverage' in sys.modules:
        script_path = tmp_path / 'script.py'
        with script_path.open('w') as f:
            f.write(script)
        return '-m', 'coverage', 'run', str(script_path)
    return '-c', script


@skip_on_win_ci
@skip_local_popups
@pytest.mark.usefixtures('qapp')
def test_trace_on_start(tmp_path: Path, perf_config, perfmon_script):
    """Make sure napari can write a perfmon trace file."""

    env = os.environ.copy()
    env.update({'NAPARI_PERFMON': str(perf_config.path), 'NAPARI_CONFIG': ''})

    # `check=False` plus an explicit assert, rather than `check=True`: this
    # spawns a whole napari process, and a `CalledProcessError` tells you only
    # the exit code. When the child dies (a GL crash on a headless runner, an
    # import error, a Qt plugin failure) the traceback it printed is the only
    # useful evidence, and letting subprocess swallow it is what turns this
    # into an undiagnosable flake worth skipping. Capture and report it.
    #
    # `timeout` is not optional here. `capture_output=True` means `run()` reads
    # the child's pipes until EOF, so a child that never exits - or a grandchild
    # that inherits the pipes and outlives it - blocks this thread forever. This
    # is a whole napari process with a GUI, on runners where a hung window
    # system is a real possibility. The value sits well below
    # `faulthandler_timeout` (600s) so that a stuck child fails *here*, with the
    # partial output below, rather than as an unexplained worker kill. Observed
    # cost of a healthy run: ~23s on macos-15.
    try:
        result = subprocess.run(
            [sys.executable, *perfmon_script],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        # TimeoutExpired carries whatever was read before the timeout; it is
        # bytes even with text=True, so decode defensively.
        def _text(stream: bytes | str | None) -> str:
            if stream is None:
                return ''
            if isinstance(stream, bytes):
                return stream.decode(errors='replace')
            return stream

        pytest.fail(
            f'perfmon script did not exit within {e.timeout}s\n'
            f'--- stdout so far ---\n{_text(e.stdout)}\n'
            f'--- stderr so far ---\n{_text(e.stderr)}'
        )

    assert result.returncode == 0, (
        f'perfmon script exited with {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n'
        f'--- stderr ---\n{result.stderr}'
    )

    # Make sure file exists and is not empty.
    assert perf_config.trace_path.exists(), 'Trace file not written'
    assert perf_config.trace_path.stat().st_size > 0, 'Trace file is empty'

    # Assert every event contains every important field.
    with perf_config.trace_path.open() as infile:
        data = json.load(infile)
        assert len(data) > 0
        for event in data:
            for field in ['pid', 'tid', 'name', 'ph', 'ts', 'args']:
                assert field in event


def test_qt_performance(qtbot, monkeypatch):
    widget = qt_performance.QtPerformance()
    widget.timer.stop()
    qtbot.addWidget(widget)
    mock = MagicMock()
    data = [
        ('test1', MockTimer(1, 1)),
        ('test2', MockTimer(20, 120)),
        ('test1', MockTimer(70, 90)),
        ('test2', MockTimer(50, 220)),
    ]
    mock.timers.items = MagicMock(return_value=data)
    monkeypatch.setattr(qt_performance.perf, 'timers', mock)
    assert widget.log.toPlainText() == ''
    widget.update()
    assert widget.log.toPlainText() == '  120ms test2\n  220ms test2\n'
    widget._change_thresh('150')
    assert widget.log.toPlainText() == ''
    widget.update()
    assert widget.log.toPlainText() == '  220ms test2\n'


@dataclasses.dataclass
class MockTimer:
    average: float
    max: float
