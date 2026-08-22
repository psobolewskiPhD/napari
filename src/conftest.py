"""Qt leak detection for *both* packages under ``src/``.

``napari/conftest.py`` defines the leak detectors, but a conftest only applies
to items below its own directory - so ``napari_builtins``' 18 Qt tests got none
of them. That is not a cosmetic gap: a widget, timer or thread leaked there is
invisible where it happens and is reported against whichever ``napari`` test
runs next on that worker, which is how ``f46d54058`` happened. Under
``--dist loadfile`` the victim is whichever file the scheduler hands that
worker next, so the symptom is a false failure in an unrelated file - the most
expensive kind to debug.

The two hooks are registered *here and only here*, delegating to helpers in
``napari/conftest.py``. Defining them in both places would register both
implementations and append every detector fixture twice (measured: pytest
dedupes the actual setup, but nothing guarantees that).

Deliberately not in the ``pytest11`` plugin (``napari.utils._testsupport``),
which every napari installer loads: the detectors auto-apply to any test using
``qapp``, so shipping them would start failing third-party plugin suites on
leaks they have always had.

If ``napari_builtins`` is ever split into its own distribution this file cannot
follow it, and the detectors belong in that plugin behind an opt-in ini flag
instead - it already has a ``pytest_addoption``, so the gate is the only new
part.
"""

import pytest

from napari.conftest import (  # noqa: F401  (fixtures: imported so pytest finds them)
    _dangling_qanimations,
    _dangling_qthread_pool,
    _dangling_qthreads,
    _dangling_qtimers,
    _find_dangling_widgets,
    apply_leak_detection_fixtures,
    force_stop_pending_qt_resources,
)


def pytest_runtest_setup(item):
    apply_leak_detection_fixtures(item)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    force_stop_pending_qt_resources(item)
