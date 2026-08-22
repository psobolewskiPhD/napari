"""Qt leak detection for ``napari_builtins``' tests.

The detectors themselves live in ``napari/conftest.py``, but a conftest only
applies to items below its own directory - so without this file
``napari_builtins``' Qt tests get no dangling-widget check, no thread/timer
checks and no dask/zarr shutdown. That gap is expensive rather than cosmetic: a
leak born here is invisible where it happens and gets reported against
whichever ``napari`` test runs next on that worker, which is how ``f46d54058``
happened. Under ``--dist loadfile`` the victim is whichever file the scheduler
hands that worker next, so the symptom is a false failure in an unrelated file.

Lives inside the package rather than at ``src/`` level on purpose. A conftest
here ships in the sdist by virtue of being in a discovered package, needing no
``MANIFEST.in`` exception, and it travels with the package if
``napari_builtins`` is ever split into its own distribution - it would keep
importing ``napari.conftest`` from napari as an installed dependency.

Not in the ``pytest11`` plugin (``napari.utils._testsupport``), which every
napari installer loads: the detectors auto-apply to any test using ``qapp``, so
shipping them there would start failing third-party plugin suites on leaks they
have always had.
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
