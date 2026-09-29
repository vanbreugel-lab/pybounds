import subprocess
import sys

import pytest

import pybounds


def _run(code):
    return subprocess.run([sys.executable, '-W', 'ignore', '-c', code], capture_output=True, text=True)


def test_star_import_without_jax():
    """With JAX unavailable, `from pybounds import *` must work and skip the JAX names."""
    code = ("import sys; sys.modules['jax'] = None\n"
            "from pybounds import *\n"
            "import pybounds\n"
            "assert not pybounds._JAX_AVAILABLE\n"
            "assert 'JaxSimulator' not in pybounds.__all__\n"
            "assert 'Simulator' in dir()\n")
    out = _run(code)
    assert out.returncode == 0, out.stderr


def test_all_names_exist():
    for name in pybounds.__all__:
        assert hasattr(pybounds, name), name


def test_star_import_with_jax():
    pytest.importorskip('jax')
    out = _run("from pybounds import *\nassert 'JaxSimulator' in dir()\n")
    assert out.returncode == 0, out.stderr
