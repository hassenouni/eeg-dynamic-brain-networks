import numpy as np
import pytest
from eeg_dynamic.algorithms import dynamic_connectivity, graph_changes


def test_window_timing_and_tail():
    rng = np.random.default_rng(7)
    result = dynamic_connectivity(rng.normal(size=(3, 105)), 10, 2, 1)
    assert result.matrices.shape == (9, 3, 3)
    np.testing.assert_array_equal(result.starts, np.arange(9))
    np.testing.assert_array_equal(result.centers, np.arange(9) + 1)


def test_detects_coupling_change():
    rng = np.random.default_rng(3)
    x = rng.normal(size=400)
    y = np.r_[x[:200], -x[200:]]
    sequence = dynamic_connectivity(np.array([x, y]), 100, 2, 2)
    times, changes = graph_changes(sequence)
    np.testing.assert_array_equal(times, [3.0])
    assert changes[0] > 2.8


def test_single_graph_has_empty_changes():
    rng = np.random.default_rng(3)
    seq = dynamic_connectivity(rng.normal(size=(2, 100)), 50, 2)
    assert graph_changes(seq)[1].shape == (0,)


def test_invalid_method():
    with pytest.raises(ValueError):
        dynamic_connectivity(np.ones((2, 100)), 10, method="unknown")
