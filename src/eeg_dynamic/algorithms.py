"""Complete-window dynamic connectivity; graphs are aligned to window centers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eeg_dataset_explorer.signals import signal_array, sampling_rate
from eeg_connectivity.algorithms import correlation, plv


@dataclass(frozen=True)
class GraphSequence:
    starts: np.ndarray
    centers: np.ndarray
    matrices: np.ndarray
    duration: float


def dynamic_connectivity(
    data: np.ndarray, fs: float, seconds: float = 2.0, stride: float = 1.0, method: str = "correlation"
) -> GraphSequence:
    """Compute correlation or PLV on full windows; discard the incomplete tail."""
    x = signal_array(data)
    sampling_rate(fs)
    if not np.isfinite([seconds, stride]).all() or min(seconds, stride) <= 0:
        raise ValueError("Positive window duration and stride required")
    width, step = round(seconds * fs), round(stride * fs)
    if width < 4 or step < 1 or width > x.shape[1]:
        raise ValueError("Window or stride incompatible with signal")
    methods = {"correlation": correlation, "plv": plv}
    if method not in methods:
        raise ValueError("Choose correlation or plv")
    indices = np.arange(0, x.shape[1] - width + 1, step)
    matrices = np.stack([methods[method](x[:, i : i + width]) for i in indices])
    duration = width / fs
    return GraphSequence(indices / fs, indices / fs + duration / 2, matrices, duration)


def graph_changes(sequence: GraphSequence) -> tuple[np.ndarray, np.ndarray]:
    """Frobenius change excluding self-edges; timestamps are later window centers."""
    matrices = np.array(sequence.matrices, copy=True, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or len(matrices) != len(sequence.centers):
        raise ValueError("Invalid graph sequence shape")
    if not np.isfinite(matrices).all() or not np.isfinite(sequence.centers).all():
        raise ValueError("Nonfinite graph sequence")
    if np.any(np.diff(sequence.centers) <= 0):
        raise ValueError("Graph centers must increase")
    idx = np.arange(matrices.shape[1])
    matrices[:, idx, idx] = 0
    return sequence.centers[1:].copy(), np.linalg.norm(np.diff(matrices, axis=0), axis=(1, 2))
