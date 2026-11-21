"""Filter amplitude and phase outliers along the time axis (issue #16)."""

from collections import deque
from math import isfinite
from statistics import median
from typing import Optional

from .csi_structs import SignalFrame


class HampelFilter:
    """Apply a causal Hampel filter independently to each amplitude and phase bin.

    Each window includes the current frame and up to window_size - 1 earlier
    frames. Missing values are ignored. A sample passes through until at least
    three valid values exist in its window. The same rule applies at the last
    frame, so no lookahead or flush is needed.

    The threshold is expressed in raw median absolute deviations (MADs).
    When MAD is zero, a value different from the median is replaced.
    """

    def __init__(self, window_size: int = 5, threshold: float = 3.0):
        if window_size < 3:
            raise ValueError("Hampel window_size must be at least 3")
        if not isfinite(threshold) or threshold <= 0:
            raise ValueError("Hampel threshold must be finite and positive")
        self._history: deque[SignalFrame] = deque(maxlen=window_size)
        self._threshold = threshold

    def process(self, frame: SignalFrame) -> SignalFrame:
        """Return the filtered frame immediately, preserving its timestamp."""
        size = len(frame.amplitude)
        if len(frame.phase_rad) != size or (
            self._history and len(self._history[-1].amplitude) != size
        ):
            raise ValueError("Hampel frames must share the same tensor size")

        self._history.append(frame)
        amplitudes = zip(*(sample.amplitude for sample in self._history))
        phases = zip(*(sample.phase_rad for sample in self._history))
        return SignalFrame(
            frame.captured_at_ns,
            tuple(self._filter(window) for window in amplitudes),
            tuple(self._filter(window) for window in phases),
        )

    def _filter(self, window: tuple[Optional[float], ...]) -> Optional[float]:
        current = window[-1]
        if current is None:
            return None

        valid = [sample for sample in window if sample is not None]
        if len(valid) < 3:
            return current

        center = median(valid)
        mad = median(abs(sample - center) for sample in valid)
        return center if abs(current - center) > self._threshold * mad else current
