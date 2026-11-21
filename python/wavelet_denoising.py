"""Denoise CSI time series with a multilevel Haar DWT (issue #17)."""

from collections.abc import Sequence
from math import copysign, isfinite, sqrt
from typing import Optional

from .csi_structs import SignalFrame

_HAAR_SCALE = 1 / sqrt(2)


def denoise_series(
    samples: Sequence[Optional[float]], *, levels: int = 2, threshold: float = 0.1
) -> tuple[Optional[float], ...]:
    """Soft-threshold the detail coefficients of a multilevel Haar DWT.

    The Haar wavelet uses pairwise sums and differences. At every level, detail
    coefficients shrink toward zero by threshold, in the input's units, while
    the coarsest approximation is retained. Each contiguous observed run is
    transformed separately. Repeat the final sample at odd lengths and trim
    padding during reconstruction so output indices match input indices.
    Short runs use only the levels supported by their length.
    """
    if levels < 1:
        raise ValueError("Wavelet levels must be at least 1")
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("Wavelet threshold must be finite and nonnegative")

    output = list(samples)
    start = 0
    for end in range(len(samples) + 1):
        if end < len(samples) and samples[end] is not None:
            continue
        run = [sample for sample in samples[start:end] if sample is not None]
        if len(run) > 1 and threshold > 0:
            output[start:end] = _denoise_run(run, levels, threshold)
        start = end + 1
    return tuple(output)


def _denoise_run(samples: list[float], levels: int, threshold: float) -> list[float]:
    approximation = samples
    details = []
    for _ in range(levels):
        length = len(approximation)
        if length < 2:
            break
        if length % 2:
            approximation = [*approximation, approximation[-1]]

        averages = []
        differences = []
        for left, right in zip(approximation[::2], approximation[1::2]):
            averages.append((left + right) * _HAAR_SCALE)
            detail = (left - right) * _HAAR_SCALE
            differences.append(copysign(max(0.0, abs(detail) - threshold), detail))
        details.append((length, differences))
        approximation = averages

    for length, differences in reversed(details):
        reconstructed = []
        for average, detail in zip(approximation, differences):
            reconstructed.extend(
                ((average + detail) * _HAAR_SCALE, (average - detail) * _HAAR_SCALE)
            )
        approximation = reconstructed[:length]
    return approximation


def denoise_frames(
    frames: Sequence[SignalFrame],
    *,
    levels: int = 2,
    amplitude_threshold: float = 0.1,
    phase_threshold: float = 0.1,
) -> tuple[SignalFrame, ...]:
    """Denoise a captured block per tensor bin and retain capture timestamps.

    Supply frames in capture order at roughly uniform intervals. The transform
    operates by sample index and uses samples on both sides within the block.
    Thresholds for linear amplitude and phase in radians are independent.
    """
    if not frames:
        return ()

    size = len(frames[0].amplitude)
    if any(len(frame.amplitude) != size or len(frame.phase_rad) != size for frame in frames):
        raise ValueError("Wavelet frames must share the same tensor size")

    amplitudes = [
        denoise_series(series, levels=levels, threshold=amplitude_threshold)
        for series in zip(*(frame.amplitude for frame in frames))
    ]
    phases = [
        denoise_series(series, levels=levels, threshold=phase_threshold)
        for series in zip(*(frame.phase_rad for frame in frames))
    ]
    return tuple(
        SignalFrame(
            frame.captured_at_ns,
            tuple(series[index] for series in amplitudes),
            tuple(series[index] for series in phases),
        )
        for index, frame in enumerate(frames)
    )
