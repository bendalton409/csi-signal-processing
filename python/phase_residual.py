"""Remove the linear trend from unwrapped CSI phase (issue #15)."""

from typing import Optional

from .csi_structs import CsiSession, SignalFrame


def remove_linear_offset(frame: SignalFrame, session: CsiSession) -> SignalFrame:
    """Subtract a least-squares phase slope and intercept per TX/RX link.

    Fit against the actual OFDM tone indices using available samples. Links
    with fewer than two valid tones pass through. This removes any affine
    phase component, including genuine channel phase that follows the trend.
    """
    layout = session.layout
    tones = layout.subcarrier_indices
    tone_count = len(tones)
    link_count = layout.tx_count * layout.rx_count
    size = link_count * tone_count
    if len(frame.phase_rad) != size or len(frame.amplitude) != size:
        raise ValueError(f"Expected {size} CSI slots for this session")

    corrected = []
    for link in range(link_count):
        start = link * tone_count
        phases = frame.phase_rad[start : start + tone_count]
        corrected.extend(_detrend_link(tones, phases))

    return SignalFrame(frame.captured_at_ns, frame.amplitude, tuple(corrected))


def _detrend_link(
    tones: tuple[int, ...], phases: tuple[Optional[float], ...]
) -> tuple[Optional[float], ...]:
    valid = [(tone, phase) for tone, phase in zip(tones, phases) if phase is not None]
    if len(valid) < 2:
        return phases

    mean_tone = sum(tone for tone, _ in valid) / len(valid)
    mean_phase = sum(phase for _, phase in valid) / len(valid)
    variance = sum((tone - mean_tone) ** 2 for tone, _ in valid)
    covariance = sum((tone - mean_tone) * (phase - mean_phase) for tone, phase in valid)
    slope = covariance / variance
    return tuple(
        None if phase is None else phase - mean_phase - slope * (tone - mean_tone)
        for tone, phase in zip(tones, phases)
    )
