"""Convert raw complex CSI into amplitude and wrapped phase."""

from math import atan2, hypot

from .csi_structs import CsiFrame, SignalFrame


def to_signal_frame(frame: CsiFrame) -> SignalFrame:
    """Convert each CSI slot to linear amplitude and wrapped phase in radians.

    Preserve the timestamp and slot order. Missing coefficients yield None in
    both tuples. A zero coefficient has amplitude zero and undefined phase.
    """

    amplitudes = []
    phases = []

    for coefficient in frame.csi.values:
        if coefficient is None:
            amplitudes.append(None)
            phases.append(None)
            continue

        mag = hypot(coefficient.real, coefficient.imaginary)
        arg = atan2(coefficient.imaginary, coefficient.real) if mag else None
        amplitudes.append(mag)
        phases.append(arg)

    return SignalFrame(frame.captured_at_ns, tuple(amplitudes), tuple(phases))
