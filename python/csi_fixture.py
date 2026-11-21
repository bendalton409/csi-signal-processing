"""Ten synthetic input frames using the proposed CSI capture contract."""

from .csi_structs import CsiSession, CsiLayout, CsiFrame, CsiTensor, Complex
from math import cos, sin
from random import Random


CSI_SESSION = CsiSession(
    layout=CsiLayout(
        tx_count=2,
        rx_count=2,
        subcarrier_indices=tuple(range(-10,11)),
    ),
    center_frequency_hz=5_180_000_000,
    channel_bandwidth_hz=20_000_000,
    subcarrier_spacing_hz=78_125,
    amplitude_convention="Linear amplitude in arbitrary synthetic units",
)


def _make_frame(frame_index: int) -> CsiFrame:
    layout = CSI_SESSION.layout
    values = []
    rng = Random(frame_index)
    amplitude = 1.0  # Flat baseline across subcarriers, links, and frames.
    noise_std = 0.03
    none_prob = 0.015

    for tx in range(layout.tx_count):
        for rx in range(layout.rx_count):
            for tone in layout.subcarrier_indices:
                # The phase crosses pi so atan2 on the raw CSI wraps.
                phase = 2.4 + 0.25 * frame_index + 0.15 * tone + 0.2 * (tx - rx)
                real = amplitude * cos(phase) + rng.gauss(0, noise_std)
                imaginary = amplitude * sin(phase) + rng.gauss(0, noise_std)
                
                # Chance for value to be missing.
                if (rng.uniform(0,1) < none_prob):
                  values.append(None)
                else:
                  values.append(Complex(real, imaginary))  

    captured_at_ns = frame_index * 10_000_000

    return CsiFrame(captured_at_ns, CsiTensor(tuple(values)))


CSI_FRAMES = tuple(_make_frame(index) for index in range(10))
