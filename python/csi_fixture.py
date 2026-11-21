"""Ten synthetic input frames using the proposed CSI capture contract."""

from dataclasses import dataclass
from math import cos, sin


@dataclass(frozen=True)
class CsiLayout:
    """Axis sizes and OFDM tone indices for a capture session."""

    tx_count: int
    rx_count: int
    subcarrier_indices: tuple[int, ...]


@dataclass(frozen=True)
class CsiSession:
    """Fixed metadata shared by all frames in one session."""

    layout: CsiLayout
    center_frequency_hz: int
    channel_bandwidth_hz: int
    subcarrier_spacing_hz: int
    amplitude_convention: str


@dataclass(frozen=True)
class ComplexCsi:
    """One raw complex channel coefficient."""

    real: float
    imaginary: float


@dataclass(frozen=True)
class CsiTensor:
    """Flat [tx_stream, rx_chain, subcarrier] samples, with subcarrier fastest.

    None marks a missing coefficient. The shape comes from CsiSession.layout.
    """

    values: tuple[ComplexCsi | None, ...]


@dataclass(frozen=True)
class CsiFrame:
    """One capture instant, timestamped in nanoseconds from session start."""

    captured_at_ns: int
    csi: CsiTensor


CSI_SESSION = CsiSession(
    layout=CsiLayout(
        tx_count=2,
        rx_count=2,
        subcarrier_indices=(-4, -3, -2, -1, 1, 2, 3, 4),
    ),
    center_frequency_hz=5_180_000_000,
    channel_bandwidth_hz=20_000_000,
    subcarrier_spacing_hz=78_125,
    amplitude_convention="Linear amplitude in arbitrary synthetic units",
)


def _make_frame(frame_index: int) -> CsiFrame:
    layout = CSI_SESSION.layout
    values = []
    for tx in range(layout.tx_count):
        for rx in range(layout.rx_count):
            for tone in layout.subcarrier_indices:
                if (frame_index, tx, rx, tone) == (6, 1, 0, 3):
                    values.append(None)
                    continue

                # Smooth amplitude drift with one isolated spike for filtering.
                amplitude = 1.0 + 0.12 * tx + 0.06 * rx + 0.02 * tone
                amplitude += 0.04 * sin(0.6 * frame_index + 0.2 * tone)
                if (frame_index, tx, rx, tone) == (4, 0, 1, -2):
                    amplitude += 2.0

                # The phase crosses pi so atan2 on the raw CSI wraps.
                phase = 2.4 + 0.25 * frame_index + 0.15 * tone + 0.2 * (tx - rx)
                values.append(ComplexCsi(amplitude * cos(phase), amplitude * sin(phase)))

    return CsiFrame(frame_index * 10_000_000, CsiTensor(tuple(values)))


CSI_FRAMES = tuple(_make_frame(index) for index in range(10))
