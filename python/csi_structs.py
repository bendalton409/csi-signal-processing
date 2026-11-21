from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


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
class Complex:
    """One raw complex channel coefficient."""

    real: float
    imaginary: float


@dataclass(frozen=True)
class CsiTensor(Generic[T]):
    """Flat [tx_stream, rx_chain, subcarrier] samples, with subcarrier fastest.
    T will either be a Complex or ProcessedSample
    None marks an unavailable sample. The shape comes from CsiSession.layout.
    """

    values: tuple[Optional[T], ...]


@dataclass(frozen=True)
class CsiFrame:
    """One capture instant, timestamped in nanoseconds from session start."""

    captured_at_ns: int
    csi: CsiTensor[Complex]


@dataclass(frozen=True)
class SignalFrame:
    """Amplitude and phase in the input frame's flat session layout.

    Phase is in radians. None marks an unavailable value.
    """

    captured_at_ns: int
    amplitude: tuple[Optional[float], ...]
    phase_rad: tuple[Optional[float], ...]


@dataclass(frozen=True)
class ProcessedSample:
    """Processed signal and feature values at one tensor position."""

    # Denoised linear amplitude before background subtraction.
    amplitude: float
    # Corrected, unwrapped phase in radians, when available.
    phase_rad: Optional[float]
    # Signed amplitude residual after background subtraction.
    dynamic_amplitude: float
    # Temporal variance. Its definition, units and window require agreement.
    variance: Optional[float]


@dataclass(frozen=True)
class ProcessedFrame:
    """DSP output for one capture, using the active session layout."""

    # Capture time in nanoseconds since the session origin.
    captured_at_ns: int
    # Unavailable tensor positions contain None.
    signals: CsiTensor[ProcessedSample]
    # Aggregate motion feature. Its formula and units require agreement.
    motion_energy: Optional[float]
