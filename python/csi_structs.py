from dataclasses import dataclass
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class CsiLayout:
    """Axis sizes and OFDM tone indices for a capture session.

    Attributes
    ----------
    tx_count
        Number of transmit streams.
    rx_count
        Number of receive chains.
    subcarrier_indices
        Actual OFDM tone indices in tensor order. 
        Indices can have gaps.
        For example, if subcarrier_indices[1] = -2, for all tx-rx links in the frame,
        CsiTensor[tx][rx][1] = CFR Complex value for frequency: 
        CsiSession.center_frequency_hz + (-2) * CsiSession.subcarrier_spacing_hz
    """

    tx_count: int
    rx_count: int
    subcarrier_indices: tuple[int, ...]


@dataclass(frozen=True)
class CsiSession:
    """Fixed metadata shared by all frames in one session.

    Attributes
    ----------
    layout
        Tensor dimensions and axis ordering for the session.
    center_frequency_hz
        Radio frequency at the centre of the channel, in hertz.
    channel_bandwidth_hz
        Nominal width of the radio channel, in hertz.
    subcarrier_spacing_hz
        Frequency spacing between adjacent OFDM tone indices, in hertz.
    """

    layout: CsiLayout
    center_frequency_hz: int
    channel_bandwidth_hz: int
    subcarrier_spacing_hz: int


@dataclass(frozen=True)
class Complex:
    """One raw complex channel coefficient.

    Attributes
    ----------
    real
        Real component of the complex number.
    imaginary
        Imaginary component of the complex number.
    """

    real: float
    imaginary: float


@dataclass(frozen=True)
class CsiTensor(Generic[T]):
    """Tensor of raw coefficients or processed samples.

    Attributes
    ----------
    values
        Flat tuple in [tx_stream, rx_chain, subcarrier] layout. 
        The shape comes from the session layout. 
        Each slot holds a Complex or a ProcessedSample according to the tensor's type,
        or None when the sample is unavailable.
    """

    values: tuple[Optional[T], ...]


@dataclass(frozen=True)
class CsiFrame:
    """Raw CSI measurements from one capture.

    Attributes
    ----------
    captured_at_ns
        Capture time in nanoseconds since the session origin.
    csi
        Complex channel coefficients arranged in the session's tensor layout.
    """

    captured_at_ns: int
    csi: CsiTensor[Complex]


@dataclass(frozen=True)
class SignalFrame:
    """Amplitude and phase in the input frame's flat session layout.

    Attributes
    ----------
    captured_at_ns
        Capture time in nanoseconds since the session origin.
    amplitude
        Amplitude at each tensor position, or None if unavailable.
    phase_rad
        Phase in radians at each tensor position, or None if unavailable.
    """

    captured_at_ns: int
    amplitude: tuple[Optional[float], ...]
    phase_rad: tuple[Optional[float], ...]


@dataclass(frozen=True)
class ProcessedSample:
    """Processed signal and feature values at one tensor position.

    Attributes
    ----------
    amplitude
        Denoised amplitude before background subtraction.
    phase_rad
        Unwrapped, corrected phase in radians, or None if unavailable.
    dynamic_amplitude
        Signed amplitude residual after background subtraction.
    variance
        Temporal variance, or None if unavailable. Its definition, units,
        and window require agreement.
    """

    amplitude: float
    phase_rad: Optional[float]
    dynamic_amplitude: float
    variance: Optional[float]


@dataclass(frozen=True)
class ProcessedFrame:
    """DSP output for one capture, using the active session layout.

    Attributes
    ----------
    captured_at_ns
        Capture time in nanoseconds since the session origin.
    signals
        Processed samples arranged in the session's tensor layout. A None
        slot marks an unavailable sample.
    motion_energy
        Aggregate motion feature, or None if unavailable. Its formula and
        units require agreement.
    """

    captured_at_ns: int
    signals: CsiTensor[ProcessedSample]
    motion_energy: Optional[float]
