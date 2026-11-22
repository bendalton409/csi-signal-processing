from dataclasses import dataclass
from typing import Optional, Union
from math import hypot, atan2


@dataclass
class CsiLayout:
  """Holds dimension sizes and tone indices for a session
  
  Attributes:
    tx_count: number of transmitters
    rx_count: number of receivers
    subcarrier_idxs: tone indices in increasing order
  """

  tx_count: int
  rx_count: int
  subcarrier_idxs: list[int]


@dataclass
class CsiSession:
  """Holds values which stay constant during the session
    
  Attributes:
    layout: dimension length + order for value tensors
    center_frequency_hz: wave frequency at the center of the channel
    channel_bandwidth_hz: width of the channel
    subcarrier_spacing_hz: spacing between adjacent tone indices
  """
  
  layout: CsiLayout
  center_frequency_hz: int
  channel_bandwidth_hz: int
  subcarrier_spacing_hz: int


@dataclass
class Complex:
  """Holds a complex number
    
  Attributes:
    re: real component
    im: imaginary component
  """

  re: float
  im: float

  def magnitude(self) -> float:
    """returns the magnitude of the complex number"""
    return hypot(self.re, self.im)

  def argument(self) -> float:
    """returns the argument of the complex number in the range (-pi, pi]"""
    return atan2(self.im, self.re)


@dataclass
class ProcessedSample:
  """Holds processed amplitude and phase, and extracted variance
    
  Attributes:
    amplitude: denoised amplitude before background subtraction
    amplitude_residual: denoised amplitude after background subtraction
    phase: unwrapped, residual phase in radians
    variance: temporal variance, yet to be agreed on definition (phase or amplitude)
  """

  amplitude: float
  amplitude_residual: float
  phase: Optional[float]
  variance: Optional[float]


@dataclass
class ValuesTensor:
  """Holds csi complex values or processed samples
    
  Attributes:
    values: values with shape defined by CsiLayout
  """

  values: list[Optional[Union[Complex, ProcessedSample]]]


@dataclass
class CsiFrame:
  """Holds the csi measurements for a frame
    
  Attributes:
    captured_at_ns: capture time in nanoseconds since session start
    csi: csi complex values
  """

  captured_at_ns: int
  csi: ValuesTensor


@dataclass
class SignalFrame:
  """Holds the amplitude and phase in separate lists
    
  Attributes:
    captured_at_ns: capture time in nanoseconds since session start
    amplitudes: amplitude of each tensor position
    phases: phase of each tensor position in radians
  """

  captured_at_ns: int
  amplitudes: list[Optional[float]]
  phases: list[Optional[float]]


@dataclass
class ProcessedFrame:
  """Holds the processed csi and extracted features
    
  Attributes:
    captured_at_ns: capture time in nanoseconds since session start
    processed_csi: processed amplitude and phase
    motion_energy: estimate of how much motion is in a frame
  """

  captured_at_ns: int
  processed_csi: ValuesTensor
  motion_energy: float
