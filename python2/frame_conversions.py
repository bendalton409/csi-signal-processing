from .structs import *

def to_signal_frame(frame: CsiFrame) -> SignalFrame:
  """Convert a csi frame to a signal frame for internal processing"""

  amplitudes = []
  phases = []

  for complex in frame.csi.values:
    amplitudes.append(None if complex is None else complex.magnitude())
    phases.append(None if complex is None else complex.argument())

  return SignalFrame(captured_at_ns=frame.captured_at_ns, amplitudes=amplitudes, phases=phases)