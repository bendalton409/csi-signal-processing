from collections import deque
from statistics import median
from typing import Optional
from copy import deepcopy
from .structs import SignalFrame
from .simulation import SESSION, simulate_frame
from .frame_conversions import to_signal_frame

TONE_COUNT: int = len(SESSION.layout.subcarrier_idxs)
LINK_COUNT: int = SESSION.layout.tx_count * SESSION.layout.rx_count
WINDOW_SIZE: int = 5
THRESHOLD: float = 3.0

history: deque[SignalFrame] = deque(maxlen=WINDOW_SIZE)


def _filter_tone(window: list[Optional[float]]) -> Optional[float]:
  current = window[-1]
  if current is None:
    return None

  valid = [sample for sample in window if sample is not None]
  if len(valid) < 3:
    return current

  med = median(valid)
  mad = median(abs(sample - med) for sample in valid)
  return med if abs(current - med) > THRESHOLD * 1.4826 * mad else current


def hampel_filter(frame: SignalFrame) -> None:
  """Filter amplitude and phase over time in place, preserving missing samples."""

  # deepcopy the current frame so that subsequent frames don't get filtered 
  # on the filtered version of the current frame

  history.append(deepcopy(frame))

  if len(history) == 1:
    return

  for tone in range(LINK_COUNT * TONE_COUNT):
    amplitude_history = [sample.amplitudes[tone] for sample in history]
    phase_history = [sample.phases[tone] for sample in history]
    frame.amplitudes[tone] = _filter_tone(amplitude_history)
    frame.phases[tone] = _filter_tone(phase_history)
