from collections import deque
from copy import deepcopy
from math import copysign, sqrt
from typing import Optional

from .simulation import SESSION
from .structs import SignalFrame

TONE_COUNT: int = len(SESSION.layout.subcarrier_idxs)
LINK_COUNT: int = SESSION.layout.tx_count * SESSION.layout.rx_count
LEVELS: int = 2
AMPLITUDE_THRESHOLD: float = 0.1
PHASE_THRESHOLD: float = 0.1
HAAR_COEFF = 1 / sqrt(2)

history: deque[SignalFrame] = deque(maxlen=2**LEVELS)


def _denoise_tone(window: list[Optional[float]], threshold: float) -> Optional[float]:
  """Return the denoised tone."""

  if window[-1] is None:
    return None

  start = len(window) - 1
  while start >=0 and window[start] is not None:
    start -= 1
  start += 1

  win_end_details = []
  win = window[start:]

  for _ in range(LEVELS):
    if len(win) < 2:
      break

    if len(win) % 2 == 1:
      win.append(win[-1])

    detail = (win[-2] - win[-1]) * HAAR_COEFF
    detail = copysign(max(0.0, abs(detail) - threshold), detail)
    win_end_details.append(detail)

    win = [(even + odd) * HAAR_COEFF for even, odd in zip(win[::2], win[1::2])]

  reconstructed = win[-1]
  for detail in reversed(win_end_details):
    reconstructed = (reconstructed - detail) * HAAR_COEFF
  return reconstructed


def wavelet_denoise(frame: SignalFrame) -> None:
  """Applies a Haar DWT to the current frame."""

  history.append(deepcopy(frame))

  if len(history) < history.maxlen:
    return

  for tone in range(LINK_COUNT * TONE_COUNT):
    amplitude_history = [sample.amplitudes[tone] for sample in history]
    phase_history = [sample.phases[tone] for sample in history]
    frame.amplitudes[tone] = _denoise_tone(amplitude_history, AMPLITUDE_THRESHOLD)
    frame.phases[tone] = _denoise_tone(phase_history, PHASE_THRESHOLD)
