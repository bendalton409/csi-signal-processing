from math import pi, tau, remainder
from typing import Optional
from .structs import *
from .simulation import SESSION

TONE_COUNT: int = len(SESSION.layout.subcarrier_idxs)
LINK_COUNT: int = SESSION.layout.tx_count * SESSION.layout.rx_count
prev_frame: Optional[SignalFrame] = None


def unwrap_over_subcarriers(frame: SignalFrame) -> None:
  """Unwraps the phase over subcarriers in place for a frame"""

  for link in range(LINK_COUNT):
    start = link * TONE_COUNT
    prev_phase = None

    for idx in range(start, start+TONE_COUNT):
      if frame.phases[idx] is None:
        continue

      if prev_phase is not None:
        frame.phases[idx] = prev_phase + remainder(frame.phases[idx] - prev_phase, tau)

      prev_phase = frame.phases[idx]


def unwrap_over_frames(frame: SignalFrame) -> None:
  """Unwraps the phase over frames in place for a frame"""

  global prev_frame

  if prev_frame is None:
    prev_frame = frame
    return

  for link in range(LINK_COUNT):
    start = link * TONE_COUNT

    must_unwrap, shift = False, 0
    for idx in range(start, start + TONE_COUNT):
      phase, prev_phase = frame.phases[idx], prev_frame.phases[idx]

      if phase is None or prev_phase is None:
        continue

      delta = phase - prev_phase

      # IMPORTANT: assumes that any jump of phase over pi rad MUST mean wrapping
      # ie there will be no frame-to-frame jump of phase over pi rad without a wrapping

      if abs(delta) > pi:
        must_unwrap = True
        shift = remainder(delta, tau) - delta
        break

    if must_unwrap:
      for idx in range(start, start + TONE_COUNT):
        if frame.phases[idx] is not None:
          frame.phases[idx] += shift

  prev_frame = frame