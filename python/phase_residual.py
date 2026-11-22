from typing import Optional
import numpy as np
from .structs import SignalFrame
from .simulation import SESSION

TONE_COUNT: int = len(SESSION.layout.subcarrier_idxs)
LINK_COUNT: int = SESSION.layout.tx_count * SESSION.layout.rx_count


def _least_squares_LOBF(frame: SignalFrame, link_start: int) -> Optional[tuple[float, float]]:
  """Return the slope and intercept for a link with at least two valid phases."""

  phases = frame.phases[link_start:link_start + TONE_COUNT]
  valid = np.array([phase is not None for phase in phases], dtype=bool)
  
  if valid.sum() < 2:
    return None

  xs = np.array(SESSION.layout.subcarrier_idxs)[valid]
  ys = np.array(phases, dtype=float)[valid]
  x_hat = xs.mean()
  y_hat = ys.mean()
  m = ((xs - x_hat) * (ys - y_hat)).sum() / ((xs - x_hat) ** 2).sum()
  b = y_hat - m * x_hat
  return m, b


def remove_linear_offset(frame: SignalFrame) -> None:
  """Subtract from phase a least squares LOBF per tx-rx link"""

  for link in range(LINK_COUNT):
    start = link * TONE_COUNT
    fit = _least_squares_LOBF(frame, start)

    if fit is None:
      continue

    m, b = fit

    for idx, x in enumerate(SESSION.layout.subcarrier_idxs, start=start):
      phase = frame.phases[idx]
      if phase is not None:
        frame.phases[idx] = phase - (m * x + b)
