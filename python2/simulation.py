from math import cos, sin, remainder, tau
from random import Random
from .structs import *

SESSION = CsiSession(
  layout = CsiLayout(tx_count=1, rx_count=1, subcarrier_idxs=tuple(range(-100,101))),
  center_frequency_hz=5_180_000_000,
  channel_bandwidth_hz=20_000_000,
  subcarrier_spacing_hz=78_125
)

def simulate_frame(frame_idx: int, amplitude: float, noise_std: float, missing_prob: float) -> CsiFrame:
  """Simulates a single frame of captured CSI data using made up formulae"""

  values = []
  rng = Random(frame_idx)
  captured_at_ns = frame_idx * 10_000

  for tx in range(SESSION.layout.tx_count):
    for rx in range(SESSION.layout.rx_count):
      for tone in SESSION.layout.subcarrier_idxs:

        phase = 1 * frame_idx # phase drift upward over time
        phase += 0.15 * tone # phase drift upward over tone
        phase += 0.2 * (tx - rx) # phase shift depending on link
        phase = remainder(phase, tau) # principal argument

        re = amplitude * cos(phase) + rng.gauss(0, noise_std)
        im = amplitude * sin(phase) + rng.gauss(0, noise_std)
        missing = rng.uniform(0, 1) < missing_prob
        values.append(None if missing else Complex(re=re, im=im))

  return CsiFrame(captured_at_ns=captured_at_ns, csi=ValuesTensor(values=values))
