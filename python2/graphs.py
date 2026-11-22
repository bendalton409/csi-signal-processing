from math import nan
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .structs import CsiLayout, SignalFrame

class Grapher:
  """Plots graphs for a single tx-rx link in a frame at each stage of the pipeline"""

  layout: CsiLayout
  output_dir: Path

  def __init__(self, layout: CsiLayout, output_dir: str = "imgs"):
    self.layout = layout
    self.output_dir = Path(output_dir)

  def _link_slice(self, tx: int, rx: int) -> slice:
    """Extracts a tx-rx link out of a frame"""

    if not 0 <= tx < self.layout.tx_count:
      raise ValueError(f"TX index {tx} is outside the session layout")
    if not 0 <= rx < self.layout.rx_count:
      raise ValueError(f"RX index {rx} is outside the session layout")

    tone_count = len(self.layout.subcarrier_idxs)
    start = (tx * self.layout.rx_count + rx) * tone_count
    return slice(start, start + tone_count)

  def plot_stage(self, frame: SignalFrame, tx: int, rx: int, stage: str, title: str) -> None:
    """Generates two graphs on a single figure: amplitude/tone and phase/tone"""

    link = self._link_slice(tx, rx)
    figure = Figure(figsize=(6, 8), layout="constrained")
    FigureCanvasAgg(figure)
    amplitude_axes, phase_axes = figure.subplots(2, 1, sharex=True)
    phase_axes.set_xlabel("Subcarrier index")
    amplitude_plot = (amplitude_axes, frame.amplitudes[link], "Amplitude")
    phase_plot = (phase_axes, frame.phases[link], "Phase (rad)")

    for axes, samples, label in (amplitude_plot, phase_plot):
      values = [nan if v is None else v for v in samples]  
      axes.plot(self.layout.subcarrier_idxs, values, marker="o", markersize=4)
      axes.set_ylabel(label)
      axes.grid(alpha=0.25)

    figure.suptitle(f"{title} · TX {tx}, RX {rx} · {frame.captured_at_ns:,} ns")
    self.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(self.output_dir / f"csi_tx{tx}_rx{rx}_{frame.captured_at_ns}ns_{stage}.png", dpi=250)

  def heatmap_stages(self, frames: list[SignalFrame], tx: int, rx: int, stage: str, title: str) -> None:
    """Saves stacked amplitude and phase heatmaps with optional phase color limits."""

    link = self._link_slice(tx, rx)
    tones = self.layout.subcarrier_idxs
    amplitude = np.array([frame.amplitudes[link] for frame in frames], dtype=float).T
    phase = np.array([frame.phases[link] for frame in frames], dtype=float).T
    cmap = colormaps["viridis"].with_extremes(bad="#dadde3")
    limits = (None, None)
    figure = Figure(figsize=(9, 8), layout="constrained")
    FigureCanvasAgg(figure)
    amplitude_axes, phase_axes = figure.subplots(2, 1, sharex=True)
    frame_ticks = range(0, len(frames), max(1, (len(frames) + 9) // 10))
    tone_ticks = range(0, len(tones), max(1, (len(tones) + 9) // 10))
    amplitude_plot = (amplitude_axes, amplitude, "Amplitude", (None, None))
    phase_plot = (phase_axes, phase, "Phase (rad)", limits)

    for axes, values, label, (vmin, vmax) in (amplitude_plot, phase_plot):
      image = axes.imshow(
        np.ma.masked_invalid(values), origin="lower", aspect="auto",
        interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
      )
      axes.set_yticks(tone_ticks, [tones[i] for i in tone_ticks])
      axes.set_ylabel("OFDM subcarrier index")
      figure.colorbar(image, ax=axes, label=label)

    phase_axes.set_xticks(frame_ticks, [f"{frames[i].captured_at_ns / 1e6:g}" for i in frame_ticks])
    phase_axes.set_xlabel("Capture time (ms, one column per frame)")
    time_span = f"{frames[0].captured_at_ns}ns-{frames[-1].captured_at_ns}ns"
    figure.suptitle(f"{title} · TX {tx}, RX {rx}")
    self.output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(self.output_dir / f"csi_tx{tx}_rx{rx}_{time_span}_{stage}_heatmap.png", dpi=250)


class PipelineGraphs(Grapher):
  """Graphs plots and heatmaps for each stage in the pipeline process"""

  STAGES = {
    "simulation": ("1-simulated", "Simulated"),
    "unwrap_subcarriers": ("2-unwrapped-subcarrier", "Phase unwrapping over subcarriers"),
    "unwrap_frames": ("3-unwrapped-frame", "Phase unwrapping over frames"),
    "phase_residuals": ("4-phase-residuals", "Phase residuals"),
    "hampel": ("5-hampel", "Hampel filtering"),
    "wavelet": ("6-wavelet", "Wavelet denoising"),
  }

  def plot_stage(self, frame: SignalFrame, tx: int, rx: int, stage: str) -> None:
    super().plot_stage(frame, tx, rx, *self.STAGES[stage])

  def heatmap_stages(self, frames: list[SignalFrame], tx: int, rx: int, stage: str) -> None:
    super().heatmap_stages(frames, tx, rx, *self.STAGES[stage])
