"""Save amplitude and phase plots for each stage of the CSI pipeline."""

from collections.abc import Sequence
from math import nan
from pathlib import Path
from typing import Optional, Union

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .simulation import SESSION
from .structs import SignalFrame

STAGES = {
  "simulation": ("1_simulated", "Simulated"),
  "unwrap_subcarriers": ("2_unwrapped-freq", "Phase unwrapping over subcarriers"),
  "unwrap_frames": ("3_unwrapped-time", "Phase unwrapping over frames"),
  "phase_residuals": ("4_residuals", "Phase residuals"),
  "hampel": ("5_hampel", "Hampel filtering"),
  "wavelet": ("6_wavelet", "Wavelet denoising"),
}
PANEL_LABELS = ("Amplitude", "Phase (rad)")


def _link_slice(tx: int, rx: int) -> slice:
  tone_count = len(SESSION.layout.subcarrier_idxs)
  start = (tx * SESSION.layout.rx_count + rx) * tone_count
  return slice(start, start + tone_count)


def _stacked_figure(width: int, xlabel: str) -> tuple[Figure, tuple[Axes, Axes]]:
  figure = Figure(figsize=(width, 8), layout="constrained")
  FigureCanvasAgg(figure)
  amplitude_axes, phase_axes = figure.subplots(2, 1, sharex=True)
  phase_axes.set_xlabel(xlabel)
  return figure, (amplitude_axes, phase_axes)


def _line_figure(
  x: Sequence[int],
  amplitudes: Sequence[Optional[float]],
  phases: Sequence[Optional[float]],
  xlabel: str,
) -> Figure:

  """Plots amplitude and phase against some x axis (time or frequency)."""
  
  figure, axes = _stacked_figure(8, xlabel)
  for panel, samples, label in zip(axes, (amplitudes, phases), PANEL_LABELS):
    values = [nan if sample is None else sample for sample in samples]
    panel.plot(x, values, marker="o", markersize=4)
    panel.set_ylabel(label)
    panel.grid(alpha=0.25)

  return figure


def _save_plot(
  figure: Figure,
  tx: int,
  rx: int,
  stage: str,
  plot_type: str,
  output_dir: Union[str, Path],
  *,
  title_detail: str = "",
) -> None:

  """Constructs the file name of a plot and saves it in output_dir."""
  
  stage_name, figure_title = STAGES[stage]

  heading = f"{figure_title} · TX {tx}, RX {rx}"
  if title_detail:
    heading += f" · {title_detail}"
  figure.suptitle(heading)

  directory = Path(output_dir)
  directory.mkdir(parents=True, exist_ok=True)
  filename = f"{stage_name}_{plot_type}.png"
  
  figure.savefig(directory / filename, dpi=250)


def _tick_positions(count: int) -> range:
  return range(0, count, max(1, (count + 9) // 10))


def plot_against_freq(
  frame: SignalFrame, 
  tx: int, 
  rx: int, 
  stage: str, 
  *, 
  output_dir: Union[str, Path] = "imgs",
) -> None:
  
  """Plot amplitude and phase across subcarriers for one frame."""

  link = _link_slice(tx, rx)
  figure = _line_figure(
    SESSION.layout.subcarrier_idxs,
    frame.amplitudes[link],
    frame.phases[link],
    "Subcarrier index",
  )
  _save_plot(
    figure, tx, rx, stage, "type=freq", output_dir,
    title_detail=f"{frame.captured_at_ns:,} ns",
  )


def plot_against_time(
  frames: Sequence[SignalFrame],
  tx: int,
  rx: int,
  tone: int,
  stage: str,
  *,
  output_dir: Union[str, Path] = "imgs",
) -> None:
  
  """Plot amplitude and phase across frames for one subcarrier position."""

  link = _link_slice(tx, rx)
  index = range(link.start, link.stop)[tone]
  subcarrier = SESSION.layout.subcarrier_idxs[tone]
  frequency = SESSION.center_frequency_hz + subcarrier * SESSION.subcarrier_spacing_hz
  figure = _line_figure(
    range(len(frames)),
    [frame.amplitudes[index] for frame in frames],
    [frame.phases[index] for frame in frames],
    "Frame",
  )
  _save_plot(
    figure, tx, rx, stage, "type=time", output_dir,
    title_detail=f"Tone Idx {tone}, Freq {frequency}",
  )


def heatmap_stages(
  frames: Sequence[SignalFrame], 
  tx: int, 
  rx: int, 
  stage: str,
  *,
  output_dir: Union[str, Path] = "imgs",
) -> None:
  
  """Plot stacked amplitude and phase heatmaps across frames and subcarriers."""

  link = _link_slice(tx, rx)
  tones = SESSION.layout.subcarrier_idxs
  amplitude = np.array([frame.amplitudes[link] for frame in frames], dtype=float).T
  phase = np.array([frame.phases[link] for frame in frames], dtype=float).T
  cmap = colormaps["viridis"].with_extremes(bad="#dadde3")

  figure, axes = _stacked_figure(9, "Capture time (ms, one column per frame)")
  tone_ticks = _tick_positions(len(tones))
  tone_labels = [tones[i] for i in tone_ticks]
  for panel, values, label in zip(axes, (amplitude, phase), PANEL_LABELS):
    image = panel.imshow(
      np.ma.masked_invalid(values), origin="lower", aspect="auto",
      interpolation="nearest", cmap=cmap,
    )
    panel.set_yticks(tone_ticks, tone_labels)
    panel.set_ylabel("OFDM subcarrier index")
    figure.colorbar(image, ax=panel, label=label)

  frame_ticks = _tick_positions(len(frames))
  frame_labels = [f"{frames[i].captured_at_ns / 1e6:g}" for i in frame_ticks]
  axes[1].set_xticks(frame_ticks, frame_labels)

  _save_plot(figure, tx, rx, stage, "type=heatmap", output_dir)


def plot_pipeline_stage(
  frames: Sequence[SignalFrame],
  tx: int,
  rx: int,
  tone: int,
  stage: str,
  *,
  output_dir: Union[str, Path] = "imgs",
) -> None:
  
  """Save the plots and heatmap for one processing stage."""

  plot_against_freq(frames[-1], tx, rx, stage, output_dir=output_dir)
  plot_against_time(frames, tx, rx, tone, stage, output_dir=output_dir)
  heatmap_stages(frames, tx, rx, stage, output_dir=output_dir)
