"""Save raw and processed CSI plots for individual TX/RX links."""

from collections.abc import Sequence
from math import nan
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .csi_structs import CsiSession, SignalFrame


class CsiPlotter:
    """Plot a single tx-rx link after each stage of the processing pipeline.
    1. Simulated
    2. Phase unwrapping
    3. Linear phase removal
    4. Hampel filter
    5. Wavelet denoising
    """

    def __init__(self, session: CsiSession, output_dir: str = "imgs"):
        self.layout = session.layout
        self.output_dir = Path(output_dir)

    def plot_simulated(self, frame: SignalFrame, tx: int, rx: int) -> Path:
        """Save amplitude and phase after simulation."""

        return self._plot_pipeline_stage(frame, tx, rx, "1-simulated", "Simulated")

    def plot_unwrapped(self, frame: SignalFrame, tx: int, rx: int) -> Path:
        """Save amplitude and phase after unwrapping."""

        return self._plot_pipeline_stage(frame, tx, rx, "2-unwrapped", "Phase unwrapping")

    def plot_phase_residuals(self, frame: SignalFrame, tx: int, rx: int) -> Path:
        """Save amplitude and phase after linear phase removal."""

        return self._plot_pipeline_stage(frame, tx, rx, "3-phase-residuals", "Phase residuals")

    def plot_hampel(self, frame: SignalFrame, tx: int, rx: int) -> Path:
        """Save amplitude and phase after Hampel filtering."""

        return self._plot_pipeline_stage(frame, tx, rx, "4-hampel", "Hampel filtering")

    def plot_wavelet(self, frame: SignalFrame, tx: int, rx: int) -> Path:
        """Save amplitude and phase after wavelet denoising."""

        return self._plot_pipeline_stage(frame, tx, rx, "5-wavelet", "Wavelet denoising")

    def plot_subcarrier_unwrapping_heatmap(
        self, frames: Sequence[SignalFrame], tx: int, rx: int, *,
        phase_limits: Optional[tuple[float, float]] = None,
    ) -> Path:
        """Save a phase heatmap from frames unwrapped across subcarriers."""
        return self._plot_phase_heatmap(
            frames, tx, rx, "subcarrier-unwrapped", "Unwrapped across subcarriers", phase_limits
        )

    def plot_time_unwrapping_heatmap(
        self, frames: Sequence[SignalFrame], tx: int, rx: int, *,
        phase_limits: Optional[tuple[float, float]] = None,
    ) -> Path:
        """Save a phase heatmap after unwrapping across subcarriers and time."""
        return self._plot_phase_heatmap(
            frames, tx, rx, "time-unwrapped", "Unwrapped across subcarriers and time", phase_limits
        )

    def _plot_phase_heatmap(
        self, frames: Sequence[SignalFrame], tx: int, rx: int, stage: str, title: str,
        phase_limits: Optional[tuple[float, float]],
    ) -> Path:
        if not frames:
            raise ValueError("A phase heatmap requires at least one frame")
        link = self._link_slice(tx, rx)
        tones = self.layout.subcarrier_indices
        size = self.layout.tx_count * self.layout.rx_count * len(tones)
        if not tones or any(len(frame.phase_rad) != size for frame in frames):
            raise ValueError("Phase frames must match the session layout")

        phase = np.array([frame.phase_rad[link] for frame in frames], dtype=float).T
        cmap = colormaps["viridis"].with_extremes(bad="#dadde3")
        vmin, vmax = phase_limits if phase_limits is not None else (None, None)
        figure = Figure(figsize=(9, 4.8), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots()
        image = axes.imshow(
            np.ma.masked_invalid(phase), origin="lower", aspect="auto",
            interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
        )

        frame_ticks = range(0, len(frames), max(1, (len(frames) + 9) // 10))
        tone_ticks = range(0, len(tones), max(1, (len(tones) + 9) // 10))
        axes.set_xticks(frame_ticks, [f"{frames[i].captured_at_ns / 1e6:g}" for i in frame_ticks])
        axes.set_yticks(tone_ticks, [tones[i] for i in tone_ticks])
        axes.set_xlabel("Capture time (ms, one column per frame)")
        axes.set_ylabel("OFDM subcarrier index")
        figure.colorbar(image, ax=axes, label="Phase (rad)")

        time_span = f"{frames[0].captured_at_ns}ns-{frames[-1].captured_at_ns}ns"
        figure.suptitle(f"{title} · TX {tx}, RX {rx}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"csi_tx{tx}_rx{rx}_{time_span}_{stage}_heatmap.png"
        figure.savefig(path, dpi=250)
        return path

    def _plot_pipeline_stage(
        self, 
        frame: SignalFrame, 
        tx: int, 
        rx: int, 
        stage: str, 
        title: str
    ) -> Path:

        """Generates two graphs for a link in a frame: 
        1. Amplitude against tone index
        2. Phase against tone index.
        """    

        if len(frame.phase_rad) != len(frame.amplitude):
            raise ValueError("Amplitude and phase must be equal length")

        link = self._link_slice(tx, rx)

        figure = Figure(figsize=(8, 6), layout="constrained")
        FigureCanvasAgg(figure)

        amplitude_axes, phase_axes = figure.subplots(2, 1, sharex=True)
        phase_axes.set_xlabel("Subcarrier index")
        amp_plot = (amplitude_axes, frame.amplitude, "Amplitude")
        phase_plot = (phase_axes, frame.phase_rad, "Phase (rad)")

        for axes, samples, label in (amp_plot, phase_plot):
            values = [nan if sample is None else sample for sample in samples[link]]
            axes.plot(self.layout.subcarrier_indices, values, marker="o", markersize=4)
            axes.set_ylabel(label)
            axes.grid(alpha=0.25)

        return self._save(figure, stage, title, frame.captured_at_ns, tx, rx)

    def _link_slice(self, tx: int, rx: int) -> slice:
        "Extracts a single tx-rx link out of a CSI frame."

        if not 0 <= tx < self.layout.tx_count:
            raise ValueError(f"TX index {tx} is outside the session layout")

        if not 0 <= rx < self.layout.rx_count:
            raise ValueError(f"RX index {rx} is outside the session layout")

        tone_count = len(self.layout.subcarrier_indices)
        tx_start = tx * self.layout.rx_count * tone_count
        rx_start = rx * tone_count
        start = tx_start + rx_start
        return slice(start, start + tone_count)

    def _save(
        self, 
        figure: Figure, 
        stage: str, 
        title: str, 
        timestamp_ns: 
        int, 
        tx: int, 
        rx: int
    ) -> Path:
        
        figure.suptitle(f"{title} · TX {tx}, RX {rx} · {timestamp_ns:,} ns")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"csi_tx{tx}_rx{rx}_{timestamp_ns}ns_{stage}.png"
        figure.savefig(path, dpi=250)
        return path
