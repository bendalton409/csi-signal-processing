"""Save raw and processed CSI plots for individual TX/RX links."""

from math import atan2, nan
from pathlib import Path

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
