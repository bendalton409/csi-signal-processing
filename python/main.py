"""Run the fixture through CSI preprocessing and save plots under imgs/."""

from .csi_conversion import to_signal_frame
from .csi_fixture import CSI_FRAMES, CSI_SESSION
from .csi_plotter import CsiPlotter
from .hampel_filter import HampelFilter
from .phase_residual import remove_linear_offset
from .phase_unwrapping import PhaseUnwrapper
from .wavelet_denoising import denoise_frames


def main() -> None:
    unwrapper = PhaseUnwrapper(CSI_SESSION)
    hampel = HampelFilter()
    plotter = CsiPlotter(CSI_SESSION)

    signal_frames = tuple(to_signal_frame(frame) for frame in CSI_FRAMES)
    subcarrier_frames = tuple(unwrapper.unwrap_across_subcarriers(frame) for frame in signal_frames)
    unwrapped_frames = tuple(unwrapper.unwrap_across_frames(frame) for frame in subcarrier_frames)
    corrected_frames = tuple(remove_linear_offset(frame, CSI_SESSION) for frame in unwrapped_frames)
    filtered_frames = tuple(hampel.process(frame) for frame in corrected_frames)
    denoised_frames = denoise_frames(filtered_frames)

    phases = [
        phase
        for frame in subcarrier_frames + unwrapped_frames + corrected_frames
        for phase in frame.phase_rad
        if phase is not None
    ]
    phase_limits = (min(phases), max(phases))

    plotter.plot_subcarrier_unwrapping_heatmap(subcarrier_frames, tx=0, rx=0, phase_limits=phase_limits)
    plotter.plot_time_unwrapping_heatmap(unwrapped_frames, tx=0, rx=0, phase_limits=phase_limits)

    layout = CSI_SESSION.layout
    for signal, unwrapped, corrected, filtered, denoised in zip(
        signal_frames, unwrapped_frames, corrected_frames, filtered_frames, denoised_frames
    ):
        for tx in range(layout.tx_count):
            for rx in range(layout.rx_count):
                plotter.plot_simulated(signal, tx, rx)
                plotter.plot_unwrapped(unwrapped, tx, rx)
                plotter.plot_phase_residuals(corrected, tx, rx)
                plotter.plot_hampel(filtered, tx, rx)
                plotter.plot_wavelet(denoised, tx, rx)
                break
            break
        break


if __name__ == "__main__":
    main()
