from .frame_conversions import to_signal_frame
from .graphs import plot_pipeline_stage
from .hampel_filter import hampel_filter
from .phase_residual import remove_linear_offset
from .phase_unwrapper import unwrap_over_frames, unwrap_over_subcarriers
from .simulation import simulate_frame
from .wavelet_denoising import wavelet_denoise

def main() -> None:
  frames = [to_signal_frame(simulate_frame(i, 1, 0.03, 0.015)) for i in range(10)]
  stages = (
    ("unwrap_subcarriers", unwrap_over_subcarriers), 
    ("unwrap_frames", unwrap_over_frames),
    ("phase_residuals", remove_linear_offset),
    ("hampel", hampel_filter),
    ("wavelet", wavelet_denoise)
  )

  plot_pipeline_stage(frames, 0, 0, 0, "simulation")

  for name, func in stages:
    for frame in frames:
      func(frame)
    plot_pipeline_stage(frames, 0, 0, 0, name)


if __name__ == "__main__":
  main()
