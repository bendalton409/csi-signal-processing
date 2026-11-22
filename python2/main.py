from .frame_conversions import to_signal_frame
from .graphs import PipelineGraphs
from .phase_residual import remove_linear_offset
from .phase_unwrapper import unwrap_over_frames, unwrap_over_subcarriers
from .simulation import SESSION, simulate_frame
from .structs import *

def main() -> None:
  grapher = PipelineGraphs(SESSION.layout)
  frames = [to_signal_frame(simulate_frame(i, 1, 0.03, 0.015)) for i in range(10)]
  stages = (
    ("unwrap_subcarriers", unwrap_over_subcarriers), 
    ("unwrap_frames", unwrap_over_frames),
    ("phase_residuals", remove_linear_offset),
  )

  grapher.plot_stage(frames[0], 0, 0, "simulation")
  grapher.heatmap_stages(frames, 0, 0, "simulation")

  for name, func in stages:
    for frame in frames:
      func(frame)
    grapher.plot_stage(frames[0], 0, 0, name)
    grapher.heatmap_stages(frames, 0, 0, name)


if __name__ == "__main__":
  main()
