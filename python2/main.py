from .frame_conversions import to_signal_frame
from .graphs import PipelineGraphs
from .simulation import SESSION, simulate_frame
from .structs import *

def main() -> None:
  grapher = PipelineGraphs(SESSION.layout)

  simulated_frames = [simulate_frame(i, 1, 0.03, 0.015) for i in range(10)]
  simulated_frames = [to_signal_frame(f) for f in simulated_frames]

  grapher.plot_stage(simulated_frames[0], 0, 0, "simulated")
  grapher.heatmap_stages(simulated_frames, 0, 0, "simulated")

if __name__ == "__main__":
  main()
