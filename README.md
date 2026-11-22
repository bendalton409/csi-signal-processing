# CSI signal processing

Python prototype for the Connected Care Wi-Fi 7 signal-processing pipeline. The demo generates synthetic channel state information (CSI), unwraps and detrends the phase, and filters and denoises both amplitude and phase. It runs without Wi-Fi hardware or a recorded capture.

## Setup and run

Run these commands from the repository root using Python 3.9. The pinned dependencies have been tested with Python 3.9.6. On macOS, you can use `/usr/bin/python3` in place of `python3.9` if `/usr/bin/python3 --version` reports Python 3.9.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r REQUIREMENTS.md
python -m python.main
```

On Windows PowerShell, create the environment with `py -3.9 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Then run the same install and launch commands.

## Pipeline

[`python/main.py`](python/main.py) configures the demo to generate 100 frames at 10 µs intervals for one TX/RX link with 100 subcarriers. The simulator adds Gaussian noise to the complex samples and independently marks samples missing with 1.5% probability.

The demo runs each processing stage over the frames in capture order before saving that stage's plots. The processing functions take one `SignalFrame` per call and update it in place. Time unwrapping, Hampel filtering, and wavelet denoising retain state between calls.

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Simulated input | [`simulation.py`](python/simulation.py), [`frame_conversions.py`](python/frame_conversions.py) | Generates complex CSI and converts it to linear amplitude and wrapped phase in radians. |
| 2. Subcarrier unwrapping | [`phase_unwrapper.py`](python/phase_unwrapper.py) | Unwraps phase along the subcarrier axis within each frame and link. |
| 3. Time unwrapping | [`phase_unwrapper.py`](python/phase_unwrapper.py) | Uses the preceding frame to align each link's phase over time. |
| 4. Phase residuals | [`phase_residual.py`](python/phase_residual.py) | Fits a least-squares line to each link's available phase samples and subtracts it. |
| 5. Hampel filtering | [`hampel_filter.py`](python/hampel_filter.py) | Checks each tone's amplitude and phase separately for outliers using a trailing five-frame window. |
| 6. Wavelet denoising | [`wavelet_denoising.py`](python/wavelet_denoising.py) | Applies two levels of Haar soft thresholding to up to four consecutive samples per tone and reconstructs the current value. |

[`structs.py`](python/structs.py) defines the session layout and frame types. A `CsiFrame` stores timestamped complex CSI. A `SignalFrame` stores the timestamp and separate amplitude and phase lists. Both use flat `[tx, rx, subcarrier]` ordering, with subcarrier varying fastest. Missing samples are represented by `None`.

## Graphs

[`graphs.py`](python/graphs.py) creates `imgs/` and writes 18 PNGs by default, three per stage. Each image shows linear amplitude and phase in radians for TX 0 / RX 0.

| Filename suffix | What it shows |
| --- | --- |
| `_type=freq.png` | The final frame plotted across subcarriers. |
| `_type=time.png` | Subcarrier 0 plotted across all frames. |
| `_type=heatmap.png` | Stacked amplitude and phase heatmaps with one column per frame and one row per subcarrier. The shared x-axis shows capture time in milliseconds. |

For example, `imgs/5_hampel_type=heatmap.png` shows the result after Hampel filtering. Missing samples appear as gaps in the line plots and grey cells in the heatmaps. Each heatmap panel scales its own colours.

## Planned work

Live stream input, feature extraction, and `ProcessedFrame` output are tracked in [`TODO.md`](TODO.md).
