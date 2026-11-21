# CSI signal processing

Python prototype for the Connected Care Wi-Fi 7 signal-processing pipeline. It takes synthetic channel state information (CSI), unwraps and corrects the phase, then filters and denoises the amplitude and phase. The demo runs without Wi-Fi hardware or a recorded capture.

## Setup and run

Run these commands from the repository root using Python 3.9. The pinned dependencies have been tested with Python 3.9.6. On macOS, you can use `/usr/bin/python3` in place of `python3.9` if `/usr/bin/python3 --version` reports Python 3.9.

```sh
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r REQUIREMENTS.md
python -m python.main
```

On Windows PowerShell, create the environment with `py -3.9 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Then run the same install and launch commands.

## Graphs

The demo saves PNGs in [`imgs/`](imgs/), which it creates on the first run. Each image shows linear amplitude and phase (radians) against the OFDM subcarrier index. Missing samples appear as gaps.

The pipeline processes all 10 fixture frames and plots the first frame for TX 0 / RX 0. Each filename includes the link, the capture time in nanoseconds, and the stage. For example, `imgs/csi_tx0_rx0_0ns_3-phase-residuals.png` shows the result after subtracting the fitted phase trend.

The five plots show the simulated input, unwrapped phase, phase residuals, Hampel-filtered signals, and wavelet-denoised signals.

Two additional PNGs ending in `_heatmap.png` compare phase after subcarrier unwrapping and after time unwrapping for TX 1 / RX 0. They show all 10 frames on the same phase colour scale. Grey cells mark missing samples.

## Project architecture

[`python/main.py`](python/main.py) runs the stages in this order.

| Stage | Module | What it does |
| --- | --- | --- |
| Simulate CSI | [`csi_fixture.py`](python/csi_fixture.py) | Generates 10 frames at 10 ms intervals. Each has two transmit streams, two receive chains, and 21 subcarrier slots per link. The fixture adds noise and occasional missing coefficients. |
| Convert to amplitude and phase | [`csi_conversion.py`](python/csi_conversion.py) | Converts each complex coefficient into linear amplitude and wrapped phase in radians. |
| Unwrap phase | [`phase_unwrapping.py`](python/phase_unwrapping.py) | Corrects phase wraps across subcarriers and successive frames for each link. |
| Compute phase residuals | [`phase_residual.py`](python/phase_residual.py) | Fits a straight line against the subcarrier indices for each link in each frame and subtracts it from the unwrapped phase. |
| Filter outliers | [`hampel_filter.py`](python/hampel_filter.py) | Uses a trailing median and median absolute deviation to filter outliers over time, separately for amplitude and phase at each tensor position. |
| Denoise | [`wavelet_denoising.py`](python/wavelet_denoising.py) | Applies a multilevel Haar wavelet transform to contiguous runs of amplitude and phase over time. |

[`csi_structs.py`](python/csi_structs.py) defines the shared session layout and frame types. A `CsiFrame` stores complex CSI and its capture timestamp. A `SignalFrame` stores amplitude and phase in separate tuples. Both use flat `[tx, rx, subcarrier]` ordering, with subcarrier varying fastest. [`csi_plotter.py`](python/csi_plotter.py) uses this layout to select a link and save its graphs.

## Tests

```sh
python -m unittest python.test_csi_dsp -v
```
