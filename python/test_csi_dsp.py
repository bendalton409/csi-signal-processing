"""Hardware-free checks for CSI preprocessing issues #15–#17."""

from math import cos, log10, sin
from unittest import TestCase

from .csi_conversion import to_signal_frame
from .csi_fixture import CSI_FRAMES, CSI_SESSION
from .csi_structs import Complex, CsiFrame, CsiLayout, CsiSession, CsiTensor, SignalFrame
from .hampel_filter import HampelFilter
from .phase_residual import remove_linear_offset
from .phase_unwrapping import PhaseUnwrapper
from .wavelet_denoising import denoise_frames, denoise_series


class PhaseTests(TestCase):
    def test_wrapped_ramp_and_affine_offset(self):
        tones = (-3, -1, 1, 3)
        residual = (0.2, -0.2, -0.2, 0.2)
        session = CsiSession(CsiLayout(1, 1, tones), 0, 0, 0, "synthetic")
        unwrap = PhaseUnwrapper(session)

        for index in range(12):
            expected = [
                2.8 + 0.4 * index + 0.25 * tone + value
                for tone, value in zip(tones, residual)
            ]
            frame = CsiFrame(
                index * 10_000_000,
                CsiTensor(tuple(Complex(cos(angle), sin(angle)) for angle in expected)),
            )
            unwrapped = unwrap.process(to_signal_frame(frame))
            corrected = remove_linear_offset(unwrapped, session)
            self.assertEqual(corrected.captured_at_ns, frame.captured_at_ns)
            self.assertEqual(corrected.amplitude, unwrapped.amplitude)
            for actual, target in zip(unwrapped.phase_rad, expected):
                self.assertAlmostEqual(actual, target, delta=1e-9)
            for actual, target in zip(corrected.phase_rad, residual):
                self.assertAlmostEqual(actual, target, delta=1e-9)


class HampelTests(TestCase):
    def test_stream_preserves_drift_and_removes_spikes_including_last_sample(self):
        filter_ = HampelFilter(window_size=5, threshold=3.0)
        samples = (0.0, 0.1, 0.2, 9.0, 0.4, 0.5, 0.6, 10.0)
        output = []
        for index, sample in enumerate(samples):
            frame = filter_.process(SignalFrame(index, (sample,), (-sample,)))
            self.assertEqual(frame.captured_at_ns, index)
            self.assertEqual(frame.phase_rad[0], -frame.amplitude[0])
            output.append(frame.amplitude[0])
        expected = (0.0, 0.1, 0.2, 0.15, 0.4, 0.5, 0.6, 0.6)
        self.assertEqual(len(output), len(expected))
        for actual, target in zip(output, expected):
            self.assertAlmostEqual(actual, target)


class WaveletTests(TestCase):
    def test_synthetic_snr_improves_by_at_least_six_db(self):
        clean = (1.0,) * 16 + (2.0,) * 16
        noisy = tuple(
            value + (0.18 if index % 2 == 0 else -0.18)
            + (0.08 if index % 4 < 2 else -0.08)
            for index, value in enumerate(clean)
        )
        denoised = denoise_series(noisy, levels=2, threshold=0.2)
        noisy_error = sum((actual - target) ** 2 for actual, target in zip(noisy, clean))
        denoised_error = sum((actual - target) ** 2 for actual, target in zip(denoised, clean))
        self.assertGreaterEqual(10 * log10(noisy_error / denoised_error), 6.0)

    def test_odd_run_lengths_and_missing_sample_keep_alignment(self):
        actual = denoise_series((1.0, 2.0, 3.0, None, 7.0), levels=1, threshold=1.0)
        self.assertEqual(len(actual), 5)
        for index, expected in ((0, 1.5), (1, 1.5), (2, 3.0), (4, 7.0)):
            self.assertAlmostEqual(actual[index], expected)
        self.assertIsNone(actual[3])

    def test_fixture_runs_through_all_stages_at_original_timestamps(self):
        phase = PhaseUnwrapper(CSI_SESSION)
        hampel = HampelFilter()
        filtered = tuple(
            hampel.process(remove_linear_offset(phase.process(to_signal_frame(frame)), CSI_SESSION))
            for frame in CSI_FRAMES
        )
        output = denoise_frames(filtered)
        self.assertEqual(
            [frame.captured_at_ns for frame in output],
            [frame.captured_at_ns for frame in CSI_FRAMES],
        )
        for result, source in zip(output, CSI_FRAMES):
            for samples in (result.amplitude, result.phase_rad):
                self.assertEqual(len(samples), len(source.csi.values))
                self.assertEqual(
                    [value is None for value in samples],
                    [value is None for value in source.csi.values],
                )
