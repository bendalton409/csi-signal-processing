"""Unwrap phase across subcarriers and successive frames (issue #15)."""

from math import pi, remainder, tau
from typing import Optional

from .csi_structs import CsiSession, SignalFrame


class PhaseUnwrapper:
    """Unwrap one session in capture order, with independent state per link.

    Unwrapping assumes less than pi radians of true change between consecutive
    available samples on each axis. A missing sample clears that tone's time
    reference.
    """

    def __init__(self, session: CsiSession):
        self._layout = session.layout
        self._previous: Optional[SignalFrame] = None
        self._tone_count = len(self._layout.subcarrier_indices)
        self._link_count = self._layout.tx_count * self._layout.rx_count

    def unwrap_across_subcarriers(self, frame: SignalFrame) -> SignalFrame:
        """Return a frame with unwrapped phase across subcarriers and the original amplitude."""

        self._validate_frame(frame)

        unwrapped = []

        for link in range(self._link_count):
            start = link * self._tone_count
            end = start + self._tone_count
            link_phase = []
            previous_tone = None

            for angle in frame.phase_rad[start:end]:
                if angle is None:
                    link_phase.append(None)
                    continue

                if previous_tone is not None:
                    angle = previous_tone + remainder(angle - previous_tone, tau)

                link_phase.append(angle)
                previous_tone = angle

            unwrapped.extend(link_phase)

        return SignalFrame(frame.captured_at_ns, frame.amplitude, tuple(unwrapped))

    def unwrap_across_frames(self, frame: SignalFrame) -> SignalFrame:
        """Return a frame with unwrapped phase across time and the original amplitude."""

        self._validate_frame(frame)
        if self._previous is None:
            self._previous = frame
            return frame

        unwrapped = []

        for link in range(self._link_count):
            start = link * self._tone_count
            end = start + self._tone_count
            link_phase = frame.phase_rad[start:end]

            for idx, prev_angle in enumerate(self._previous.phase_rad[start:end]):
                angle = link_phase[idx]

                if angle is not None and prev_angle is not None:
                    # Align the whole link to an overlapping tone so returning
                    # tones use the same phase branch as their neighbours.
                    delta = angle - prev_angle


                    if abs(delta) > pi:
                        shift = remainder(delta, tau) - delta
                        link_phase = tuple(
                            value if value is None else value + shift for value in link_phase
                        )
                    break

            unwrapped.extend(link_phase)

        result = SignalFrame(frame.captured_at_ns, frame.amplitude, tuple(unwrapped))
        self._previous = result
        return result

    def _validate_frame(self, frame: SignalFrame) -> None:
        size = self._link_count * self._tone_count
        if len(frame.phase_rad) != size or len(frame.amplitude) != size:
            raise ValueError(f"Expected {size} CSI slots for this session")

    def process(self, frame: SignalFrame) -> SignalFrame:
        """Return a frame with unwrapped phase across subcarriers and time and the original amplitude."""

        frame = self.unwrap_across_subcarriers(frame)
        frame = self.unwrap_across_frames(frame)
        return frame
