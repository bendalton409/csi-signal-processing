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
        self._previous: Optional[list[Optional[float]]] = None

    def process(self, frame: SignalFrame) -> SignalFrame:
        """Return a frame with unwrapped phase and the original amplitude."""

        tone_count = len(self._layout.subcarrier_indices)
        link_count = self._layout.tx_count * self._layout.rx_count

        if len(frame.phase_rad) != len(frame.amplitude):
            raise ValueError(f"{frame} contains different amount of phase and amplitude numbers.")

        unwrapped = []
        
        for link in range(link_count):
            start = link * tone_count
            end = start + tone_count
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

            if self._previous is not None:
                for index, previous in enumerate(self._previous[start:end]):
                    angle = link_phase[index]
                    if angle is not None and previous is not None:
                        # Align the whole link to an overlapping tone so returning
                        # tones use the same phase branch as their neighbours.
                        delta = angle - previous
                        if abs(delta) > pi:
                            shift = remainder(delta, tau) - delta
                            link_phase = [
                                value if value is None else value + shift for value in link_phase
                            ]
                        break

            unwrapped.extend(link_phase)

        self._previous = unwrapped

        return SignalFrame(frame.captured_at_ns, frame.amplitude, tuple(unwrapped))
