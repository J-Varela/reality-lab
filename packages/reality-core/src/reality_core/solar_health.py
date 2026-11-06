"""Causal solar underproduction detection from interval telemetry."""

from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True)
class SolarAlarm:
    time_s: float
    residual_w: float
    evidence_duration_s: float


class SolarPowerMonitor:
    """Confirm a shortfall over consecutive observed intervals in sunlight.

    Readings summarize [start_s, end_s) and become available at end_s. The
    caller supplies an independent healthy prediction. Gaps, eclipse, and
    healthy readings reset pending evidence. The first alarm stays latched;
    update returns an event only when that alarm is first raised.
    """

    def __init__(
        self, *, threshold_w: float = 5.0, confirmation_s: float = 10.0
    ) -> None:
        if not isfinite(threshold_w) or threshold_w <= 0:
            raise ValueError("threshold_w must be finite and positive.")
        if not isfinite(confirmation_s) or confirmation_s <= 0:
            raise ValueError("confirmation_s must be finite and positive.")
        self.threshold_w = threshold_w
        self.confirmation_s = confirmation_s
        self._last_end_s: float | None = None
        self._evidence_s = 0.0
        self._alarm: SolarAlarm | None = None

    @property
    def alarm(self) -> SolarAlarm | None:
        return self._alarm

    def update(
        self, *, start_s: float, end_s: float, observed_w: float, expected_w: float
    ) -> SolarAlarm | None:
        """Consume one interval; invalid readings reset evidence and raise."""
        if (
            not all(isfinite(x) for x in (start_s, end_s, observed_w, expected_w))
            or start_s < 0
            or end_s <= start_s
            or expected_w < 0
        ):
            self._evidence_s = 0.0
            raise ValueError(
                "Telemetry must be finite with valid times and expected power."
            )

        if self._last_end_s is not None:
            adjacent = isclose(start_s, self._last_end_s, rel_tol=0, abs_tol=1e-9)
            if start_s < self._last_end_s and not adjacent:
                self._evidence_s = 0.0
                raise ValueError(
                    "Telemetry intervals must not overlap or arrive out of order."
                )
            if not adjacent:
                self._evidence_s = 0.0
        self._last_end_s = end_s

        residual_w = observed_w - expected_w
        if expected_w == 0 or residual_w >= -self.threshold_w:
            self._evidence_s = 0.0
            return None

        self._evidence_s += end_s - start_s
        if self._alarm is None and (
            self._evidence_s >= self.confirmation_s
            or isclose(self._evidence_s, self.confirmation_s, rel_tol=0, abs_tol=1e-9)
        ):
            self._alarm = SolarAlarm(end_s, residual_w, self._evidence_s)
            return self._alarm
        return None
