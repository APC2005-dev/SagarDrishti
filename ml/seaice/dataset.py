"""Build training/evaluation samples from stored sea-ice entries.

THE WINDOW RULE: a sample's input is the LAST 7 CHRONOLOGICAL DATABASE ENTRIES
ending at the anchor — seven stored rows, whatever dates they carry. It is not
"the previous 7 calendar days", and a gap in the official source never causes a
day to be invented to pad the window.

Targets, by contrast, are defined in *days*: horizon h means the official entry
dated ``anchor_date + h days``. A sample is built only when every required
target actually exists; otherwise it is skipped, never interpolated. On a
gapless daily series this is identical to the notebook's array-offset indexing,
and the ``span_days`` diagnostic records when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ml.seaice.constants import HORIZONS, WINDOW
from ml.seaice.preprocessing import build_input_tensor, first_target_date


@dataclass(frozen=True)
class SeaIceSample:
    anchor_date: date
    entry_dates: tuple[date, ...]
    x: np.ndarray  # (IN_CHANNELS, H, W)
    y: np.ndarray  # (len(HORIZONS), H, W)
    mask: np.ndarray  # (len(HORIZONS), H, W)

    @property
    def span_days(self) -> int:
        """Calendar span of the 7 entries. Equals WINDOW-1 only on a daily cadence."""
        return (self.entry_dates[-1] - self.entry_dates[0]).days

    @property
    def daily_cadence(self) -> bool:
        return self.span_days == WINDOW - 1


@dataclass
class SampleBuildReport:
    built: int = 0
    skipped_incomplete_window: int = 0
    skipped_missing_target: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "built": self.built,
            "skipped_incomplete_window": self.skipped_incomplete_window,
            "skipped_missing_target": self.skipped_missing_target,
        }


def build_samples(
    dates: list[date],
    concentration: np.ndarray,
    mask: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
    window: int = WINDOW,
) -> tuple[list[SeaIceSample], SampleBuildReport]:
    """``dates`` ascending, ``concentration``/``mask`` shaped ``(len(dates), H, W)``."""
    index = {d: i for i, d in enumerate(dates)}
    report = SampleBuildReport()
    samples: list[SeaIceSample] = []
    for position in range(len(dates)):
        if position + 1 < window:
            report.skipped_incomplete_window += 1
            continue
        window_slice = slice(position + 1 - window, position + 1)
        entry_dates = tuple(dates[window_slice])
        anchor = entry_dates[-1]
        target_positions = [index.get(anchor + timedelta(days=h)) for h in horizons]
        if any(p is None for p in target_positions):
            report.skipped_missing_target += 1
            continue
        samples.append(
            SeaIceSample(
                anchor_date=anchor,
                entry_dates=entry_dates,
                x=build_input_tensor(
                    concentration[window_slice], mask[window_slice], first_target_date(anchor)
                ),
                y=np.stack([concentration[p] for p in target_positions]).astype(np.float32),
                mask=np.stack([mask[p] for p in target_positions]).astype(np.float32),
            )
        )
        report.built += 1
    return samples, report


def chronological_split(
    samples: list[SeaIceSample], validation_fraction: float
) -> tuple[list[SeaIceSample], list[SeaIceSample]]:
    """Split by time, never at random: the tail becomes validation.

    A random split would leak future ice state into training through
    overlapping windows.
    """
    if not samples:
        return [], []
    ordered = sorted(samples, key=lambda s: s.anchor_date)
    cut = max(1, int(round(len(ordered) * (1.0 - validation_fraction)))) if len(ordered) > 1 else 1
    return ordered[:cut], ordered[cut:]
