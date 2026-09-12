"""Official Copernicus Marine source for the sea-ice model.

Reads the **same** product the base model was trained on
(``osisaf_obs-si_glo_phy-sic-south_nrt_amsr2_l4_P1D-m``, EUMETSAT OSI SAF AMSR2
L4 daily sea-ice concentration, Southern Hemisphere). It is deliberately not
interchangeable with the CMEMS *model* ``siconc`` fields used by the
environmental feature pipeline, nor with ERA5: a different product would change
the grid and the statistics the weights were fitted to.

Access goes through the toolbox's lazy ARCO reader, so only the requested days
are transferred. Credentials come from settings/environment and are never
written to disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

import numpy as np

from ml.seaice.constants import AUTHORITY, DATASET_ID, VARIABLE


class SeaIceSourceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceEntry:
    """One official daily field as it exists at the source."""

    observation_date: date
    source_time: datetime | None
    data_array: Any  # xarray.DataArray for a single time step


class DatasetOpener(Protocol):
    def __call__(self, dataset_id: str) -> Any: ...


def _default_opener(dataset_id: str, username: str | None, password: str | None) -> Any:
    try:
        import copernicusmarine
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SeaIceSourceUnavailableError("copernicusmarine package not installed") from exc
    try:
        # Passed explicitly: the toolbox otherwise falls back to an interactive
        # prompt, which would hang a scheduled job.
        return copernicusmarine.open_dataset(dataset_id=dataset_id, username=username, password=password)
    except Exception as exc:  # network / auth / service errors from the toolbox
        raise SeaIceSourceUnavailableError(f"{dataset_id}: {type(exc).__name__}: {exc}") from exc


class SeaIceSource:
    def __init__(
        self,
        dataset_id: str = DATASET_ID,
        username: str | None = None,
        password: str | None = None,
        opener: DatasetOpener | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self._username = username
        self._password = password
        self._opener = opener

    def is_configured(self) -> tuple[bool, str]:
        if self._opener is not None:
            return True, "ok"
        try:
            import copernicusmarine  # noqa: F401
        except ImportError:
            return False, "copernicusmarine package not installed"
        if not (self._username and self._password):
            return False, "COPERNICUS_MARINE_USERNAME / COPERNICUS_MARINE_PASSWORD not set"
        return True, "ok"

    @contextmanager
    def open(self) -> Iterator[Any]:
        ok, reason = self.is_configured()
        if not ok:
            raise SeaIceSourceUnavailableError(reason)
        if self._opener is not None:
            yield self._opener(self.dataset_id)
            return
        yield _default_opener(self.dataset_id, self._username, self._password)

    @staticmethod
    def available_dates(dataset: Any) -> list[date]:
        """Every observation date the source currently offers, ascending."""
        return [_as_date(t) for t in np.asarray(dataset["time"].values)]

    @staticmethod
    def iter_entries(dataset: Any, days: list[date], chunk_size: int = 20) -> Iterator[SourceEntry]:
        """Yield the requested days, fetching contiguous runs in single reads.

        Requesting one day at a time costs a full round trip per day (~16 s
        against the live service); reading a contiguous time slice amortises
        that over the whole chunk (~0.65 s/day measured), which is what makes a
        multi-year historical backfill practical. Days are yielded in
        chronological order and the caller still stores them one at a time.
        """
        times = np.asarray(dataset["time"].values)
        position: dict[date, int] = {}
        for index, stamp in enumerate(times):
            position.setdefault(_as_date(stamp), index)
        wanted = sorted((position[day], day) for day in days if day in position)
        variable = dataset[VARIABLE]
        for group in _contiguous_groups(wanted, chunk_size):
            low, high = group[0][0], group[-1][0]
            block = variable.isel(time=slice(low, high + 1)).load()
            for index, day in group:
                yield SourceEntry(
                    observation_date=day,
                    source_time=_as_datetime(times[index]),
                    data_array=block.isel(time=index - low),
                )

    @property
    def authority(self) -> str:
        return AUTHORITY


def _as_date(value: Any) -> date:
    return np.datetime64(value, "D").astype("datetime64[D]").item()


def _as_datetime(value: Any) -> datetime | None:
    try:
        stamp = np.datetime64(value, "s").astype("datetime64[s]").item()
        return stamp if isinstance(stamp, datetime) else None
    except (ValueError, TypeError):
        return None


def _contiguous_groups(
    wanted: list[tuple[int, date]], chunk_size: int
) -> Iterator[list[tuple[int, date]]]:
    """Split ``(time_index, date)`` pairs into runs of consecutive indices.

    A gap (a day already stored, or absent from the source) ends the run, so a
    slice never drags in time steps that were not asked for.
    """
    group: list[tuple[int, date]] = []
    for item in wanted:
        if group and (item[0] != group[-1][0] + 1 or len(group) >= chunk_size):
            yield group
            group = []
        group.append(item)
    if group:
        yield group
