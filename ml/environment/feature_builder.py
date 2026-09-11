"""Leakage-safe environmental features for trajectory entries.

AS-OF RULE (no future information)
----------------------------------
A forecast is made at prediction time ``T`` (a UTC date). For an entry observed
on ``d`` (always ``d <= T``) and an environmental group whose *operational*
feed has latency ``L`` days, the value used is the field valid on::

    valid_date = min(d, T - L)

i.e. the newest field that the operational feed would already have published
at ``T``. Fields dated after ``T - L`` are never read — not for the last
entries, not for the forecast days. If ``d - valid_date`` exceeds
``max_staleness_days`` the value is treated as missing rather than silently
carried forward. The same rule is applied when building *training* samples
(``T`` = the sample's anchor date), even when the values come from a
reanalysis: the reanalysis only substitutes a better estimate of a past state
that the operational feed would already have delivered by ``T``.

Forecast-period environment (D+1…D+7) is **not** an input of the concatenated
architecture: no leakage-free archive of *issued* forecasts exists for the
historical training period. The platform archives the forecast fields it can
fetch at forecast time (``environmental.forecast_snapshots``) so a future
architecture can be trained on genuinely issued forecasts.

PROVIDER CHAINS
---------------
Each group can have several providers (e.g. reanalysis first, then the
operational feed). The first provider whose coverage contains ``valid_date``
is used, and its name is recorded with the value.

MISSING DATA
------------
Values are never fabricated. Missing values are NaN in the feature block and
the sample is marked incomplete; training excludes incomplete samples
(complete-case) and live forecasting falls back to a trajectory-only model for
that iceberg, recording the reason.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from numpy.typing import NDArray

from ml.environment.cache import EnvironmentalCache
from ml.environment.providers.base import EnvironmentalProvider, ProviderUnavailableError
from ml.environment.sampler import INTERPOLATION, sample_point
from ml.environment.types import GROUP_VARIABLES, EnvGroup, EnvSample
from ml.features.schemas import ENV_VARIABLES


@dataclass
class EnvFeatureBlock:
    variables: tuple[str, ...]
    values: NDArray[np.float32]  # (n_entries, n_variables), NaN = missing
    samples: list[dict[EnvGroup, EnvSample]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(np.isfinite(self.values).all())

    def missing_by_variable(self) -> dict[str, int]:
        return {v: int((~np.isfinite(self.values[:, i])).sum()) for i, v in enumerate(self.variables)}

    def missing_reasons(self) -> list[str]:
        out: list[str] = []
        for i, entry in enumerate(self.samples):
            for g, s in entry.items():
                if s.missing:
                    out.append(f"entry {i + 1} {g.value}: {s.reason}")
        return out

    def provenance(self) -> list[dict[str, dict[str, object]]]:
        return [{g.value: s.as_dict() for g, s in entry.items()} for entry in self.samples]

    def newest_valid_date(self) -> date | None:
        dates = [s.valid_date for entry in self.samples for s in entry.values() if s.valid_date and not s.missing]
        return max(dates) if dates else None


ProviderChain = EnvironmentalProvider | Sequence[EnvironmentalProvider]


class EnvironmentalFeatureBuilder:
    def __init__(
        self,
        providers: Mapping[EnvGroup, ProviderChain],
        cache: EnvironmentalCache,
        as_of_latency_days: Mapping[EnvGroup, float] | None = None,
        max_staleness_days: int = 3,
    ) -> None:
        self.chains: dict[EnvGroup, list[EnvironmentalProvider]] = {
            g: ([p] if isinstance(p, EnvironmentalProvider) else list(p)) for g, p in providers.items()
        }
        self.cache = cache
        self.latency = {
            g: float(as_of_latency_days[g]) if as_of_latency_days and g in as_of_latency_days else chain[-1].spec.latency_days
            for g, chain in self.chains.items()
        }
        self.max_staleness_days = max_staleness_days
        self._memo: dict[tuple[str, EnvGroup, float, float, date], tuple[dict[str, float | None], int, str | None, str | None]] = {}

    @property
    def providers(self) -> dict[EnvGroup, EnvironmentalProvider]:
        """Primary provider per group (first in chain) — used for reporting."""
        return {g: chain[0] for g, chain in self.chains.items()}

    def valid_date_for(self, group: EnvGroup, entry_date: date, as_of: date) -> date:
        return min(entry_date, as_of - timedelta(days=math.ceil(self.latency[group])))

    def _provider_for(self, group: EnvGroup, day: date) -> EnvironmentalProvider | None:
        return next((p for p in self.chains.get(group, []) if p.covers(day)), None)

    def sample(self, group: EnvGroup, lat: float, lon: float, entry_date: date, as_of: date) -> EnvSample:
        if entry_date > as_of:
            raise ValueError(f"entry {entry_date} is after prediction time {as_of}: would leak future information")
        variables = GROUP_VARIABLES[group]
        if group not in self.chains:
            return self._missing(group, variables, entry_date, as_of, None, "no provider configured", "-", "-")
        valid = self.valid_date_for(group, entry_date, as_of)
        if (entry_date - valid).days > self.max_staleness_days:
            p = self.chains[group][-1]
            return self._missing(group, variables, entry_date, as_of, valid, "latest available field too stale", p.spec.name, p.spec.dataset_id)
        provider = self._provider_for(group, valid)
        if provider is None:
            p = self.chains[group][0]
            return self._missing(group, variables, entry_date, as_of, valid, "date outside source coverage", p.spec.name, p.spec.dataset_id)
        spec = provider.spec

        key = (spec.name, group, round(lat, 4), round(lon, 4), valid)
        if key not in self._memo:
            try:
                ds, cache_id = self.cache.get(provider, lat, lon, valid)
                pv = sample_point(ds, variables, lat, lon, valid)
                self._memo[key] = (pv.values, pv.valid_neighbours, pv.reason, cache_id)
            except ProviderUnavailableError as exc:
                return self._missing(group, variables, entry_date, as_of, valid, f"provider unavailable: {exc}", spec.name, spec.dataset_id)
        values, n_valid, reason, cache_id = self._memo[key]
        return EnvSample(
            group=group, values=values, requested_date=entry_date, as_of=as_of, valid_date=valid,
            provider=spec.name, dataset_id=spec.dataset_id, interpolation=INTERPOLATION, valid_neighbours=n_valid,
            missing=any(v is None for v in values.values()), reason=reason, cache_key=cache_id,
            quality_flags=self._quality_flags(values, n_valid),
        )

    @staticmethod
    def _quality_flags(values: Mapping[str, float | None], n_valid: int) -> tuple[str, ...]:
        flags: list[str] = []
        if 0 < n_valid < 4:
            flags.append(f"partial_neighbourhood_{n_valid}_of_4")
        for name, v in values.items():
            lo, hi = ENV_VARIABLES[name].valid_range
            if v is not None and not lo <= v <= hi:
                flags.append(f"{name}_out_of_physical_range")
        return tuple(flags)

    @staticmethod
    def _missing(group, variables, entry_date, as_of, valid, reason, provider, dataset_id) -> EnvSample:  # type: ignore[no-untyped-def]
        return EnvSample(
            group=group, values={v: None for v in variables}, requested_date=entry_date, as_of=as_of, valid_date=valid,
            provider=provider, dataset_id=dataset_id, interpolation=INTERPOLATION, valid_neighbours=0,
            missing=True, reason=reason,
        )

    def build(
        self, entries: Sequence[tuple[float, float, date]], as_of: date, variables: Sequence[str]
    ) -> EnvFeatureBlock:
        """Feature block for chronological entries (lat, lon, date) as known at ``as_of``."""
        variables = tuple(variables)
        groups: list[EnvGroup] = []
        for v in variables:
            g = EnvGroup(ENV_VARIABLES[v].group)
            if g not in groups:
                groups.append(g)
        values = np.full((len(entries), len(variables)), np.nan, dtype=np.float32)
        samples: list[dict[EnvGroup, EnvSample]] = []
        for i, (lat, lon, d) in enumerate(entries):
            per_group = {g: self.sample(g, lat, lon, d, as_of) for g in groups}
            samples.append(per_group)
            for j, v in enumerate(variables):
                val = per_group[EnvGroup(ENV_VARIABLES[v].group)].values.get(v)
                if val is not None:
                    values[i, j] = val
        return EnvFeatureBlock(variables=variables, values=values, samples=samples)
