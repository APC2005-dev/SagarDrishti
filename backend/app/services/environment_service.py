"""Backend side of the environmental pipeline.

* builds providers from configuration (``ENV_*_PROVIDER``) and credentials
* indexes every cached tile × month subset in ``environmental.cache_entries``
  and logs fetches in ``environmental.ingestion_runs``
* aligns environmental state to official observations (``observation_features``)
* keeps coarse overlay grids for the map
* archives forecast fields *as issued* at forecast time (``forecast_snapshots``)

Nothing here is used unless an environmental model version exists or is being
trained; the base model and trajectory versions never call it.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.environment import EnvCacheEntry, EnvIngestionRun, ForecastEnvironmentSnapshot, ObservationEnvironment
from app.models.tracking import Iceberg, Observation
from app.services.geo import ewkt_point
from ml.environment.cache import CacheRecord, EnvironmentalCache
from ml.environment.feature_builder import EnvironmentalFeatureBuilder
from ml.environment.providers.base import EnvironmentalProvider, ProviderUnavailableError
from ml.environment.providers.registry import Credentials, build_role
from ml.environment.sampler import sample_point
from ml.environment.types import GROUP_VARIABLES, BBox, EnvGroup, ProviderRole
from ml.features.schemas import FeatureSchema

log = get_logger(__name__)
OVERLAY_BBOX = BBox(-80.0, -45.0, -180.0, 180.0)


def _polygon_ewkt(b: dict[str, float]) -> str:
    x0, x1, y0, y1 = b["lon_min"], b["lon_max"], b["lat_min"], b["lat_max"]
    return f"SRID=4326;POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


@dataclass(frozen=True)
class GroupStatus:
    group: str
    role: str
    provider: str | None
    spec: dict[str, Any] | None
    configured: bool
    reason: str


class EnvironmentService:
    def __init__(
        self,
        session: Session,
        settings: Settings,
        providers: dict[ProviderRole, dict[EnvGroup, EnvironmentalProvider]] | None = None,
    ) -> None:
        self.session = session
        self.settings = settings
        self._providers = providers
        self._cache: EnvironmentalCache | None = None

    # ------------------------------------------------------------ configuration
    def credentials(self) -> Credentials:
        s = self.settings
        return Credentials(
            copernicus_marine_username=s.copernicus_marine_username.get_secret_value() if s.copernicus_marine_username else None,
            copernicus_marine_password=s.copernicus_marine_password.get_secret_value() if s.copernicus_marine_password else None,
            cds_api_key=s.cds_api_key.get_secret_value() if s.cds_api_key else None,
            cds_api_url=s.cds_api_url,
        )

    def families(self, role: ProviderRole) -> dict[EnvGroup, str]:
        s = self.settings
        if role == ProviderRole.OPERATIONAL:
            return {EnvGroup.WIND: s.env_wind_provider, EnvGroup.CURRENT: s.env_current_provider, EnvGroup.SEA_ICE: s.env_sea_ice_provider}
        return {EnvGroup.WIND: s.env_historical_wind_provider, EnvGroup.CURRENT: s.env_historical_current_provider,
                EnvGroup.SEA_ICE: s.env_historical_sea_ice_provider}

    def providers(self, role: ProviderRole) -> dict[EnvGroup, EnvironmentalProvider]:
        if self._providers is not None:
            return self._providers.get(role, {})
        return build_role(self.families(role), role, self.credentials())

    def configured_providers(self, role: ProviderRole) -> dict[EnvGroup, EnvironmentalProvider]:
        return {g: p for g, p in self.providers(role).items() if p.is_configured()[0]}

    def status(self) -> list[GroupStatus]:
        out: list[GroupStatus] = []
        for role in (ProviderRole.OPERATIONAL, ProviderRole.HISTORICAL):
            try:
                providers = self.providers(role)
            except ValueError as exc:
                out += [GroupStatus(g.value, role.value, None, None, False, str(exc)) for g in EnvGroup]
                continue
            for g in EnvGroup:
                p = providers.get(g)
                if p is None:
                    out.append(GroupStatus(g.value, role.value, None, None, False, "no provider configured"))
                    continue
                ok, reason = p.is_configured()
                ok = ok and self.settings.env_enabled
                out.append(GroupStatus(g.value, role.value, p.spec.name, p.spec.as_dict(), ok,
                                       reason if self.settings.env_enabled else "ENV_ENABLED=false"))
        return out

    def schema_trainable(self, schema: FeatureSchema) -> tuple[bool, str]:
        """Training needs a historical provider; evaluation/inference needs the operational one."""
        if not self.settings.env_enabled:
            return False, "ENV_ENABLED=false"
        hist = self.configured_providers(ProviderRole.HISTORICAL)
        oper = self.configured_providers(ProviderRole.OPERATIONAL)
        missing = [g for g in schema.groups if EnvGroup(g) not in hist or EnvGroup(g) not in oper]
        if missing:
            reasons = []
            for g in missing:
                for role, provs in ((ProviderRole.HISTORICAL, self.providers(ProviderRole.HISTORICAL)),
                                    (ProviderRole.OPERATIONAL, self.providers(ProviderRole.OPERATIONAL))):
                    p = provs.get(EnvGroup(g))
                    ok, why = p.is_configured() if p else (False, "no provider")
                    if not ok:
                        reasons.append(f"{g} {role.value}: {why}")
            return False, "; ".join(reasons)
        return True, "ok"

    # -------------------------------------------------------------------- cache
    def cache(self) -> EnvironmentalCache:
        if self._cache is None:
            self._cache = EnvironmentalCache(Path(self.settings.env_cache_dir), on_fetched=self._on_fetched)
        return self._cache

    def _group_of(self, provider_name: str) -> str:
        for role in (ProviderRole.OPERATIONAL, ProviderRole.HISTORICAL):
            for g, p in self.providers(role).items():
                if p.spec.name == provider_name:
                    return g.value
        return "unknown"

    def _on_fetched(self, record: CacheRecord) -> None:
        group = self._group_of(record.provider)
        values = dict(
            provider=record.provider, dataset_id=record.dataset_id, group=group, variables=record.variables,
            tile_key=record.tile_key, period_start=date.fromisoformat(record.period_start),
            period_end=date.fromisoformat(record.period_end), bbox=record.bbox, geom=_polygon_ewkt(record.bbox),
            spatial_resolution_deg=record.spatial_resolution_deg, file_path=record.path, sha256=record.sha256,
            bytes=record.bytes, max_valid_date=date.fromisoformat(record.max_valid_date) if record.max_valid_date else None,
            complete=record.complete, fetched_at=datetime.fromisoformat(record.fetched_at),
        )
        stmt = insert(EnvCacheEntry).values(**values)
        self.session.execute(stmt.on_conflict_do_update(
            constraint="uq_env_cache_entries_tile_period",
            set_={k: stmt.excluded[k] for k in ("file_path", "sha256", "bytes", "max_valid_date", "complete", "fetched_at")},
        ))
        self.session.add(EnvIngestionRun(
            kind="tile_fetch", group=group, provider=record.provider, dataset_id=record.dataset_id, status="success",
            request={"tile": record.tile_key, "bbox": record.bbox, "period": [record.period_start, record.period_end]},
            records=1, bytes=record.bytes, cache_path=record.path, completed_at=datetime.now(UTC),
        ))
        self.session.flush()
        log.info("environment_tile_cached", provider=record.provider, tile=record.tile_key, period=record.period_start,
                 bytes=record.bytes, complete=record.complete)

    # ------------------------------------------------------------------ builders
    def operational_builder(self) -> EnvironmentalFeatureBuilder:
        return EnvironmentalFeatureBuilder(self.providers(ProviderRole.OPERATIONAL), self.cache(),
                                           max_staleness_days=self.settings.env_max_staleness_days)

    def training_builder(self) -> EnvironmentalFeatureBuilder:
        """Historical source first, operational feed for dates after its coverage;
        as-of latency always from the operational feed (see feature_builder)."""
        hist = self.providers(ProviderRole.HISTORICAL)
        oper = self.providers(ProviderRole.OPERATIONAL)
        chains: dict[EnvGroup, list[EnvironmentalProvider]] = {}
        for g in EnvGroup:
            chain = [p for p in (hist.get(g), oper.get(g)) if p is not None]
            if chain:
                chains[g] = chain
        latency = {g: oper[g].spec.latency_days for g in oper}
        return EnvironmentalFeatureBuilder(chains, self.cache(), as_of_latency_days=latency,
                                           max_staleness_days=self.settings.env_max_staleness_days)

    def data_sources(self, groups: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        hist = self.providers(ProviderRole.HISTORICAL)
        oper = self.providers(ProviderRole.OPERATIONAL)
        return {
            g: {"historical": hist[EnvGroup(g)].spec.as_dict() if EnvGroup(g) in hist else None,
                "operational": oper[EnvGroup(g)].spec.as_dict() if EnvGroup(g) in oper else None}
            for g in groups
        }

    # ----------------------------------------------------------------- alignment
    def align_observations(self, observation_ids: list[int] | None = None, as_of: date | None = None) -> dict[str, Any]:
        """Environmental state at official observations, as available at ``as_of`` (default: today, UTC)."""
        as_of = as_of or datetime.now(UTC).date()
        started = time.monotonic()
        providers = self.configured_providers(ProviderRole.OPERATIONAL)
        run = EnvIngestionRun(kind="alignment", role="operational", status="running",
                              request={"observation_ids": observation_ids[:200] if observation_ids else None, "as_of": as_of.isoformat()})
        self.session.add(run)
        self.session.flush()
        if not self.settings.env_enabled or not providers:
            run.status, run.error_message = "skipped", ("ENV_ENABLED=false" if not self.settings.env_enabled else "no operational provider configured")
            run.completed_at = datetime.now(UTC)
            self.session.commit()
            return {"status": run.status, "aligned": 0, "reason": run.error_message}

        q = select(Observation).where(Observation.provenance == "official_usnic")
        if observation_ids:
            q = q.where(Observation.id.in_(observation_ids))
        else:
            latest = (
                select(Observation.id).where(Observation.provenance == "official_usnic")
                .join(Iceberg, Iceberg.iceberg_id == Observation.iceberg_id).where(Iceberg.status == "active")
                .distinct(Observation.iceberg_id).order_by(Observation.iceberg_id, Observation.observation_date.desc())
            )
            q = q.where(Observation.id.in_(latest))
        observations = list(self.session.execute(q).scalars())
        builder = EnvironmentalFeatureBuilder(providers, self.cache(), max_staleness_days=self.settings.env_max_staleness_days)
        stored, missing, reasons = 0, 0, []
        for obs in observations:
            if obs.observation_date > as_of:
                continue
            for g in providers:
                s = builder.sample(g, obs.latitude, obs.longitude, obs.observation_date, as_of)
                missing += int(s.missing)
                if s.missing and s.reason:
                    reasons.append(f"{obs.iceberg_id} {g.value}: {s.reason}")
                spec = providers[g].spec
                res = self.session.execute(insert(ObservationEnvironment).values(
                    observation_id=obs.id, iceberg_id=obs.iceberg_id, group=g.value, provider=s.provider, dataset_id=s.dataset_id,
                    values=s.values, units={v: spec.units.get(v, "") for v in GROUP_VARIABLES[g]}, observation_date=obs.observation_date,
                    as_of=as_of, valid_date=s.valid_date, staleness_days=s.staleness_days, latitude=obs.latitude,
                    longitude=obs.longitude, geom=ewkt_point(obs.latitude, obs.longitude), interpolation=s.interpolation,
                    valid_neighbours=s.valid_neighbours, missing=s.missing, reason=s.reason,
                    quality_flags=list(s.quality_flags), cache_path=s.cache_key,
                ).on_conflict_do_nothing(constraint="uq_env_observation_features").returning(ObservationEnvironment.id))
                stored += int(res.first() is not None)  # rowcount is unreliable for ON CONFLICT DO NOTHING
        run.records = stored
        run.status = "success" if missing == 0 else ("partial" if stored else "failed")
        run.error_message = "; ".join(reasons[:10]) or None
        run.duration_ms = int((time.monotonic() - started) * 1000)
        run.completed_at = datetime.now(UTC)
        self.session.commit()
        log.info("environment_alignment_completed", observations=len(observations), stored=stored, missing=missing, status=run.status)
        return {"status": run.status, "observations": len(observations), "aligned": stored, "missing": missing}

    # ---------------------------------------------------------- forecast archive
    def archive_forecast_fields(self, forecast_set_id: int, lat: float, lon: float, issued_on: date, lead_days: int = 7) -> int:
        """Store the forecast fields available *now* at the anchor location (lead 1..7)."""
        stored = 0
        for g, provider in self.configured_providers(ProviderRole.OPERATIONAL).items():
            if not provider.spec.supports_forecast:
                continue
            run = EnvIngestionRun(kind="forecast_snapshot", group=g.value, provider=provider.spec.name,
                                  dataset_id=provider.spec.dataset_id, role="operational", status="running",
                                  request={"forecast_set_id": forecast_set_id, "issued_on": issued_on.isoformat()})
            self.session.add(run)
            try:
                pad = 2 * provider.spec.spatial_resolution_deg
                ds = provider.fetch_forecast(BBox(lat - pad, lat + pad, max(-180.0, lon - pad), min(180.0, lon + pad)), issued_on, lead_days)
                for t in ds["time"].values.astype("datetime64[D]"):
                    valid = date.fromisoformat(str(t))
                    lead = (valid - issued_on).days
                    if not 1 <= lead <= lead_days:
                        continue
                    pv = sample_point(ds, GROUP_VARIABLES[g], lat, lon, valid)
                    self.session.execute(insert(ForecastEnvironmentSnapshot).values(
                        forecast_set_id=forecast_set_id, group=g.value, provider=provider.spec.name,
                        dataset_id=provider.spec.dataset_id, issued_on=issued_on, lead_day=lead, valid_date=valid,
                        latitude=lat, longitude=lon, location_basis="anchor", values=pv.values,
                    ).on_conflict_do_nothing(constraint="uq_env_forecast_snapshots"))
                    stored += 1
                run.status, run.records = "success", stored
            except ProviderUnavailableError as exc:
                run.status, run.error_message = "failed", str(exc)
            run.completed_at = datetime.now(UTC)
        self.session.flush()
        return stored

    # ------------------------------------------------------------------ overlay
    def overlay_dir(self, group: str) -> Path:
        return Path(self.settings.env_cache_dir) / "overlay" / group

    def refresh_overlay(self) -> dict[str, str]:
        """Coarse Southern Ocean field (latest available day) per configured operational group."""
        results: dict[str, str] = {}
        today = datetime.now(UTC).date()
        for g, provider in self.configured_providers(ProviderRole.OPERATIONAL).items():
            day = provider.last_available_date(today)
            target = self.overlay_dir(g.value) / f"{day.isoformat()}.nc"
            if target.exists():
                results[g.value] = f"cached {day}"
                continue
            run = EnvIngestionRun(kind="overlay", group=g.value, provider=provider.spec.name, dataset_id=provider.spec.dataset_id,
                                  role="operational", status="running", request={"day": day.isoformat(), "bbox": OVERLAY_BBOX.as_dict()})
            self.session.add(run)
            started = time.monotonic()
            try:
                ds = provider.fetch_region(OVERLAY_BBOX, day, day)
                k = max(1, round(self.settings.env_overlay_resolution_deg / provider.spec.spatial_resolution_deg))
                coarse = ds.coarsen(lat=k, lon=k, boundary="trim").mean(skipna=True)
                coarse.attrs.update({"provider": provider.spec.name, "dataset_id": provider.spec.dataset_id,
                                     "valid_date": day.isoformat(), "resolution_deg": str(self.settings.env_overlay_resolution_deg)})
                target.parent.mkdir(parents=True, exist_ok=True)
                coarse.to_netcdf(target)
                run.status, run.cache_path, run.bytes = "success", str(target), target.stat().st_size
                results[g.value] = f"refreshed {day}"
            except ProviderUnavailableError as exc:
                run.status, run.error_message = "failed", str(exc)
                results[g.value] = f"failed: {exc}"
            run.duration_ms = int((time.monotonic() - started) * 1000)
            run.completed_at = datetime.now(UTC)
        self.session.commit()
        return results

    def overlay_field(self, group: str, day: date | None = None, max_points: int = 6000) -> dict[str, Any] | None:
        import xarray as xr

        folder = self.overlay_dir(group)
        if not folder.exists():
            return None
        files = sorted(p for p in folder.glob("*.nc") if day is None or p.stem <= day.isoformat())
        if not files:
            return None
        with xr.open_dataset(files[-1]) as src:
            ds = src.load()
        variables = GROUP_VARIABLES[EnvGroup(group)]
        lat, lon = ds["lat"].values, ds["lon"].values
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        cols = [ds[v].values[0] for v in variables]
        ok = np.all([np.isfinite(c) for c in cols], axis=0)
        idx = np.argwhere(ok)
        step = max(1, len(idx) // max_points)
        points = [[round(float(LAT[i, j]), 3), round(float(LON[i, j]), 3), *[round(float(c[i, j]), 4) for c in cols]]
                  for i, j in idx[::step]]
        return {
            "group": group, "variables": list(variables), "valid_date": ds.attrs.get("valid_date", files[-1].stem),
            "provider": ds.attrs.get("provider"), "dataset_id": ds.attrs.get("dataset_id"),
            "resolution_deg": float(ds.attrs.get("resolution_deg", self.settings.env_overlay_resolution_deg)), "points": points,
        }

    # ------------------------------------------------------------------- queries
    def last_runs(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for g in [*[x.value for x in EnvGroup], None]:
            q = select(EnvIngestionRun).order_by(EnvIngestionRun.started_at.desc()).limit(1)
            q = q.where(EnvIngestionRun.group == g) if g else q.where(EnvIngestionRun.kind == "alignment")
            r = self.session.execute(q).scalar_one_or_none()
            ok = self.session.execute(
                (select(func.max(EnvIngestionRun.completed_at)).where(EnvIngestionRun.status.in_(("success", "partial"))))
                .where(EnvIngestionRun.group == g if g else EnvIngestionRun.kind == "alignment")
            ).scalar()
            out[g or "alignment"] = {
                "last_status": r.status if r else None, "last_run_at": r.started_at.isoformat() if r else None,
                "last_error": r.error_message if r else None, "last_success_at": ok.isoformat() if ok else None,
            }
        return out


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=str)
