import { motion } from 'framer-motion';
import { useState } from 'react';

import { ApiError } from '../../api/client';
import { useHistory, useIceberg, useIcebergEvaluations, useIcebergForecast } from '../../hooks/queries';
import { useUi } from '../../stores/uiStore';
import { fmtDate, fmtDateTime, fmtKm, fmtNum } from '../../utils/format';
import { formatLat, formatLon, haversineKm } from '../../utils/projection';
import { KV, SectionTitle } from '../common/Metric';
import { QueryState } from '../common/QueryState';
import { ProvenanceChip, RunStatusChip, StaleChip } from '../common/StatusChip';
import { EnvironmentSection } from './EnvironmentSection';
import { TrackChart } from './TrackChart';

export function IcebergDetailPanel({ icebergId }: { icebergId: string }) {
  const select = useUi((s) => s.select);
  const horizon = useUi((s) => s.horizon);
  const detail = useIceberg(icebergId);
  const forecast = useIcebergForecast(icebergId);
  const evaluations = useIcebergEvaluations(icebergId);
  const history = useHistory(icebergId, true);
  const [showInputs, setShowInputs] = useState(false);

  const forecastMissing = forecast.isError && forecast.error instanceof ApiError && forecast.error.status === 404;

  return (
    <motion.aside
      className="detail-panel glass"
      initial={{ opacity: 0, x: 24 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 24 }}
      transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      aria-label={`Iceberg ${icebergId} details`}
    >
      <div className="detail-head">
        <div>
          <span className="label">Iceberg</span>
          <div className="detail-id mono">{icebergId}</div>
          {detail.data && (
            <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
              <RunStatusChip status={detail.data.status} />
              {detail.data.provenance && <ProvenanceChip provenance={detail.data.provenance} />}
              {detail.data.isStale && <StaleChip />}
            </div>
          )}
        </div>
        <button className="close-btn" onClick={() => select(null)} aria-label="Close details">
          ×
        </button>
      </div>

      <div className="detail-body">
        <QueryState query={detail} notFoundText="Iceberg not found.">
          {(d) => (
            <>
              <SectionTitle right={<ProvenanceChip provenance="official_usnic" />}>Latest official state</SectionTitle>
              <div className="kv-grid">
                <KV k="Latitude" v={formatLat(d.latitude, 3)} />
                <KV k="Longitude" v={formatLon(d.longitude, 3)} />
                <KV k="USNIC last update" v={`${fmtDate(d.lastUpdate)}${d.daysSinceUpdate != null ? ` (${d.daysSinceUpdate} d)` : ''}`} />
                <KV k="Source" v={d.source ?? '—'} />
                <KV k="Length × width" v={d.lengthNm != null ? `${fmtNum(d.lengthNm)} × ${fmtNum(d.widthNm)} nm` : '—'} />
                <KV k="Area" v={d.areaSqKm != null ? `${fmtNum(d.areaSqKm, 1)} km² · ${fmtNum(d.areaSqNm, 1)} nm²` : '—'} />
                <KV k="Official fixes" v={fmtNum(d.officialObservationCount)} />
                <KV k="Historical positions" v={fmtNum(d.historicalObservationCount)} />
              </div>
            </>
          )}
        </QueryState>

        <SectionTitle>Track</SectionTitle>
        <QueryState query={history} compact emptyText="No stored track." empty={(h) => h.items.length === 0}>
          {(h) => (
            <TrackChart
              history={h.items}
              forecast={forecast.data?.points.filter((p) => p.horizonDays <= horizon)}
              anchor={forecast.data?.anchor}
            />
          )}
        </QueryState>

        <SectionTitle right={<ProvenanceChip provenance="predicted" />}>Forecast D+1…D+7</SectionTitle>
        {forecastMissing ? (
          <div className="note">
            No forecast for this iceberg. The adapter needs 14 chronological entries (official USNIC + historical training
            positions); icebergs without enough history are skipped rather than forecast from invented positions.
          </div>
        ) : (
          <QueryState query={forecast} compact>
            {(f) => (
              <>
                <div className="kv-grid three" style={{ marginBottom: 10 }}>
                  <KV k="Model" v={`${f.modelVersion}${f.isChampion ? ' (champion)' : ''}`} />
                  <KV k="Issued" v={fmtDateTime(f.generatedAt)} />
                  <KV k="From official fix" v={fmtDate(f.latestObservationDate)} />
                  <KV k="Feature schema" v={f.featureSchemaVersion} />
                  <KV k="Environment as of" v={f.environmentAsOf ? fmtDate(f.environmentAsOf) : 'not used'} />
                  <KV
                    k="Env. sources"
                    v={f.anchorEnvironment ? Object.values(f.anchorEnvironment).map((e) => e.dataset_id).join(', ') : '—'}
                  />
                </div>
                {f.fallback && (
                  <div className="note warn" style={{ marginBottom: 10 }}>
                    Champion {f.fallback.champion} ({f.fallback.champion_schema}) could not be used: {f.fallback.reason} (
                    {f.fallback.missing_count} missing values). This forecast was produced by trajectory-only model{' '}
                    {f.fallback.used_model}.
                  </div>
                )}
                <table className="data">
                  <thead>
                    <tr>
                      <th>D+</th>
                      <th>DATE</th>
                      <th>POSITION</th>
                      <th className="num">MOVED</th>
                      <th className="num">P90 R</th>
                    </tr>
                  </thead>
                  <tbody>
                    {f.points.map((p) => (
                      <tr key={p.forecastId} style={{ opacity: p.horizonDays <= horizon ? 1 : 0.4 }}>
                        <td className="mono">{p.horizonDays}</td>
                        <td className="mono">{p.forecastDate}</td>
                        <td className="mono">
                          {formatLat(p.predictedLatitude)} {formatLon(p.predictedLongitude)}
                        </td>
                        <td className="mono num">{fmtKm(haversineKm(f.anchor.latitude, f.anchor.longitude, p.predictedLatitude, p.predictedLongitude))}</td>
                        <td className="mono num">{fmtKm(p.riskRadiusKmP90)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 10 }}>
                  <div className="note">
                    One GRU inference produced all seven days; the {horizon}-day view highlights D+1…D+{horizon}. Input:{' '}
                    {f.diagnostics.official_entries} official + {f.diagnostics.historical_entries} historical entries, gaps{' '}
                    {f.diagnostics.min_gap_days}–{f.diagnostics.max_gap_days} d (adapter {f.adapterStrategy}).
                  </div>
                  {!f.diagnostics.daily_cadence && (
                    <div className="note warn">
                      Input cadence is not daily. The base model was trained on consecutive daily positions; velocities are
                      computed per elapsed day, but accuracy under weekly cadence is established only by operational evaluation.
                    </div>
                  )}
                  {f.isStale && <div className="note warn">Anchor observation is stale; this forecast may no longer be current.</div>}
                  <button className="btn-ghost" style={{ alignSelf: 'flex-start' }} onClick={() => setShowInputs((v) => !v)}>
                    {showInputs ? 'Hide' : 'Show'} 14 input entries
                  </button>
                  {showInputs && f.inputEntries && (
                    <table className="data">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>DATE</th>
                          <th>POSITION</th>
                          <th className="num">Δt</th>
                          <th>SOURCE</th>
                        </tr>
                      </thead>
                      <tbody>
                        {f.inputEntries.map((e, i) => (
                          <tr key={i}>
                            <td className="mono dim">{i + 1}</td>
                            <td className="mono">{e.date}</td>
                            <td className="mono">
                              {formatLat(e.latitude)} {formatLon(e.longitude)}
                            </td>
                            <td className="mono num">{e.elapsedDays ? `${e.elapsedDays} d` : '—'}</td>
                            <td>
                              <ProvenanceChip provenance={e.provenance} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </>
            )}
          </QueryState>
        )}

        <EnvironmentSection icebergId={icebergId} />

        <SectionTitle>Prediction vs actual</SectionTitle>
        <QueryState
          query={evaluations}
          compact
          empty={(e) => e.length === 0}
          emptyText="No evaluations yet. Past forecasts are scored when a later official USNIC observation falls on a forecast date (typically D+7 for the weekly product)."
        >
          {(evs) => (
            <table className="data">
              <thead>
                <tr>
                  <th>DATE</th>
                  <th>D+</th>
                  <th>MODEL</th>
                  <th className="num">ERROR</th>
                </tr>
              </thead>
              <tbody>
                {evs.map((e) => (
                  <tr key={e.evaluationId}>
                    <td className="mono">{e.forecastDate}</td>
                    <td className="mono">{e.horizonDays}</td>
                    <td className="mono">{e.modelVersion}</td>
                    <td className="mono num">{fmtKm(e.errorKm, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </QueryState>
      </div>
    </motion.aside>
  );
}
