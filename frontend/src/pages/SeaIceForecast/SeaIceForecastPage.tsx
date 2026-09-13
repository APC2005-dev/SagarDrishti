import { motion } from 'framer-motion';
import { useState } from 'react';

import { Metric, SectionTitle } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { Chip } from '../../components/common/StatusChip';
import { ConcentrationMap } from '../../components/seaice/ConcentrationMap';
import { useSeaIceField, useSeaIceStatus } from '../../hooks/queries';
import type { SeaIceStatus } from '../../types/api';
import { fmtDate, fmtDateTime } from '../../utils/format';

/**
 * Sea ice is a CORE model family, independent of the iceberg trajectory model:
 * its own base artifact (U-Net Residual v4, PyTorch), its own official feed
 * (Copernicus Marine / OSI SAF), its own champion and its own version lineage.
 *
 * Everything on this screen comes from the backend. When a model, the data
 * window or a forecast is genuinely unavailable the backend says why and that
 * reason is shown — this page never renders simulated ice.
 */

const toneFor = (status: string) => (status === 'deployed' ? 'ok' : status === 'rejected' ? 'bad' : 'neutral');

function ModelCard({ status }: { status: SeaIceStatus }) {
  const model = status.model;
  if (!model) {
    return <div className="placeholder-card mono">{status.modelUnavailableReason ?? 'No sea-ice model deployed.'}</div>;
  }
  return (
    <div className="panel card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 10 }}>
        <h3 className="mono" style={{ margin: 0 }}>{model.version}</h3>
        <Chip tone={toneFor(model.status)}>{model.status.toUpperCase()}</Chip>
      </div>
      <div className="seaice-model-metrics" style={{ marginTop: 12 }}>
        <Metric label="Architecture" value={model.architecture} sub={model.architectureVersion} />
        <Metric label="Input window" value={`${model.inputWindowEntries} entries`} />
      </div>
      <div className="seaice-model-rmse" style={{ marginTop: 10 }}>
        <Metric label="Day-1 RMSE" value={model.day1Rmse?.toFixed(5) ?? '—'} />
        <Metric label="Day-3 RMSE" value={model.day3Rmse?.toFixed(5) ?? '—'} />
        <Metric label="Day-7 RMSE" value={model.day7Rmse?.toFixed(5) ?? '—'} />
      </div>
      {model.statusReason && (
        <div className="dim" style={{ fontSize: 11, marginTop: 10 }}>{model.statusReason}</div>
      )}
    </div>
  );
}

export default function SeaIceForecastPage() {
  const status = useSeaIceStatus();
  const [horizon, setHorizon] = useState<number | null>(7);
  const model = status.data?.model ?? null;
  const horizons = model?.forecastHorizonsDays ?? [1, 3, 7];
  const canRenderField = !!status.data && (horizon === null ? status.data.observationCount > 0 : status.data.latestForecasts.length > 0);
  const field = useSeaIceField(horizon, canRenderField);

  return (
    <div className="page-scroll">
      <QueryState query={status}>
        {(s) => (
          <>
            <div className="page-head">
              <div>
                <h1>Sea-ice forecast</h1>
                <span className="dim mono" style={{ fontSize: 11 }}>
                  {s.authority} · {s.datasetId}
                </span>
              </div>
              <Chip tone={s.sourceConfigured ? 'ok' : 'neutral'}>
                {s.sourceConfigured ? 'SOURCE CONFIGURED' : 'NO SOURCE CONFIGURED'}
              </Chip>
            </div>

            {!s.sourceConfigured && <div className="note">{s.sourceReason}</div>}

            <div className="seaice-layout">
              <div>
                <SectionTitle>Deployed model</SectionTitle>
                <ModelCard status={s} />

                <SectionTitle>Official data window</SectionTitle>
                <div className="panel card">
                  <div className="seaice-window-metrics">
                    <Metric label="Stored observations" value={s.observationCount.toLocaleString()} />
                    <Metric label="Latest official" value={s.latestObservation ? fmtDate(s.latestObservation.observationDate) : '—'} />
                    <Metric label="Window" value={`${s.window.length}/${s.windowEntriesRequired} entries`} />
                  </div>
                  <div className="dim" style={{ fontSize: 11, marginTop: 10 }}>
                    The input is the last {s.windowEntriesRequired} chronological database entries — not the previous{' '}
                    {s.windowEntriesRequired} calendar days. Gaps in the official record are preserved, never filled.
                  </div>
                  {s.window.length > 0 && (
                    <div className="mono" style={{ fontSize: 11, marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                      {s.window.map((w, i) => (
                        <span key={w.observationDate} className="dim" style={{ border: '1px solid var(--line)', borderRadius: 3, padding: '1px 6px' }}>
                          {i + 1}. {w.observationDate}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <SectionTitle>Latest forecast run</SectionTitle>
                {s.latestForecasts.length === 0 ? (
                  <div className="placeholder-card mono">
                    {s.forecastUnavailableReason ?? 'No sea-ice forecast available.'}
                  </div>
                ) : (
                  <div className="panel card">
                    <div className="table-wrap">
                      <table className="data" style={{ minWidth: '100%' }}>
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>Anchor</th>
                            <th>Horizon</th>
                            <th>Target</th>
                            <th className="num" style={{ textAlign: 'right', paddingRight: 12 }}>Mean SIC</th>
                          </tr>
                        </thead>
                        <tbody>
                          {s.latestForecasts.map((f) => (
                            <tr key={f.horizonDays}>
                              <td className="mono">{f.modelVersion}</td>
                              <td>{fmtDate(f.anchorDate)}</td>
                              <td>D+{f.horizonDays}</td>
                              <td>{fmtDate(f.targetDate)}</td>
                              <td className="mono num" style={{ textAlign: 'right', paddingRight: 12, fontWeight: 500 }}>
                                {f.meanConcentration != null ? `${(f.meanConcentration * 100).toFixed(1)}% (${f.meanConcentration.toFixed(4)})` : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {(() => {
                      const head = s.latestForecasts[0];
                      if (!head) return null;
                      const dates = head.inputEntryDates;
                      return (
                        <div className="dim" style={{ fontSize: 11, marginTop: 8 }}>
                          Window {dates[0]} → {dates[dates.length - 1]} · span {head.inputSpanDays} d ·{' '}
                          {head.dailyCadence ? 'daily cadence' : 'irregular cadence'}
                        </div>
                      );
                    })()}
                  </div>
                )}

                {s.recentEvaluations.length > 0 && (
                  <>
                    <SectionTitle>Evaluations vs official data</SectionTitle>
                    <div className="panel card">
                      <table className="data">
                        <thead>
                          <tr><th>Model</th><th>Horizon</th><th>Target</th><th>RMSE</th><th>Persistence</th></tr>
                        </thead>
                        <tbody>
                          {s.recentEvaluations.map((e, i) => (
                            <tr key={`${e.modelVersion}-${e.horizonDays}-${e.targetDate}-${i}`}>
                              <td className="mono">{e.modelVersion}</td>
                              <td>D+{e.horizonDays}</td>
                              <td>{fmtDate(e.targetDate)}</td>
                              <td className="mono">{e.rmse.toFixed(5)}</td>
                              <td className="mono dim">{e.persistenceRmse?.toFixed(5) ?? '—'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>

              <div>
                <SectionTitle>Concentration field</SectionTitle>
                <div className="panel card">
                  <div className="segmented seaice-field-selector" role="radiogroup" aria-label="Sea-ice field" style={{ marginBottom: 12 }}>
                    {([null, ...horizons] as (number | null)[]).map((h) => (
                      <button
                        key={h ?? 'observed'}
                        role="radio"
                        aria-checked={horizon === h}
                        className={`seg${horizon === h ? ' active' : ''}`}
                        onClick={() => setHorizon(h)}
                      >
                        {horizon === h && (
                          <motion.span
                            layoutId="seaice-field-pill"
                            className="seg-pill"
                            transition={{ type: 'spring', stiffness: 420, damping: 36 }}
                          />
                        )}
                        <span className="seg-text mono">{h === null ? 'OBSERVED' : `D+${h}`}</span>
                      </button>
                    ))}
                  </div>
                  {!canRenderField ? (
                    <div className="placeholder-card mono">
                      {horizon === null
                        ? 'No official sea-ice observation stored yet.'
                        : (s.forecastUnavailableReason ?? 'No sea-ice forecast available.')}
                    </div>
                  ) : field.isError ? (
                    <div className="placeholder-card mono">
                      {(field.error as Error)?.message ?? 'Field unavailable.'}
                    </div>
                  ) : field.data ? (
                    <>
                      <ConcentrationMap field={field.data} />
                      <div className="dim" style={{ fontSize: 11, marginTop: 10, lineHeight: 1.7 }}>
                        {field.data.kind === 'forecast'
                          ? `Forecast · ${field.data.modelVersion} · anchor ${fmtDate(field.data.anchorDate)} · valid ${fmtDate(field.data.validDate)}`
                          : `Official observation · valid ${fmtDate(field.data.validDate)}`}
                        <br />
                        {field.data.resolutionDeg}° grid · cells below {Math.round(field.data.minConcentration * 100)}%
                        concentration and cells the source did not retrieve are not drawn.
                      </div>
                    </>
                  ) : (
                    <div className="placeholder-card mono">Loading field…</div>
                  )}
                </div>

                <SectionTitle>Recent runs</SectionTitle>
                <div className="panel card">
                  {s.recentRuns.length === 0 ? (
                    <div className="dim mono" style={{ fontSize: 12 }}>No sea-ice runs yet.</div>
                  ) : (
                    <table className="data">
                      <thead>
                        <tr><th>Kind</th><th>Status</th><th>New</th><th>Started</th></tr>
                      </thead>
                      <tbody>
                        {s.recentRuns.map((r) => (
                          <tr key={r.id}>
                            <td>{r.kind}</td>
                            <td>
                              <Chip tone={r.status === 'failed' ? 'bad' : r.status === 'success' ? 'ok' : 'neutral'}>
                                {r.status.toUpperCase()}
                              </Chip>
                            </td>
                            <td className="mono">{r.entriesNew}</td>
                            <td className="dim">{fmtDateTime(r.startedAt)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>
          </>
        )}
      </QueryState>
    </div>
  );
}
