import { KV, Metric, SectionTitle } from '../../components/common/Metric';
import { EnvironmentPanel } from '../../components/operations/EnvironmentPanel';
import { ExperimentPanel } from '../../components/operations/ExperimentPanel';
import { QueryState } from '../../components/common/QueryState';
import { Chip, FeedStateChip, RunStatusChip } from '../../components/common/StatusChip';
import { useForecastRuns, useIngestion, useModels, useRetraining } from '../../hooks/queries';
import type { ModelVersion } from '../../types/api';
import { fmtDate, fmtDateTime, fmtKm, fmtNum, relTime, shortHash } from '../../utils/format';

function Lineage({ versions }: { versions: ModelVersion[] }) {
  const children = (parent: string | null) => versions.filter((v) => v.parentVersion === parent);
  const render = (v: ModelVersion, depth: number): React.ReactNode => (
    <div key={v.version}>
      <div className="lineage-node" style={{ paddingLeft: depth * 18 }}>
        <span className="dim">{depth ? '└─' : ''}</span>
        <span style={{ minWidth: 44 }}>{v.version}</span>
        <RunStatusChip status={v.status} />
        <span className="dim" title={v.modelType}>{v.featureSchemaVersion}</span>
        <span className="dim">D+7 {fmtKm(v.day7Error, 2)}</span>
        {v.statusReason && (
          <span
            className="dim"
            style={{
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              maxWidth: 420,
            }}
          >
            {v.statusReason}
          </span>
        )}
      </div>
      {children(v.version).map((c) => render(c, depth + 1))}
    </div>
  );
  return <div className="lineage">{children(null).map((r) => render(r, 0))}</div>;
}

export default function OperationsPage() {
  const ingestion = useIngestion();
  const forecastRuns = useForecastRuns();
  const models = useModels();
  const retraining = useRetraining();

  return (
    <div className="page-scroll">
      <div className="page-head">
        <h1>Operations</h1>
        <span className="mono dim" style={{ fontSize: 11 }}>
          polling every 30 s · jobs run in background workers
        </span>
      </div>

      <SectionTitle right={ingestion.data && <FeedStateChip state={ingestion.data.feedState} />}>
        USNIC ingestion
      </SectionTitle>
      <QueryState query={ingestion}>
        {(s) => (
          <>
            <div className="cards" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 140px), 1fr))' }}>
              <Metric
                label="Last fetch"
                value={relTime(s.latestRun?.fetchedAt)}
                sub={s.latestRun ? <RunStatusChip status={s.latestRun.status} /> : 'never'}
              />
              <Metric label="Last success" value={relTime(s.lastSuccessfulRun?.fetchedAt)} />
              <Metric label="Last obs. date" value={fmtDate(s.latestOfficialObservationDate)} tone="official" />
              <Metric label="Rows received" value={fmtNum(s.latestRun?.rowCount)} />
              <Metric label="New obs." value={fmtNum(s.latestRun?.newObservations)} tone="ok" />
              <Metric label="Duplicates" value={fmtNum(s.latestRun?.duplicateObservations)} />
              <Metric label="Corrections" value={fmtNum(s.latestRun?.updatedObservations)} />
              <Metric
                label="Failed rows"
                value={fmtNum(s.latestRun?.failedRows)}
                tone={s.latestRun?.failedRows ? 'bad' : 'default'}
              />
            </div>
            {s.stateReasons.length > 0 && (
              <div className="note warn" style={{ marginTop: 10 }}>
                {s.stateReasons.join(' · ')}
              </div>
            )}
            <div className="panel" style={{ marginTop: 12 }}>
              <div className="table-wrap" style={{ maxHeight: 300 }}>
                <table className="data">
                  <thead>
                    <tr>
                      <th>RUN</th>
                      <th>FETCHED</th>
                      <th>STATUS</th>
                      <th className="num">ROWS</th>
                      <th className="num">NEW</th>
                      <th className="num">DUP</th>
                      <th className="num">FAIL</th>
                      <th className="num">MS</th>
                      <th>SOURCE UPDATE</th>
                      <th>SHA-256</th>
                      <th>ERROR</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.recentRuns.map((r) => (
                      <tr key={r.id}>
                        <td className="mono dim">#{r.id}</td>
                        <td className="mono">{fmtDateTime(r.fetchedAt)}</td>
                        <td>
                          <RunStatusChip status={r.status} />
                        </td>
                        <td className="mono num">{r.rowCount}</td>
                        <td className="mono num">{r.newObservations}</td>
                        <td className="mono num">{r.duplicateObservations}</td>
                        <td className="mono num">{r.failedRows}</td>
                        <td className="mono num dim">{r.durationMs ?? '—'}</td>
                        <td className="mono">{fmtDate(r.sourceLatestUpdate)}</td>
                        <td className="mono dim" title={r.checksumSha256 ?? ''}>
                          {shortHash(r.checksumSha256)}
                        </td>
                        <td
                          className="mono"
                          style={{
                            color: 'var(--bad)',
                            maxWidth: 280,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                          }}
                          title={r.errorMessage ?? ''}
                        >
                          {r.errorMessage ?? ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            {s.latestRowErrors.length > 0 && (
              <>
                <SectionTitle>Rejected rows (latest run)</SectionTitle>
                <div className="panel pad">
                  {s.latestRowErrors.map((e) => (
                    <div key={e.rowNumber} className="mono" style={{ fontSize: 11, marginBottom: 6 }}>
                      <span className="dim">row {e.rowNumber}</span> {Object.values(e.rawRow).join(', ')} —{' '}
                      <span style={{ color: 'var(--bad)' }}>{e.errors.join('; ')}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </>
        )}
      </QueryState>

      <SectionTitle>Forecast generation</SectionTitle>
      <QueryState query={forecastRuns} empty={(r) => r.length === 0} emptyText="No forecast runs yet.">
        {(runs) => (
          <div className="panel">
            <div className="table-wrap" style={{ maxHeight: 320 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>RUN</th>
                    <th>STARTED</th>
                    <th>MODEL</th>
                    <th>TRIGGER</th>
                    <th>STATUS</th>
                    <th className="num">CONSIDERED</th>
                    <th className="num">NEW SETS</th>
                    <th className="num">ALREADY</th>
                    <th>SKIPPED</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.slice(0, 10).map((r) => (
                    <tr key={r.id}>
                      <td className="mono dim">#{r.id}</td>
                      <td className="mono">{fmtDateTime(r.startedAt)}</td>
                      <td className="mono">{r.modelVersion ?? '—'}</td>
                      <td className="mono dim">{r.trigger}</td>
                      <td>
                        <RunStatusChip status={r.status} />
                      </td>
                      <td className="mono num">{r.icebergsConsidered}</td>
                      <td className="mono num">{r.forecastSetsCreated}</td>
                      <td className="mono num">{r.alreadyForecast}</td>
                      <td className="mono dim">
                        {r.errorMessage ??
                          (Object.entries(r.skipped ?? {})
                            .map(([k, v]) => `${k.replace(/_/g, ' ')}: ${v}`)
                            .join(' · ') ||
                            '—')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </QueryState>

      <SectionTitle>Model registry & lineage</SectionTitle>
      <QueryState
        query={models}
        empty={(m) => m.length === 0}
        emptyText="No model versions registered. Place the base artifact in models/base and run make bootstrap."
      >
        {(versions) => (
          <div className="cards">
            <div className="panel card">
              <h3>Lineage</h3>
              <span className="dim mono" style={{ fontSize: 11 }}>
                base → v1 → … → vN (rejected branches kept)
              </span>
              <div style={{ marginTop: 12 }}>
                <Lineage versions={versions} />
              </div>
            </div>
            {versions
              .filter((v) => v.status === 'deployed')
              .map((v) => (
                <div key={v.version} className="panel card">
                  <h3>
                    Champion {v.version} <Chip tone="ok">DEPLOYED</Chip>
                  </h3>
                  <div className="kv-grid" style={{ marginTop: 12 }}>
                    <KV k="Architecture" v={v.architectureVersion} />
                    <KV k="Feature schema" v={`${v.featureSchemaVersion} (${v.modelType})`} />
                    <KV k="Env. data cutoff" v={fmtDate(v.environmentalDataCutoff)} />
                    <KV k="Input semantics" v={v.inputSemantics} />
                    <KV k="Adapter" v={v.adapterStrategy ?? '—'} />
                    <KV k="Origin" v={v.artifactOrigin} />
                    <KV k="Data cutoff" v={fmtDate(v.trainingDataCutoff)} />
                    <KV k="Deployed" v={fmtDateTime(v.deployedAt)} />
                    <KV
                      k="D+1 / D+3 / D+7"
                      v={`${fmtKm(v.day1Error, 2)} / ${fmtKm(v.day3Error, 2)} / ${fmtKm(v.day7Error, 2)}`}
                    />
                    <KV k="model.keras sha256" v={shortHash(Object.values(v.artifactSha256)[0])} />
                  </div>
                </div>
              ))}
          </div>
        )}
      </QueryState>

      <SectionTitle
        right={retraining.data?.latestRun && <RunStatusChip status={retraining.data.latestRun.status} />}
      >
        Retraining
      </SectionTitle>
      <QueryState query={retraining}>
        {(r) => (
          <div className="cards">
            <div className="panel card">
              <h3>
                Eligibility{' '}
                {r.eligibility?.eligible ? <Chip tone="ok">ELIGIBLE</Chip> : <Chip tone="neutral">WAITING</Chip>}
              </h3>
              <span className="dim mono" style={{ fontSize: 11 }}>
                since {fmtDateTime(r.eligibility?.since)}
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 12 }}>
                {Object.entries(r.eligibility?.checks ?? {}).map(([k, c]) => (
                  <div
                    key={k}
                    className="mono"
                    style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}
                  >
                    <span className="muted">{k.replace(/_/g, ' ')}</span>
                    <span style={{ color: c.met ? 'var(--ok)' : 'var(--text-3)' }}>
                      {String(c.value ?? '—')}
                      {c.threshold != null ? ` / ${c.threshold}` : ''} {c.met ? '✓' : '·'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="panel card">
              <h3>Promotion policy</h3>
              <div className="kv-grid" style={{ marginTop: 12 }}>
                {Object.entries(r.policy.promotion ?? {}).map(([k, v]) => (
                  <KV key={k} k={k.replace(/_/g, ' ')} v={String(Array.isArray(v) ? v.join(', ') : v)} />
                ))}
              </div>
            </div>
            <div className="panel card" style={{ gridColumn: '1 / -1' }}>
              <h3>Recent runs</h3>
              {r.recentRuns.length === 0 ? (
                <div className="qs-empty" style={{ marginTop: 8 }}>
                  No retraining runs yet.
                </div>
              ) : (
                <div className="table-wrap" style={{ maxHeight: 300 }}>
                  <table className="data" style={{ marginTop: 8 }}>
                    <thead>
                      <tr>
                        <th>STARTED</th>
                        <th>STATUS</th>
                        <th>CHAMPION</th>
                        <th>CANDIDATE</th>
                        <th className="num">SAMPLES</th>
                        <th>DECISION / REASON</th>
                      </tr>
                    </thead>
                    <tbody>
                      {r.recentRuns.map((run) => (
                        <tr key={run.id}>
                          <td className="mono">{fmtDateTime(run.startedAt)}</td>
                          <td>
                            <RunStatusChip status={run.status} />
                          </td>
                          <td className="mono">{run.championVersion ?? '—'}</td>
                          <td className="mono">{run.candidateVersion ?? '—'}</td>
                          <td className="mono num">{fmtNum(run.sampleCount)}</td>
                          <td className="mono dim" style={{ whiteSpace: 'normal' }}>
                            {run.decision?.reasons.join('; ') ?? run.failureReason ?? '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            {r.latestRun && <ExperimentPanel run={r.latestRun} />}
          </div>
        )}
      </QueryState>

      <EnvironmentPanel />
    </div>
  );
}
