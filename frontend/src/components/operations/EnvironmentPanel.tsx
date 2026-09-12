import { useEnvironmentRuns, useEnvironmentStatus } from '../../hooks/queries';
import { fmtDate, fmtDateTime, fmtNum, relTime } from '../../utils/format';
import { SectionTitle } from '../common/Metric';
import { QueryState } from '../common/QueryState';
import { Chip, RunStatusChip } from '../common/StatusChip';

/** Environmental sources: which dataset feeds each role, whether it is configured, latency and coverage. */
export function EnvironmentPanel() {
  const status = useEnvironmentStatus();
  const runs = useEnvironmentRuns();
  return (
    <>
      <SectionTitle right={status.data && <Chip tone={status.data.enabled ? 'ok' : 'neutral'}>{status.data.enabled ? 'ENABLED' : 'DISABLED'}</Chip>}>
        Environmental sources
      </SectionTitle>
      <QueryState query={status}>
        {(s) => (
          <>
            <div className="panel">
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>GROUP</th>
                      <th>ROLE</th>
                      <th>DATASET</th>
                      <th>LEVEL</th>
                      <th className="num">RES °</th>
                      <th className="num">LATENCY</th>
                      <th>COVERAGE</th>
                      <th>FORECAST</th>
                      <th>STATE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.sources.map((src) => (
                      <tr key={`${src.group}-${src.role}`} title={src.notes ?? ''}>
                        <td className="mono">{src.group}</td>
                        <td className="mono dim">{src.role}</td>
                        <td className="mono" style={{ whiteSpace: 'normal' }}>
                          {src.datasetId ?? '—'}
                          <div className="dim" style={{ fontSize: 10 }}>{src.authority}</div>
                        </td>
                        <td className="mono dim" style={{ whiteSpace: 'normal', maxWidth: 180 }}>{src.depth ?? (src.group === 'wind' ? '10 m' : 'surface')}</td>
                        <td className="mono num">{src.spatialResolutionDeg != null ? src.spatialResolutionDeg.toFixed(3) : '—'}</td>
                        <td className="mono num">{src.latencyDays != null ? `${src.latencyDays} d` : '—'}</td>
                        <td className="mono dim">
                          {fmtDate(src.coverageStart)} → {src.coverageEnd ? fmtDate(src.coverageEnd) : 'ongoing'}
                        </td>
                        <td className="mono dim">{src.supportsForecast ? `+${src.forecastLeadDays} d` : '—'}</td>
                        <td>{src.configured ? <Chip tone="ok">CONFIGURED</Chip> : <Chip tone="neutral" title={src.reason}>NOT CONFIGURED</Chip>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

          </>
        )}
      </QueryState>
      <SectionTitle>Environmental runs</SectionTitle>
      <QueryState query={runs} empty={(r) => r.length === 0} emptyText="No environmental fetch, alignment or overlay run yet.">
        {(list) => (
          <div className="panel">
            <div className="table-wrap" style={{ maxHeight: 260 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>STARTED</th>
                    <th>KIND</th>
                    <th>GROUP</th>
                    <th>DATASET</th>
                    <th>STATUS</th>
                    <th className="num">RECORDS</th>
                    <th>ERROR</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((r) => (
                    <tr key={r.id}>
                      <td className="mono">{fmtDateTime(r.startedAt)}</td>
                      <td className="mono dim">{r.kind}</td>
                      <td className="mono">{r.group ?? '—'}</td>
                      <td className="mono dim">{r.datasetId ?? '—'}</td>
                      <td><RunStatusChip status={r.status} /></td>
                      <td className="mono num">{fmtNum(r.records)}</td>
                      <td className="mono" style={{ color: 'var(--bad)', maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }} title={r.errorMessage ?? ''}>
                        {r.errorMessage ?? ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </QueryState>
    </>
  );
}
