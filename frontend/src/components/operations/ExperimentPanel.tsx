import type { AblationEffect, RetrainingRun } from '../../types/api';
import { fmtKm, fmtNum } from '../../utils/format';
import { Chip, RunStatusChip } from '../common/StatusChip';

const EFFECT_LABEL: Record<string, string> = {
  wind: 'Adding wind',
  ocean_current: 'Adding ocean current',
  wind_plus_current: 'Adding wind + current',
  sea_ice_given_wind_current: 'Adding sea ice (given wind + current)',
};

function verdictTone(v: string) {
  return v === 'improved' ? 'ok' : v === 'worse' ? 'bad' : 'neutral';
}

interface TestReport {
  protocol: string;
  samples: number;
  models: Record<string, { schema: string; by_horizon: Record<string, { mae_km: number | null; p90_km: number | null; n: number }> }>;
  effects: Record<string, AblationEffect>;
}

/** Feature-schema experiment of one retraining run: candidates, selection, and variable effects. */
export function ExperimentPanel({ run }: { run: RetrainingRun }) {
  const exp = run.experiment;
  const test = (run.metrics as { test?: TestReport } | null)?.test;
  if (!exp?.candidates?.length) {
    const unavailable = exp?.environmental_schemas_unavailable ?? exp?.unavailable;
    return unavailable && Object.keys(unavailable).length ? (
      <div className="note">
        Environmental schemas were not trained in this run:
        {Object.entries(unavailable).map(([k, v]) => (
          <div key={k} className="mono">
            {k}: {v}
          </div>
        ))}
      </div>
    ) : null;
  }
  const labels = test ? Object.keys(test.models) : [];
  return (
    <div className="panel card" style={{ gridColumn: '1 / -1' }}>
      <h3>
        Feature-schema experiment <span className="dim mono" style={{ fontSize: 11 }}>{exp.selection?.criterion}</span>
      </h3>
      <div className="table-wrap">
        <table className="data" style={{ marginTop: 8 }}>
          <thead>
            <tr>
              <th>VERSION</th>
              <th>SCHEMA</th>
              <th className="num">TRAIN</th>
              <th className="num">VAL D+7 MAE</th>
              <th>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {exp.candidates.map((c) => (
              <tr key={c.version} className={c.schema === exp.selection?.selected ? 'selected' : undefined}>
                <td className="mono">{c.version}</td>
                <td className="mono">{c.schema}</td>
                <td className="mono num">{fmtNum(c.train_samples)}</td>
                <td className="mono num">{fmtKm(c.validation_primary_mae_km, 3)}</td>
                <td><RunStatusChip status={c.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {Object.keys(exp.failures ?? {}).length > 0 && (
        <div className="note warn" style={{ marginTop: 8 }}>
          {Object.entries(exp.failures ?? {}).map(([k, v]) => (
            <div key={k}>{k}: {v}</div>
          ))}
        </div>
      )}
      {test && (
        <>
          <div className="label" style={{ margin: '14px 0 6px' }}>
            Test protocol {test.protocol} · {fmtNum(test.samples)} identical samples · mean error / p90 (km)
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>MODEL</th>
                  {[1, 3, 7].map((h) => (
                    <th key={h} className="num">D+{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {labels.map((l) => (
                  <tr key={l}>
                    <td className="mono">
                      {l} <span className="dim">({test.models[l]!.schema})</span>
                    </td>
                    {['1', '3', '7'].map((h) => {
                      const m = test.models[l]!.by_horizon[h];
                      return (
                        <td key={h} className="mono num">
                          {m ? `${fmtKm(m.mae_km, 2)} / ${fmtKm(m.p90_km, 1)}` : '—'}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {Object.keys(test.effects).length > 0 && (
            <>
              <div className="label" style={{ margin: '14px 0 6px' }}>Did the variable help? (paired, bootstrap 95 % CI of MAE difference)</div>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>EFFECT</th>
                      {[1, 3, 7].map((h) => (
                        <th key={h}>D+{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(test.effects).map(([name, e]) => (
                      <tr key={name}>
                        <td className="mono">
                          {EFFECT_LABEL[name] ?? name}
                          <div className="dim" style={{ fontSize: 10 }}>
                            {e.without} → {e.with}
                          </div>
                        </td>
                        {['1', '3', '7'].map((h) => {
                          const b = e.by_horizon[h];
                          const v = e.verdict[h] ?? 'no data';
                          return (
                            <td key={h} title={b ? `ΔMAE ${b.mae_diff_km?.toFixed(3)} km [${b.ci95_low_km?.toFixed(3)}, ${b.ci95_high_km?.toFixed(3)}], better in ${((b.candidate_better_share ?? 0) * 100).toFixed(0)} % of samples` : ''}>
                              <Chip tone={verdictTone(v)}>{v.toUpperCase()}</Chip>
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
