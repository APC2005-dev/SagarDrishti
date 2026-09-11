import { useIcebergEnvironment } from '../../hooks/queries';
import type { EnvGroupValue } from '../../types/api';
import { fmtDate } from '../../utils/format';
import { KV, SectionTitle } from '../common/Metric';
import { QueryState } from '../common/QueryState';
import { Chip } from '../common/StatusChip';

const GROUP_LABEL: Record<string, string> = { wind: 'Wind (10 m)', current: 'Ocean current (0.5 m)', sea_ice: 'Sea ice' };

function value(g: EnvGroupValue): string {
  if (g.missing) return 'not available';
  if (g.group === 'sea_ice') {
    const c = g.values.sea_ice_concentration;
    return c == null ? '—' : `${(c * 100).toFixed(0)} %`;
  }
  if (!g.vector) return '—';
  const dir = `${g.vector.directionDeg.toFixed(0)}°`;
  return `${g.vector.speedMS.toFixed(2)} m/s · ${g.vector.directionConvention === 'from' ? 'from' : 'towards'} ${dir}`;
}

/** Environmental conditions aligned to the latest official fix — model inputs, not observations of the iceberg. */
export function EnvironmentSection({ icebergId }: { icebergId: string }) {
  const env = useIcebergEnvironment(icebergId);
  return (
    <>
      <SectionTitle right={<Chip tone="neutral">ANALYSIS</Chip>}>Environment at latest official fix</SectionTitle>
      <QueryState query={env} compact notFoundText="No official observation to align to.">
        {(e) =>
          !e.aligned ? (
            <div className="note">{e.reason}</div>
          ) : (
            <>
              <div className="kv-grid">
                {e.groups.map((g) => (
                  <KV key={g.group} k={GROUP_LABEL[g.group] ?? g.group} v={value(g)} />
                ))}
              </div>
              <div className="note" style={{ marginTop: 8 }}>
                {e.groups.map((g) => (
                  <div key={g.group}>
                    {g.group}: {g.datasetId} · valid {fmtDate(g.validDate)} (as of {fmtDate(g.asOf)}
                    {g.stalenessDays ? `, ${g.stalenessDays} d before the fix` : ''}) · {g.interpolation}
                    {g.qualityFlags.length ? ` · ${g.qualityFlags.join(', ')}` : ''}
                    {g.missing && g.reason ? ` · ${g.reason}` : ''}
                  </div>
                ))}
              </div>
            </>
          )
        }
      </QueryState>
    </>
  );
}
