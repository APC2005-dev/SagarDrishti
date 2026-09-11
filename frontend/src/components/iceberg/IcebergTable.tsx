import { useEffect, useRef } from 'react';

import { useUi } from '../../stores/uiStore';
import type { IcebergSummary } from '../../types/api';
import { fmtNum } from '../../utils/format';
import { formatLat, formatLon } from '../../utils/projection';
import { Chip, StaleChip } from '../common/StatusChip';

export function IcebergTable({ rows }: { rows: IcebergSummary[] }) {
  const selected = useUi((s) => s.selectedIcebergId);
  const select = useUi((s) => s.select);
  const hover = useUi((s) => s.hover);
  const rowRefs = useRef<Record<string, HTMLTableRowElement | null>>({});

  useEffect(() => {
    if (selected) rowRefs.current[selected]?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [selected]);

  return (
    <div className="table-wrap">
      <table className="data clickable">
        <thead>
          <tr>
            <th>DESIGNATOR</th>
            <th>POSITION (OFFICIAL)</th>
            <th className="num">AREA KM²</th>
            <th className="num">L × W NM</th>
            <th>USNIC UPDATE</th>
            <th>FORECAST</th>
            <th>STATE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => (
            <tr
              key={b.icebergId}
              ref={(el) => {
                rowRefs.current[b.icebergId] = el;
              }}
              className={b.icebergId === selected ? 'selected' : undefined}
              onClick={() => select(b.icebergId === selected ? null : b.icebergId)}
              onMouseEnter={() => hover(b.icebergId)}
              onMouseLeave={() => hover(null)}
            >
              <td className="designator">{b.icebergId}</td>
              <td className="mono muted">
                {formatLat(b.latitude)} {formatLon(b.longitude)}
              </td>
              <td className="mono num">{fmtNum(b.areaSqKm)}</td>
              <td className="mono num muted">{b.lengthNm != null ? `${fmtNum(b.lengthNm)} × ${fmtNum(b.widthNm)}` : '—'}</td>
              <td className="mono muted">{b.lastUpdate ?? '—'}</td>
              <td>{b.hasForecast ? <Chip tone="forecast">{b.latestForecastModelVersion}</Chip> : <span className="dim mono">—</span>}</td>
              <td>{b.isStale ? <StaleChip /> : <Chip tone="official">OFFICIAL</Chip>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
