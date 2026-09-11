import { useMemo } from 'react';

import type { ForecastPoint, Observation } from '../../types/api';
import { toPolar } from '../../utils/projection';

interface Props {
  history: Observation[];
  forecast?: ForecastPoint[];
  anchor?: { latitude: number; longitude: number } | null;
  height?: number;
}

/**
 * Local track plot in EPSG:3031 km (north-up = towards 0° longitude). Past
 * positions solid, official fixes as cyan diamonds, forecast dashed violet.
 */
export function TrackChart({ history, forecast = [], anchor, height = 200 }: Props) {
  const data = useMemo(() => {
    const hist = [...history]
      .sort((a, b) => a.observationDate.localeCompare(b.observationDate))
      .map((o) => ({ p: toPolar(o.latitude, o.longitude), official: o.provenance === 'official_usnic', date: o.observationDate }));
    const fc = forecast.map((f) => toPolar(f.predictedLatitude, f.predictedLongitude));
    const a = anchor ? toPolar(anchor.latitude, anchor.longitude) : null;
    const all = [...hist.map((h) => h.p), ...fc, ...(a ? [a] : [])];
    if (all.length === 0) return null;
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const span = Math.max(maxX - minX, maxY - minY, 20_000) * 1.15;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return { hist, fc, a, cx, cy, span };
  }, [history, forecast, anchor]);

  if (!data) return <div className="qs qs-empty compact">No positions to plot.</div>;
  const W = 400, H = height;
  const scale = Math.min(W, H) / data.span;
  const sx = (x: number) => W / 2 + (x - data.cx) * scale;
  const sy = (y: number) => H / 2 - (y - data.cy) * scale;
  const km = data.span / 1000;
  const barKm = [1, 2, 5, 10, 20, 50, 100, 200, 500].find((v) => v > km / 6) ?? 1000;
  const histPath = data.hist.map((h, i) => `${i ? 'L' : 'M'}${sx(h.p[0]).toFixed(1)},${sy(h.p[1]).toFixed(1)}`).join(' ');
  const fcChain = data.a ? [data.a, ...data.fc] : data.fc;
  const fcPath = fcChain.map((p, i) => `${i ? 'L' : 'M'}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Track plot" style={{ background: 'var(--bg-2)', borderRadius: 4 }}>
      <path d={histPath} fill="none" stroke="#6f7a99" strokeWidth={1.2} />
      {data.hist
        .filter((h) => h.official)
        .map((h) => (
          <rect key={h.date} x={sx(h.p[0]) - 3} y={sy(h.p[1]) - 3} width={6} height={6} fill="var(--official)" transform={`rotate(45 ${sx(h.p[0])} ${sy(h.p[1])})`} />
        ))}
      {fcChain.length > 1 && <path d={fcPath} fill="none" stroke="var(--forecast)" strokeWidth={1.3} strokeDasharray="4 3" />}
      {data.fc.map((p, i) => (
        <circle key={i} cx={sx(p[0])} cy={sy(p[1])} r={3.2} fill="none" stroke="var(--forecast)" strokeWidth={1.2} />
      ))}
      <g transform={`translate(12 ${H - 14})`}>
        <line x1={0} x2={barKm * 1000 * scale} y1={0} y2={0} stroke="var(--text-3)" />
        <text x={0} y={-5} fill="var(--text-3)" fontSize={9} fontFamily="var(--font-mono)">
          {barKm} km · EPSG:3031
        </text>
      </g>
    </svg>
  );
}
