import { useEffect, useRef } from 'react';

import type { SeaIceField } from '../../types/api';
import { toPolar } from '../../utils/projection';

/**
 * Sea-ice concentration rendered in EPSG:3031 (the projection the backend and
 * the model use), from the [lat, lon, concentration] points the API returns.
 *
 * Every cell drawn comes from a stored grid — an official observation or a
 * persisted model output. Nothing is interpolated or synthesised here; cells
 * the source never retrieved are simply absent from the payload and are not
 * drawn.
 */

/** Extent of the plotted area in EPSG:3031 metres (comfortably covers the Southern Ocean). */
const EXTENT_M = 5_200_000;

function colour(concentration: number): string {
  // Open water -> deep blue, full ice cover -> white. Matches the legend below
  // the map; the scale is linear in concentration so it can be read directly.
  const t = Math.min(1, Math.max(0, concentration));
  const r = Math.round(28 + t * 227);
  const g = Math.round(52 + t * 203);
  const b = Math.round(105 + t * 150);
  return `rgb(${r},${g},${b})`;
}

export function ConcentrationMap({ field, size = 420 }: { field: SeaIceField; size?: number }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    // Ocean backdrop so absent cells read as "no data", not as zero ice.
    ctx.fillStyle = 'rgba(12, 22, 48, 0.55)';
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
    ctx.fill();

    const scale = size / (2 * EXTENT_M);
    // A cell spans resolutionDeg in both axes, but on a polar projection its
    // WIDTH shrinks towards the pole (arc length = radius x delta-longitude)
    // while its height stays roughly constant. Sizing every cell identically
    // leaves radial gaps at the outer edge and piles cells up near the pole,
    // so each is sized from its own projected radius.
    const dLon = (field.resolutionDeg * Math.PI) / 180;
    const cellHeight = Math.max(1.2, field.resolutionDeg * 111_320 * scale);
    for (const point of field.points) {
      const [lat, lon, value] = point;
      if (lat === undefined || lon === undefined || value === undefined) continue;
      const [x, y] = toPolar(lat, lon);
      const radius = Math.hypot(x, y);
      const cellWidth = Math.max(1.2, radius * dLon * scale);
      const px = size / 2 + x * scale;
      const py = size / 2 - y * scale;
      ctx.fillStyle = colour(value);
      // Overlap slightly so neighbouring cells tile without seams.
      ctx.fillRect(px - cellWidth / 2, py - cellHeight / 2, cellWidth + 0.6, cellHeight + 0.6);
    }

    // Graticule: the pole and two reference latitude circles.
    ctx.strokeStyle = 'rgba(233, 236, 245, 0.16)';
    ctx.lineWidth = 1;
    for (const lat of [-60, -75]) {
      const [, ry] = toPolar(lat, 0);
      ctx.beginPath();
      ctx.arc(size / 2, size / 2, Math.abs(ry) * scale, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2);
    ctx.stroke();
  }, [field, size]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, alignItems: 'center' }}>
      <canvas
        ref={ref}
        style={{ width: size, height: size, maxWidth: '100%' }}
        aria-label={`Sea-ice concentration, ${field.kind}, valid ${field.validDate}`}
      />
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 11 }} className="mono">
        <span className="dim">{Math.round(field.minConcentration * 100)}%</span>
        <span
          style={{
            width: 140,
            height: 8,
            borderRadius: 2,
            background: `linear-gradient(90deg, ${colour(field.minConcentration)}, ${colour(1)})`,
          }}
        />
        <span className="dim">100% ice</span>
        <span className="dim">· {field.points.length.toLocaleString()} cells · EPSG:3031</span>
      </div>
    </div>
  );
}
