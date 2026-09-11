import { describe, expect, it } from 'vitest';

import { bearingDeg, formatLat, formatLon, haversineKm, scaleFactor, toPolar } from './projection';

describe('EPSG:3031 projection', () => {
  it('matches pyproj output used by the notebook (A01, 1978-10-22)', () => {
    const [x, y] = toPolar(-56.0, -34.2);
    expect(x).toBeCloseTo(-2.137135e6, -2);
    expect(y).toBeCloseTo(3.144699e6, -2);
  });

  it('maps the pole to the origin', () => {
    const [x, y] = toPolar(-90, 123);
    expect(Math.abs(x)).toBeLessThan(1e-6);
    expect(Math.abs(y)).toBeLessThan(1e-6);
  });

  it('is true to scale at 71°S', () => {
    expect(scaleFactor(-71)).toBeCloseTo(1, 6);
    expect(scaleFactor(-90)).toBeLessThan(1);
  });

  it('formats coordinates with hemispheres', () => {
    expect(formatLat(-62.92)).toBe('62.92°S');
    expect(formatLon(-47.22)).toBe('47.22°W');
    expect(formatLon(83.39)).toBe('83.39°E');
  });

  it('computes distances and bearings', () => {
    expect(haversineKm(-60, 10, -61, 10)).toBeCloseTo(111.195, 2);
    expect(bearingDeg(-60, 10, -59, 10)).toBeCloseTo(0, 5);
    expect(bearingDeg(-60, 10, -60, 11)).toBeGreaterThan(80);
  });
});
