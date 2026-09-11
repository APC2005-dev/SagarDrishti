import { describe, expect, it } from 'vitest';

import { isSeamVertex, planarArea, planarPolygon } from './antarcticaGeometry';

// A coarse Antarctic coastline ring (lon, lat) around the pole.
const coast = Array.from({ length: 36 }, (_, i) => [-180 + i * 10, -70] as [number, number]);
coast.push(coast[0]!);
// d3/world-atlas spherical winding: first ring hugs the pole, coastline comes second.
const poleRing = Array.from({ length: 37 }, (_, i) => [180 - i * 10, -89.999] as [number, number]);

describe('Antarctica planar geometry', () => {
  it('treats pole and antimeridian-seam vertices as artefacts', () => {
    expect(isSeamVertex([12, -89.999])).toBe(true);
    expect(isSeamVertex([180, -85])).toBe(true);
    expect(isSeamVertex([180, -78])).toBe(false); // Ross Ice Shelf edge is real coastline
    expect(isSeamVertex([45, -67])).toBe(false);
  });

  it('recovers the coastline when it is stored as the second (spherical) ring', () => {
    const planar = planarPolygon([poleRing, coast]);
    expect(planar).not.toBeNull();
    expect(planar!.outer.length).toBeGreaterThanOrEqual(35);
    expect(planar!.holes).toHaveLength(0);
    // ~ area of a disc of radius rho(-70°) in scene units (1 unit = 1000 km)
    expect(planarArea(planar!.outer)).toBeGreaterThan(9);
  });

  it('keeps a normally wound island unchanged', () => {
    const island = [[-60, -63], [-58, -63], [-58, -62], [-60, -62], [-60, -63]] as [number, number][];
    const planar = planarPolygon([island]);
    expect(planar!.outer).toHaveLength(4);
  });

  it('drops a polygon that is entirely seam', () => {
    expect(planarPolygon([poleRing])).toBeNull();
  });
});
