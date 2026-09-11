/**
 * Real Antarctic coastline: Natural Earth 1:50m (via the `world-atlas` package),
 * feature ISO 3166-1 numeric 010. Projected to EPSG:3031 for rendering.
 *
 * world-atlas follows d3-geo's *spherical* winding convention. For Antarctica
 * that means the polygon's first ring is a tiny circle around the pole
 * (lat -89.999) and the real coastline is stored as its second ring. In a
 * planar projection that pole ring is degenerate, so every ring is projected
 * and cleaned first, and the ring with the largest planar area is taken as the
 * exterior. Vertices on the pole / ±180° seam (artefacts of the source's
 * equirectangular closure) are removed; they would collapse onto a line in
 * polar stereographic.
 */
import type { Feature, FeatureCollection, MultiPolygon, Polygon, Position } from 'geojson';
import * as THREE from 'three';
import { feature } from 'topojson-client';
import type { GeometryCollection, Topology } from 'topojson-specification';
import atlasUrl from 'world-atlas/countries-50m.json?url';

import { toScene } from '../../utils/projection';

export interface LandGeometry {
  shapes: THREE.Shape[];
  /** Closed coastline loops in scene coordinates [X, Z] */
  outlines: [number, number][][];
}

export function isSeamVertex([lon, lat]: Position): boolean {
  if (lat === undefined || lon === undefined) return true;
  return lat <= -89.5 || (Math.abs(lon) >= 179.999 && lat < -79);
}

export function projectRing(ring: Position[]): [number, number][] {
  const out: [number, number][] = [];
  for (const p of ring) {
    if (isSeamVertex(p)) continue;
    const [x, z] = toScene(p[1]!, p[0]!);
    const last = out[out.length - 1];
    if (last && Math.abs(last[0] - x) < 1e-5 && Math.abs(last[1] - z) < 1e-5) continue;
    out.push([x, z]);
  }
  // GeoJSON rings repeat the first vertex at the end; THREE.Shape closes itself.
  const first = out[0];
  const last = out[out.length - 1];
  if (out.length > 1 && first && last && Math.abs(first[0] - last[0]) < 1e-5 && Math.abs(first[1] - last[1]) < 1e-5) out.pop();
  return out;
}

export function planarArea(ring: [number, number][]): number {
  let a = 0;
  for (let i = 0; i < ring.length; i++) {
    const [x1, z1] = ring[i]!;
    const [x2, z2] = ring[(i + 1) % ring.length]!;
    a += x1 * z2 - x2 * z1;
  }
  return Math.abs(a) / 2;
}

/** Split one (spherically wound) polygon into a planar exterior + holes. */
export function planarPolygon(poly: Position[][]): { outer: [number, number][]; holes: [number, number][][] } | null {
  const rings = poly.map(projectRing).filter((r) => r.length >= 4);
  if (rings.length === 0) return null;
  rings.sort((a, b) => planarArea(b) - planarArea(a));
  const [outer, ...holes] = rings;
  return { outer: outer!, holes };
}

/** Shape points are in map plane (x, y_map) = (X, −Z); the mesh is rotated onto the ground. */
function toShapePoints(ring: [number, number][]): THREE.Vector2[] {
  return ring.map(([x, z]) => new THREE.Vector2(x, -z));
}

let cache: Promise<LandGeometry> | null = null;

export function loadAntarctica(): Promise<LandGeometry> {
  cache ??= (async () => {
    const res = await fetch(atlasUrl);
    if (!res.ok) throw new Error(`coastline data HTTP ${res.status}`);
    const topo = (await res.json()) as Topology;
    const countries = feature(topo, topo.objects.countries as GeometryCollection) as FeatureCollection;
    const ant = countries.features.find(
      (f: Feature) => String(f.id) === '010' || (f.properties as { name?: string } | null)?.name === 'Antarctica',
    );
    if (!ant) throw new Error('Antarctica feature not found in Natural Earth dataset');
    const geom = ant.geometry as Polygon | MultiPolygon;
    const polygons: Position[][][] = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;

    const shapes: THREE.Shape[] = [];
    const outlines: [number, number][][] = [];
    for (const poly of polygons) {
      const planar = planarPolygon(poly);
      if (!planar) continue;
      const shape = new THREE.Shape(toShapePoints(planar.outer));
      for (const h of planar.holes) shape.holes.push(new THREE.Path(toShapePoints(h)));
      shapes.push(shape);
      outlines.push(planar.outer, ...planar.holes);
    }
    if (!shapes.length) throw new Error('coastline produced no polygons');
    return { shapes, outlines };
  })();
  cache.catch(() => {
    cache = null;
  });
  return cache;
}
