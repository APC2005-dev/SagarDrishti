import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';

import { COLORS } from '../../constants';
import { type LandGeometry, loadAntarctica } from './antarcticaGeometry';

export const LAND_HEIGHT = 0.045;

/**
 * The continent as a 3D slab: Natural Earth 1:50m outline extruded upward.
 * Top face = EPSG:3031 satellite mosaic (when available), sides = ice cliffs.
 * The slab height is a visual device, not measured elevation.
 */
export function Land({ map, onError }: { map: THREE.Texture | null; onError?: (msg: string) => void }) {
  const [land, setLand] = useState<LandGeometry | null>(null);

  useEffect(() => {
    let alive = true;
    loadAntarctica()
      .then((g) => alive && setLand(g))
      .catch((e: unknown) => onError?.(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, [onError]);

  const extruded = useMemo(() => {
    if (!land) return null;
    // Default UV generator gives cap UVs = shape (x, y_map) in scene units; the
    // basemap texture's repeat/offset maps those onto the EPSG:3031 mosaic.
    const g = new THREE.ExtrudeGeometry(land.shapes, { depth: LAND_HEIGHT, bevelEnabled: false, curveSegments: 1 });
    g.computeVertexNormals();
    return g;
  }, [land]);

  const coast = useMemo(() => {
    if (!land) return null;
    const pos: number[] = [];
    for (const ring of land.outlines) {
      for (let i = 0; i < ring.length; i++) {
        const a = ring[i]!;
        const b = ring[(i + 1) % ring.length]!;
        pos.push(a[0], LAND_HEIGHT + 0.001, a[1], b[0], LAND_HEIGHT + 0.001, b[1]);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, [land]);

  // ExtrudeGeometry groups: 0 = caps (top/bottom), 1 = side walls.
  const materials = useMemo(() => {
    const top = map
      ? new THREE.MeshBasicMaterial({ map, toneMapped: false })
      : new THREE.MeshStandardMaterial({ color: COLORS.land, roughness: 0.95, metalness: 0, emissive: '#1c2233', emissiveIntensity: 0.35 });
    const sides = new THREE.MeshStandardMaterial({ color: '#c9d4e6', roughness: 0.85, metalness: 0, emissive: '#223055', emissiveIntensity: 0.4 });
    return [top, sides];
  }, [map]);

  useEffect(() => () => extruded?.dispose(), [extruded]);
  useEffect(() => () => coast?.dispose(), [coast]);
  useEffect(() => () => materials.forEach((m) => m.dispose()), [materials]);

  if (!extruded || !coast) return null;
  return (
    <group>
      <mesh geometry={extruded} material={materials} rotation={[-Math.PI / 2, 0, 0]} />
      <lineSegments geometry={coast}>
        <lineBasicMaterial color={COLORS.landEdge} transparent opacity={map ? 0.3 : 0.55} />
      </lineSegments>
    </group>
  );
}
