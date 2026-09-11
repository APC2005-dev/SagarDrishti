import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import type { EnvField } from '../../types/api';
import { toScene } from '../../utils/projection';

const Y = 0.012; // just above the ocean, below land and markers
const tmp = new THREE.Object3D();

/** Flat arrow along +X in the XZ plane (length 1, width ~0.18). */
function arrowGeometry(): THREE.BufferGeometry {
  const s = 0.035; // half shaft width
  const h = 0.09; // half head width
  const verts = [
    // shaft
    0, 0, -s, 0.62, 0, -s, 0.62, 0, s,
    0, 0, -s, 0.62, 0, s, 0, 0, s,
    // head
    0.62, 0, -h, 1, 0, 0, 0.62, 0, h,
  ];
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  g.computeVertexNormals();
  return g;
}

/**
 * East/north unit vectors of the projected plane at (lat, lon): polar
 * stereographic rotates "north" with longitude, so a (u, v) vector must be
 * mapped through the local basis to point the right way on the map.
 */
function localBasis(lat: number, lon: number): { east: [number, number]; north: [number, number] } {
  const [x0, z0] = toScene(lat, lon);
  const [xn, zn] = toScene(Math.min(lat + 0.1, -0.1), lon);
  const [xe, ze] = toScene(lat, lon + 0.1);
  const n = Math.hypot(xn - x0, zn - z0) || 1;
  const e = Math.hypot(xe - x0, ze - z0) || 1;
  return { north: [(xn - x0) / n, (zn - z0) / n], east: [(xe - x0) / e, (ze - z0) / e] };
}

interface VectorProps {
  field: EnvField;
  color: string;
  /** scene length for a 1 m/s vector, capped */
  scale: number;
  maxLength: number;
}

export function VectorFieldLayer({ field, color, scale, maxLength }: VectorProps) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const geometry = useMemo(arrowGeometry, []);
  const points = field.points.filter((p) => p.length >= 4 && Number.isFinite(p[2]) && Number.isFinite(p[3]));

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    points.forEach(([lat, lon, u, v], i) => {
      const [x, z] = toScene(lat!, lon!);
      const { east, north } = localBasis(lat!, lon!);
      const dx = u! * east[0] + v! * north[0];
      const dz = u! * east[1] + v! * north[1];
      const speed = Math.hypot(u!, v!);
      const len = Math.min(maxLength, speed * scale);
      tmp.position.set(x, Y, z);
      tmp.rotation.set(0, Math.atan2(-dz, dx), 0);
      tmp.scale.set(Math.max(len, 0.004), 1, Math.max(len, 0.004) * 0.9);
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
    });
    m.count = points.length;
    m.instanceMatrix.needsUpdate = true;
    m.computeBoundingSphere();
  }, [points, scale, maxLength]);

  useEffect(() => () => geometry.dispose(), [geometry]);
  if (!points.length) return null;
  return (
    <instancedMesh ref={mesh} args={[geometry, undefined, points.length]} frustumCulled={false}>
      <meshBasicMaterial color={color} transparent opacity={0.5} side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </instancedMesh>
  );
}

/** Sea-ice concentration as faint cells; cells below 15 % (ice edge convention) are not drawn. */
export function SeaIceLayer({ field }: { field: EnvField }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const cells = field.points.filter((p) => Number.isFinite(p[2]) && p[2]! >= 0.15);
  const size = (field.resolutionDeg / 90) * 5; // rough scene size of a cell near 65°S

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const c = new THREE.Color();
    cells.forEach(([lat, lon, conc], i) => {
      const [x, z] = toScene(lat!, lon!);
      tmp.position.set(x, Y - 0.002, z);
      tmp.rotation.set(-Math.PI / 2, 0, 0);
      const s = size * Math.cos(((lat! + 90) * Math.PI) / 180 + 0.2) + size * 0.4;
      tmp.scale.set(s, s, s);
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
      m.setColorAt(i, c.setRGB(0.55 + 0.45 * conc!, 0.62 + 0.38 * conc!, 0.75 + 0.25 * conc!));
    });
    m.count = cells.length;
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
  }, [cells, size]);

  if (!cells.length) return null;
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, cells.length]} frustumCulled={false}>
      <planeGeometry args={[1, 1]} />
      <meshBasicMaterial transparent opacity={0.28} depthWrite={false} toneMapped={false} />
    </instancedMesh>
  );
}
