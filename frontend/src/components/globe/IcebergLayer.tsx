import { Html } from '@react-three/drei';
import type { ThreeEvent } from '@react-three/fiber';
import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { COLORS } from '../../constants';
import type { IcebergSummary } from '../../types/api';
import { formatLat, formatLon } from '../../utils/projection';
import { toScene } from '../../utils/projection';

export const MARKER_Y = 0.085;

export interface PlottableIceberg extends IcebergSummary {
  latitude: number;
  longitude: number;
}

interface Props {
  icebergs: PlottableIceberg[];
  selectedId: string | null;
  hoveredId: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string | null) => void;
}

const tmp = new THREE.Object3D();
const cOfficial = new THREE.Color(COLORS.official);
const cStale = new THREE.Color(COLORS.officialStale);
const cSelected = new THREE.Color(COLORS.selected);

/**
 * Current OFFICIAL positions: one InstancedMesh (a single draw call for all
 * icebergs). Solid diamonds = observed; forecasts use hollow rings instead.
 */
export function IcebergLayer({ icebergs, selectedId, hoveredId, onHover, onSelect }: Props) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const halo = useRef<THREE.Mesh>(null);
  const positions = useMemo(() => icebergs.map((b) => toScene(b.latitude, b.longitude)), [icebergs]);
  const selectedIdx = icebergs.findIndex((b) => b.icebergId === selectedId);
  const hoveredIdx = icebergs.findIndex((b) => b.icebergId === hoveredId);

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    icebergs.forEach((b, i) => {
      const [x, z] = positions[i]!;
      const s = i === selectedIdx ? 2.0 : i === hoveredIdx ? 1.5 : 1.2;
      tmp.position.set(x, MARKER_Y, z);
      tmp.rotation.set(0, Math.PI / 4, 0);
      tmp.scale.set(s, s * 0.8, s);
      tmp.updateMatrix();
      m.setMatrixAt(i, tmp.matrix);
      m.setColorAt(i, i === selectedIdx ? cSelected : b.isStale ? cStale : cOfficial);
    });
    m.count = icebergs.length;
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.computeBoundingSphere();
  }, [icebergs, positions, selectedIdx, hoveredIdx]);

  useFrame(({ clock }) => {
    if (!halo.current) return;
    const t = clock.getElapsedTime();
    const s = 1 + 0.25 * Math.sin(t * 2.2);
    halo.current.scale.set(s, s, s);
    (halo.current.material as THREE.MeshBasicMaterial).opacity = 0.55 - 0.25 * Math.sin(t * 2.2);
  });

  const idAt = (e: ThreeEvent<PointerEvent | MouseEvent>) => (e.instanceId !== undefined ? icebergs[e.instanceId]?.icebergId ?? null : null);
  const hovered = hoveredIdx >= 0 ? icebergs[hoveredIdx] : undefined;
  const selectedPos = selectedIdx >= 0 ? positions[selectedIdx] : undefined;

  return (
    <group>
      <instancedMesh
        ref={mesh}
        args={[undefined, undefined, Math.max(icebergs.length, 1)]}
        onPointerMove={(e) => {
          e.stopPropagation();
          onHover(idAt(e));
        }}
        onPointerOut={() => onHover(null)}
        onClick={(e) => {
          e.stopPropagation();
          onSelect(idAt(e));
        }}
      >
        <octahedronGeometry args={[0.062, 0]} />
        <meshBasicMaterial toneMapped={false} />
      </instancedMesh>

      {selectedPos && (
        <mesh ref={halo} position={[selectedPos[0], MARKER_Y - 0.01, selectedPos[1]]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.075, 0.085, 48]} />
          <meshBasicMaterial color={COLORS.selected} transparent opacity={0.5} depthWrite={false} />
        </mesh>
      )}

      {hovered && (
        <Html position={[positions[hoveredIdx]![0], MARKER_Y, positions[hoveredIdx]![1]]} zIndexRange={[20, 10]}>
          <div className="map-tooltip glass">
            <span className="designator">{hovered.icebergId}</span>{' '}
            <span className="mono dim">
              {formatLat(hovered.latitude)} {formatLon(hovered.longitude)}
            </span>
            <br />
            <span className="mono dim">OFFICIAL · USNIC {hovered.lastUpdate}</span>
          </div>
        </Html>
      )}
    </group>
  );
}
