import { useEffect, useMemo } from 'react';
import * as THREE from 'three';

import { COLORS } from '../../constants';
import type { Observation } from '../../types/api';
import { toScene } from '../../utils/projection';
import { MARKER_Y } from './IcebergLayer';

/** Past track of the selected iceberg: solid grey line, official fixes as small cyan diamonds. */
export function HistoryTrail({ observations }: { observations: Observation[] }) {
  const sorted = useMemo(() => [...observations].sort((a, b) => a.observationDate.localeCompare(b.observationDate)), [observations]);

  const line = useMemo(() => {
    const pos = sorted.flatMap((o) => {
      const [x, z] = toScene(o.latitude, o.longitude);
      return [x, MARKER_Y - 0.012, z];
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, [sorted]);

  const official = sorted.filter((o) => o.provenance === 'official_usnic');
  const officialGeom = useMemo(() => {
    const pos = official.flatMap((o) => {
      const [x, z] = toScene(o.latitude, o.longitude);
      return [x, MARKER_Y - 0.01, z];
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, [official]);

  useEffect(() => () => (line.dispose(), officialGeom.dispose()), [line, officialGeom]);
  if (sorted.length < 2) return null;
  return (
    <group>
      <line>
        <primitive object={line} attach="geometry" />
        <lineBasicMaterial color={COLORS.history} transparent opacity={0.85} />
      </line>
      <points geometry={officialGeom}>
        <pointsMaterial color={COLORS.official} size={4} sizeAttenuation={false} />
      </points>
    </group>
  );
}
