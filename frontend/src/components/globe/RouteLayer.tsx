import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';

import type { RouteWaypoint } from '../../types/api';
import { toScene } from '../../utils/projection';
import { MARKER_Y } from './IcebergLayer';

/**
 * The computed route on the existing Antarctic scene.
 *
 * Geometry comes from the backend's A* waypoints and is rendered with the same
 * `toScene` projection every other layer uses, so the route lands in the same
 * place as the icebergs and the coastline. The drawn polyline follows the
 * authoritative waypoints exactly — no smoothing, no invented control points.
 * The vessel marker is animated along those waypoints using their own
 * `elapsedHours`, so its motion reflects the route's timing rather than an
 * arbitrary frontend clock.
 */

const ROUTE_Y = MARKER_Y - 0.004;
const ROUTE_COLOUR = '#5fc79a';
const DEPARTURE_COLOUR = '#00e5ff';
const DESTINATION_COLOUR = '#ff9f45';
/** Seconds of wall clock per hour of voyage — the whole voyage replays smoothly. */
const REPLAY_SECONDS_PER_HOUR = 0.05;

interface Props {
  waypoints: RouteWaypoint[];
  departureName?: string;
  destinationName?: string;
}

function Endpoint({ position, colour }: { position: [number, number]; colour: string }) {
  const [x, z] = position;
  return (
    <group position={[x, ROUTE_Y + 0.002, z]}>
      <mesh>
        <sphereGeometry args={[0.05, 20, 20]} />
        <meshBasicMaterial color={colour} toneMapped={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.07, 0.095, 32]} />
        <meshBasicMaterial color={colour} toneMapped={false} transparent opacity={0.8} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

export function RouteLayer({ waypoints }: Props) {
  const points = useMemo(
    () => waypoints.map((w) => toScene(w.latitude, w.longitude)),
    [waypoints],
  );

  const lineGeometry = useMemo(() => {
    const positions: number[] = [];
    for (let i = 0; i < points.length - 1; i++) {
      const a = points[i];
      const b = points[i + 1];
      if (!a || !b) continue;
      positions.push(a[0], ROUTE_Y, a[1], b[0], ROUTE_Y, b[1]);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    return geometry;
  }, [points]);

  // Direction is shown by a marker that travels the route in voyage order.
  const vessel = useRef<THREE.Mesh>(null);
  const totalHours = waypoints.length ? (waypoints[waypoints.length - 1]?.elapsedHours ?? 0) : 0;

  useFrame(({ clock }) => {
    if (!vessel.current || points.length < 2 || totalHours <= 0) return;
    const cycle = totalHours * REPLAY_SECONDS_PER_HOUR;
    const hour = ((clock.getElapsedTime() % cycle) / cycle) * totalHours;
    let index = 0;
    while (index < waypoints.length - 2 && (waypoints[index + 1]?.elapsedHours ?? 0) < hour) index++;
    const a = points[index];
    const b = points[index + 1];
    const ha = waypoints[index]?.elapsedHours ?? 0;
    const hb = waypoints[index + 1]?.elapsedHours ?? ha + 1;
    if (!a || !b) return;
    const t = hb > ha ? Math.min(1, Math.max(0, (hour - ha) / (hb - ha))) : 0;
    vessel.current.position.set(a[0] + (b[0] - a[0]) * t, ROUTE_Y + 0.01, a[1] + (b[1] - a[1]) * t);
  });

  if (points.length < 2) return null;
  const first = points[0]!;
  const last = points[points.length - 1]!;

  return (
    <group>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial color={ROUTE_COLOUR} toneMapped={false} transparent opacity={0.95} />
      </lineSegments>
      <Endpoint position={first} colour={DEPARTURE_COLOUR} />
      <Endpoint position={last} colour={DESTINATION_COLOUR} />
      <mesh ref={vessel}>
        <coneGeometry args={[0.035, 0.09, 12]} />
        <meshBasicMaterial color="#ffffff" toneMapped={false} />
      </mesh>
    </group>
  );
}
