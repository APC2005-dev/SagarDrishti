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
const ROUTE_COLOUR = '#9184d9';
const DEPARTURE_COLOUR = '#10b981';
const DESTINATION_COLOUR = '#ec4899';
/** Seconds of wall clock per hour of voyage — the whole voyage replays smoothly. */
const REPLAY_SECONDS_PER_HOUR = 0.05;

interface Props {
  waypoints: RouteWaypoint[];
  departureName?: string;
  destinationName?: string;
}

/**
 * Marker size is derived from the route's own extent. One scene unit is 1000 km,
 * so a fixed 0.05 marker is a 50 km blob: on a 33 km voyage that is larger than
 * the whole route and the two endpoints merge into one dot. Scaling with the
 * route keeps departure and destination separable at any length.
 */
function markerRadius(extentSceneUnits: number): number {
  // The lower bound is a visibility floor: below ~12 km a marker is sub-pixel on
  // a whole-continent view. The upper bound stops a basin crossing from being
  // capped by two huge blobs.
  return Math.min(0.035, Math.max(0.012, extentSceneUnits * 0.25));
}

function Endpoint({
  position,
  colour,
  radius,
}: {
  position: [number, number];
  colour: string;
  radius: number;
}) {
  const [x, z] = position;
  return (
    <group position={[x, ROUTE_Y + 0.002, z]}>
      <mesh>
        <sphereGeometry args={[radius, 20, 20]} />
        <meshBasicMaterial color={colour} toneMapped={false} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[radius * 1.5, radius * 2.1, 32]} />
        <meshBasicMaterial color={colour} toneMapped={false} transparent opacity={0.85} side={THREE.DoubleSide} />
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
  // Longest span the route actually covers, so a short hop is not drawn with
  // markers sized for a basin crossing.
  let extent = 0;
  for (const [x, z] of points) extent = Math.max(extent, Math.hypot(x - first[0], z - first[1]));
  extent = Math.max(extent, Math.hypot(last[0] - first[0], last[1] - first[1]));
  const radius = markerRadius(extent);

  return (
    <group>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial color={ROUTE_COLOUR} toneMapped={false} transparent opacity={0.95} />
      </lineSegments>
      <Endpoint position={first} colour={DEPARTURE_COLOUR} radius={radius} />
      <Endpoint position={last} colour={DESTINATION_COLOUR} radius={radius} />
      <mesh ref={vessel}>
        <coneGeometry args={[radius * 0.8, radius * 2, 12]} />
        <meshBasicMaterial color="#ffffff" toneMapped={false} />
      </mesh>
    </group>
  );
}
