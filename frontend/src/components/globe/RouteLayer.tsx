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

  // Direction and heading orientation of the ship along the route
  const vesselGroup = useRef<THREE.Group>(null);
  const totalHours = waypoints.length ? (waypoints[waypoints.length - 1]?.elapsedHours ?? 0) : 0;

  useFrame(({ clock }) => {
    if (!vesselGroup.current || points.length < 2 || totalHours <= 0) return;
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
    const posX = a[0] + (b[0] - a[0]) * t;
    const posZ = a[1] + (b[1] - a[1]) * t;
    vesselGroup.current.position.set(posX, ROUTE_Y + 0.01, posZ);

    // Compute heading angle (in scene XZ plane) so the ship faces forward along the route
    const dx = b[0] - a[0];
    const dz = b[1] - a[1];
    if (Math.abs(dx) > 1e-6 || Math.abs(dz) > 1e-6) {
      const headingAngle = Math.atan2(dx, dz);
      vesselGroup.current.rotation.y = headingAngle;
    }
  });

  // Custom hydrodynamic ship hull shape (pointed bow, wide beam midship, tapered stern)
  const shipHullGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0, 1.4);                 // Bow tip
    shape.quadraticCurveTo(0.5, 0.8, 0.45, 0.0);  // Starboard curve
    shape.lineTo(0.4, -1.2);               // Midship to stern
    shape.lineTo(-0.4, -1.2);              // Stern transom width
    shape.lineTo(-0.45, 0.0);              // Port midship
    shape.quadraticCurveTo(-0.5, 0.8, 0, 1.4);    // Port curve back to bow
    const extrudeSettings = { depth: 0.45, bevelEnabled: true, bevelSegments: 3, steps: 1, bevelSize: 0.06, bevelThickness: 0.08 };
    const geom = new THREE.ExtrudeGeometry(shape, extrudeSettings);
    geom.rotateX(Math.PI / 2);
    geom.center();
    return geom;
  }, []);

  if (points.length < 2) return null;
  const first = points[0]!;
  const last = points[points.length - 1]!;
  // Longest span the route actually covers
  let extent = 0;
  for (const [x, z] of points) extent = Math.max(extent, Math.hypot(x - first[0], z - first[1]));
  extent = Math.max(extent, Math.hypot(last[0] - first[0], last[1] - first[1]));
  const radius = markerRadius(extent);
  const shipScale = radius * 1.5;

  return (
    <group>
      <lineSegments geometry={lineGeometry}>
        <lineBasicMaterial color={ROUTE_COLOUR} toneMapped={false} transparent opacity={0.95} />
      </lineSegments>
      <Endpoint position={first} colour={DEPARTURE_COLOUR} radius={radius} />
      <Endpoint position={last} colour={DESTINATION_COLOUR} radius={radius} />

      {/* Realistic Hydrodynamic Ship 3D Model */}
      <group ref={vesselGroup}>
        <group scale={[shipScale, shipScale, shipScale]}>
          {/* Main Ship Hull */}
          <mesh geometry={shipHullGeometry} position={[0, 0.22, 0]}>
            <meshStandardMaterial color="#0284c7" metalness={0.7} roughness={0.2} emissive="#0369a1" emissiveIntensity={0.25} />
          </mesh>

          {/* White Upper Deck Housing */}
          <mesh position={[0, 0.5, -0.1]}>
            <boxGeometry args={[0.65, 0.25, 1.6]} />
            <meshStandardMaterial color="#f8fafc" roughness={0.2} metalness={0.3} />
          </mesh>

          {/* Elevated Command Bridge / Wheelhouse */}
          <mesh position={[0, 0.75, 0.2]}>
            <boxGeometry args={[0.55, 0.35, 0.7]} />
            <meshStandardMaterial color="#e2e8f0" roughness={0.1} />
          </mesh>

          {/* Wrap-around Tinted Glass Windows */}
          <mesh position={[0, 0.82, 0.56]}>
            <boxGeometry args={[0.52, 0.16, 0.05]} />
            <meshStandardMaterial color="#0f172a" metalness={0.95} roughness={0.05} />
          </mesh>

          {/* Red Exhaust Funnel / Smokestack */}
          <mesh position={[0, 0.88, -0.4]} rotation={[0.1, 0, 0]}>
            <cylinderGeometry args={[0.08, 0.11, 0.45, 12]} />
            <meshStandardMaterial color="#ef4444" metalness={0.4} roughness={0.3} />
          </mesh>

          {/* Radar Mast & Glowing Beacon */}
          <mesh position={[0, 1.15, 0.2]}>
            <cylinderGeometry args={[0.02, 0.03, 0.5, 8]} />
            <meshBasicMaterial color="#38bdf8" toneMapped={false} />
          </mesh>
          <mesh position={[0, 1.4, 0.2]}>
            <sphereGeometry args={[0.08, 12, 12]} />
            <meshBasicMaterial color="#38bdf8" toneMapped={false} />
          </mesh>

          {/* Hydrodynamic V-Wake Water Ripple Trail */}
          <mesh position={[0, 0.02, -1.5]} rotation={[-Math.PI / 2, 0, 0]}>
            <planeGeometry args={[1.2, 1.8]} />
            <meshBasicMaterial color="#38bdf8" transparent opacity={0.4} side={THREE.DoubleSide} />
          </mesh>

          {/* Bright Navigation Lights */}
          <pointLight color="#38bdf8" intensity={2.5} distance={1.8} position={[0, 0.8, 1.2]} />
          <pointLight color="#10b981" intensity={1.5} distance={0.8} position={[0.4, 0.6, 0.2]} />
          <pointLight color="#ef4444" intensity={1.5} distance={0.8} position={[-0.4, 0.6, 0.2]} />
        </group>
      </group>
    </group>
  );
}
