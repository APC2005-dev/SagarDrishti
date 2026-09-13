import { OrbitControls } from '@react-three/drei';
import { useFrame, useThree } from '@react-three/fiber';
import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';

const HOME_TARGET = new THREE.Vector3(0, 0, 0.3);
const HOME_POSITION = new THREE.Vector3(0, 7.4, 6.2);

/**
 * Smooth fly-to on selection; any user drag cancels the flight so the
 * camera never fights the operator.
 */
export function CameraRig({ focus }: { focus: [number, number] | null }) {
  const controls = useRef<OrbitControlsImpl>(null);
  const { camera } = useThree();
  const flight = useRef<{ target: THREE.Vector3; position: THREE.Vector3 } | null>(null);

  useEffect(() => {
    if (focus) {
      const isHomeFocus = focus[0] === 0 && focus[1] === 0;
      if (isHomeFocus) {
        flight.current = { target: HOME_TARGET.clone(), position: HOME_POSITION.clone() };
      } else {
        const target = new THREE.Vector3(focus[0], 0, focus[1]);
        const dir = new THREE.Vector3(focus[0] * 0.25, 2.6, focus[1] * 0.25 + 1.9);
        flight.current = { target, position: target.clone().add(dir) };
      }
    } else {
      flight.current = { target: HOME_TARGET.clone(), position: HOME_POSITION.clone() };
    }
  }, [focus]);

  useEffect(() => {
    const c = controls.current;
    if (!c) return;
    const cancel = () => {
      flight.current = null;
    };
    c.addEventListener('start', cancel);
    return () => c.removeEventListener('start', cancel);
  }, []);

  useFrame((_, dt) => {
    const f = flight.current;
    const c = controls.current;
    if (!f || !c) return;
    const k = 1 - Math.exp(-dt * 3);
    c.target.lerp(f.target, k);
    camera.position.lerp(f.position, k);
    c.update();
    if (c.target.distanceTo(f.target) < 0.002 && camera.position.distanceTo(f.position) < 0.005) flight.current = null;
  });

  return (
    <OrbitControls
      ref={controls}
      makeDefault
      target={HOME_TARGET}
      enableDamping
      dampingFactor={0.08}
      minDistance={0.8}
      maxDistance={15}
      minPolarAngle={0.05}
      maxPolarAngle={1.15}
      screenSpacePanning={false}
    />
  );
}
