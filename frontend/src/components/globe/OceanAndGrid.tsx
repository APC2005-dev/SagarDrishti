import { Html } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { COLORS } from '../../constants';
import { toScene } from '../../utils/projection';

export const OUTER_LAT = -40;
export const OCEAN_RADIUS = Math.hypot(...toScene(OUTER_LAT, 0));

const oceanVertex = /* glsl */ `
  varying vec2 vPos;
  void main() {
    vPos = position.xy; // (x, y) in EPSG:3031 scene units (mesh is rotated onto the ground)
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

// Basemap imagery where available (inside the GIBS square), otherwise a deep
// navy depth gradient with a very slow, faint swell.
const oceanFragment = /* glsl */ `
  uniform float uTime;
  uniform float uRadius;
  uniform vec3 uDeep;
  uniform vec3 uShallow;
  uniform sampler2D uMap;
  uniform float uHasMap;
  uniform float uExtent;
  varying vec2 vPos;
  float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
  float noise(vec2 p) {
    vec2 i = floor(p); vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
  }
  void main() {
    float r = length(vPos) / uRadius;
    vec3 col = mix(uShallow, uDeep, smoothstep(0.15, 1.0, r));
    float n = noise(vPos * 3.0 + vec2(uTime * 0.02, -uTime * 0.015)) * 0.5 + noise(vPos * 7.0 - uTime * 0.03) * 0.5;
    col += (n - 0.5) * 0.012;
    if (uHasMap > 0.5) {
      vec2 uv = vPos / (2.0 * uExtent) + 0.5;
      float e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
      float m = smoothstep(0.0, 0.035, e);
      vec3 img = texture2D(uMap, clamp(uv, 0.0, 1.0)).rgb;
      col = mix(col, img, m);
    }
    float edge = 1.0 - smoothstep(0.93, 1.0, r);
    gl_FragColor = linearToOutputTexel(vec4(col, edge));
  }
`;

export function Ocean({ map, extentUnits }: { map: THREE.Texture | null; extentUnits: number }) {
  const mat = useRef<THREE.ShaderMaterial>(null);
  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uRadius: { value: OCEAN_RADIUS },
      uDeep: { value: new THREE.Color('#040820') },
      uShallow: { value: new THREE.Color('#0b1745') },
      uMap: { value: null as THREE.Texture | null },
      uHasMap: { value: 0 },
      uExtent: { value: extentUnits },
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );
  useEffect(() => {
    uniforms.uMap.value = map;
    uniforms.uHasMap.value = map ? 1 : 0;
    uniforms.uExtent.value = extentUnits;
  }, [map, extentUnits, uniforms]);
  useFrame((_, dt) => {
    if (mat.current) mat.current.uniforms.uTime!.value += dt;
  });
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.002, 0]}>
      <circleGeometry args={[OCEAN_RADIUS, 160]} />
      <shaderMaterial ref={mat} vertexShader={oceanVertex} fragmentShader={oceanFragment} uniforms={uniforms} transparent depthWrite={false} />
    </mesh>
  );
}

const RING_LATS = [-50, -60, -70, -80];
const MERIDIANS = Array.from({ length: 12 }, (_, i) => -180 + i * 30);

function lonLabel(lon: number): string {
  if (lon === 0) return '0°';
  if (lon === -180 || lon === 180) return '180°';
  return `${Math.abs(lon)}°${lon < 0 ? 'W' : 'E'}`;
}

export function Graticule() {
  const geometry = useMemo(() => {
    const pos: number[] = [];
    const y = 0.001;
    for (const lat of [...RING_LATS, OUTER_LAT]) {
      for (let lon = -180; lon < 180; lon += 2) {
        const [x1, z1] = toScene(lat, lon);
        const [x2, z2] = toScene(lat, lon + 2);
        pos.push(x1, y, z1, x2, y, z2);
      }
    }
    for (const lon of MERIDIANS) {
      for (let lat = OUTER_LAT; lat > -86; lat -= 1) {
        const [x1, z1] = toScene(lat, lon);
        const [x2, z2] = toScene(lat - 1, lon);
        pos.push(x1, y, z1, x2, y, z2);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }, []);

  return (
    <group>
      <lineSegments geometry={geometry}>
        <lineBasicMaterial color={COLORS.grid} transparent opacity={0.35} depthWrite={false} />
      </lineSegments>
      {RING_LATS.map((lat) => {
        const [x, z] = toScene(lat, 22);
        return (
          <Html key={lat} position={[x, 0.01, z]} center zIndexRange={[1, 0]}>
            <span className="graticule-label">{Math.abs(lat)}°S</span>
          </Html>
        );
      })}
      {MERIDIANS.filter((l) => l % 60 === 0).map((lon) => {
        const [x, z] = toScene(OUTER_LAT + 1.8, lon);
        return (
          <Html key={lon} position={[x, 0.01, z]} center zIndexRange={[1, 0]}>
            <span className="graticule-label">{lonLabel(lon)}</span>
          </Html>
        );
      })}
    </group>
  );
}
