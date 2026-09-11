import { useFrame } from '@react-three/fiber';
import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';

import { COLORS } from '../../constants';
import type { RiskConeMode } from '../../stores/uiStore';
import type { ForecastSet } from '../../types/api';
import { kmToScene, toScene } from '../../utils/projection';
import { MARKER_Y } from './IcebergLayer';

interface Props {
  forecasts: ForecastSet[];
  /** Target horizon (1, 3 or 7). The layer animates smoothly towards it. */
  horizon: number;
  selectedId: string | null;
  riskMode: RiskConeMode;
}

const LINE_Y = MARKER_Y - 0.005;

/*
 * Every forecast set is drawn from a single geometry. Each vertex carries its
 * horizon index (aD: 0 = official anchor, 1..7 = D+1..D+7) and set index. The
 * shader discards fragments beyond the animated progress uniform, so switching
 * 1 -> 3 -> 7 days extends/retracts the same trajectories instead of reloading.
 */
const lineVertex = /* glsl */ `
  attribute float aD;
  attribute float aSet;
  attribute float aLen;
  varying float vD;
  varying float vSet;
  varying float vLen;
  void main() {
    vD = aD; vSet = aSet; vLen = aLen;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;
const lineFragment = /* glsl */ `
  uniform float uProgress;
  uniform float uTime;
  uniform float uSelected;
  uniform float uDashed;
  uniform vec3 uColor;
  varying float vD;
  varying float vSet;
  varying float vLen;
  void main() {
    if (vD > uProgress + 0.0001) discard;
    if (uDashed > 0.5 && fract(vLen * 55.0 - uTime * 0.6) > 0.55) discard;
    bool anySel = uSelected >= 0.0;
    bool isSel = abs(vSet - uSelected) < 0.5;
    float a = anySel ? (isSel ? 1.0 : 0.22) : 0.75;
    // fade out towards the far horizon: uncertainty grows with lead time
    a *= mix(1.0, 0.55, clamp(vD / 7.0, 0.0, 1.0));
    gl_FragColor = vec4(uColor, a);
  }
`;

function makeLineMaterial(color: string, dashed: boolean) {
  return new THREE.ShaderMaterial({
    vertexShader: lineVertex,
    fragmentShader: lineFragment,
    uniforms: {
      uProgress: { value: 0 },
      uTime: { value: 0 },
      uSelected: { value: -1 },
      uDashed: { value: dashed ? 1 : 0 },
      uColor: { value: new THREE.Color(color) },
    },
    transparent: true,
    depthWrite: false,
  });
}

const tmp = new THREE.Object3D();
const cPoint = new THREE.Color(COLORS.forecast);
const cDim = new THREE.Color(COLORS.forecast).multiplyScalar(0.35);
const cSel = new THREE.Color('#ffffff');

export function ForecastLayer({ forecasts, horizon, selectedId, riskMode }: Props) {
  const progress = useRef(0);
  const points = useRef<THREE.InstancedMesh>(null);
  const lineMat = useMemo(() => makeLineMaterial(COLORS.forecastLine, true), []);
  const riskMat = useMemo(() => makeLineMaterial(COLORS.risk, false), []);
  const selectedIdx = forecasts.findIndex((f) => f.icebergId === selectedId);

  const scenePts = useMemo(
    () =>
      forecasts.map((f) => [
        toScene(f.anchor.latitude, f.anchor.longitude),
        ...f.points.map((p) => toScene(p.predictedLatitude, p.predictedLongitude)),
      ]),
    [forecasts],
  );

  const lineGeom = useMemo(() => {
    const pos: number[] = [];
    const d: number[] = [];
    const set: number[] = [];
    const len: number[] = [];
    scenePts.forEach((chain, s) => {
      let acc = 0;
      for (let i = 0; i < chain.length - 1; i++) {
        const [x1, z1] = chain[i]!;
        const [x2, z2] = chain[i + 1]!;
        const segLen = Math.hypot(x2 - x1, z2 - z1);
        pos.push(x1, LINE_Y, z1, x2, LINE_Y, z2);
        d.push(i, i + 1);
        set.push(s, s);
        len.push(acc, acc + segLen);
        acc += segLen;
      }
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aD', new THREE.Float32BufferAttribute(d, 1));
    g.setAttribute('aSet', new THREE.Float32BufferAttribute(set, 1));
    g.setAttribute('aLen', new THREE.Float32BufferAttribute(len, 1));
    return g;
  }, [scenePts]);

  const riskGeom = useMemo(() => {
    const pos: number[] = [];
    const d: number[] = [];
    const set: number[] = [];
    const SEG = 48;
    forecasts.forEach((f, s) => {
      if (riskMode === 'off' || (riskMode === 'selected' && s !== selectedIdx)) return;
      f.points.forEach((p, i) => {
        if (p.riskRadiusKmP90 == null) return; // no empirical radius for this horizon -> no cone drawn
        const r = kmToScene(p.riskRadiusKmP90, p.predictedLatitude);
        const [cx, cz] = scenePts[s]![i + 1]!;
        for (let k = 0; k < SEG; k++) {
          const a1 = (k / SEG) * Math.PI * 2;
          const a2 = ((k + 1) / SEG) * Math.PI * 2;
          pos.push(cx + r * Math.cos(a1), LINE_Y, cz + r * Math.sin(a1), cx + r * Math.cos(a2), LINE_Y, cz + r * Math.sin(a2));
          d.push(p.horizonDays, p.horizonDays);
          set.push(s, s);
        }
      });
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('aD', new THREE.Float32BufferAttribute(d, 1));
    g.setAttribute('aSet', new THREE.Float32BufferAttribute(set, 1));
    g.setAttribute('aLen', new THREE.Float32BufferAttribute(new Array(d.length).fill(0), 1));
    return g;
  }, [forecasts, scenePts, riskMode, selectedIdx]);

  useEffect(() => () => lineGeom.dispose(), [lineGeom]);
  useEffect(() => () => riskGeom.dispose(), [riskGeom]);
  useEffect(() => () => (lineMat.dispose(), riskMat.dispose()), [lineMat, riskMat]);

  const nPoints = forecasts.reduce((n, f) => n + f.points.length, 0);

  const layoutPoints = (p: number) => {
    const m = points.current;
    if (!m) return;
    let k = 0;
    forecasts.forEach((f, s) => {
      f.points.forEach((pt, i) => {
        const [x, z] = scenePts[s]![i + 1]!;
        const reveal = THREE.MathUtils.clamp(p - (pt.horizonDays - 1), 0, 1);
        const sc = reveal * (s === selectedIdx ? 1.35 : 1);
        tmp.position.set(x, LINE_Y + 0.001, z);
        tmp.rotation.set(-Math.PI / 2, 0, 0);
        tmp.scale.set(sc, sc, sc);
        tmp.updateMatrix();
        m.setMatrixAt(k, tmp.matrix);
        m.setColorAt(k, s === selectedIdx ? cSel : selectedIdx >= 0 ? cDim : cPoint);
        k++;
      });
    });
    m.count = k;
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  };

  useEffect(() => {
    lineMat.uniforms.uSelected!.value = selectedIdx;
    riskMat.uniforms.uSelected!.value = selectedIdx;
    layoutPoints(progress.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIdx, forecasts, scenePts]);

  useFrame((_, dt) => {
    lineMat.uniforms.uTime!.value += dt;
    const target = horizon;
    const p = progress.current;
    if (Math.abs(target - p) > 0.001) {
      progress.current = p + (target - p) * Math.min(1, dt * 3.2);
      lineMat.uniforms.uProgress!.value = progress.current;
      riskMat.uniforms.uProgress!.value = progress.current;
      layoutPoints(progress.current);
    }
  });

  if (!forecasts.length) return null;
  return (
    <group>
      <lineSegments geometry={lineGeom} material={lineMat} />
      <lineSegments geometry={riskGeom} material={riskMat} />
      <instancedMesh ref={points} args={[undefined, undefined, Math.max(nPoints, 1)]} frustumCulled={false}>
        <ringGeometry args={[0.016, 0.025, 20]} />
        <meshBasicMaterial toneMapped={false} side={THREE.DoubleSide} transparent opacity={0.95} />
      </instancedMesh>
    </group>
  );
}
