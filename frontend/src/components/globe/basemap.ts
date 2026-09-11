/**
 * EPSG:3031 basemap mosaic (NASA Blue Marble shaded relief + bathymetry).
 *
 * Tiles come from the backend proxy (/api/v1/basemap/...), never from NASA
 * directly. They are stitched into one square canvas covering the full GIBS
 * extent (±4 194 304 m): level 1 first (fast, coarse), then level 2 (~2 km/px)
 * drawn over it. One texture is shared by every scene on every page.
 *
 * Because the imagery, the coastline and the iceberg positions are all in the
 * same projection, they register without any warping.
 */
import * as THREE from 'three';
import { create } from 'zustand';

import { api } from '../../api/client';
import type { BasemapInfo } from '../../types/api';
import { SCENE_UNITS_PER_METRE } from '../../utils/projection';

export type BasemapStatus = 'idle' | 'loading' | 'ready' | 'partial' | 'unavailable' | 'disabled';

interface BasemapState {
  status: BasemapStatus;
  texture: THREE.CanvasTexture | null;
  level: number | null;
  failedTiles: number;
  /** Half-width of the mosaic in scene units. */
  extentUnits: number;
  attribution: string | null;
}

export const useBasemap = create<BasemapState>(() => ({
  status: 'idle',
  texture: null,
  level: null,
  failedTiles: 0,
  extentUnits: 4.194304,
  attribution: null,
}));

const LEVELS = [1, 2];
let startedFor: string | null = null;

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(url));
    img.src = url;
  });
}

async function drawLevel(ctx: CanvasRenderingContext2D, layer: string, z: number, size: number) {
  const n = 2 ** (z + 1);
  const px = size / n;
  let ok = 0;
  let failed = 0;
  const jobs: Promise<void>[] = [];
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      jobs.push(
        loadImage(api.basemapTileUrl(layer, z, row, col))
          .then((img) => {
            ctx.drawImage(img, col * px, row * px, px, px);
            ok++;
          })
          .catch(() => {
            failed++;
          }),
      );
    }
  }
  await Promise.all(jobs);
  return { ok, failed };
}

export function ensureBasemap(info: BasemapInfo): void {
  if (!info.enabled) {
    useBasemap.setState({ status: 'disabled' });
    return;
  }
  const layer = info.defaultLayer;
  if (startedFor === layer) return;
  startedFor = layer;

  const size = info.tileSize * 2 ** (Math.max(...LEVELS) + 1);
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    useBasemap.setState({ status: 'unavailable' });
    return;
  }
  ctx.fillStyle = '#060b26';
  ctx.fillRect(0, 0, size, size);

  const extentUnits = info.extentM * SCENE_UNITS_PER_METRE;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  // Land caps carry UVs in scene units (x, y_map); map them onto the mosaic square.
  texture.repeat.set(1 / (2 * extentUnits), 1 / (2 * extentUnits));
  texture.offset.set(0.5, 0.5);

  useBasemap.setState({ status: 'loading', extentUnits, attribution: info.attribution });

  void (async () => {
    let failedTotal = 0;
    for (const z of LEVELS) {
      const { ok, failed } = await drawLevel(ctx, layer, z, size);
      failedTotal = failed;
      if (ok === 0) continue;
      texture.needsUpdate = true;
      useBasemap.setState({ texture, level: z, status: failed ? 'partial' : 'ready', failedTiles: failed });
    }
    if (!useBasemap.getState().texture) {
      useBasemap.setState({ status: 'unavailable', failedTiles: failedTotal });
      startedFor = null; // allow a retry on the next scene mount
    }
  })();
}
