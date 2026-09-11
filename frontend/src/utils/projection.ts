/**
 * EPSG:3031 — WGS84 Antarctic Polar Stereographic (true scale at 71°S, lon0 = 0).
 *
 * The API serves EPSG:4326 degrees; the scene is rendered in EPSG:3031 so it
 * matches the projection the model and the backend use (pyproj). Snyder (1987)
 * eq. 21-33/21-34 in the south-polar form: x = ρ·sin λ, y = ρ·cos λ.
 *
 * Scene mapping: 1 scene unit = 1000 km. Map +x -> scene +X, map +y -> scene −Z,
 * so the Greenwich meridian points "up" the screen as in standard 3031 maps.
 */

const A = 6378137.0;
const E = 0.0818191908426215;
const PHI_C = (71 * Math.PI) / 180;
const DEG = Math.PI / 180;

function tFn(phi: number): number {
  const s = E * Math.sin(phi);
  return Math.tan(Math.PI / 4 - phi / 2) / Math.pow((1 - s) / (1 + s), E / 2);
}

const T_C = tFn(PHI_C);
const M_C = Math.cos(PHI_C) / Math.sqrt(1 - E * E * Math.sin(PHI_C) ** 2);

/** WGS84 degrees -> EPSG:3031 metres [x, y]. */
export function toPolar(lat: number, lon: number): [number, number] {
  const phi = -lat * DEG; // south pole -> positive
  const rho = (A * M_C * tFn(phi)) / T_C;
  const lam = lon * DEG;
  return [rho * Math.sin(lam), rho * Math.cos(lam)];
}

/** Point scale factor k at a latitude (for sizing km-based radii in the projected plane). */
export function scaleFactor(lat: number): number {
  const phi = -lat * DEG;
  const m = Math.cos(phi) / Math.sqrt(1 - E * E * Math.sin(phi) ** 2);
  const rho = (A * M_C * tFn(phi)) / T_C;
  return rho / (A * m);
}

export const SCENE_UNITS_PER_METRE = 1 / 1_000_000;

/** WGS84 degrees -> scene [X, Z] on the ground plane. */
export function toScene(lat: number, lon: number): [number, number] {
  const [x, y] = toPolar(lat, lon);
  return [x * SCENE_UNITS_PER_METRE, -y * SCENE_UNITS_PER_METRE];
}

/** A distance in km at a given latitude -> scene units on the projected plane. */
export function kmToScene(km: number, lat: number): number {
  return km * 1000 * scaleFactor(lat) * SCENE_UNITS_PER_METRE;
}

export function formatLat(lat: number | null | undefined, digits = 2): string {
  if (lat == null) return '—';
  return `${Math.abs(lat).toFixed(digits)}°${lat <= 0 ? 'S' : 'N'}`;
}

export function formatLon(lon: number | null | undefined, digits = 2): string {
  if (lon == null) return '—';
  return `${Math.abs(lon).toFixed(digits)}°${lon < 0 ? 'W' : 'E'}`;
}

/** Great-circle distance in km (same sphere as the backend haversine). */
export function haversineKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371.0088;
  const dLat = (lat2 - lat1) * DEG;
  const dLon = (lon2 - lon1) * DEG;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * DEG) * Math.cos(lat2 * DEG) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(a)));
}

/** Initial bearing in degrees true (0 = north). */
export function bearingDeg(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const p1 = lat1 * DEG;
  const p2 = lat2 * DEG;
  const dl = (lon2 - lon1) * DEG;
  const y = Math.sin(dl) * Math.cos(p2);
  const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
  return ((Math.atan2(y, x) / DEG) + 360) % 360;
}
