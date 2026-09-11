import { Canvas } from '@react-three/fiber';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { CRS } from '../../constants';
import { useBasemapInfo, useHistory } from '../../hooks/queries';
import { ensureBasemap, useBasemap } from './basemap';
import { type RiskConeMode, useUi } from '../../stores/uiStore';
import type { ForecastSet, IcebergSummary } from '../../types/api';
import { toScene } from '../../utils/projection';
import { CameraRig } from './CameraRig';
import { ForecastLayer } from './ForecastLayer';
import { HistoryTrail } from './HistoryTrail';
import { IcebergLayer, type PlottableIceberg } from './IcebergLayer';
import { Land } from './Land';
import { Graticule, Ocean } from './OceanAndGrid';

interface Props {
  icebergs: IcebergSummary[];
  forecasts?: ForecastSet[];
  /** Horizon to display; forecasts are always the full 7-day run filtered to 1..horizon. */
  horizon?: number;
  riskMode?: RiskConeMode;
  showLegend?: boolean;
  showHistory?: boolean;
  children?: React.ReactNode;
}

function Legend({ hasForecasts }: { hasForecasts: boolean }) {
  return (
    <div className="map-overlay legend glass">
      <div className="legend-row">
        <span className="legend-swatch sw-official" />
        OFFICIAL position (USNIC)
      </div>
      <div className="legend-row">
        <span className="legend-swatch sw-stale" />
        OFFICIAL · stale (&gt; threshold)
      </div>
      {hasForecasts && (
        <>
          <div className="legend-row">
            <span className="legend-swatch sw-forecast" />
            FORECAST position D+n (GRU)
          </div>
          <div className="legend-row">
            <span className="legend-swatch sw-line" />
            Forecast trajectory
          </div>
          <div className="legend-row">
            <span className="legend-swatch sw-risk" />
            p90 error radius
          </div>
        </>
      )}
      <div className="legend-row">
        <span className="legend-swatch sw-history" />
        Past track (selected)
      </div>
    </div>
  );
}

export function AntarcticScene({ icebergs, forecasts = [], horizon = 7, riskMode, showLegend = true, showHistory = true, children }: Props) {
  const selectedId = useUi((s) => s.selectedIcebergId);
  const hoveredId = useUi((s) => s.hoveredIcebergId);
  const select = useUi((s) => s.select);
  const hover = useUi((s) => s.hover);
  const storeRisk = useUi((s) => s.riskCones);
  const [landError, setLandError] = useState<string | null>(null);
  const onLandError = useCallback((m: string) => setLandError(m), []);
  const history = useHistory(showHistory ? selectedId : null, true);
  const basemapInfo = useBasemapInfo();
  const basemap = useBasemap();
  useEffect(() => {
    if (basemapInfo.data) ensureBasemap(basemapInfo.data);
  }, [basemapInfo.data]);
  const basemapLabel =
    basemap.status === 'ready' || basemap.status === 'partial'
      ? `BASEMAP NASA BLUE MARBLE (STATIC COMPOSITE — NOT CURRENT CONDITIONS)${basemap.status === 'partial' ? ` · ${basemap.failedTiles} TILES MISSING` : ''}`
      : basemap.status === 'loading' || basemap.status === 'idle'
        ? 'BASEMAP LOADING…'
        : basemap.status === 'disabled'
          ? 'BASEMAP DISABLED'
          : 'BASEMAP UNAVAILABLE — COASTLINE ONLY';

  const plottable = useMemo(
    () => icebergs.filter((b): b is PlottableIceberg => b.latitude != null && b.longitude != null),
    [icebergs],
  );
  const selected = plottable.find((b) => b.icebergId === selectedId);
  const focus = useMemo(() => (selected ? toScene(selected.latitude, selected.longitude) : null), [selected]);

  return (
    <div className="scene-root" style={{ position: 'absolute', inset: 0 }}>
      <Canvas
        camera={{ position: [0, 7.4, 6.2], fov: 38, near: 0.01, far: 100 }}
        dpr={[1, 2]}
        gl={{ antialias: true, powerPreference: 'high-performance' }}
        onPointerMissed={() => select(null)}
      >
        <color attach="background" args={['#030615']} />
        <fog attach="fog" args={['#030615', 12, 26]} />
        <ambientLight intensity={0.65} />
        <directionalLight position={[-4, 8, 3]} intensity={1.1} />
        <Ocean map={basemap.texture} extentUnits={basemap.extentUnits} />
        <Graticule />
        <Land map={basemap.texture} onError={onLandError} />
        {showHistory && history.data && <HistoryTrail observations={history.data.items} />}
        <ForecastLayer forecasts={forecasts} horizon={horizon} selectedId={selectedId} riskMode={riskMode ?? storeRisk} />
        <IcebergLayer icebergs={plottable} selectedId={selectedId} hoveredId={hoveredId} onHover={hover} onSelect={select} />
        <CameraRig focus={focus} />
      </Canvas>

      <div className="map-overlay map-meta mono">
        <span>RENDER {CRS.render} · POLAR STEREOGRAPHIC (TRUE SCALE 71°S)</span>
        <span>DATA {CRS.data} · COASTLINE NATURAL EARTH 1:50m</span>
        <span style={basemap.status === 'unavailable' ? { color: 'var(--warn)' } : undefined}>{basemapLabel}</span>
        <span>
          {plottable.length} OFFICIAL POSITIONS{forecasts.length ? ` · ${forecasts.length} FORECAST SETS · D+1…D+${horizon}` : ''}
        </span>
        {landError && <span style={{ color: 'var(--warn)' }}>COASTLINE UNAVAILABLE: {landError}</span>}
      </div>
      {showLegend && <Legend hasForecasts={forecasts.length > 0} />}
      {children}
    </div>
  );
}
