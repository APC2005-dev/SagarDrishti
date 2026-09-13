import { Canvas } from '@react-three/fiber';
import { AnimatePresence, motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { CRS } from '../../constants';
import { useBasemapInfo, useEnvironmentField, useHistory } from '../../hooks/queries';
import { ensureBasemap, useBasemap } from './basemap';
import { SeaIceLayer, VectorFieldLayer } from './EnvironmentLayer';
import { LayerToggles } from './LayerToggles';
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
  showLayerToggles?: boolean;
  /** Overlay markup rendered above the canvas (HTML). */
  children?: React.ReactNode;
  /** Extra 3D layers rendered INSIDE the canvas, where R3F hooks work. */
  sceneChildren?: React.ReactNode;
  /** Fly the camera here instead of to the selected iceberg (scene X/Z). */
  focusOverride?: [number, number] | null;
}

function Legend({ hasForecasts }: { hasForecasts: boolean }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="map-overlay legend-container">
      <AnimatePresence mode="wait" initial={false}>
        {!isOpen ? (
          <motion.button
            key="legend-toggle"
            className="legend-toggle-btn"
            onClick={() => setIsOpen(true)}
            title="Expand legend"
            aria-label="Expand map legend"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            transition={{ duration: 0.15, ease: 'easeOut' }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="18 15 12 9 6 15" />
            </svg>
          </motion.button>
        ) : (
          <motion.div
            key="legend-panel"
            className="legend glass"
            initial={{ opacity: 0, scale: 0.9, originX: 0, originY: 1 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            <button
              className="legend-close-btn"
              onClick={() => setIsOpen(false)}
              title="Close legend"
              aria-label="Close map legend"
            >
              ✕
            </button>
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
              </>
            )}
            <div className="legend-row">
              <span className="legend-swatch sw-dep-port" />
              Departure port
            </div>
            <div className="legend-row">
              <span className="legend-swatch sw-dest-port" />
              Destination port
            </div>
            <div className="legend-row">
              <span className="legend-swatch sw-history" />
              Past track (selected)
            </div>
            <div className="legend-row dim">Wind / current: analysis fields (toggle, bottom right)</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function AntarcticScene({
  icebergs,
  forecasts = [],
  horizon = 7,
  riskMode,
  showLegend = true,
  showHistory = true,
  showLayerToggles = true,
  children,
  sceneChildren,
  focusOverride = null,
}: Props) {
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
  const layers = useUi((s) => s.layers);
  const windField = useEnvironmentField('wind', layers.wind);
  const currentField = useEnvironmentField('current', layers.current);
  const iceField = useEnvironmentField('sea_ice', layers.seaIce);
  const envLabels = [
    layers.wind && windField.data && `WIND ${windField.data.datasetId} · VALID ${windField.data.validDate}`,
    layers.current && currentField.data && `CURRENT (0.5 m) ${currentField.data.datasetId} · VALID ${currentField.data.validDate}`,
    layers.seaIce && iceField.data && `SEA ICE ${iceField.data.datasetId} · VALID ${iceField.data.validDate}`,
  ].filter(Boolean) as string[];
  useEffect(() => {
    if (basemapInfo.data) ensureBasemap(basemapInfo.data);
  }, [basemapInfo.data]);

  const plottable = useMemo(
    () => icebergs.filter((b): b is PlottableIceberg => b.latitude != null && b.longitude != null),
    [icebergs],
  );
  const selected = plottable.find((b) => b.icebergId === selectedId);
  const selectedFocus = useMemo(
    () => (selected ? toScene(selected.latitude, selected.longitude) : null),
    [selected],
  );
  const focus = focusOverride ?? selectedFocus;

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
        {layers.seaIce && iceField.data && <SeaIceLayer field={iceField.data} />}
        {layers.current && currentField.data && <VectorFieldLayer field={currentField.data} color="#6fe0d0" scale={0.35} maxLength={0.28} />}
        {layers.wind && windField.data && <VectorFieldLayer field={windField.data} color="#cfe3ff" scale={0.012} maxLength={0.22} />}
        {showHistory && history.data && <HistoryTrail observations={history.data.items} />}
        {layers.forecast && (
          <ForecastLayer forecasts={forecasts} horizon={horizon} selectedId={selectedId} riskMode={riskMode ?? storeRisk} />
        )}
        {layers.icebergs && (
          <IcebergLayer icebergs={plottable} selectedId={selectedId} hoveredId={hoveredId} onHover={hover} onSelect={select} />
        )}
        {sceneChildren}
        <CameraRig focus={focus} />
      </Canvas>

      <div className="map-overlay map-meta mono">
        <span>RENDER {CRS.render} · POLAR STEREOGRAPHIC (TRUE SCALE 71°S)</span>
        <span>DATA {CRS.data} · COASTLINE NATURAL EARTH 1:50m</span>
        {landError && <span style={{ color: 'var(--warn)' }}>COASTLINE UNAVAILABLE: {landError}</span>}
        {envLabels.map((l) => (
          <span key={l}>{l} · ANALYSIS, COARSENED</span>
        ))}
      </div>
      {showLayerToggles && (
        <div className="map-overlay" style={{ bottom: 14, right: 14 }}>
          <LayerToggles />
        </div>
      )}
      {showLegend && <Legend hasForecasts={forecasts.length > 0} />}
      {children}
    </div>
  );
}
