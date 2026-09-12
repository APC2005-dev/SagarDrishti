import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useMemo } from 'react';
import { Navigate, useNavigate, useParams } from 'react-router-dom';

import { KV, SectionTitle } from '../../components/common/Metric';
import { QueryState } from '../../components/common/QueryState';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { Chip, StaleChip } from '../../components/common/StatusChip';
import { HorizonSelector } from '../../components/forecast/HorizonSelector';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { IcebergDetailPanel } from '../../components/iceberg/IcebergDetailPanel';
import { HORIZON_ROUTES } from '../../constants';
import { useIcebergs, useLatestForecasts, useModels } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';
import { useUi } from '../../stores/uiStore';
import type { ForecastHorizon, ForecastSet } from '../../types/api';
import { fmtDate, fmtDateTime, fmtKm } from '../../utils/format';
import { bearingDeg, haversineKm } from '../../utils/projection';

const PARAM_TO_H: Record<string, ForecastHorizon> = { '1d': 1, '3d': 3, '7d': 7 };

function atHorizon(f: ForecastSet, h: number) {
  const p = f.points.find((x) => x.horizonDays === h);
  if (!p) return null;
  return {
    point: p,
    km: haversineKm(f.anchor.latitude, f.anchor.longitude, p.predictedLatitude, p.predictedLongitude),
    bearing: bearingDeg(f.anchor.latitude, f.anchor.longitude, p.predictedLatitude, p.predictedLongitude),
  };
}

export default function ForecastPage() {
  const { horizon: param = '7d' } = useParams();
  const horizon = PARAM_TO_H[param];
  const navigate = useNavigate();
  const setHorizon = useUi((s) => s.setHorizon);
  const select = useUi((s) => s.select);
  const selected = useUi((s) => s.selectedIcebergId);
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();
  const models = useModels();

  const {
    sidebarWidth,
    sidebarHeight,
    isDragging,
    isMobile,
    layoutMode,
    setLayoutMode,
    startResize,
    resetSize,
  } = useResizableSidebar({
    storageKey: 'sagar_sidebar_forecast',
    defaultWidth: 440,
    minWidth: 320,
  });

  useEffect(() => {
    if (horizon) setHorizon(horizon);
  }, [horizon, setHorizon]);

  const rows = useMemo(
    () =>
      (forecasts.data ?? [])
        .map((f) => ({ f, h: atHorizon(f, horizon ?? 7) }))
        .sort((a, b) => (b.h?.km ?? 0) - (a.h?.km ?? 0)),
    [forecasts.data, horizon]
  );
  const versions = useMemo(() => [...new Set((forecasts.data ?? []).map((f) => f.modelVersion))], [forecasts.data]);
  const champion = models.data?.find((m) => m.status === 'deployed');
  const issued = useMemo(() => {
    const t = (forecasts.data ?? []).map((f) => f.generatedAt).sort();
    return t.length ? { first: t[0]!, last: t[t.length - 1]! } : null;
  }, [forecasts.data]);

  if (!horizon) return <Navigate to="/forecast/7d" replace />;

  return (
    <div
      className={`split-resizable${isMobile ? ' mobile-stacked' : ''}`}
      style={
        isMobile
          ? undefined
          : { gridTemplateColumns: `${sidebarWidth}px 8px 1fr` }
      }
    >
      {/* Sidebar summary panel */}
      {(!isMobile || layoutMode !== 'map') && (
        <section
          className="split-left scroll pad"
          aria-label="Forecast summary"
          style={
            isMobile
              ? {
                  height: layoutMode === 'content' ? '100%' : `${sidebarHeight}px`,
                  flexShrink: 0,
                }
              : undefined
          }
        >
          <div className="page-head">
            <div>
              <span className="label">Trajectory forecast</span>
              <AnimatePresence mode="wait" initial={false}>
                <motion.h1
                  key={horizon}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.18 }}
                >
                  {horizon}-day view · D+1…D+{horizon}
                </motion.h1>
              </AnimatePresence>
            </div>

            {isMobile && (
              <div className="mobile-view-toggle">
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'content' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('content')}
                >
                  📋 List
                </button>
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'split' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('split')}
                >
                  ◫ Split
                </button>
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'map' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('map')}
                >
                  🌐 Map
                </button>
              </div>
            )}
          </div>

          <HorizonSelector value={horizon} onChange={(h) => navigate(HORIZON_ROUTES[h])} />
          <div className="note" style={{ marginTop: 12 }}>
            One GRU inference per iceberg produces D+1…D+7. The 1/3/7-day views filter that same run — they are not separate
            models. Forecast points are predictions, never observations.
          </div>

          <SectionTitle>Model</SectionTitle>
          <div className="kv-grid">
            <KV k="Champion" v={champion ? `${champion.version} · ${champion.architectureVersion}` : 'none deployed'} />
            <KV k="Feature schema" v={champion ? `${champion.featureSchemaVersion} (${champion.modelType})` : '—'} />
            <KV k="Versions in view" v={versions.join(', ') || '—'} />
            <KV k="Issued" v={issued ? fmtDateTime(issued.last) : '—'} />
            <KV
              k={`Benchmark D+${horizon} MAE`}
              v={fmtKm(champion ? (horizon === 1 ? champion.day1Error : horizon === 3 ? champion.day3Error : champion.day7Error) : null)}
            />
          </div>

          <SectionTitle right={<span className="mono dim">{rows.length} sets</span>}>
            Forecasts at D+{horizon}
          </SectionTitle>
          <QueryState
            query={forecasts}
            empty={(d) => d.length === 0}
            emptyText="No forecasts available. Forecasts are generated after each ingestion when a model is deployed."
          >
            {() => (
              <div className="table-wrap">
                <table className="data clickable">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>FROM FIX</th>
                      <th className="num">MOVE</th>
                      <th className="num">BRG</th>
                      <th className="num">P90</th>
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ f, h }) => (
                      <tr
                        key={f.forecastSetId}
                        className={f.icebergId === selected ? 'selected' : undefined}
                        onClick={() => select(f.icebergId === selected ? null : f.icebergId)}
                      >
                        <td className="designator">{f.icebergId}</td>
                        <td className="mono muted">{fmtDate(f.latestObservationDate)}</td>
                        <td className="mono num">{fmtKm(h?.km)}</td>
                        <td className="mono num muted">{h ? `${h.bearing.toFixed(0)}°` : '—'}</td>
                        <td className="mono num muted">{fmtKm(h?.point.riskRadiusKmP90)}</td>
                        <td>{f.isStale ? <StaleChip /> : <Chip tone="forecast">{f.modelVersion}</Chip>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </QueryState>
        </section>
      )}

      {/* Resize Handle */}
      {(!isMobile || layoutMode === 'split') && (
        <ResizeHandle
          width={sidebarWidth}
          height={sidebarHeight}
          isDragging={isDragging}
          isMobile={isMobile}
          layoutMode={layoutMode}
          onLayoutModeChange={setLayoutMode}
          onPointerDown={startResize}
          onDoubleClick={resetSize}
          label="Resize forecast split"
        />
      )}

      {/* 3D Map panel */}
      {(!isMobile || layoutMode !== 'content') && (
        <section
          className="split-right"
          aria-label="Forecast map"
          style={
            isMobile
              ? {
                  flex: 1,
                  minHeight: 0,
                  height: layoutMode === 'map' ? '100%' : 'auto',
                }
              : undefined
          }
        >
          {isMobile && layoutMode === 'map' && (
            <div className="mobile-floating-toggle">
              <button
                type="button"
                className="mobile-floating-btn"
                onClick={() => setLayoutMode('split')}
              >
                ◫ Show Forecasts
              </button>
            </div>
          )}

          <AntarcticScene icebergs={icebergs.data?.items ?? []} forecasts={forecasts.data ?? []} horizon={horizon} />
          <AnimatePresence>{selected && <IcebergDetailPanel key={selected} icebergId={selected} />}</AnimatePresence>
        </section>
      )}
    </div>
  );
}
