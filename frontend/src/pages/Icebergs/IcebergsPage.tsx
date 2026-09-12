import { AnimatePresence } from 'framer-motion';
import { useDeferredValue, useMemo, useState } from 'react';

import { QueryState } from '../../components/common/QueryState';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { HorizonSelector } from '../../components/forecast/HorizonSelector';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { IcebergDetailPanel } from '../../components/iceberg/IcebergDetailPanel';
import { IcebergTable } from '../../components/iceberg/IcebergTable';
import { useIcebergs, useLatestForecasts } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';
import { useUi } from '../../stores/uiStore';

export default function IcebergsPage() {
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();
  const selected = useUi((s) => s.selectedIcebergId);
  const horizon = useUi((s) => s.horizon);
  const setHorizon = useUi((s) => s.setHorizon);
  const riskCones = useUi((s) => s.riskCones);
  const setRiskCones = useUi((s) => s.setRiskCones);
  const [filter, setFilter] = useState('');
  const q = useDeferredValue(filter.trim().toUpperCase().replace(/[^A-Z0-9]/g, ''));

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
    storageKey: 'sagar_sidebar_icebergs',
    defaultWidth: 540,
    minWidth: 320,
  });

  const rows = useMemo(
    () => (icebergs.data?.items ?? []).filter((b) => !q || b.icebergId.includes(q)),
    [icebergs.data, q]
  );
  const stats = useMemo(() => {
    const items = icebergs.data?.items ?? [];
    return {
      total: items.length,
      forecast: items.filter((b) => b.hasForecast).length,
      stale: items.filter((b) => b.isStale).length,
    };
  }, [icebergs.data]);

  return (
    <div
      className={`split-resizable${isMobile ? ' mobile-stacked' : ''}`}
      style={
        isMobile
          ? undefined
          : { gridTemplateColumns: `${sidebarWidth}px 8px 1fr` }
      }
    >
      {/* Sidebar / table panel */}
      {(!isMobile || layoutMode !== 'map') && (
        <section
          className="split-left"
          aria-label="Tracked icebergs"
          style={
            isMobile
              ? {
                  height: layoutMode === 'content' ? '100%' : `${sidebarHeight}px`,
                  flexShrink: 0,
                }
              : undefined
          }
        >
          <div className="toolbar">
            <input
              className="search"
              placeholder="Filter by designator (e.g. A23A, D15)"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Filter icebergs"
            />
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', justifyContent: 'space-between' }}>
              <span className="mono dim" style={{ fontSize: 11 }}>
                {stats.total} tracked · {stats.forecast} with forecasts · {stats.stale} stale
              </span>

              {isMobile && (
                <div className="mobile-view-toggle">
                  <button
                    type="button"
                    className={`mobile-view-btn${layoutMode === 'content' ? ' active' : ''}`}
                    onClick={() => setLayoutMode('content')}
                  >
                    📋 Table
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
          </div>

          <QueryState
            query={icebergs}
            empty={(d) => d.items.length === 0}
            emptyText="No tracked icebergs yet. Run a USNIC ingestion (make ingest) to populate official positions."
          >
            {() => <IcebergTable rows={rows} />}
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
          label="Resize icebergs split"
        />
      )}

      {/* 3D Globe panel */}
      {(!isMobile || layoutMode !== 'content') && (
        <section
          className="split-right"
          aria-label="Map"
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
                ◫ Show Table
              </button>
            </div>
          )}

          <AntarcticScene icebergs={icebergs.data?.items ?? []} forecasts={forecasts.data ?? []} horizon={horizon}>
            <div className="map-overlay map-controls">
              <HorizonSelector value={horizon} onChange={setHorizon} />
              <div className="segmented" role="radiogroup" aria-label="Risk radius display">
                {(['selected', 'all', 'off'] as const).map((m) => (
                  <button key={m} className={`btn-ghost${riskCones === m ? ' active' : ''}`} onClick={() => setRiskCones(m)}>
                    P90 {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          </AntarcticScene>

          <AnimatePresence>{selected && <IcebergDetailPanel key={selected} icebergId={selected} />}</AnimatePresence>
        </section>
      )}
    </div>
  );
}
