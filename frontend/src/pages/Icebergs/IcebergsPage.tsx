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

  const { sidebarWidth, isDragging, startResize, resetWidth } = useResizableSidebar({
    storageKey: 'sagar_sidebar_width_icebergs',
    defaultWidth: 540,
    minWidth: 320,
  });

  const rows = useMemo(() => (icebergs.data?.items ?? []).filter((b) => !q || b.icebergId.includes(q)), [icebergs.data, q]);
  const stats = useMemo(() => {
    const items = icebergs.data?.items ?? [];
    return { total: items.length, forecast: items.filter((b) => b.hasForecast).length, stale: items.filter((b) => b.isStale).length };
  }, [icebergs.data]);

  return (
    <div className="split-resizable" style={{ gridTemplateColumns: `${sidebarWidth}px 6px 1fr` }}>
      <section className="split-left" aria-label="Tracked icebergs">
        <div className="toolbar">
          <input
            className="search"
            placeholder="Filter by designator (e.g. A23A, D15)"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            aria-label="Filter icebergs"
          />
          <span className="mono dim" style={{ fontSize: 11 }}>
            {stats.total} tracked · {stats.forecast} with forecasts · {stats.stale} stale
          </span>
        </div>
        <QueryState
          query={icebergs}
          empty={(d) => d.items.length === 0}
          emptyText="No tracked icebergs yet. Run a USNIC ingestion (make ingest) to populate official positions."
        >
          {() => <IcebergTable rows={rows} />}
        </QueryState>
      </section>
      <ResizeHandle
        width={sidebarWidth}
        isDragging={isDragging}
        onPointerDown={startResize}
        onDoubleClick={resetWidth}
        label="Resize icebergs sidebar"
      />
      <section className="split-right" aria-label="Map">
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
    </div>
  );
}
