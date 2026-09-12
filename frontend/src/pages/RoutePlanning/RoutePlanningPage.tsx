import { SectionTitle } from '../../components/common/Metric';
import { ResizeHandle } from '../../components/common/ResizeHandle';
import { Chip } from '../../components/common/StatusChip';
import { AntarcticScene } from '../../components/globe/AntarcticScene';
import { useIcebergs, useLatestForecasts } from '../../hooks/queries';
import { useResizableSidebar } from '../../hooks/useResizableSidebar';

const INPUTS = [
  { name: 'Iceberg forecast positions', status: 'available', detail: 'GET /api/v1/forecasts/latest — D+1…D+7 per iceberg with model version.' },
  { name: 'Danger zones', status: 'available', detail: 'p90 error radius around each forecast point (empirical, per horizon).' },
  { name: 'Sea-ice constraints', status: 'planned', detail: 'Requires a sea-ice concentration feed (not configured).' },
  { name: 'Vessel route', status: 'planned', detail: 'Waypoints input and ice-class parameters.' },
  { name: 'Routing engine', status: 'planned', detail: 'Not implemented in this release; no routes are computed or suggested.' },
] as const;

export default function RoutePlanningPage() {
  const icebergs = useIcebergs();
  const forecasts = useLatestForecasts();

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
    storageKey: 'sagar_sidebar_route_planning',
    defaultWidth: 440,
    minWidth: 320,
  });

  return (
    <div
      className={`split-resizable${isMobile ? ' mobile-stacked' : ''}`}
      style={
        isMobile
          ? undefined
          : { gridTemplateColumns: `${sidebarWidth}px 8px 1fr` }
      }
    >
      {/* Sidebar inputs panel */}
      {(!isMobile || layoutMode !== 'map') && (
        <section
          className="split-left scroll pad"
          aria-label="Route planning inputs"
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
              <h1>Route planning</h1>
            </div>

            {isMobile && (
              <div className="mobile-view-toggle">
                <button
                  type="button"
                  className={`mobile-view-btn${layoutMode === 'content' ? ' active' : ''}`}
                  onClick={() => setLayoutMode('content')}
                >
                  📋 Inputs
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

          <div className="note">
            This screen defines the inputs a maritime routing engine will consume. It shows the danger zones derived from the
            current iceberg forecasts; it does not compute routes.
          </div>
          <SectionTitle>Inputs</SectionTitle>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {INPUTS.map((i) => (
              <div key={i.name} className={i.status === 'available' ? 'panel card' : 'placeholder-card'}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <span>{i.name}</span>
                  <Chip tone={i.status === 'available' ? 'ok' : 'neutral'}>{i.status.toUpperCase()}</Chip>
                </div>
                <div className="dim" style={{ fontSize: 12, marginTop: 6 }}>{i.detail}</div>
              </div>
            ))}
          </div>
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
          label="Resize route planning split"
        />
      )}

      {/* 3D Map panel */}
      {(!isMobile || layoutMode !== 'content') && (
        <section
          className="split-right"
          aria-label="Route planning map"
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
                ◫ Show Inputs
              </button>
            </div>
          )}

          <AntarcticScene icebergs={icebergs.data?.items ?? []} forecasts={forecasts.data ?? []} horizon={7} riskMode="all" showHistory={false} />
        </section>
      )}
    </div>
  );
}
