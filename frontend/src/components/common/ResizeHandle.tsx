import type { LayoutMode } from '../../hooks/useResizableSidebar';

interface Props {
  width?: number;
  height?: number;
  isDragging: boolean;
  isMobile?: boolean;
  layoutMode?: LayoutMode;
  onLayoutModeChange?: (mode: LayoutMode) => void;
  onPointerDown: (e: React.PointerEvent) => void;
  onDoubleClick: () => void;
  label?: string;
}

export function ResizeHandle({
  width,
  height,
  isDragging,
  isMobile = false,
  layoutMode = 'split',
  onLayoutModeChange,
  onPointerDown,
  onDoubleClick,
  label = 'Resize panel split',
}: Props) {
  if (isMobile) {
    return (
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-valuenow={height}
        aria-label={label}
        tabIndex={0}
        className={`resize-handle horizontal${isDragging ? ' dragging' : ''}`}
        onPointerDown={onPointerDown}
        onDoubleClick={onDoubleClick}
        title="Drag up/down to adjust split height · Double-tap to reset"
      >
        <div className="resize-handle-content">
          {onLayoutModeChange && (
            <button
              type="button"
              className="resize-snap-btn"
              onClick={(e) => {
                e.stopPropagation();
                onLayoutModeChange(layoutMode === 'content' ? 'split' : 'content');
              }}
              title={layoutMode === 'content' ? 'Restore split' : 'Maximize content panel'}
            >
              {layoutMode === 'content' ? '◫ Split' : '▲ Maximize'}
            </button>
          )}

          <div className="resize-grip-indicator">
            <span className="resize-grip-pill" />
          </div>

          {onLayoutModeChange && (
            <button
              type="button"
              className="resize-snap-btn"
              onClick={(e) => {
                e.stopPropagation();
                onLayoutModeChange(layoutMode === 'map' ? 'split' : 'map');
              }}
              title={layoutMode === 'map' ? 'Restore split' : 'Maximize 3D map'}
            >
              {layoutMode === 'map' ? '◫ Split' : '▼ 3D Map'}
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={width}
      aria-label={label}
      tabIndex={0}
      className={`resize-handle vertical${isDragging ? ' dragging' : ''}`}
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      title="Drag left/right to resize sidebar · Double-click to reset"
    >
      <div className="resize-handle-bar" />
    </div>
  );
}
