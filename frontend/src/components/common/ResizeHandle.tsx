interface Props {
  width?: number;
  height?: number;
  isDragging: boolean;
  isMobile?: boolean;
  onPointerDown: (e: React.PointerEvent) => void;
  onDoubleClick: () => void;
  label?: string;
}

export function ResizeHandle({
  width,
  height,
  isDragging,
  isMobile = false,
  onPointerDown,
  onDoubleClick,
  label = 'Resize panel split',
}: Props) {
  return (
    <div
      role="separator"
      aria-orientation={isMobile ? 'horizontal' : 'vertical'}
      aria-valuenow={isMobile ? height : width}
      aria-label={label}
      tabIndex={0}
      className={`resize-handle ${isMobile ? 'horizontal' : 'vertical'}${isDragging ? ' dragging' : ''}`}
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      title={
        isMobile
          ? 'Drag up/down to adjust split height · Double-tap to reset'
          : 'Drag left/right to resize sidebar · Double-click to reset'
      }
    >
      <div className="resize-handle-bar" />
    </div>
  );
}
