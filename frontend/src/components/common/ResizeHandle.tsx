interface Props {
  width: number;
  isDragging: boolean;
  isMobile?: boolean;
  onPointerDown: (e: React.PointerEvent) => void;
  onDoubleClick: () => void;
  label?: string;
}

export function ResizeHandle({ width, isDragging, isMobile = false, onPointerDown, onDoubleClick, label = 'Resize sidebar' }: Props) {
  if (isMobile) return null;

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={width}
      aria-label={label}
      className={`resize-handle${isDragging ? ' dragging' : ''}`}
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      title="Drag to resize sidebar · Double-click to reset"
    >
      <div className="resize-handle-bar" />
    </div>
  );
}
