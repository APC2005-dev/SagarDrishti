interface Props {
  width: number;
  isDragging: boolean;
  onPointerDown: (e: React.PointerEvent) => void;
  onDoubleClick: () => void;
  label?: string;
}

export function ResizeHandle({ width, isDragging, onPointerDown, onDoubleClick, label = 'Resize sidebar' }: Props) {
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
