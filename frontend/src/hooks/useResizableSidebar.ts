import { useCallback, useEffect, useRef, useState } from 'react';

interface UseResizableSidebarOptions {
  storageKey?: string;
  defaultWidth?: number;
  minWidth?: number;
  maxWidthRatio?: number;
}

export function useResizableSidebar({
  storageKey,
  defaultWidth = 500,
  minWidth = 320,
  maxWidthRatio = 0.8,
}: UseResizableSidebarOptions = {}) {
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (storageKey) {
      try {
        const saved = localStorage.getItem(storageKey);
        if (saved) {
          const parsed = Number(saved);
          if (!isNaN(parsed) && parsed >= minWidth) return parsed;
        }
      } catch {
        // ignore localStorage errors
      }
    }
    return defaultWidth;
  });

  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  const startResize = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      setIsDragging(true);
      isDraggingRef.current = true;

      const handlePointerMove = (moveEvent: PointerEvent) => {
        if (!isDraggingRef.current) return;
        const maxWidth = window.innerWidth * maxWidthRatio;
        const newWidth = Math.min(Math.max(moveEvent.clientX, minWidth), maxWidth);
        setSidebarWidth(newWidth);
      };

      const handlePointerUp = () => {
        setIsDragging(false);
        isDraggingRef.current = false;
        window.removeEventListener('pointermove', handlePointerMove);
        window.removeEventListener('pointerup', handlePointerUp);
        document.body.style.removeProperty('user-select');
        document.body.style.removeProperty('cursor');
      };

      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'col-resize';
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    },
    [minWidth, maxWidthRatio]
  );

  useEffect(() => {
    if (storageKey) {
      try {
        localStorage.setItem(storageKey, String(sidebarWidth));
      } catch {
        // ignore localStorage errors
      }
    }
  }, [sidebarWidth, storageKey]);

  const resetWidth = useCallback(() => {
    setSidebarWidth(defaultWidth);
  }, [defaultWidth]);

  return {
    sidebarWidth,
    isDragging,
    startResize,
    resetWidth,
  };
}
