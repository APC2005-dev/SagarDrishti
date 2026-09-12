import { useCallback, useEffect, useRef, useState } from 'react';

export type LayoutMode = 'split' | 'content' | 'map';

interface UseResizableSidebarOptions {
  storageKey?: string;
  defaultWidth?: number;
  minWidth?: number;
  maxWidthRatio?: number;
  defaultHeight?: number;
  minHeight?: number;
  maxHeightRatio?: number;
  mobileBreakpoint?: number;
}

export function useResizableSidebar({
  storageKey,
  defaultWidth = 520,
  minWidth = 320,
  maxWidthRatio = 0.75,
  defaultHeight,
  minHeight = 120,
  maxHeightRatio = 0.82,
  mobileBreakpoint = 768,
}: UseResizableSidebarOptions = {}) {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < mobileBreakpoint : false
  );

  const [layoutMode, setLayoutMode] = useState<LayoutMode>('split');

  // Width for horizontal layout (desktop/tablet)
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        const saved = localStorage.getItem(`${storageKey}_w`);
        if (saved) {
          const parsed = Number(saved);
          if (!isNaN(parsed) && parsed >= minWidth) return parsed;
        }
      } catch {
        // ignore
      }
    }
    return defaultWidth;
  });

  // Height for vertical layout (mobile)
  const [sidebarHeight, setSidebarHeight] = useState<number>(() => {
    const fallbackH =
      defaultHeight ??
      (typeof window !== 'undefined' ? Math.round((window.innerHeight - 52) * 0.44) : 320);
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        const saved = localStorage.getItem(`${storageKey}_h`);
        if (saved) {
          const parsed = Number(saved);
          if (!isNaN(parsed) && parsed >= minHeight) return parsed;
        }
      } catch {
        // ignore
      }
    }
    return fallbackH;
  });

  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  // Viewport breakpoint listener
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${mobileBreakpoint - 1}px)`);
    const handler = (e: MediaQueryListEvent) => {
      setIsMobile(e.matches);
    };
    if (mq.addEventListener) {
      mq.addEventListener('change', handler);
    } else {
      mq.addListener(handler);
    }
    return () => {
      if (mq.removeEventListener) {
        mq.removeEventListener('change', handler);
      } else {
        mq.removeListener(handler);
      }
    };
  }, [mobileBreakpoint]);

  const startResize = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      setIsDragging(true);
      isDraggingRef.current = true;

      const topbarH = isMobile ? 52 : 56;

      const handlePointerMove = (moveEvent: PointerEvent) => {
        if (!isDraggingRef.current) return;
        if (isMobile) {
          // Vertical resize on mobile
          const availH = window.innerHeight - topbarH;
          const maxH = availH * maxHeightRatio;
          const newH = Math.min(Math.max(moveEvent.clientY - topbarH, minHeight), maxH);
          setSidebarHeight(Math.round(newH));
        } else {
          // Horizontal resize on desktop
          const maxW = window.innerWidth * maxWidthRatio;
          const newW = Math.min(Math.max(moveEvent.clientX, minWidth), maxW);
          setSidebarWidth(Math.round(newW));
        }
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
      document.body.style.cursor = isMobile ? 'row-resize' : 'col-resize';
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    },
    [isMobile, minWidth, maxWidthRatio, minHeight, maxHeightRatio]
  );

  useEffect(() => {
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(`${storageKey}_w`, String(sidebarWidth));
        localStorage.setItem(`${storageKey}_h`, String(sidebarHeight));
      } catch {
        // ignore
      }
    }
  }, [sidebarWidth, sidebarHeight, storageKey]);

  const resetSize = useCallback(() => {
    if (isMobile) {
      const defH = defaultHeight ?? Math.round((window.innerHeight - 52) * 0.44);
      setSidebarHeight(defH);
    } else {
      setSidebarWidth(defaultWidth);
    }
  }, [isMobile, defaultWidth, defaultHeight]);

  return {
    sidebarWidth,
    sidebarHeight,
    isDragging,
    isMobile,
    layoutMode,
    setLayoutMode,
    startResize,
    resetSize,
    resetWidth: resetSize,
  };
}
