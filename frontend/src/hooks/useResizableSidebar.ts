import { useCallback, useEffect, useRef, useState } from 'react';

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
  minHeight = 160,
  maxHeightRatio = 0.75,
  mobileBreakpoint = 768,
}: UseResizableSidebarOptions = {}) {
  const [isMobile, setIsMobile] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth < mobileBreakpoint : false
  );

  // Width for horizontal side-by-side layout (desktop/tablet)
  const [sidebarWidth, setSidebarWidth] = useState<number>(() => {
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        const saved = localStorage.getItem(`${storageKey}_w`);
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

  // Height for vertical stacked layout (mobile)
  const [sidebarHeight, setSidebarHeight] = useState<number>(() => {
    const fallbackH =
      defaultHeight ??
      (typeof window !== 'undefined' ? Math.round((window.innerHeight - 56) * 0.46) : 320);
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        const saved = localStorage.getItem(`${storageKey}_h`);
        if (saved) {
          const parsed = Number(saved);
          if (!isNaN(parsed) && parsed >= minHeight) return parsed;
        }
      } catch {
        // ignore localStorage errors
      }
    }
    return fallbackH;
  });

  const [isDragging, setIsDragging] = useState(false);
  const isDraggingRef = useRef(false);

  // Track viewport width changes
  useEffect(() => {
    const mq = window.matchMedia(`(max-width: ${mobileBreakpoint - 1}px)`);
    const handler = (e: MediaQueryListEvent) => setIsMobile(e.matches);
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
          // Vertical resize (mobile)
          const maxHeight = (window.innerHeight - topbarH) * maxHeightRatio;
          const newHeight = Math.min(Math.max(moveEvent.clientY - topbarH, minHeight), maxHeight);
          setSidebarHeight(Math.round(newHeight));
        } else {
          // Horizontal resize (desktop)
          const maxWidth = window.innerWidth * maxWidthRatio;
          const newWidth = Math.min(Math.max(moveEvent.clientX, minWidth), maxWidth);
          setSidebarWidth(Math.round(newWidth));
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

  // Save changes to localStorage
  useEffect(() => {
    if (storageKey && typeof localStorage !== 'undefined') {
      try {
        localStorage.setItem(`${storageKey}_w`, String(sidebarWidth));
        localStorage.setItem(`${storageKey}_h`, String(sidebarHeight));
      } catch {
        // ignore localStorage errors
      }
    }
  }, [sidebarWidth, sidebarHeight, storageKey]);

  const resetSize = useCallback(() => {
    if (isMobile) {
      const defH = defaultHeight ?? Math.round((window.innerHeight - 52) * 0.46);
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
    startResize,
    resetSize,
    resetWidth: resetSize,
  };
}
