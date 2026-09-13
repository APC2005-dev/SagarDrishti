import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState } from 'react';

const SCAN_MESSAGES = [
  'Analyzing sea-ice conditions...',
  'Checking iceberg hazards...',
  'Evaluating fuel efficiency...',
  'Calculating optimal route...',
  'Finalizing safest passage...',
];

interface RouteScannerProps {
  active: boolean;
}

export function RouteScanner({ active }: RouteScannerProps) {
  const [msgIndex, setMsgIndex] = useState(0);

  useEffect(() => {
    if (!active) {
      setMsgIndex(0);
      return;
    }
    const timer = setInterval(() => {
      setMsgIndex((prev) => (prev + 1) % SCAN_MESSAGES.length);
    }, 2400);
    return () => clearInterval(timer);
  }, [active]);

  return (
    <AnimatePresence>
      {active && (
        <motion.div
          className="route-scanner-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.35, ease: 'easeInOut' }}
        >
          {/* Radar Sweep Effect */}
          <div className="radar-container">
            <div className="radar-circle ring-1" />
            <div className="radar-circle ring-2" />
            <div className="radar-circle ring-3" />
            <div className="radar-crosshair-h" />
            <div className="radar-crosshair-v" />
            <div className="radar-sweep" />
            <div className="radar-center-ping" />
          </div>

          {/* Status HUD Overlay */}
          <div className="radar-hud-card glass">
            <div className="radar-hud-header">
              <span className="radar-hud-dot pulse" />
              <span className="radar-hud-title mono">SCANNING ROUTE</span>
            </div>
            <div className="radar-hud-message mono">
              <motion.span
                key={msgIndex}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.3 }}
              >
                {SCAN_MESSAGES[msgIndex]}
              </motion.span>
            </div>
            {/* Animated Indeterminate Progress Bar */}
            <div className="radar-progress-bar">
              <div className="radar-progress-indicator" />
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
