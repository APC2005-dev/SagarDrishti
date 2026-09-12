import { AnimatePresence, motion } from 'framer-motion';
import { Suspense, lazy } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';

import { SystemBanner } from './components/common/SystemBanner';
import { TopBar } from './components/navigation/TopBar';

const Overview = lazy(() => import('./pages/Overview/OverviewPage'));
const Icebergs = lazy(() => import('./pages/Icebergs/IcebergsPage'));
const Operations = lazy(() => import('./pages/Operations/OperationsPage'));
const Feeds = lazy(() => import('./pages/Feeds/FeedsPage'));
const RoutePlanning = lazy(() => import('./pages/RoutePlanning/RoutePlanningPage'));
const SeaIce = lazy(() => import('./pages/SeaIceForecast/SeaIceForecastPage'));

export default function App() {
  const location = useLocation();
  return (
    <div className="app">
      <TopBar />
      <SystemBanner />
      <main className="app-main">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={location.pathname}
            className="page-frame"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
          >
            <Suspense fallback={<div className="page-loading label">Loading module…</div>}>
              <Routes location={location}>
                <Route path="/" element={<Overview />} />
                <Route path="/icebergs" element={<Icebergs />} />
                <Route path="/operations" element={<Operations />} />
                <Route path="/feeds" element={<Feeds />} />
                <Route path="/route-planning" element={<RoutePlanning />} />
                <Route path="/sea-ice" element={<SeaIce />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
