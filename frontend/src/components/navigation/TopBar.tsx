import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

import { CRS } from '../../constants';
import { useIngestion } from '../../hooks/queries';
import { relTime } from '../../utils/format';
import { FeedStateChip } from '../common/StatusChip';

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/icebergs', label: 'Icebergs' },
  { to: '/operations', label: 'Operations' },
  { to: '/feeds', label: 'Feeds' },
  { to: '/route-planning', label: 'Route planning' },
  { to: '/sea-ice', label: 'Sea-ice' },
];

function UtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 15_000);
    return () => clearInterval(t);
  }, []);
  return <span className="mono dim">{now.toISOString().slice(11, 16)} UTC</span>;
}

export function TopBar() {
  const { pathname } = useLocation();
  const ingestion = useIngestion();
  const lastSuccess = ingestion.data?.lastSuccessfulRun?.completedAt ?? null;

  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark" aria-hidden>
          ◇
        </span>
        <span className="brand-name">SAGAR DRISHTI</span>
        <span className="brand-sub mono">ANTARCTIC ICEBERG INTELLIGENCE · v1.0</span>
      </div>
      <nav className="nav" aria-label="Primary">
        {NAV.map((item) => {
          const active = item.end ? pathname === item.to : pathname.startsWith(item.to);
          return (
            <NavLink key={item.to} to={item.to} end={item.end} className={`nav-link${active ? ' active' : ''}`}>
              {item.label}
              {active && <motion.span layoutId="nav-underline" className="nav-underline" transition={{ duration: 0.25 }} />}
            </NavLink>
          );
        })}
      </nav>
      <div className="topbar-status">
        {ingestion.isError ? (
          <FeedStateChip state="FAILED" label="API OFFLINE" />
        ) : (
          <FeedStateChip state={ingestion.data?.feedState ?? 'UNKNOWN'} label={`USNIC ${ingestion.data?.feedState ?? '…'}`} />
        )}
        <span className="mono dim">synced {relTime(lastSuccess)}</span>
        <UtcClock />
        <span className="crs-chip mono" title="Data in EPSG:4326; rendered in EPSG:3031 polar stereographic">
          {CRS.data} → {CRS.render}
        </span>
      </div>
    </header>
  );
}
