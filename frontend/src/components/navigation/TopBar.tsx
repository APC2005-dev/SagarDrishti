import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';

import logoImg from '../../assets/logo.jpg';

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
  return <span className="mono dim topbar-clock">{now.toISOString().slice(11, 16)} UTC</span>;
}

function HamburgerIcon({ open }: { open: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden>
      {open ? (
        <>
          <line x1="2" y1="2" x2="14" y2="14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="14" y1="2" x2="2" y2="14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </>
      ) : (
        <>
          <line x1="2" y1="4" x2="14" y2="4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="2" y1="8" x2="14" y2="8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <line x1="2" y1="12" x2="14" y2="12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        </>
      )}
    </svg>
  );
}

export function TopBar() {
  const { pathname } = useLocation();
  const ingestion = useIngestion();
  const lastSuccess = ingestion.data?.lastSuccessfulRun?.completedAt ?? null;
  const [menuOpen, setMenuOpen] = useState(false);
  const drawerRef = useRef<HTMLDivElement>(null);

  // Close menu on route change
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (drawerRef.current && !drawerRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    // Small delay so the hamburger-click itself doesn't immediately close
    const id = setTimeout(() => document.addEventListener('click', handleClick), 10);
    return () => {
      clearTimeout(id);
      document.removeEventListener('click', handleClick);
    };
  }, [menuOpen]);

  return (
    <>
      <header className="topbar">
        <button
          className="topbar-hamburger"
          onClick={() => setMenuOpen((o) => !o)}
          aria-label={menuOpen ? 'Close navigation menu' : 'Open navigation menu'}
          aria-expanded={menuOpen}
          id="topbar-hamburger-btn"
        >
          <HamburgerIcon open={menuOpen} />
        </button>

        <div className="brand">
          <span className="brand-mark" aria-hidden style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
            <img src={logoImg} alt="Sagar Drishti Logo" style={{ width: 28, height: 28, borderRadius: '50%', objectFit: 'cover' }} />
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
          <span className="mono dim topbar-sync">synced {relTime(lastSuccess)}</span>
          <UtcClock />
          <span className="crs-chip mono" title="Data in EPSG:4326; rendered in EPSG:3031 polar stereographic">
            {CRS.data} → {CRS.render}
          </span>
        </div>
      </header>

      {/* Mobile nav drawer */}
      <div className={`mobile-nav-overlay${menuOpen ? ' open' : ''}`} aria-hidden={!menuOpen} ref={drawerRef}>
        <div className="mobile-nav-backdrop" onClick={() => setMenuOpen(false)} />
        <nav className="mobile-nav-drawer" aria-label="Mobile navigation">
          {NAV.map((item) => {
            const active = item.end ? pathname === item.to : pathname.startsWith(item.to);
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={`mobile-nav-link${active ? ' active' : ''}`}
                onClick={() => setMenuOpen(false)}
              >
                {item.label}
              </NavLink>
            );
          })}
        </nav>
      </div>
    </>
  );
}
