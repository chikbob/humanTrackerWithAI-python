import { useMemo } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Activity, Camera, LayoutDashboard, Radio, Settings, ShieldAlert, Users } from "lucide-react";
import { DashboardPage } from "./pages/DashboardPage";
import { IncidentsPage } from "./pages/IncidentsPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { SourcesPage } from "./pages/SourcesPage";
import { SettingsPage } from "./pages/SettingsPage";
import { DirectoryPage } from "./pages/DirectoryPage";
import { AuditPage } from "./pages/AuditPage";

type NavItem = {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
};

const navItems: NavItem[] = [
  { to: "/dashboard", label: "Ситуационный центр", icon: LayoutDashboard },
  { to: "/monitoring", label: "Операторский мониторинг", icon: Radio },
  { to: "/incidents", label: "Инциденты", icon: ShieldAlert },
  { to: "/sources", label: "Источники", icon: Camera },
  { to: "/directory", label: "Персонал", icon: Users },
  { to: "/settings", label: "Настройки", icon: Settings },
  { to: "/audit", label: "Аудит", icon: Activity }
];

export function App() {
  const buildStamp = useMemo(() => new Date().toLocaleString("ru-RU"), []);

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="brand-block">
          <span className="brand-eyebrow">Flexible Control Stack</span>
          <h1>Human Tracker</h1>
          <p>FastAPI backend + React operator SPA instead of a monolithic Streamlit surface.</p>
        </div>
        <nav className="nav-list">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) => `nav-item${isActive ? " is-active" : ""}`}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
        <div className="sidebar-footnote">
          <span>Build switch</span>
          <strong>{buildStamp}</strong>
        </div>
      </aside>

      <main className="app-main">
        <header className="app-topbar">
          <div>
            <span className="topbar-label">Migration branch</span>
            <strong>`flexible-stack-migration`</strong>
          </div>
          <div className="topbar-chip">Fast UI path</div>
        </header>

        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/incidents" element={<IncidentsPage />} />
          <Route path="/sources" element={<SourcesPage />} />
          <Route path="/directory" element={<DirectoryPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/audit" element={<AuditPage />} />
        </Routes>
      </main>
    </div>
  );
}
