import { useMemo } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Activity, Camera, LayoutDashboard, Radio, ScanFace, Settings, ShieldAlert, Users } from "lucide-react";
import { DashboardPage } from "./pages/DashboardPage";
import { IncidentsPage } from "./pages/IncidentsPage";
import { MonitoringPage } from "./pages/MonitoringPage";
import { SourcesPage } from "./pages/SourcesPage";
import { SettingsPage } from "./pages/SettingsPage";
import { DirectoryPage } from "./pages/DirectoryPage";
import { AuditPage } from "./pages/AuditPage";
import { AttendancePage } from "./pages/AttendancePage";

type NavItem = {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
};

const navItems: NavItem[] = [
  { to: "/dashboard", label: "Ситуационный центр", icon: LayoutDashboard },
  { to: "/monitoring", label: "Операторский мониторинг", icon: Radio },
  { to: "/checkpoint", label: "КПП сотрудников", icon: ScanFace },
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
          <span className="brand-eyebrow">NeuroGate</span>
          <h1>Система мониторинга и интеллектуального анализа объектов</h1>
          <p>Контур видеомониторинга, проходной сотрудников и аналитики в реальном времени на основе нейросетевых моделей YOLOv8.</p>
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
          <span>Актуальная сборка</span>
          <strong>{buildStamp}</strong>
        </div>
      </aside>

      <main className="app-main">
        <header className="app-topbar">
          <div>
            <span className="topbar-label">NeuroGate</span>
            <strong>Контроль проходной, мониторинг камер и событий</strong>
          </div>
          <div className="topbar-chip">YOLOv8 · FastAPI · React</div>
        </header>

        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/checkpoint" element={<AttendancePage />} />
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
