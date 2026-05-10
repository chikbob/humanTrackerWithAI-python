import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Camera, DoorOpen, Radar } from "lucide-react";
import { apiClient } from "../lib/api";

function formatTimestamp(value?: number) {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString("ru-RU");
}

export function DashboardPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ["dashboard-summary"], queryFn: apiClient.dashboardSummary, refetchInterval: 10_000 });

  if (isLoading) return <div className="page-state">Загружаю ситуационный центр…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить summary.</div>;

  const kpi = [
    { label: "Онлайн-камер", value: data.summary.online_cameras, icon: Camera },
    { label: "Событий за день", value: data.summary.total_events_today, icon: Radar },
    { label: "Проходов за день", value: data.summary.entries_today, icon: DoorOpen },
    { label: "Активных инцидентов", value: data.incidents_summary.active, icon: AlertTriangle }
  ];

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Control Center</span>
        <h2>Ситуационный центр</h2>
        <p>Новый frontend читает агрегаты напрямую из API и не блокирует весь экран на каждый navigation rerender.</p>
      </div>

      <div className="metrics-grid">
        {kpi.map((item) => {
          const Icon = item.icon;
          return (
            <article key={item.label} className="metric-card">
              <div className="metric-icon"><Icon size={18} /></div>
              <div className="metric-value">{item.value}</div>
              <div className="metric-label">{item.label}</div>
            </article>
          );
        })}
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-header">
            <h3>Очередь инцидентов</h3>
            <span>{data.recent_incidents.length} записей</span>
          </div>
          <div className="list-stack">
            {data.recent_incidents.slice(0, 8).map((incident) => (
              <article key={incident.id} className="queue-item">
                <div>
                  <strong>{incident.incident_type}</strong>
                  <p>{incident.source_name || incident.zone_name || "Источник не указан"}</p>
                </div>
                <div className={`severity-pill severity-${incident.severity}`}>{incident.severity}</div>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>Последние события</h3>
            <span>{data.recent_events.length} записей</span>
          </div>
          <div className="table-scroll">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Источник</th>
                  <th>Время</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_events.slice(0, 8).map((event) => (
                  <tr key={event.event_id}>
                    <td>{event.event_type || "—"}</td>
                    <td>{event.source_name || "—"}</td>
                    <td>{formatTimestamp(event.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </section>
  );
}
