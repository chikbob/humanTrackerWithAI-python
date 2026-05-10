import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";

export function AuditPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ["audit-logs"], queryFn: apiClient.auditLogs, refetchInterval: 15_000 });

  if (isLoading) return <div className="page-state">Загружаю аудит…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить аудит.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Security Trace</span>
        <h2>Доступ и аудит</h2>
      </div>
      <div className="list-stack panel">
        {data.items.map((item, index) => (
          <article key={`${item.id || "audit"}-${index}`} className="queue-item">
            <div>
              <strong>{String(item.action || "—")}</strong>
              <p>{String(item.actor_name || "unknown")} · {String(item.resource_type || "resource")}</p>
            </div>
            <div className="muted-tag">{String(item.actor_role || "—")}</div>
          </article>
        ))}
      </div>
    </section>
  );
}
