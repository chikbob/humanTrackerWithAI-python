import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Incident } from "../lib/api";

const nextStatuses = ["new", "acknowledged", "in_progress", "resolved", "false_positive"];

export function IncidentsPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["incidents"], queryFn: apiClient.incidents, refetchInterval: 10_000 });
  const [drafts, setDrafts] = useState<Record<number, string>>({});

  const mutation = useMutation({
    mutationFn: ({ incidentId, status }: { incidentId: number; status: string }) =>
      apiClient.updateIncidentStatus(incidentId, { status, actor_name: "react-ui", actor_role: "admin" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["incidents"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  if (isLoading) return <div className="page-state">Загружаю инциденты…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить инциденты.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Incident Ops</span>
        <h2>Журнал инцидентов</h2>
        <p>Изменения статусов теперь идут точечными API-запросами, а не через полный rerun всего интерфейса.</p>
      </div>

      <div className="table-scroll panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Инцидент</th>
              <th>Источник</th>
              <th>Severity</th>
              <th>Статус</th>
              <th>Действие</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((incident: Incident) => {
              const value = drafts[incident.id] ?? incident.status;
              return (
                <tr key={incident.id}>
                  <td>{incident.id}</td>
                  <td>{incident.incident_type}</td>
                  <td>{incident.source_name || incident.zone_name || "—"}</td>
                  <td><span className={`severity-pill severity-${incident.severity}`}>{incident.severity}</span></td>
                  <td>
                    <select
                      className="input like-select"
                      value={value}
                      onChange={(event) => setDrafts((current) => ({ ...current, [incident.id]: event.target.value }))}
                    >
                      {nextStatuses.map((status) => <option key={status} value={status}>{status}</option>)}
                    </select>
                  </td>
                  <td>
                    <button
                      className="button secondary"
                      onClick={() => mutation.mutate({ incidentId: incident.id, status: value })}
                      disabled={mutation.isPending}
                    >
                      Сохранить
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
