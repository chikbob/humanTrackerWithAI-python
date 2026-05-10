import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Incident } from "../lib/api";

const nextStatuses = ["new", "acknowledged", "in_progress", "resolved", "false_positive"];
const pageSize = 10;

type SortKey = "incident_type" | "source_name" | "severity" | "status";

export function IncidentsPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["incidents"], queryFn: apiClient.incidents, refetchInterval: 10_000 });
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("status");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(1);

  const mutation = useMutation({
    mutationFn: ({ incidentId, status }: { incidentId: number; status: string }) =>
      apiClient.updateIncidentStatus(incidentId, { status, actor_name: "react-ui", actor_role: "admin" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["incidents"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  const items = useMemo(() => data?.items || [], [data]);
  const statusOptions = useMemo(() => Array.from(new Set(items.map((item) => item.status).filter(Boolean))), [items]);
  const severityOptions = useMemo(() => Array.from(new Set(items.map((item) => item.severity).filter(Boolean))), [items]);

  const filteredItems = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return items
      .filter((incident) => statusFilter === "all" || incident.status === statusFilter)
      .filter((incident) => severityFilter === "all" || incident.severity === severityFilter)
      .filter((incident) => {
        if (!normalizedSearch) return true;
        const haystack = [
          incident.incident_type,
          incident.source_name,
          incident.zone_name,
          incident.status,
          incident.severity
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(normalizedSearch);
      })
      .sort((left, right) => {
        const leftValue = String(left[sortKey] || left.zone_name || "");
        const rightValue = String(right[sortKey] || right.zone_name || "");
        const result = leftValue.localeCompare(rightValue, "ru");
        return sortDirection === "asc" ? result : -result;
      });
  }, [items, search, severityFilter, sortDirection, sortKey, statusFilter]);

  const pageCount = Math.max(1, Math.ceil(filteredItems.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const pageItems = filteredItems.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDirection((value) => (value === "asc" ? "desc" : "asc"));
      return;
    }
    setSortKey(key);
    setSortDirection("asc");
  }

  if (isLoading) return <div className="page-state">Загружаю инциденты…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить инциденты.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Incident Ops</span>
        <h2>Журнал инцидентов</h2>
        <p>Журнал с поиском, фильтрацией, сортировкой и постраничной навигацией для операторской обработки событий.</p>
      </div>

      <section className="panel">
        <div className="table-toolbar">
          <input
            className="input"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
            placeholder="Поиск по типу, источнику, статусу"
          />
          <select className="input like-select" value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }}>
            <option value="all">Все статусы</option>
            {statusOptions.map((status) => <option key={status} value={status}>{status}</option>)}
          </select>
          <select className="input like-select" value={severityFilter} onChange={(event) => { setSeverityFilter(event.target.value); setPage(1); }}>
            <option value="all">Все уровни</option>
            {severityOptions.map((severity) => <option key={severity} value={severity}>{severity}</option>)}
          </select>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>№</th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("incident_type")}>Инцидент</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("source_name")}>Источник</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("severity")}>Severity</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("status")}>Статус</button></th>
                <th>Действие</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((incident: Incident, index) => {
                const value = drafts[incident.id] ?? incident.status;
                return (
                  <tr key={incident.id}>
                    <td>{(currentPage - 1) * pageSize + index + 1}</td>
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

        <div className="table-footer">
          <span className="table-meta">Показано {pageItems.length} из {filteredItems.length}</span>
          <div className="pagination">
            <button className="button secondary" type="button" disabled={currentPage <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>
              Назад
            </button>
            <span className="page-indicator">{currentPage} / {pageCount}</span>
            <button className="button secondary" type="button" disabled={currentPage >= pageCount} onClick={() => setPage((value) => Math.min(pageCount, value + 1))}>
              Вперёд
            </button>
          </div>
        </div>
      </section>
    </section>
  );
}
