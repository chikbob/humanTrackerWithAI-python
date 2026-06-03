import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Employee, EventItem } from "../lib/api";

const pageSize = 25;

const identificationStatuses = [
  "unknown",
  "unlinked",
  "pending_operator_confirmation",
  "linked_from_directory",
  "linked_from_access_control",
  "inactive_employee"
];

function formatTimestamp(value?: number | null) {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString("ru-RU");
}

function formatDateInput(value?: number | null) {
  if (!value) return "";
  return new Date(value * 1000).toISOString().slice(0, 10);
}

function eventSeverity(event: EventItem) {
  const type = event.event_type || "";
  if (type === "stream_offline") return "critical";
  if (["unknown_person_detected", "repeated_entry_attempt", "prolonged_presence_near_entry"].includes(type)) return "high";
  if (type === "camera_reconnected") return "low";
  return "medium";
}

function eventSeverityLabel(level: string) {
  return { low: "Низкий", medium: "Средний", high: "Высокий", critical: "Критический" }[level] || level;
}

function identificationStatusLabel(status?: string) {
  return {
    unknown: "unknown",
    unlinked: "unlinked",
    pending_operator_confirmation: "pending_operator_confirmation",
    linked_from_directory: "linked_from_directory",
    linked_from_access_control: "linked_from_access_control",
    inactive_employee: "inactive_employee"
  }[status || "unknown"] || status || "unknown";
}

function toCsv(rows: EventItem[]) {
  const header = ["timestamp", "event_type", "source_name", "severity", "employee_name", "confidence", "identification_status", "message"];
  const lines = rows.map((event) => [
    formatTimestamp(event.timestamp),
    event.event_type || "",
    event.source_name || "",
    eventSeverity(event),
    event.employee_name || "",
    String(event.confidence ?? ""),
    event.identification_status || "",
    (event.message || "").replace(/\n/g, " ")
  ]);
  return [header, ...lines].map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
}

export function AuditPage() {
  const queryClient = useQueryClient();
  const { data: eventsData, isLoading, error } = useQuery({ queryKey: ["events-journal"], queryFn: () => apiClient.events(2000), refetchInterval: 15_000 });
  const { data: employeesData } = useQuery({ queryKey: ["employees"], queryFn: apiClient.employees, refetchInterval: 20_000 });
  const { data: auditData } = useQuery({ queryKey: ["audit-logs"], queryFn: apiClient.auditLogs, refetchInterval: 20_000 });
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState("all");
  const [sourceFilter, setSourceFilter] = useState("all");
  const [severityFilter, setSeverityFilter] = useState("all");
  const [employeeFilter, setEmployeeFilter] = useState("all");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [selectedEventId, setSelectedEventId] = useState("");
  const [linkEmployeeId, setLinkEmployeeId] = useState<number>(0);
  const [linkStatus, setLinkStatus] = useState("pending_operator_confirmation");
  const [linkNote, setLinkNote] = useState("");
  const [page, setPage] = useState(1);

  const events = useMemo(() => eventsData?.items || [], [eventsData]);
  const employees = useMemo(() => employeesData?.items || [], [employeesData]);

  const filteredEvents = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return events.filter((event) => {
      const severity = eventSeverity(event);
      const eventDate = formatDateInput(event.timestamp);
      const haystack = [
        event.event_type,
        event.source_name,
        event.employee_name,
        event.identification_status,
        event.message
      ].filter(Boolean).join(" ").toLowerCase();
      return (
        (!normalizedSearch || haystack.includes(normalizedSearch)) &&
        (typeFilter === "all" || event.event_type === typeFilter) &&
        (sourceFilter === "all" || (event.source_name || "не указан") === sourceFilter) &&
        (severityFilter === "all" || severity === severityFilter) &&
        (employeeFilter === "all" || (event.employee_name || "—") === employeeFilter) &&
        (!dateFrom || eventDate >= dateFrom) &&
        (!dateTo || eventDate <= dateTo)
      );
    });
  }, [dateFrom, dateTo, employeeFilter, events, search, severityFilter, sourceFilter, typeFilter]);

  const selectedEvent = useMemo(
    () => filteredEvents.find((event) => event.event_id === selectedEventId) || filteredEvents[0] || null,
    [filteredEvents, selectedEventId]
  );
  const pageCount = Math.max(1, Math.ceil(filteredEvents.length / pageSize));
  const currentPage = Math.min(page, pageCount);
  const pageItems = filteredEvents.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const linkMutation = useMutation({
    mutationFn: ({ eventId, employeeId, identificationStatus, note }: { eventId: string; employeeId: number; identificationStatus: string; note: string }) =>
      apiClient.linkEventToEmployee(eventId, {
        employee_id: employeeId,
        identification_status: identificationStatus,
        note,
        actor_name: "react-ui",
        actor_role: "operator"
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["events-journal"] });
      queryClient.invalidateQueries({ queryKey: ["audit-logs"] });
    }
  });

  if (isLoading) return <div className="page-state">Загружаю журнал событий…</div>;
  if (error || !eventsData) return <div className="page-state error">Не удалось загрузить журнал событий.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Event Journal</span>
        <h2>Журнал событий</h2>
        <p>Табличный журнал проходной зоны с фильтрацией, карточкой события, ручной привязкой к сотруднику и экспортом текущей выборки в CSV.</p>
      </div>

      <section className="panel">
        <div className="table-toolbar">
          <input className="input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Поиск по типу, источнику, сотруднику, сообщению" />
          <input className="input" type="date" value={dateFrom} onChange={(event) => { setDateFrom(event.target.value); setPage(1); }} />
          <input className="input" type="date" value={dateTo} onChange={(event) => { setDateTo(event.target.value); setPage(1); }} />
          <select className="input like-select" value={typeFilter} onChange={(event) => { setTypeFilter(event.target.value); setPage(1); }}>
            <option value="all">Все типы событий</option>
            {Array.from(new Set(events.map((event) => event.event_type).filter(Boolean))).sort().map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <select className="input like-select" value={sourceFilter} onChange={(event) => { setSourceFilter(event.target.value); setPage(1); }}>
            <option value="all">Все источники</option>
            {Array.from(new Set(events.map((event) => event.source_name || "не указан"))).sort().map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <select className="input like-select" value={severityFilter} onChange={(event) => { setSeverityFilter(event.target.value); setPage(1); }}>
            <option value="all">Все уровни</option>
            {["low", "medium", "high", "critical"].map((item) => <option key={item} value={item}>{eventSeverityLabel(item)}</option>)}
          </select>
          <select className="input like-select" value={employeeFilter} onChange={(event) => { setEmployeeFilter(event.target.value); setPage(1); }}>
            <option value="all">Все сотрудники</option>
            {Array.from(new Set(events.map((event) => event.employee_name || "—"))).sort().map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <a
            className="button secondary"
            href={`data:text/csv;charset=utf-8,${encodeURIComponent(toCsv(filteredEvents))}`}
            download="event_journal.csv"
          >
            Экспорт CSV
          </a>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Дата и время</th>
                <th>Тип события</th>
                <th>Источник</th>
                <th>Уровень</th>
                <th>Сотрудник</th>
                <th>Confidence</th>
                <th>Статус связи</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((event) => (
                <tr key={event.event_id} onClick={() => setSelectedEventId(event.event_id)} className={selectedEvent?.event_id === event.event_id ? "row-selected" : ""}>
                  <td>{formatTimestamp(event.timestamp)}</td>
                  <td>{event.event_type || "—"}</td>
                  <td>{event.source_name || "—"}</td>
                  <td><span className={`severity-pill severity-${eventSeverity(event)}`}>{eventSeverityLabel(eventSeverity(event))}</span></td>
                  <td>{event.employee_name || "—"}</td>
                  <td>{event.confidence ? Number(event.confidence).toFixed(3) : "—"}</td>
                  <td>{identificationStatusLabel(event.identification_status)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="table-footer">
          <span className="table-meta">Показано {pageItems.length} из {filteredEvents.length}</span>
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

      {selectedEvent && (
        <div className="content-grid two-columns">
          <section className="panel">
            <div className="panel-header"><h3>Карточка события</h3><span>{selectedEvent.event_id}</span></div>
            <div className="list-stack">
              <article className="stat-line"><strong>Время</strong><span>{formatTimestamp(selectedEvent.timestamp)}</span></article>
              <article className="stat-line"><strong>Тип</strong><span>{selectedEvent.event_type || "—"}</span></article>
              <article className="stat-line"><strong>Источник</strong><span>{selectedEvent.source_name || "—"}</span></article>
              <article className="stat-line"><strong>Точка доступа</strong><span>{selectedEvent.access_point_name || "—"}</span></article>
              <article className="stat-line"><strong>Сотрудник</strong><span>{selectedEvent.employee_name || "—"}</span></article>
              <article className="stat-line"><strong>Статус связи</strong><span>{identificationStatusLabel(selectedEvent.identification_status)}</span></article>
              <article className="stat-line"><strong>Confidence</strong><span>{selectedEvent.confidence ? Number(selectedEvent.confidence).toFixed(3) : "—"}</span></article>
              <article className="stat-line"><strong>Идентификационная достоверность</strong><span>{selectedEvent.identification_confidence ? Number(selectedEvent.identification_confidence).toFixed(3) : "—"}</span></article>
              <article className="stat-line"><strong>Уровень события</strong><span>{eventSeverityLabel(eventSeverity(selectedEvent))}</span></article>
              <article className="status-card"><strong>Сообщение</strong><span>{selectedEvent.message || "—"}</span></article>
            </div>
          </section>

          <section className="panel">
            <div className="panel-header"><h3>Ручная привязка к сотруднику</h3><span>{employees.length} сотрудников</span></div>
            <div className="form-grid">
              <label className="field-block">
                <span>Сотрудник</span>
                <select className="input like-select" value={linkEmployeeId} onChange={(event) => setLinkEmployeeId(Number(event.target.value))}>
                  <option value={0}>Выберите сотрудника</option>
                  {employees.map((employee: Employee) => (
                    <option key={employee.id} value={employee.id}>{employee.display_name || employee.full_name}</option>
                  ))}
                </select>
              </label>
              <label className="field-block">
                <span>Статус связи</span>
                <select className="input like-select" value={linkStatus} onChange={(event) => setLinkStatus(event.target.value)}>
                  {identificationStatuses.map((status) => <option key={status} value={status}>{status}</option>)}
                </select>
              </label>
              <label className="field-block">
                <span>Комментарий</span>
                <textarea className="input" rows={4} value={linkNote} onChange={(event) => setLinkNote(event.target.value)} placeholder="Примечание оператора о ручной привязке" />
              </label>
              <div className="button-row">
                <button
                  className="button"
                  disabled={!linkEmployeeId || linkMutation.isPending}
                  onClick={() => selectedEvent && linkMutation.mutate({ eventId: selectedEvent.event_id, employeeId: linkEmployeeId, identificationStatus: linkStatus, note: linkNote })}
                >
                  {linkMutation.isPending ? "Сохраняю..." : "Привязать событие"}
                </button>
              </div>
              {linkMutation.error instanceof Error && <div className="inline-warning">{linkMutation.error.message}</div>}
            </div>
          </section>
        </div>
      )}

      <section className="panel">
        <div className="panel-header"><h3>Последние действия аудита</h3><span>{auditData?.items.length || 0}</span></div>
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Время</th><th>Действие</th><th>Actor</th><th>Ресурс</th></tr></thead>
            <tbody>{(auditData?.items || []).slice(0, 20).map((item, index) => (
              <tr key={`${item.id || "audit"}-${index}`}>
                <td>{formatTimestamp(Number(item.created_at || 0))}</td>
                <td>{String(item.action || "—")}</td>
                <td>{String(item.actor_name || "unknown")} · {String(item.actor_role || "—")}</td>
                <td>{String(item.resource_type || "—")} / {String(item.resource_id || "—")}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
