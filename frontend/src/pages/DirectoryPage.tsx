import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Employee } from "../lib/api";

const pageSize = 10;
type SortKey = "display_name" | "employee_number" | "department" | "status" | "presence_status";

type EmployeeFormState = {
  full_name: string;
  last_name: string;
  first_name: string;
  middle_name: string;
  employee_number: string;
  department: string;
  position: string;
  status: string;
  hire_date: string;
  profile_photo_url: string;
};

function formatDateTime(value?: number | null) {
  if (!value) return "—";
  return new Date(Number(value) * 1000).toLocaleString("ru-RU");
}

function formatDate(value?: number | null) {
  if (!value) return "—";
  return new Date(Number(value) * 1000).toLocaleDateString("ru-RU");
}

function formatDateInput(value?: number | null) {
  if (!value) return "";
  return new Date(Number(value) * 1000).toISOString().slice(0, 10);
}

function emptyForm(): EmployeeFormState {
  return {
    full_name: "",
    last_name: "",
    first_name: "",
    middle_name: "",
    employee_number: "",
    department: "",
    position: "",
    status: "active",
    hire_date: "",
    profile_photo_url: ""
  };
}

function employeeToForm(employee: Employee): EmployeeFormState {
  return {
    full_name: employee.full_name || "",
    last_name: employee.last_name || "",
    first_name: employee.first_name || "",
    middle_name: employee.middle_name || "",
    employee_number: employee.employee_number || "",
    department: employee.department || "",
    position: employee.position || "",
    status: employee.status || "active",
    hire_date: formatDateInput(employee.hire_date),
    profile_photo_url: ""
  };
}

function buildPayload(form: EmployeeFormState) {
  return {
    ...form,
    hire_date: form.hire_date ? new Date(`${form.hire_date}T00:00:00`).getTime() / 1000 : null,
    actor_name: "react-ui",
    actor_role: "admin"
  };
}

export function DirectoryPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["employees"], queryFn: apiClient.employees, refetchInterval: 20_000 });
  const { data: syncData } = useQuery({ queryKey: ["employee-sync-state"], queryFn: apiClient.employeeSyncState, refetchInterval: 20_000 });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [presenceFilter, setPresenceFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("display_name");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(1);
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<number>(0);
  const [form, setForm] = useState<EmployeeFormState>(emptyForm());

  const items = useMemo(() => data?.items || [], [data]);
  const selectedEmployee = useMemo(() => items.find((item) => item.id === selectedEmployeeId) || null, [items, selectedEmployeeId]);
  const statusOptions = useMemo(() => Array.from(new Set(items.map((item) => item.status).filter(Boolean))), [items]);

  const createMutation = useMutation({
    mutationFn: () => apiClient.createEmployee(buildPayload(form)),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      queryClient.invalidateQueries({ queryKey: ["audit-logs"] });
      setForm(emptyForm());
    }
  });
  const updateMutation = useMutation({
    mutationFn: () => selectedEmployeeId ? apiClient.updateEmployee(selectedEmployeeId, buildPayload(form)) : Promise.reject(new Error("employee_not_selected")),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      queryClient.invalidateQueries({ queryKey: ["audit-logs"] });
    }
  });
  const statusMutation = useMutation({
    mutationFn: ({ employeeId, status }: { employeeId: number; status: string }) =>
      apiClient.updateEmployeeStatus(employeeId, { status, actor_name: "react-ui", actor_role: "admin" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["employees"] });
      queryClient.invalidateQueries({ queryKey: ["audit-logs"] });
    }
  });

  const filteredItems = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return items
      .filter((employee) => statusFilter === "all" || employee.status === statusFilter)
      .filter((employee) => presenceFilter === "all" || employee.presence_status === presenceFilter)
      .filter((employee) => {
        if (!normalizedSearch) return true;
        const haystack = [
          employee.display_name,
          employee.full_name,
          employee.employee_number,
          employee.department,
          employee.position
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();
        return haystack.includes(normalizedSearch);
      })
      .sort((left, right) => {
        const leftValue = String(left[sortKey] || left.full_name || "");
        const rightValue = String(right[sortKey] || right.full_name || "");
        const result = leftValue.localeCompare(rightValue, "ru");
        return sortDirection === "asc" ? result : -result;
      });
  }, [items, presenceFilter, search, sortDirection, sortKey, statusFilter]);

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

  if (isLoading) return <div className="page-state">Загружаю справочник…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить сотрудников.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Employee Directory</span>
        <h2>Справочник персонала</h2>
        <p>Список сотрудников, карточка, создание и редактирование записей, смена статуса без удаления, а также контроль синхронизации и fallback-режима справочника.</p>
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-header"><h3>Состояние синхронизации</h3><span>{syncData?.item?.sync_status || "local_only"}</span></div>
          <div className="list-stack">
            <article className="stat-line"><strong>Источник данных</strong><span>{syncData?.item?.data_source || "sqlite"}</span></article>
            <article className="stat-line"><strong>Статус синхронизации</strong><span>{syncData?.item?.sync_status || "local_only"}</span></article>
            <article className="stat-line"><strong>Последнее обновление</strong><span>{formatDateTime(syncData?.item?.last_synced_at)}</span></article>
            <article className="stat-line"><strong>Режим кэша</strong><span>{syncData?.item?.cache_mode || "read_write"}</span></article>
            <article className="status-card">
              <strong>Fallback / ошибки</strong>
              <span>{syncData?.item?.last_error || "Ошибки синхронизации не зафиксированы. При недоступности внешнего справочника используется локальный кэш и режим read-only."}</span>
            </article>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header"><h3>{selectedEmployee ? "Карточка сотрудника" : "Новая запись"}</h3><span>{selectedEmployee ? selectedEmployee.display_name || selectedEmployee.full_name : "create"}</span></div>
          <div className="form-grid">
            <div className="triple-grid">
              <label className="field-block"><span>Фамилия</span><input className="input" value={form.last_name} onChange={(event) => setForm((current) => ({ ...current, last_name: event.target.value }))} /></label>
              <label className="field-block"><span>Имя</span><input className="input" value={form.first_name} onChange={(event) => setForm((current) => ({ ...current, first_name: event.target.value }))} /></label>
              <label className="field-block"><span>Отчество</span><input className="input" value={form.middle_name} onChange={(event) => setForm((current) => ({ ...current, middle_name: event.target.value }))} /></label>
            </div>
            <label className="field-block"><span>ФИО</span><input className="input" value={form.full_name} onChange={(event) => setForm((current) => ({ ...current, full_name: event.target.value }))} /></label>
            <div className="triple-grid">
              <label className="field-block"><span>Табельный номер</span><input className="input" value={form.employee_number} onChange={(event) => setForm((current) => ({ ...current, employee_number: event.target.value }))} /></label>
              <label className="field-block"><span>Подразделение</span><input className="input" value={form.department} onChange={(event) => setForm((current) => ({ ...current, department: event.target.value }))} /></label>
              <label className="field-block"><span>Должность</span><input className="input" value={form.position} onChange={(event) => setForm((current) => ({ ...current, position: event.target.value }))} /></label>
            </div>
            <div className="triple-grid">
              <label className="field-block"><span>Кадровый статус</span>
                <select className="input like-select" value={form.status} onChange={(event) => setForm((current) => ({ ...current, status: event.target.value }))}>
                  {["active", "inactive", "on_leave", "blocked"].map((status) => <option key={status} value={status}>{status}</option>)}
                </select>
              </label>
              <label className="field-block"><span>Дата приема</span><input className="input" type="date" value={form.hire_date} onChange={(event) => setForm((current) => ({ ...current, hire_date: event.target.value }))} /></label>
              <label className="field-block"><span>Источник фото</span><input className="input" value={form.profile_photo_url} onChange={(event) => setForm((current) => ({ ...current, profile_photo_url: event.target.value }))} /></label>
            </div>
            <div className="button-row">
              <button className="button secondary" type="button" onClick={() => { setSelectedEmployeeId(0); setForm(emptyForm()); }}>Очистить форму</button>
              <button className="button secondary" type="button" disabled={!selectedEmployee || statusMutation.isPending} onClick={() => selectedEmployee && statusMutation.mutate({ employeeId: selectedEmployee.id, status: form.status })}>Сменить статус</button>
              <button className="button" type="button" disabled={createMutation.isPending || updateMutation.isPending} onClick={() => selectedEmployee ? updateMutation.mutate() : createMutation.mutate()}>
                {selectedEmployee ? "Сохранить изменения" : "Создать сотрудника"}
              </button>
            </div>
          </div>
        </section>
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
            placeholder="Поиск по ФИО, табельному номеру, отделу"
          />
          <select className="input like-select" value={statusFilter} onChange={(event) => { setStatusFilter(event.target.value); setPage(1); }}>
            <option value="all">Все кадровые статусы</option>
            {statusOptions.map((status) => <option key={status} value={status}>{status}</option>)}
          </select>
          <select className="input like-select" value={presenceFilter} onChange={(event) => { setPresenceFilter(event.target.value); setPage(1); }}>
            <option value="all">Любое присутствие</option>
            <option value="on_site">На работе</option>
            <option value="off_duty">Вне смены</option>
          </select>
        </div>

        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>№</th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("display_name")}>Сотрудник</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("employee_number")}>Табельный</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("department")}>Отдел</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("status")}>Кадровый статус</button></th>
                <th><button className="sort-button" type="button" onClick={() => toggleSort("presence_status")}>Присутствие</button></th>
                <th>Дата приема</th>
                <th>Источник данных</th>
                <th>Последний sync</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((employee: Employee, index) => (
                <tr key={`${employee.id || "emp"}-${index}`} onClick={() => { setSelectedEmployeeId(employee.id); setForm(employeeToForm(employee)); }} className={selectedEmployeeId === employee.id ? "row-selected" : ""}>
                  <td>{(currentPage - 1) * pageSize + index + 1}</td>
                  <td>{String(employee.display_name || employee.full_name || "—")}</td>
                  <td>{String(employee.employee_number || "—")}</td>
                  <td>{String(employee.department || "—")}</td>
                  <td>{String(employee.status || "—")}</td>
                  <td>{employee.presence_status === "on_site" ? "На работе" : "Вне смены"}</td>
                  <td>{formatDate(employee.hire_date)}</td>
                  <td>{String(employee.source_system || "local")}</td>
                  <td>{formatDateTime(employee.last_synced_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {selectedEmployee && (
          <div className="content-grid two-columns">
            <section className="panel">
              <div className="panel-header"><h3>Карточка сотрудника</h3><span>{selectedEmployee.employee_number || selectedEmployee.id}</span></div>
              <div className="list-stack">
                <article className="stat-line"><strong>ФИО</strong><span>{selectedEmployee.display_name || selectedEmployee.full_name}</span></article>
                <article className="stat-line"><strong>Подразделение</strong><span>{selectedEmployee.department || "—"}</span></article>
                <article className="stat-line"><strong>Должность</strong><span>{selectedEmployee.position || "—"}</span></article>
                <article className="stat-line"><strong>Дата приема</strong><span>{formatDate(selectedEmployee.hire_date)}</span></article>
                <article className="stat-line"><strong>Источник данных</strong><span>{selectedEmployee.source_system || "local"}</span></article>
                <article className="stat-line"><strong>Последний вход</strong><span>{formatDateTime(selectedEmployee.last_check_in_at)}</span></article>
              </div>
            </section>
          </div>
        )}

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
