import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, Employee } from "../lib/api";

const pageSize = 10;
type SortKey = "display_name" | "employee_number" | "department" | "status" | "presence_status";

function formatDateTime(value?: number | null) {
  if (!value) return "—";
  return new Date(Number(value) * 1000).toLocaleString("ru-RU");
}

export function DirectoryPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ["employees"], queryFn: apiClient.employees, refetchInterval: 20_000 });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [presenceFilter, setPresenceFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("display_name");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(1);

  const items = useMemo(() => data?.items || [], [data]);
  const statusOptions = useMemo(() => Array.from(new Set(items.map((item) => item.status).filter(Boolean))), [items]);

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
        <p>Справочник сотрудников с поиском, сортировкой, фильтрацией и контролем фактического присутствия на предприятии.</p>
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
                <th>Последний вход</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((employee: Employee, index) => (
                <tr key={`${employee.id || "emp"}-${index}`}>
                  <td>{(currentPage - 1) * pageSize + index + 1}</td>
                  <td>{String(employee.display_name || employee.full_name || "—")}</td>
                  <td>{String(employee.employee_number || "—")}</td>
                  <td>{String(employee.department || "—")}</td>
                  <td>{String(employee.status || "—")}</td>
                  <td>{employee.presence_status === "on_site" ? "На работе" : "Вне смены"}</td>
                  <td>{formatDateTime(employee.last_check_in_at)}</td>
                </tr>
              ))}
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
