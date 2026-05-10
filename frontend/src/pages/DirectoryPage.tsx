import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";

export function DirectoryPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ["employees"], queryFn: apiClient.employees, refetchInterval: 20_000 });

  if (isLoading) return <div className="page-state">Загружаю справочник…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить сотрудников.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Employee Directory</span>
        <h2>Справочник персонала</h2>
        <p>Справочник сотрудников и их фактический статус на предприятии по данным проходной.</p>
      </div>
      <div className="table-scroll panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>Сотрудник</th>
              <th>Табельный</th>
              <th>Отдел</th>
              <th>Кадровый статус</th>
              <th>Присутствие</th>
              <th>Последний вход</th>
            </tr>
          </thead>
          <tbody>
            {data.items.map((employee, index) => (
              <tr key={`${employee.id || "emp"}-${index}`}>
                <td>{String(employee.display_name || employee.full_name || "—")}</td>
                <td>{String(employee.employee_number || "—")}</td>
                <td>{String(employee.department || "—")}</td>
                <td>{String(employee.status || "—")}</td>
                <td>{employee.presence_status === "on_site" ? "На работе" : "Вне смены"}</td>
                <td>{employee.last_check_in_at ? new Date(Number(employee.last_check_in_at) * 1000).toLocaleString("ru-RU") : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
