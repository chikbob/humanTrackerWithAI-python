import { useMemo } from "react";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient, EventItem } from "../lib/api";

const hourlyPageSize = 24;
const dailyPageSize = 24;

function formatDateHour(timestamp?: number) {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
}

function formatDateDay(timestamp?: number) {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleDateString("ru-RU");
}

function countBy<T extends string>(values: T[]) {
  return Array.from(values.reduce((map, value) => map.set(value, (map.get(value) || 0) + 1), new Map<T, number>()).entries());
}

function buildRows(entries: Array<[string, number]>, key: string, value: string) {
  return entries.map(([label, count]) => ({ [key]: label, [value]: count }));
}

export function AnalyticsPage() {
  const [hourlyPage, setHourlyPage] = useState(1);
  const [dailyPage, setDailyPage] = useState(1);
  const { data, isLoading, error } = useQuery({ queryKey: ["events-analytics"], queryFn: () => apiClient.events(2500), refetchInterval: 15_000 });
  const items = useMemo(() => data?.items || [], [data]);

  const derived = useMemo(() => {
    const domainEvents = items.filter((event) => event.event_scope === "domain");
    const prolongedPresence = domainEvents.filter((event) => event.event_type === "prolonged_presence_near_entry");
    const offlineEvents = domainEvents.filter((event) => event.event_type === "stream_offline");
    const entries = domainEvents.filter((event) => event.event_type === "person_entered_entry_zone");

    const hourly = buildRows(
      countBy(domainEvents.map((event) => formatDateHour(event.timestamp)).filter((value) => value !== "—")).sort((left, right) => left[0].localeCompare(right[0], "ru")),
      "Час",
      "Событий"
    );
    const dailyEntries = buildRows(
      countBy(entries.map((event) => formatDateDay(event.timestamp)).filter((value) => value !== "—")).sort((left, right) => left[0].localeCompare(right[0], "ru")),
      "Дата",
      "Входов"
    );
    const byAccessPoint = buildRows(
      countBy(domainEvents.map((event) => event.access_point_name || "не задана")).sort((left, right) => right[1] - left[1]),
      "Точка доступа",
      "Событий"
    );
    const offlineBySource = buildRows(
      countBy(offlineEvents.map((event) => event.source_name || "не указан")).sort((left, right) => right[1] - left[1]),
      "Источник",
      "Offline-событий"
    );
    const topTypes = buildRows(
      countBy(domainEvents.map((event) => event.event_type || "unknown")).sort((left, right) => right[1] - left[1]).slice(0, 8),
      "Тип события",
      "Количество"
    );

    return {
      domainEvents,
      prolongedPresence,
      offlineEvents,
      hourly,
      dailyEntries,
      byAccessPoint,
      offlineBySource,
      topTypes
    };
  }, [items]);

  const hourlyPageCount = Math.max(1, Math.ceil(derived.hourly.length / hourlyPageSize));
  const currentHourlyPage = Math.min(hourlyPage, hourlyPageCount);
  const hourlyRows = derived.hourly.slice((currentHourlyPage - 1) * hourlyPageSize, currentHourlyPage * hourlyPageSize);
  const dailyPageCount = Math.max(1, Math.ceil(derived.dailyEntries.length / dailyPageSize));
  const currentDailyPage = Math.min(dailyPage, dailyPageCount);
  const dailyRows = derived.dailyEntries.slice((currentDailyPage - 1) * dailyPageSize, currentDailyPage * dailyPageSize);

  if (isLoading) return <div className="page-state">Загружаю аналитику…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить аналитику.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Access Analytics</span>
        <h2>Аналитика событий проходной зоны</h2>
        <p>Обобщенная картина работы контура: активность по времени, входы по дням, распределение по точкам доступа, длительные присутствия, offline-события и наиболее частые типы событий.</p>
      </div>

      <div className="metrics-grid">
        <article className="metric-card">
          <div className="metric-value">{derived.domainEvents.filter((event) => event.event_type === "person_detected_near_entry").length}</div>
          <div className="metric-label">Обнаружения людей</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{derived.domainEvents.filter((event) => event.event_type === "person_entered_entry_zone").length}</div>
          <div className="metric-label">Входы в зону</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{derived.prolongedPresence.length}</div>
          <div className="metric-label">Длительные присутствия</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{derived.offlineEvents.length}</div>
          <div className="metric-label">Offline-события камер</div>
        </article>
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-header"><h3>Обнаружения по часам</h3><span>{derived.hourly.length}</span></div>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Час</th><th>Событий</th></tr></thead>
              <tbody>{hourlyRows.map((row) => <tr key={row["Час"]}><td>{row["Час"]}</td><td>{row["Событий"]}</td></tr>)}</tbody>
            </table>
          </div>
          <div className="table-footer">
            <span className="table-meta">Показано {hourlyRows.length} из {derived.hourly.length}</span>
            <div className="pagination">
              <button className="button secondary" type="button" disabled={currentHourlyPage <= 1} onClick={() => setHourlyPage((value) => Math.max(1, value - 1))}>
                Назад
              </button>
              <span className="page-indicator">{currentHourlyPage} / {hourlyPageCount}</span>
              <button className="button secondary" type="button" disabled={currentHourlyPage >= hourlyPageCount} onClick={() => setHourlyPage((value) => Math.min(hourlyPageCount, value + 1))}>
                Вперёд
              </button>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header"><h3>Входы в зону по дням</h3><span>{derived.dailyEntries.length}</span></div>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Дата</th><th>Входов</th></tr></thead>
              <tbody>{dailyRows.map((row) => <tr key={row["Дата"]}><td>{row["Дата"]}</td><td>{row["Входов"]}</td></tr>)}</tbody>
            </table>
          </div>
          <div className="table-footer">
            <span className="table-meta">Показано {dailyRows.length} из {derived.dailyEntries.length}</span>
            <div className="pagination">
              <button className="button secondary" type="button" disabled={currentDailyPage <= 1} onClick={() => setDailyPage((value) => Math.max(1, value - 1))}>
                Назад
              </button>
              <span className="page-indicator">{currentDailyPage} / {dailyPageCount}</span>
              <button className="button secondary" type="button" disabled={currentDailyPage >= dailyPageCount} onClick={() => setDailyPage((value) => Math.min(dailyPageCount, value + 1))}>
                Вперёд
              </button>
            </div>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header"><h3>События по точкам доступа</h3><span>{derived.byAccessPoint.length}</span></div>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Точка доступа</th><th>Событий</th></tr></thead>
              <tbody>{derived.byAccessPoint.map((row) => <tr key={row["Точка доступа"]}><td>{row["Точка доступа"]}</td><td>{row["Событий"]}</td></tr>)}</tbody>
            </table>
          </div>
        </section>

        <section className="panel">
          <div className="panel-header"><h3>Топ типов событий</h3><span>{derived.topTypes.length}</span></div>
          <div className="table-scroll">
            <table className="data-table">
              <thead><tr><th>Тип события</th><th>Количество</th></tr></thead>
              <tbody>{derived.topTypes.map((row) => <tr key={row["Тип события"]}><td>{row["Тип события"]}</td><td>{row["Количество"]}</td></tr>)}</tbody>
            </table>
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header"><h3>Offline-события по камерам</h3><span>{derived.offlineBySource.length}</span></div>
        <div className="table-scroll">
          <table className="data-table">
            <thead><tr><th>Источник</th><th>Offline-событий</th></tr></thead>
            <tbody>
              {derived.offlineBySource.length ? derived.offlineBySource.map((row) => (
                <tr key={row["Источник"]}><td>{row["Источник"]}</td><td>{row["Offline-событий"]}</td></tr>
              )) : <tr><td colSpan={2}>Offline-события по камерам пока не зафиксированы.</td></tr>}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
