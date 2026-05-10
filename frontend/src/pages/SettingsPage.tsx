import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Dictionary } from "../lib/api";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["system-settings"], queryFn: apiClient.settings });
  const [draft, setDraft] = useState<Dictionary>({});

  useEffect(() => {
    if (data?.items) setDraft(data.items);
  }, [data]);

  const mutation = useMutation({
    mutationFn: () => apiClient.updateSettings(draft),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["system-settings"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    mutation.mutate();
  }

  if (isLoading) return <div className="page-state">Загружаю настройки…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить настройки.</div>;

  const editableKeys = [
    "model_name",
    "confidence_threshold",
    "inference_size",
    "tracker_type",
    "incident_score_threshold",
    "notifications_enabled",
    "incident_notify_min_severity"
  ];

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">System Config</span>
        <h2>Настройки системы</h2>
        <p>Редактирование идёт пакетом через API и не вызывает полный rerun интерфейса.</p>
      </div>

      <section className="panel">
        <form className="form-grid settings-grid" onSubmit={onSubmit}>
          {editableKeys.map((key) => (
            <label key={key} className="field-block">
              <span>{key}</span>
              <input className="input" value={draft[key] || ""} onChange={(event) => setDraft((current) => ({ ...current, [key]: event.target.value }))} />
            </label>
          ))}
          <div className="button-row">
            <button className="button" type="submit" disabled={mutation.isPending}>Сохранить настройки</button>
          </div>
        </form>
      </section>
    </section>
  );
}
