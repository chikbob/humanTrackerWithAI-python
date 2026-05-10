import { FormEvent, useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, Dictionary } from "../lib/api";

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["system-settings"], queryFn: apiClient.settings });
  const { data: modelsData } = useQuery({ queryKey: ["models"], queryFn: apiClient.models });
  const { data: accessPointsData } = useQuery({ queryKey: ["access-points"], queryFn: apiClient.accessPoints });
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
    "active_access_point_id",
    "notifications_enabled",
    "incident_notify_min_severity"
  ];

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Runtime Configuration</span>
        <h2>Настройки системы</h2>
        <p>Управление рабочей моделью, параметрами анализа кадров и активной проходной без возврата к legacy Streamlit-контуру.</p>
      </div>

      <section className="panel">
        <form className="form-grid settings-grid" onSubmit={onSubmit}>
          <label className="field-block">
            <span>Модель YOLO по умолчанию</span>
            <select className="input like-select" value={draft.model_name || ""} onChange={(event) => setDraft((current) => ({ ...current, model_name: event.target.value }))}>
              {(modelsData?.items || []).filter((item) => item.available).map((item) => (
                <option key={item.name} value={item.name}>{item.label}</option>
              ))}
            </select>
          </label>
          <label className="field-block">
            <span>Трекер</span>
            <select className="input like-select" value={draft.tracker_type || ""} onChange={(event) => setDraft((current) => ({ ...current, tracker_type: event.target.value }))}>
              <option value="bytetrack">ByteTrack</option>
              <option value="botsort">BoT-SORT</option>
              <option value="detect_only">Только детекция</option>
            </select>
          </label>
          <label className="field-block">
            <span>Основная точка доступа</span>
            <select className="input like-select" value={draft.active_access_point_id || ""} onChange={(event) => setDraft((current) => ({ ...current, active_access_point_id: event.target.value }))}>
              <option value="">Не выбрана</option>
              {(accessPointsData?.items || []).map((item) => (
                <option key={item.id} value={String(item.id)}>{item.name}</option>
              ))}
            </select>
          </label>
          {editableKeys.filter((key) => !["model_name", "tracker_type", "active_access_point_id"].includes(key)).map((key) => (
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
