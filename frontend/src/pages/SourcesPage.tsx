import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, VideoSource } from "../lib/api";

type SourceFormState = {
  name: string;
  source_type: string;
  source_url: string;
  location: string;
  description: string;
  is_active: boolean;
  enable_roi: boolean;
  roi_x: number;
  roi_y: number;
  roi_w: number;
  roi_h: number;
  rule_count_enabled: boolean;
  rule_n: number;
  rule_t: number;
  rule_disappear_enabled: boolean;
  rule_disappear_seconds: number;
  prolonged_presence_seconds: number;
  ai_profile_override: string;
  conf_threshold_override: number | null;
  inference_size_override: number | null;
  tracker_type_override: string;
  incident_threshold_override: number | null;
  actor_name: string;
  actor_role: string;
};

const defaultForm: SourceFormState = {
  name: "",
  source_type: "rtsp",
  source_url: "",
  location: "",
  description: "",
  is_active: true,
  enable_roi: true,
  roi_x: 20,
  roi_y: 20,
  roi_w: 60,
  roi_h: 60,
  rule_count_enabled: false,
  rule_n: 3,
  rule_t: 10,
  rule_disappear_enabled: true,
  rule_disappear_seconds: 5,
  prolonged_presence_seconds: 10,
  ai_profile_override: "",
  conf_threshold_override: null,
  inference_size_override: null,
  tracker_type_override: "",
  incident_threshold_override: null,
  actor_name: "react-ui",
  actor_role: "admin"
};

export function SourcesPage() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({ queryKey: ["video-sources"], queryFn: apiClient.sources });
  const [form, setForm] = useState<SourceFormState>(defaultForm);
  const [editingId, setEditingId] = useState<number | null>(null);

  const createMutation = useMutation({
    mutationFn: () => apiClient.createSource(form),
    onSuccess: () => {
      setForm(defaultForm);
      queryClient.invalidateQueries({ queryKey: ["video-sources"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  const updateMutation = useMutation({
    mutationFn: ({ sourceId, payload }: { sourceId: number; payload: Record<string, unknown> }) => apiClient.updateSource(sourceId, payload),
    onSuccess: () => {
      setEditingId(null);
      setForm(defaultForm);
      queryClient.invalidateQueries({ queryKey: ["video-sources"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  const toggleMutation = useMutation({
    mutationFn: ({ sourceId, isActive }: { sourceId: number; isActive: boolean }) => apiClient.setSourceActive(sourceId, isActive),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["video-sources"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    }
  });

  const submitLabel = editingId === null ? "Добавить источник" : "Сохранить изменения";

  function beginEdit(source: VideoSource) {
    setEditingId(source.id);
    setForm({
      ...defaultForm,
      ...source,
      actor_name: "react-ui",
      actor_role: "admin"
    });
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (editingId === null) {
      createMutation.mutate();
      return;
    }
    updateMutation.mutate({ sourceId: editingId, payload: form });
  }

  if (isLoading) return <div className="page-state">Загружаю источники…</div>;
  if (error || !data) return <div className="page-state error">Не удалось загрузить источники.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Source Registry</span>
        <h2>Источники видеоданных</h2>
        <p>Новый SPA управляет источниками без блокирующего полного перерендера всей системы.</p>
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-header">
            <h3>{editingId === null ? "Новый источник" : `Редактирование #${editingId}`}</h3>
          </div>
          <form className="form-grid" onSubmit={handleSubmit}>
            <input className="input" placeholder="Название" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} />
            <select className="input like-select" value={form.source_type} onChange={(event) => setForm({ ...form, source_type: event.target.value })}>
              <option value="rtsp">RTSP/IP</option>
              <option value="stream_url">HLS / HTTP</option>
              <option value="usb_camera">USB / local server cam</option>
              <option value="browser_camera">Device browser camera</option>
            </select>
            <input className="input" placeholder="URL / index / browser_camera" value={form.source_url} onChange={(event) => setForm({ ...form, source_url: event.target.value })} />
            <input className="input" placeholder="Локация" value={form.location} onChange={(event) => setForm({ ...form, location: event.target.value })} />
            <textarea className="input textarea" placeholder="Описание" value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} />
            <label className="toggle-row">
              <input type="checkbox" checked={form.is_active} onChange={(event) => setForm({ ...form, is_active: event.target.checked })} />
              <span>Активировать сразу</span>
            </label>
            <div className="button-row">
              <button className="button" type="submit" disabled={createMutation.isPending || updateMutation.isPending}>{submitLabel}</button>
              {editingId !== null && (
                <button className="button secondary" type="button" onClick={() => { setEditingId(null); setForm(defaultForm); }}>
                  Отменить
                </button>
              )}
            </div>
          </form>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>Реестр</h3>
            <span>{data.items.length} источников</span>
          </div>
          <div className="list-stack">
            {data.items.map((source) => (
              <article key={source.id} className="source-card">
                <div>
                  <strong>{source.name}</strong>
                  <p>{source.source_type} · {source.location || "без локации"}</p>
                </div>
                <div className="card-actions">
                  <button className="button ghost" onClick={() => beginEdit(source)}>Редактировать</button>
                  <button className="button secondary" onClick={() => toggleMutation.mutate({ sourceId: source.id, isActive: !source.is_active })}>
                    {source.is_active ? "Деактивировать" : "Активировать"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}
