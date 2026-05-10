export type Dictionary = Record<string, string>;

export type DashboardSummary = {
  summary: {
    detections_today: number;
    entries_today: number;
    suspicious_today: number;
    online_cameras: number;
    total_events_today: number;
  };
  incidents_summary: {
    total: number;
    active: number;
    critical: number;
    high: number;
    overdue_active: number;
  };
  incident_queue: Array<Record<string, string | number>>;
  video_sources: VideoSource[];
  worker_statuses: WorkerStatus[];
  recent_incidents: Incident[];
  recent_events: EventItem[];
};

export type VideoSource = {
  id: number;
  name: string;
  source_type: string;
  source_url: string;
  location?: string;
  description?: string;
  is_active: boolean;
  enable_roi?: boolean;
  roi_x?: number;
  roi_y?: number;
  roi_w?: number;
  roi_h?: number;
  rule_count_enabled?: boolean;
  rule_n?: number;
  rule_t?: number;
  rule_disappear_enabled?: boolean;
  rule_disappear_seconds?: number;
  prolonged_presence_seconds?: number;
  ai_profile_override?: string;
  conf_threshold_override?: number | null;
  inference_size_override?: number | null;
  tracker_type_override?: string;
  incident_threshold_override?: number | null;
};

export type WorkerStatus = {
  source_id: number;
  status?: string;
  is_connected?: boolean;
  last_heartbeat?: number | null;
  last_frame_at?: number | null;
  fps?: number | null;
  reconnect_count?: number;
  last_error?: string;
};

export type Incident = {
  id: number;
  event_id?: string;
  source_id?: number | null;
  source_name?: string;
  zone_name?: string;
  incident_type: string;
  severity: string;
  status: string;
  confidence?: number;
  operator_comment?: string;
  assigned_to?: string;
  resolution_code?: string;
  resolution_notes?: string;
  started_at?: number;
  updated_at?: number;
};

export type EventItem = {
  event_id: string;
  source_name?: string;
  access_point_name?: string;
  event_type?: string;
  timestamp?: number;
  message?: string;
};

export type TelemetryPayload = {
  telemetry: Record<string, unknown>;
  operational: {
    readiness: string;
    issues: string[];
    coverage_ratio: number;
  };
};

const jsonHeaders = {
  "Content-Type": "application/json"
};

async function api<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `request_failed:${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const apiClient = {
  dashboardSummary: () => api<DashboardSummary>("/api/v1/dashboard/summary?event_limit=400"),
  telemetry: () => api<TelemetryPayload>("/api/v1/telemetry"),
  incidents: () => api<{ items: Incident[]; summary: Record<string, unknown> }>("/api/v1/incidents?limit=500"),
  sources: () => api<{ items: VideoSource[] }>("/api/v1/video-sources"),
  workerStatuses: () => api<{ items: WorkerStatus[] }>("/api/v1/worker-status"),
  settings: () => api<{ items: Dictionary }>("/api/v1/system/settings"),
  employees: () => api<{ items: Array<Record<string, unknown>> }>("/api/v1/employees"),
  auditLogs: () => api<{ items: Array<Record<string, unknown>> }>("/api/v1/audit-logs?limit=200"),
  createSource: (payload: Record<string, unknown>) =>
    api<{ ok: boolean }>("/api/v1/video-sources", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }),
  updateSource: (sourceId: number, payload: Record<string, unknown>) =>
    api<{ ok: boolean; source_id: number }>(`/api/v1/video-sources/${sourceId}`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  setSourceActive: (sourceId: number, isActive: boolean) =>
    api<{ source_id: number; is_active: boolean }>(`/api/v1/video-sources/${sourceId}/active?is_active=${isActive}`, { method: "PUT" }),
  updateIncidentStatus: (incidentId: number, payload: Record<string, unknown>) =>
    api<{ incident_id: number }>(`/api/v1/incidents/${incidentId}/status`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  updateSettings: (items: Dictionary) =>
    api<{ items: Dictionary }>("/api/v1/system/settings", {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify({ items })
    })
};
