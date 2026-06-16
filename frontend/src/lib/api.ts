export type Dictionary = Record<string, string>;

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
  session_id?: string;
  event_scope?: string;
  source_name?: string;
  source_type?: string;
  access_point_name?: string;
  event_type?: string;
  timestamp?: number;
  message?: string;
  employee_name?: string;
  employee_id?: number | null;
  employee_number?: string;
  employee_department?: string;
  employee_status?: string;
  confidence?: number | null;
  identification_status?: string;
  identification_confidence?: number | null;
  incident_score?: number | null;
  snapshot_path?: string;
};

export type Employee = {
  id: number;
  full_name: string;
  display_name?: string;
  last_name?: string;
  first_name?: string;
  middle_name?: string;
  employee_number?: string;
  department?: string;
  position?: string;
  status?: string;
  hire_date?: number | null;
  source_system?: string;
  last_synced_at?: number | null;
  presence_status?: string;
  last_check_in_at?: number | null;
  last_check_out_at?: number | null;
};

export type EmployeeSyncState = {
  data_source?: string;
  sync_status?: string;
  last_synced_at?: number | null;
  last_error?: string;
  cache_mode?: string;
  updated_at?: number | null;
};

export type AccessPoint = {
  id: number;
  name: string;
  location?: string;
  description?: string;
};

export type ModelInfo = {
  name: string;
  label: string;
  available: boolean;
  path: string;
};

export type AttendanceRecord = {
  id: number;
  employee_id: number;
  employee_name: string;
  employee_number?: string;
  department?: string;
  position?: string;
  access_point_id?: number | null;
  access_point_name?: string;
  check_in_at: number;
  check_out_at?: number | null;
  status: string;
  model_name?: string;
  source_type?: string;
  detection_confidence?: number | null;
  duration_seconds: number;
};

export type AttendanceTodayPayload = {
  day: string;
  summary: {
    check_ins: number;
    check_outs: number;
    currently_on_site: number;
    average_duration_minutes: number;
  };
  items: AttendanceRecord[];
};

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
  attendance_today: AttendanceTodayPayload;
  incident_queue: Array<Record<string, string | number>>;
  video_sources: VideoSource[];
  worker_statuses: WorkerStatus[];
  recent_incidents: Incident[];
  recent_events: EventItem[];
};

export type TelemetryPayload = {
  telemetry: Record<string, unknown>;
  operational: {
    readiness: string;
    issues: string[];
    coverage_ratio: number;
  };
};

export type FrameAnalysisResponse = {
  model_name: string;
  processing_time_ms: number;
  person_count: number;
  detections: Array<{ class_name: string; confidence: number; box: number[] }>;
  image_width: number;
  image_height: number;
  annotated_image_base64: string;
};

export type AttendanceCheckpointResponse = {
  attendance_session_id: number;
  attendance_status: "check_in" | "check_out";
  event_id: string;
  event_type: string;
  timestamp: number;
  message: string;
  employee: {
    id: number;
    full_name: string;
    employee_number?: string;
    department?: string;
    position?: string;
  };
  access_point_name?: string;
  analysis: FrameAnalysisResponse;
};

const jsonHeaders = {
  "Content-Type": "application/json"
};

function mapApiError(detail: string) {
  if (detail.startsWith("employee_inactive:")) return "Выбранный сотрудник не активен и не может быть отмечен.";
  if (detail.startsWith("employee_not_found:")) return "Сотрудник не найден.";
  if (detail === "employee_not_found") return "Сотрудник не найден.";
  if (detail.startsWith("access_point_not_found:")) return "Точка доступа не найдена.";
  if (detail === "person_not_detected") return "Человек в кадре не обнаружен.";
  if (detail === "multiple_people_detected") return "В кадре должно быть только одно лицо/один человек.";
  return detail;
}

async function api<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const text = await response.text();
    let parsedDetail = "";
    try {
      const payload = JSON.parse(text) as { detail?: string };
      if (payload.detail) {
        parsedDetail = payload.detail;
      }
    } catch {
      // Fall through to plain-text error handling.
    }
    throw new Error(mapApiError(parsedDetail || text || `request_failed:${response.status}`));
  }
  return response.json() as Promise<T>;
}

export const apiClient = {
  dashboardSummary: () => api<DashboardSummary>("/api/v1/dashboard/summary?event_limit=400"),
  telemetry: () => api<TelemetryPayload>("/api/v1/telemetry"),
  incidents: () => api<{ items: Incident[]; summary: Record<string, unknown> }>("/api/v1/incidents?limit=500"),
  events: (limit = 1000) => api<{ items: EventItem[] }>(`/api/v1/events?limit=${limit}`),
  sources: () => api<{ items: VideoSource[] }>("/api/v1/video-sources"),
  workerStatuses: () => api<{ items: WorkerStatus[] }>("/api/v1/worker-status"),
  settings: () => api<{ items: Dictionary }>("/api/v1/system/settings"),
  models: () => api<{ items: ModelInfo[] }>("/api/v1/models"),
  accessPoints: () => api<{ items: AccessPoint[] }>("/api/v1/access-points"),
  attendanceToday: () => api<AttendanceTodayPayload>("/api/v1/attendance/today"),
  employees: () => api<{ items: Employee[] }>("/api/v1/employees"),
  employeeSyncState: () => api<{ item: EmployeeSyncState | null }>("/api/v1/employees/sync-state"),
  auditLogs: () => api<{ items: Array<Record<string, unknown>> }>("/api/v1/audit-logs?limit=200"),
  createEmployee: (payload: Record<string, unknown>) =>
    api<{ ok: boolean }>("/api/v1/employees", { method: "POST", headers: jsonHeaders, body: JSON.stringify(payload) }),
  updateEmployee: (employeeId: number, payload: Record<string, unknown>) =>
    api<{ ok: boolean; employee_id: number }>(`/api/v1/employees/${employeeId}`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  updateEmployeeStatus: (employeeId: number, payload: Record<string, unknown>) =>
    api<{ ok: boolean; employee_id: number; status: string }>(`/api/v1/employees/${employeeId}/status`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  deleteEmployee: (employeeId: number, actorName = "react-ui", actorRole = "admin") =>
    api<{ ok: boolean; employee_id: number; deleted: Record<string, number> }>(
      `/api/v1/employees/${employeeId}?actor_name=${encodeURIComponent(actorName)}&actor_role=${encodeURIComponent(actorRole)}`,
      { method: "DELETE" }
    ),
  linkEventToEmployee: (eventId: string, payload: Record<string, unknown>) =>
    api<{ event_id: string; employee_id: number; identification_status: string }>(`/api/v1/events/${eventId}/link`, {
      method: "PUT",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
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
    }),
  analyzeFrame: (payload: Record<string, unknown>) =>
    api<FrameAnalysisResponse>("/api/v1/analyze-frame", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    }),
  attendanceCheckpoint: (payload: Record<string, unknown>) =>
    api<AttendanceCheckpointResponse>("/api/v1/attendance/checkpoint", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify(payload)
    })
};
