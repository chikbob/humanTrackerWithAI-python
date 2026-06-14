import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient, AttendanceCheckpointResponse, Employee } from "../lib/api";
import { resolveWindowsCameraFallback } from "../lib/localCameraBridge";
import {
  CameraDiagnosticsReport,
  DeviceOption,
  formatCameraDiagnosticsReport,
  getCameraStartErrorMessage,
  stopCameraStream,
  startCameraStream,
  syncVideoInputDevices
} from "../lib/mediaDevices";

function captureFrame(element: HTMLVideoElement | HTMLImageElement | null): string | null {
  const width = element instanceof HTMLVideoElement ? element.videoWidth : element?.naturalWidth || 0;
  const height = element instanceof HTMLVideoElement ? element.videoHeight : element?.naturalHeight || 0;
  if (!element || !width || !height) return null;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.drawImage(element, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.84);
}

async function waitForVideoFrame(video: HTMLVideoElement) {
  const startedAt = performance.now();
  while (performance.now() - startedAt < 2500) {
    if (video.videoWidth > 0 && video.videoHeight > 0 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      return true;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 120));
  }
  return false;
}

function formatTimestamp(value?: number | null) {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString("ru-RU");
}

function buildEmployeeLabel(employee: Employee) {
  const parts = [
    employee.display_name || employee.full_name,
    employee.employee_number ? `таб. ${employee.employee_number}` : "",
    employee.department || ""
  ].filter(Boolean);
  return parts.join(" · ");
}

export function AttendancePage() {
  const queryClient = useQueryClient();
  const { data: employeesData } = useQuery({ queryKey: ["employees"], queryFn: apiClient.employees, refetchInterval: 20_000 });
  const { data: accessPointsData } = useQuery({ queryKey: ["access-points"], queryFn: apiClient.accessPoints });
  const { data: modelsData } = useQuery({ queryKey: ["models"], queryFn: apiClient.models });
  const { data: attendanceData, isLoading, error } = useQuery({
    queryKey: ["attendance-today"],
    queryFn: apiClient.attendanceToday,
    refetchInterval: 10_000
  });
  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState("");
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<number>(0);
  const [selectedAccessPointId, setSelectedAccessPointId] = useState<number>(0);
  const [selectedModelName, setSelectedModelName] = useState("yolov8s.pt");
  const [search, setSearch] = useState("");
  const [lastResult, setLastResult] = useState<AttendanceCheckpointResponse | null>(null);
  const [cameraError, setCameraError] = useState("");
  const [cameraDebugReport, setCameraDebugReport] = useState<CameraDiagnosticsReport | null>(null);
  const [serverCameraMode, setServerCameraMode] = useState(false);
  const [serverCameraProbe, setServerCameraProbe] = useState("");
  const [serverCameraError, setServerCameraError] = useState("");
  const [serverCameraStreamUrl, setServerCameraStreamUrl] = useState("");
  const [serverCameraSourceLabel, setServerCameraSourceLabel] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraStartRequestRef = useRef(0);
  const selectedCameraIndex = useMemo(() => {
    const index = devices.findIndex((device) => device.deviceId === activeDeviceId);
    return index >= 0 ? index : 0;
  }, [activeDeviceId, devices]);

  const availableEmployees = useMemo(
    () => (employeesData?.items || []).filter((employee) => (employee.status || "").trim() === "active"),
    [employeesData]
  );
  const filteredEmployees = useMemo(() => {
    const normalized = search.trim().toLowerCase();
    if (!normalized) return availableEmployees;
    return availableEmployees.filter((employee) => {
      const haystack = [employee.display_name, employee.full_name, employee.employee_number, employee.department].join(" ").toLowerCase();
      return haystack.includes(normalized);
    });
  }, [availableEmployees, search]);

  useEffect(() => {
    if (!availableEmployees.length) {
      setSelectedEmployeeId(0);
      return;
    }
    if (!selectedEmployeeId || availableEmployees.every((employee) => employee.id !== selectedEmployeeId)) {
      setSelectedEmployeeId(availableEmployees[0].id);
    }
  }, [availableEmployees, selectedEmployeeId]);

  useEffect(() => {
    if (!selectedAccessPointId && accessPointsData?.items[0]) {
      setSelectedAccessPointId(accessPointsData.items[0].id);
    }
  }, [accessPointsData, selectedAccessPointId]);

  useEffect(() => {
    if (!selectedModelName && modelsData?.items[0]) {
      setSelectedModelName(modelsData.items[0].name);
    }
  }, [modelsData, selectedModelName]);

  useEffect(() => {
    async function readDevices() {
      await syncVideoInputDevices(activeDeviceId, setDevices, setActiveDeviceId, {
        skipPermissionProbe: cameraEnabled || Boolean(streamRef.current)
      });
    }

    void readDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", readDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", readDevices);
  }, [activeDeviceId, cameraEnabled]);

  useEffect(() => () => {
    stopCameraStream(streamRef.current);
    streamRef.current = null;
  }, []);

  useEffect(() => {
    async function startCamera() {
      if (!cameraEnabled) {
        stopCameraStream(streamRef.current);
        streamRef.current = null;
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setServerCameraMode(false);
        setServerCameraError("");
        setServerCameraStreamUrl("");
        setServerCameraSourceLabel("");
        return;
      }

      const requestId = cameraStartRequestRef.current + 1;
      cameraStartRequestRef.current = requestId;
      stopCameraStream(streamRef.current);
      streamRef.current = null;
      setCameraError("");
      setCameraDebugReport(null);
      setServerCameraProbe("");
      setServerCameraError("");
      setServerCameraStreamUrl("");
      setServerCameraSourceLabel("");
      try {
        const stream = await startCameraStream({
          activeDeviceId,
          allowAlternateDevices: false,
          onDebugReport: setCameraDebugReport
        });
        if (cameraStartRequestRef.current !== requestId) {
          stopCameraStream(stream);
          return;
        }

        streamRef.current = stream;
        setServerCameraMode(false);
        const [videoTrack] = stream.getVideoTracks();
        const resolvedDeviceId = videoTrack?.getSettings().deviceId || activeDeviceId;
        const resolvedLabel = videoTrack?.label || "Камера устройства";
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          const hasFrame = await waitForVideoFrame(videoRef.current);
          if (!hasFrame) {
            stopCameraStream(stream);
            streamRef.current = null;
            setCameraError("Камера открылась, но браузер не получил ни одного видеокадра. Для встроенной камеры Windows это обычно означает сбой драйвера или пустой поток устройства.");
            return;
          }
        }
        const cameras = await syncVideoInputDevices(resolvedDeviceId, setDevices, setActiveDeviceId, {
          fallbackLabel: resolvedLabel,
          skipPermissionProbe: true
        });
        if (cameraStartRequestRef.current !== requestId) {
          stopCameraStream(stream);
          return;
        }
        if (!cameras.length && resolvedDeviceId && resolvedDeviceId !== activeDeviceId) {
          setActiveDeviceId(resolvedDeviceId);
        }
      } catch (error) {
        if (cameraStartRequestRef.current !== requestId) {
          return;
        }
        const message = getCameraStartErrorMessage(error);
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        const isWindows = /win/i.test(navigator.platform || navigator.userAgent);
        if (isWindows) {
          setServerCameraMode(true);
          setCameraError(`${message} Переключаюсь на Windows fallback.`);
          resolveWindowsCameraFallback(selectedCameraIndex)
            .then((resolved) => {
              setServerCameraProbe(resolved.probeText);
              setServerCameraStreamUrl(resolved.streamUrl);
              setServerCameraSourceLabel(resolved.sourceLabel);
              if (!resolved.probe.ok) {
                setServerCameraError(`${resolved.sourceLabel} не смог открыть локальную камеру.`);
              }
            })
            .catch((probeError) => {
              setServerCameraProbe("");
              setServerCameraError(probeError instanceof Error ? probeError.message : "Windows fallback недоступен.");
            });
          return;
        }
        setCameraError(message);
      }
    }

    void startCamera();
    return () => {
      cameraStartRequestRef.current += 1;
    };
  }, [activeDeviceId, cameraEnabled]);

  const checkpointMutation = useMutation({
    mutationFn: async () => {
      const imageBase64 = captureFrame(videoRef.current);
      const fallbackImageBase64 = imageBase64 || captureFrame(imageRef.current);
      if (!fallbackImageBase64) {
        throw new Error("Не удалось получить кадр с камеры.");
      }
      return apiClient.attendanceCheckpoint({
        employee_id: selectedEmployeeId,
        access_point_id: selectedAccessPointId || null,
        image_base64: fallbackImageBase64,
        model_name: selectedModelName,
        actor_name: "employee-kiosk",
        actor_role: "operator"
      });
    },
    onSuccess: (payload) => {
      setLastResult(payload);
      queryClient.invalidateQueries({ queryKey: ["attendance-today"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
      queryClient.invalidateQueries({ queryKey: ["employees"] });
    }
  });

  if (isLoading) return <div className="page-state">Загружаю контур КПП…</div>;
  if (error || !attendanceData) return <div className="page-state error">Не удалось загрузить данные проходной.</div>;

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">NeuroGate Access</span>
        <h2>КПП сотрудников</h2>
        <p>Сотрудник проходит к камере, система проверяет наличие человека в кадре выбранной моделью и фиксирует вход или выход в журнале предприятия.</p>
      </div>

      <div className="metrics-grid">
        <article className="metric-card">
          <div className="metric-value">{attendanceData.summary.check_ins}</div>
          <div className="metric-label">Приходов сегодня</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{attendanceData.summary.check_outs}</div>
          <div className="metric-label">Уходов сегодня</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{attendanceData.summary.currently_on_site}</div>
          <div className="metric-label">Сейчас на предприятии</div>
        </article>
        <article className="metric-card">
          <div className="metric-value">{attendanceData.summary.average_duration_minutes}</div>
          <div className="metric-label">Среднее время на смене, мин</div>
        </article>
      </div>

      <div className="content-grid two-columns">
        <section className="panel media-panel">
          <div className="panel-header">
            <div>
              <h3>Окно входа сотрудника</h3>
              <p className="panel-subtitle">Камера устройства используется как терминал самоотметки.</p>
            </div>
          </div>

          <div className="form-grid">
            <label className="field-block">
              <span>Поиск сотрудника</span>
              <input className="input" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="ФИО, табельный номер, отдел" />
            </label>
            <label className="field-block">
              <span>Сотрудник</span>
              <select className="input like-select" value={selectedEmployeeId} onChange={(event) => setSelectedEmployeeId(Number(event.target.value))}>
                {filteredEmployees.map((employee) => (
                  <option key={employee.id} value={employee.id}>{buildEmployeeLabel(employee)}</option>
                ))}
              </select>
            </label>
            <div className="triple-grid">
              <label className="field-block">
                <span>Точка доступа</span>
                <select className="input like-select" value={selectedAccessPointId} onChange={(event) => setSelectedAccessPointId(Number(event.target.value))}>
                  {(accessPointsData?.items || []).map((item) => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))}
                </select>
              </label>
              <label className="field-block">
                <span>Модель YOLO</span>
                <select className="input like-select" value={selectedModelName} onChange={(event) => setSelectedModelName(event.target.value)}>
                  {(modelsData?.items || []).filter((item) => item.available).map((item) => (
                    <option key={item.name} value={item.name}>{item.label}</option>
                  ))}
                </select>
              </label>
              <label className="field-block">
                <span>Камера</span>
                <select className="input like-select" value={activeDeviceId} onChange={(event) => setActiveDeviceId(event.target.value)}>
                  {!devices.length && <option value="">Сначала включите камеру</option>}
                  {devices.map((device) => (
                    <option key={device.deviceId} value={device.deviceId}>{device.label}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>

          <div className="button-row">
            <button className="button secondary" onClick={() => setCameraEnabled((value) => !value)}>
              {cameraEnabled ? "Остановить камеру" : "Включить камеру"}
            </button>
            <button className="button ghost" type="button" onClick={() => void syncVideoInputDevices(activeDeviceId, setDevices, setActiveDeviceId, { skipPermissionProbe: cameraEnabled || Boolean(streamRef.current) })}>
              Обновить камеры
            </button>
            <button className="button" disabled={!cameraEnabled || checkpointMutation.isPending || !selectedEmployeeId} onClick={() => checkpointMutation.mutate()}>
              {checkpointMutation.isPending ? "Распознаю..." : "Отметить вход / выход"}
            </button>
          </div>

          <div className="video-frame">
            {serverCameraMode ? (
              <img
                ref={imageRef}
                src={serverCameraStreamUrl}
                alt="Server local camera stream"
                onError={() => setServerCameraError(`${serverCameraSourceLabel || "Windows fallback"} не смог открыть поток камеры.`)}
              />
            ) : (
              <video ref={videoRef} muted playsInline />
            )}
          </div>

          {cameraError && (
            <div className="inline-warning">{cameraError}</div>
          )}
          {serverCameraError && (
            <div className="inline-warning">{serverCameraError}</div>
          )}
          {serverCameraProbe && (
            <details className="debug-panel">
              <summary>Диагностика fallback: {serverCameraSourceLabel || "camera bridge"}</summary>
              <pre>{serverCameraProbe}</pre>
            </details>
          )}
          {cameraDebugReport && (
            <details className="debug-panel">
              <summary>Диагностика камеры</summary>
              <pre>{formatCameraDiagnosticsReport(cameraDebugReport)}</pre>
            </details>
          )}
          {checkpointMutation.error instanceof Error && (
            <div className="inline-warning">{checkpointMutation.error.message}</div>
          )}
          {!availableEmployees.length && (
            <div className="inline-warning">Нет активных сотрудников для отметки входа и выхода.</div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Последний результат</h3>
              <p className="panel-subtitle">Подтверждение действия и аннотированный кадр.</p>
            </div>
          </div>
          {lastResult ? (
            <div className="list-stack">
              <article className="status-card">
                <strong>{lastResult.attendance_status === "check_in" ? "Сотрудник отмечен на работе" : "Сотрудник завершил смену"}</strong>
                <span>{lastResult.employee.full_name}</span>
              </article>
              <article className="stat-line">
                <strong>Точка доступа</strong>
                <span>{lastResult.access_point_name || "—"}</span>
              </article>
              <article className="stat-line">
                <strong>Модель</strong>
                <span>{lastResult.analysis.model_name}</span>
              </article>
              <article className="stat-line">
                <strong>Людей в кадре</strong>
                <span>{lastResult.analysis.person_count}</span>
              </article>
              <article className="stat-line">
                <strong>Время анализа</strong>
                <span>{lastResult.analysis.processing_time_ms} мс</span>
              </article>
              <img className="analysis-preview" src={lastResult.analysis.annotated_image_base64} alt="Annotated checkpoint frame" />
              <p className="compact-note">{lastResult.message}</p>
            </div>
          ) : (
            <div className="page-state">После первой отметки здесь появится подтверждение прохода и аннотированный кадр.</div>
          )}
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <div>
            <h3>Статистика и журнал за день</h3>
            <p className="panel-subtitle">Полная картина прихода и ухода сотрудников по текущей дате.</p>
          </div>
          <span>{attendanceData.items.length} записей</span>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Сотрудник</th>
                <th>КПП</th>
                <th>Приход</th>
                <th>Уход</th>
                <th>Статус</th>
                <th>Модель</th>
              </tr>
            </thead>
            <tbody>
              {attendanceData.items.map((item) => (
                <tr key={item.id}>
                  <td>{item.employee_name}</td>
                  <td>{item.access_point_name || "—"}</td>
                  <td>{formatTimestamp(item.check_in_at)}</td>
                  <td>{formatTimestamp(item.check_out_at)}</td>
                  <td>{item.status === "on_site" ? "На работе" : "Смена завершена"}</td>
                  <td>{item.model_name || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </section>
  );
}
