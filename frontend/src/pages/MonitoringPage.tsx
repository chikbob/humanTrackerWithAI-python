import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient, FrameAnalysisResponse } from "../lib/api";
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
  return canvas.toDataURL("image/jpeg", 0.82);
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

export function MonitoringPage() {
  const { data: sources } = useQuery({ queryKey: ["video-sources"], queryFn: apiClient.sources, refetchInterval: 10_000 });
  const { data: telemetry } = useQuery({ queryKey: ["telemetry"], queryFn: apiClient.telemetry, refetchInterval: 10_000 });
  const { data: modelsData } = useQuery({ queryKey: ["models"], queryFn: apiClient.models });
  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState("");
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);
  const [isAnalysisEnabled, setIsAnalysisEnabled] = useState(false);
  const [activeModel, setActiveModel] = useState("yolov8s.pt");
  const [lastAnalysis, setLastAnalysis] = useState<FrameAnalysisResponse | null>(null);
  const [cameraError, setCameraError] = useState("");
  const [cameraDebugReport, setCameraDebugReport] = useState<CameraDiagnosticsReport | null>(null);
  const [serverCameraMode, setServerCameraMode] = useState(false);
  const [serverCameraProbe, setServerCameraProbe] = useState<string>("");
  const [serverCameraError, setServerCameraError] = useState("");
  const [serverCameraStreamUrl, setServerCameraStreamUrl] = useState("");
  const [serverCameraSourceLabel, setServerCameraSourceLabel] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const analysisLockRef = useRef(false);
  const cameraStartRequestRef = useRef(0);

  const browserSources = useMemo(() => (sources?.items || []).filter((item) => item.source_type === "browser_camera"), [sources]);
  const serverSources = useMemo(() => (sources?.items || []).filter((item) => item.source_type !== "browser_camera"), [sources]);
  const selectedCameraIndex = useMemo(() => {
    const index = devices.findIndex((device) => device.deviceId === activeDeviceId);
    return index >= 0 ? index : 0;
  }, [activeDeviceId, devices]);

  useEffect(() => {
    async function readDevices() {
      await syncVideoInputDevices(activeDeviceId, setDevices, setActiveDeviceId, {
        skipPermissionProbe: isCameraEnabled || Boolean(streamRef.current)
      });
    }

    void readDevices();
    navigator.mediaDevices?.addEventListener?.("devicechange", readDevices);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", readDevices);
  }, [activeDeviceId, isCameraEnabled]);

  useEffect(() => () => {
    stopCameraStream(streamRef.current);
    streamRef.current = null;
  }, []);

  useEffect(() => {
    async function startCamera() {
      if (!isCameraEnabled) {
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
        setIsAnalysisEnabled(false);
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
  }, [activeDeviceId, isCameraEnabled]);

  const analyzeMutation = useMutation({
    mutationFn: async () => {
      const imageBase64 = captureFrame(videoRef.current);
      const fallbackImageBase64 = imageBase64 || captureFrame(imageRef.current);
      if (!fallbackImageBase64) {
        throw new Error("Нет доступного кадра для анализа.");
      }
      return apiClient.analyzeFrame({
        image_base64: fallbackImageBase64,
        model_name: activeModel
      });
    },
    onSuccess: (payload) => {
      setLastAnalysis(payload);
    },
    onSettled: () => {
      analysisLockRef.current = false;
    }
  });

  useEffect(() => {
    if (!isAnalysisEnabled || !isCameraEnabled) return;
    const timer = window.setInterval(() => {
      if (analysisLockRef.current) return;
      analysisLockRef.current = true;
      analyzeMutation.mutate();
    }, 2500);
    return () => window.clearInterval(timer);
  }, [analyzeMutation, isAnalysisEnabled, isCameraEnabled]);

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Realtime Monitoring</span>
        <h2>Операторский мониторинг</h2>
        <p>Экран снова выполняет основную задачу системы: получает видеокадр, прогоняет его через выбранную YOLO-модель и показывает обнаружение человека в реальном времени.</p>
      </div>

      <div className="content-grid two-columns">
        <section className="panel media-panel">
          <div className="panel-header">
            <div>
              <h3>Живой видеопоток устройства</h3>
              <p className="panel-subtitle">Камера MacBook, внешняя USB-камера или мобильный браузер по HTTPS.</p>
            </div>
            <span>{devices.length} камер</span>
          </div>
          <div className="triple-grid">
            <label className="field-block">
              <span>Камера</span>
              <select className="input like-select" value={activeDeviceId} onChange={(event) => setActiveDeviceId(event.target.value)}>
                {!devices.length && <option value="">Сначала запустите поток</option>}
                {devices.map((device) => <option key={device.deviceId} value={device.deviceId}>{device.label}</option>)}
              </select>
            </label>
            <label className="field-block">
              <span>Модель</span>
              <select className="input like-select" value={activeModel} onChange={(event) => setActiveModel(event.target.value)}>
                {(modelsData?.items || []).filter((item) => item.available).map((item) => (
                  <option key={item.name} value={item.name}>{item.label}</option>
                ))}
              </select>
            </label>
            <label className="field-block">
              <span>Режим</span>
              <select className="input like-select" value={isAnalysisEnabled ? "analysis" : "preview"} onChange={(event) => setIsAnalysisEnabled(event.target.value === "analysis")}>
                <option value="preview">Только превью</option>
                <option value="analysis">Непрерывный анализ</option>
              </select>
            </label>
          </div>
          <div className="button-row">
            <button className="button secondary" onClick={() => setIsCameraEnabled((value) => !value)}>
              {isCameraEnabled ? "Остановить поток" : "Запустить поток"}
            </button>
            <button className="button ghost" type="button" onClick={() => void syncVideoInputDevices(activeDeviceId, setDevices, setActiveDeviceId, { skipPermissionProbe: isCameraEnabled || Boolean(streamRef.current) })}>
              Обновить камеры
            </button>
            <button className="button" disabled={!isCameraEnabled || analyzeMutation.isPending} onClick={() => analyzeMutation.mutate()}>
              {analyzeMutation.isPending ? "Анализ..." : "Проверить текущий кадр"}
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
          {cameraError && <div className="inline-warning">{cameraError}</div>}
          {serverCameraError && <div className="inline-warning">{serverCameraError}</div>}
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
          {analyzeMutation.error instanceof Error && <div className="inline-warning">{analyzeMutation.error.message}</div>}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h3>Результат нейросетевого анализа</h3>
              <p className="panel-subtitle">Аннотированный кадр, количество людей и производительность модели.</p>
            </div>
          </div>
          {lastAnalysis ? (
            <div className="list-stack">
              <img className="analysis-preview" src={lastAnalysis.annotated_image_base64} alt="Annotated monitoring frame" />
              <article className="stat-line">
                <strong>Модель</strong>
                <span>{lastAnalysis.model_name}</span>
              </article>
              <article className="stat-line">
                <strong>Людей в кадре</strong>
                <span>{lastAnalysis.person_count}</span>
              </article>
              <article className="stat-line">
                <strong>Время анализа</strong>
                <span>{lastAnalysis.processing_time_ms} мс</span>
              </article>
              <article className="stat-line">
                <strong>Детекций</strong>
                <span>{lastAnalysis.detections.length}</span>
              </article>
            </div>
          ) : (
            <div className="page-state">Выполните анализ кадра, чтобы увидеть результат распознавания.</div>
          )}
        </section>
      </div>

      <div className="content-grid two-columns">
        <section className="panel">
          <div className="panel-header">
            <h3>Источники браузерных камер</h3>
            <span>{browserSources.length}</span>
          </div>
          <div className="list-stack">
            {browserSources.map((source) => (
              <article key={source.id} className="source-card">
                <div>
                  <strong>{source.name}</strong>
                  <p>{source.location || "Без локации"}</p>
                </div>
                <div className="muted-tag">{source.is_active ? "активен" : "выключен"}</div>
              </article>
            ))}
          </div>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>Состояние контура</h3>
            <span>{telemetry?.operational.readiness || "unknown"}</span>
          </div>
          <div className="list-stack">
            <article className="stat-line">
              <strong>Серверные источники</strong>
              <span>{serverSources.length}</span>
            </article>
            <article className="stat-line">
              <strong>Browser/WebRTC источники</strong>
              <span>{browserSources.length}</span>
            </article>
            <article className="stat-line">
              <strong>Readiness</strong>
              <span>{telemetry?.operational.readiness || "—"}</span>
            </article>
            {(telemetry?.operational.issues || []).map((issue) => (
              <article key={issue} className="inline-warning">{issue}</article>
            ))}
          </div>
        </section>
      </div>
    </section>
  );
}
