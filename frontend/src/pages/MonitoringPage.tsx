import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "../lib/api";

type DeviceOption = {
  deviceId: string;
  label: string;
};

export function MonitoringPage() {
  const { data: sources } = useQuery({ queryKey: ["video-sources"], queryFn: apiClient.sources, refetchInterval: 10_000 });
  const { data: telemetry } = useQuery({ queryKey: ["telemetry"], queryFn: apiClient.telemetry, refetchInterval: 10_000 });
  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [activeDeviceId, setActiveDeviceId] = useState("");
  const [isCameraEnabled, setIsCameraEnabled] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const browserSources = useMemo(() => (sources?.items || []).filter((item) => item.source_type === "browser_camera"), [sources]);
  const serverSources = useMemo(() => (sources?.items || []).filter((item) => item.source_type !== "browser_camera"), [sources]);

  useEffect(() => {
    async function readDevices() {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const items = await navigator.mediaDevices.enumerateDevices();
      const cameras = items
        .filter((item) => item.kind === "videoinput")
        .map((item, index) => ({ deviceId: item.deviceId, label: item.label || `Камера ${index + 1}` }));
      setDevices(cameras);
      if (!activeDeviceId && cameras[0]) setActiveDeviceId(cameras[0].deviceId);
    }
    void readDevices();
  }, [activeDeviceId]);

  useEffect(() => {
    async function startCamera() {
      if (!isCameraEnabled || !activeDeviceId || !navigator.mediaDevices?.getUserMedia) return;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: { ideal: activeDeviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    }

    void startCamera();
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, [activeDeviceId, isCameraEnabled]);

  return (
    <section className="page-grid">
      <div className="page-heading">
        <span className="eyebrow">Operator Workspace</span>
        <h2>Операторский мониторинг</h2>
        <p>
          Локальная камера MacBook и камера телефона на проде теперь рассматриваются как browser-device sources:
          захват идёт у клиента, а не через блокирующий серверный rerun.
        </p>
      </div>

      <div className="content-grid two-columns">
        <section className="panel media-panel">
          <div className="panel-header">
            <h3>Камера текущего устройства</h3>
            <span>{devices.length} доступно</span>
          </div>
          <div className="device-toolbar">
            <select className="input like-select" value={activeDeviceId} onChange={(event) => setActiveDeviceId(event.target.value)}>
              {devices.map((device) => <option key={device.deviceId} value={device.deviceId}>{device.label}</option>)}
            </select>
            <button className="button" onClick={() => setIsCameraEnabled((value) => !value)}>
              {isCameraEnabled ? "Остановить preview" : "Запустить preview"}
            </button>
          </div>
          <div className="video-frame">
            <video ref={videoRef} muted playsInline />
          </div>
          <p className="compact-note">
            На локальном MacBook это даст встроенную камеру. На проде через HTTPS тот же экран сможет открыть
            камеру ноутбука или телефона пользователя без перезапуска всего UI.
          </p>
        </section>

        <section className="panel">
          <div className="panel-header">
            <h3>Состояние контура</h3>
            <span>{telemetry?.operational.readiness || "unknown"}</span>
          </div>
          <div className="list-stack">
            <article className="stat-line">
              <strong>Browser/WebRTC sources</strong>
              <span>{browserSources.length}</span>
            </article>
            <article className="stat-line">
              <strong>Server-managed sources</strong>
              <span>{serverSources.length}</span>
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
