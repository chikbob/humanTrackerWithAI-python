export type DeviceOption = {
  deviceId: string;
  label: string;
};

export type CameraDiagnosticsDevice = {
  deviceId: string;
  groupId: string;
  kind: string;
  label: string;
};

export type CameraStartAttempt = {
  label: string;
  constraints: MediaStreamConstraints;
  startedAt: string;
  durationMs?: number;
  ok: boolean;
  errorName?: string;
  errorMessage?: string;
};

export type CameraDiagnosticsReport = {
  startedAt: string;
  finishedAt?: string;
  userAgent: string;
  platform: string;
  secureContext: boolean;
  visibilityState: string;
  mediaDevicesAvailable: boolean;
  enumerateDevicesAvailable: boolean;
  permissionsApiAvailable: boolean;
  permissionState: string;
  devicesBefore: CameraDiagnosticsDevice[];
  devicesAfter: CameraDiagnosticsDevice[];
  attempts: CameraStartAttempt[];
  selectedDeviceId: string;
  finalTrackLabel?: string;
  finalTrackSettings?: MediaTrackSettings;
  finalTrackConstraints?: MediaTrackConstraints;
  finalErrorName?: string;
  finalErrorMessage?: string;
};

type StartCameraOptions = {
  activeDeviceId?: string;
  preferredWidth?: number;
  preferredHeight?: number;
  onDebugReport?: (report: CameraDiagnosticsReport) => void;
};

type ListDevicesOptions = {
  skipPermissionProbe?: boolean;
};

const CAMERA_RELEASE_DELAY_MS = 1200;
let lastCameraReleaseAt = 0;
let permissionProbeCompleted = false;

function normalizeCameraLabel(label: string, index: number) {
  const normalized = label.trim();
  return normalized || `Видеоустройство ${index + 1}`;
}

function navigatorPlatform() {
  const userAgentData = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData;
  return userAgentData?.platform || navigator.platform || "unknown";
}

function isWindowsClient() {
  const platformValue = navigatorPlatform();
  return /win/i.test(platformValue);
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function markCameraReleased() {
  lastCameraReleaseAt = Date.now();
}

async function waitForCameraRelease() {
  const elapsed = Date.now() - lastCameraReleaseAt;
  if (elapsed < CAMERA_RELEASE_DELAY_MS) {
    await wait(CAMERA_RELEASE_DELAY_MS - elapsed);
  }
}

function stopMediaStream(stream: MediaStream | null | undefined) {
  if (!stream) return;
  stream.getTracks().forEach((track) => track.stop());
  markCameraReleased();
}

function sanitizeDevices(devices: MediaDeviceInfo[]): CameraDiagnosticsDevice[] {
  return devices
    .filter((item) => item.kind === "videoinput")
    .map((item, index) => ({
      deviceId: item.deviceId || `index-${index}`,
      groupId: item.groupId || "",
      kind: item.kind,
      label: normalizeCameraLabel(item.label, index)
    }));
}

async function readCameraPermissionState() {
  if (!navigator.permissions?.query) {
    return "unsupported";
  }

  try {
    const status = await navigator.permissions.query({ name: "camera" as PermissionName });
    return status.state;
  } catch {
    return "unavailable";
  }
}

async function collectVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  return sanitizeDevices(await navigator.mediaDevices.enumerateDevices());
}

function cloneConstraints(constraints: MediaStreamConstraints): MediaStreamConstraints {
  return JSON.parse(JSON.stringify(constraints)) as MediaStreamConstraints;
}

export function formatCameraDiagnosticsReport(report: CameraDiagnosticsReport) {
  return JSON.stringify(report, null, 2);
}

export function stopCameraStream(stream: MediaStream | null | undefined) {
  stopMediaStream(stream);
}

export async function listVideoInputDevices(options: ListDevicesOptions = {}): Promise<DeviceOption[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }

  if (isWindowsClient()) {
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices
      .filter((item) => item.kind === "videoinput")
      .map((item, index) => ({
        deviceId: item.deviceId,
        label: normalizeCameraLabel(item.label, index)
      }));
  }

  let temporaryStream: MediaStream | null = null;
  try {
    const initialDevices = await navigator.mediaDevices.enumerateDevices();
    const hasHiddenLabels = initialDevices.some((item) => item.kind === "videoinput" && !item.label.trim());
    if (hasHiddenLabels && navigator.mediaDevices.getUserMedia && !options.skipPermissionProbe && !permissionProbeCompleted) {
      temporaryStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      permissionProbeCompleted = true;
    }
  } catch {
    // Ignore permission and device-access errors here; fallback labels will still be used.
  } finally {
    stopMediaStream(temporaryStream);
    await waitForCameraRelease();
  }

  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((item) => item.kind === "videoinput")
    .map((item, index) => ({
      deviceId: item.deviceId,
      label: normalizeCameraLabel(item.label, index)
    }));
}

function buildCameraAttempts({ activeDeviceId, preferredWidth = 1280, preferredHeight = 720 }: StartCameraOptions): MediaStreamConstraints[] {
  const attempts: MediaStreamConstraints[] = [];
  const lowResolutionConstraints = {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 24, max: 30 }
  };
  const sizedConstraints = {
    width: { ideal: preferredWidth },
    height: { ideal: preferredHeight },
    frameRate: { ideal: 24, max: 30 }
  };

  if (activeDeviceId && activeDeviceId !== "default") {
    attempts.push({
      video: {
        deviceId: { exact: activeDeviceId },
        ...lowResolutionConstraints
      },
      audio: false
    });
    attempts.push({
      video: {
        deviceId: { ideal: activeDeviceId },
        ...sizedConstraints
      },
      audio: false
    });
    attempts.push({
      video: {
        deviceId: { ideal: activeDeviceId }
      },
      audio: false
    });
  }

  attempts.push({
    video: lowResolutionConstraints,
    audio: false
  });
  attempts.push({
    video: sizedConstraints,
    audio: false
  });
  attempts.push({
    video: true,
    audio: false
  });

  return attempts;
}

function buildAlternateDeviceAttempts(devices: CameraDiagnosticsDevice[], activeDeviceId: string): MediaStreamConstraints[] {
  const alternates = devices.filter((device) => device.deviceId && device.deviceId !== activeDeviceId);
  const attempts: MediaStreamConstraints[] = [];
  for (const device of alternates) {
    attempts.push({
      video: {
        deviceId: { exact: device.deviceId },
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 24, max: 30 }
      },
      audio: false
    });
    attempts.push({
      video: {
        deviceId: { exact: device.deviceId }
      },
      audio: false
    });
  }
  return attempts;
}

function canRetryCameraStart(error: unknown) {
  if (!(error instanceof DOMException)) {
    return false;
  }

  return ["NotReadableError", "OverconstrainedError", "AbortError"].includes(error.name);
}

export function getCameraStartErrorMessage(error: unknown) {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") {
      return "Браузер не получил доступ к камере. Проверьте разрешение на использование камеры для этого сайта.";
    }
    if (error.name === "NotFoundError") {
      return "Камера не найдена. Проверьте, что устройство подключено и доступно в системе.";
    }
    if (error.name === "NotReadableError") {
      return "Не удалось запустить видеопоток. На Windows это обычно означает, что камера занята другим приложением или драйвер не принимает текущий режим.";
    }
    if (error.name === "OverconstrainedError") {
      return "Браузер не смог подобрать совместимые параметры камеры. Поток будет запущен только после выбора другого устройства или режима.";
    }
  }

  return "Не удалось запустить камеру.";
}

export async function startCameraStream(options: StartCameraOptions): Promise<MediaStream> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Браузер не поддерживает доступ к камере.");
  }

  await waitForCameraRelease();
  const devicesBefore = await collectVideoDevices();
  const attempts = [
    ...buildCameraAttempts(options),
    ...buildAlternateDeviceAttempts(devicesBefore, options.activeDeviceId || "")
  ];
  let lastError: unknown = null;
  const report: CameraDiagnosticsReport = {
    startedAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    platform: navigatorPlatform(),
    secureContext: window.isSecureContext,
    visibilityState: document.visibilityState,
    mediaDevicesAvailable: Boolean(navigator.mediaDevices),
    enumerateDevicesAvailable: Boolean(navigator.mediaDevices?.enumerateDevices),
    permissionsApiAvailable: Boolean(navigator.permissions?.query),
    permissionState: await readCameraPermissionState(),
    devicesBefore,
    devicesAfter: [],
    attempts: [],
    selectedDeviceId: options.activeDeviceId || "default"
  };

  for (let index = 0; index < attempts.length; index += 1) {
    const constraints = attempts[index];
    const attemptLabel = `attempt_${index + 1}`;
    const attempt: CameraStartAttempt = {
      label: attemptLabel,
      constraints: cloneConstraints(constraints),
      startedAt: new Date().toISOString(),
      ok: false
    };
    const startedAt = performance.now();
    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      attempt.durationMs = Math.round(performance.now() - startedAt);
      if (!stream.getVideoTracks().length) {
        stopMediaStream(stream);
        throw new DOMException("Browser returned a stream without video tracks.", "NotReadableError");
      }
      const [videoTrack] = stream.getVideoTracks();
      report.finishedAt = new Date().toISOString();
      report.devicesAfter = await collectVideoDevices();
      report.finalTrackLabel = videoTrack.label;
      report.finalTrackSettings = videoTrack.getSettings();
      report.finalTrackConstraints = videoTrack.getConstraints();
      attempt.ok = true;
      report.attempts.push(attempt);
      options.onDebugReport?.(report);
      console.info("[camera-debug] startCameraStream success", report);
      return stream;
    } catch (error) {
      attempt.durationMs = Math.round(performance.now() - startedAt);
      attempt.errorName = error instanceof DOMException ? error.name : error instanceof Error ? error.name : "UnknownError";
      attempt.errorMessage = error instanceof Error ? error.message : String(error);
      report.attempts.push(attempt);
      lastError = error;
      if (!canRetryCameraStart(error)) {
        report.finishedAt = new Date().toISOString();
        report.devicesAfter = await collectVideoDevices();
        report.finalErrorName = attempt.errorName;
        report.finalErrorMessage = attempt.errorMessage;
        options.onDebugReport?.(report);
        console.error("[camera-debug] startCameraStream failed", report);
        throw error;
      }
      if (isWindowsClient()) {
        await wait(900 + index * 250);
      }
    }
  }

  report.finishedAt = new Date().toISOString();
  report.devicesAfter = await collectVideoDevices();
  report.finalErrorName = lastError instanceof DOMException ? lastError.name : lastError instanceof Error ? lastError.name : "UnknownError";
  report.finalErrorMessage = lastError instanceof Error ? lastError.message : String(lastError);
  options.onDebugReport?.(report);
  console.error("[camera-debug] startCameraStream failed", report);
  throw lastError ?? new Error("Не удалось запустить камеру.");
}

export async function syncVideoInputDevices(
  activeDeviceId: string,
  setDevices: (devices: DeviceOption[]) => void,
  setActiveDeviceId: (deviceId: string) => void,
  options: ListDevicesOptions & { fallbackLabel?: string } = {}
) {
  const cameras = await listVideoInputDevices(options);
  if (cameras.length > 0) {
    setDevices(cameras);
    if (!activeDeviceId || cameras.every((camera) => camera.deviceId !== activeDeviceId)) {
      setActiveDeviceId(cameras[0].deviceId);
    }
    return cameras;
  }

  if (options.fallbackLabel) {
    const fallbackDevice = { deviceId: "default", label: options.fallbackLabel };
    setDevices([fallbackDevice]);
    if (!activeDeviceId) {
      setActiveDeviceId(fallbackDevice.deviceId);
    }
    return [fallbackDevice];
  }

  setDevices([]);
  return [];
}
