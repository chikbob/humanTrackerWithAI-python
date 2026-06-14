export type DeviceOption = {
  deviceId: string;
  label: string;
};

type StartCameraOptions = {
  activeDeviceId?: string;
  preferredWidth?: number;
  preferredHeight?: number;
};

type ListDevicesOptions = {
  skipPermissionProbe?: boolean;
};

function normalizeCameraLabel(label: string, index: number) {
  const normalized = label.trim();
  return normalized || `Видеоустройство ${index + 1}`;
}

export async function listVideoInputDevices(options: ListDevicesOptions = {}): Promise<DeviceOption[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }

  let temporaryStream: MediaStream | null = null;
  try {
    const initialDevices = await navigator.mediaDevices.enumerateDevices();
    const hasHiddenLabels = initialDevices.some((item) => item.kind === "videoinput" && !item.label.trim());
    if (hasHiddenLabels && navigator.mediaDevices.getUserMedia && !options.skipPermissionProbe) {
      temporaryStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  } catch {
    // Ignore permission and device-access errors here; fallback labels will still be used.
  } finally {
    temporaryStream?.getTracks().forEach((track) => track.stop());
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
  const sizedConstraints = {
    width: { ideal: preferredWidth },
    height: { ideal: preferredHeight }
  };

  if (activeDeviceId && activeDeviceId !== "default") {
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
    video: sizedConstraints,
    audio: false
  });
  attempts.push({
    video: true,
    audio: false
  });

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

  const attempts = buildCameraAttempts(options);
  let lastError: unknown = null;

  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (!canRetryCameraStart(error)) {
        throw error;
      }
    }
  }

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
