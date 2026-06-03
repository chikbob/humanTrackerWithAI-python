export type DeviceOption = {
  deviceId: string;
  label: string;
};

function normalizeCameraLabel(label: string, index: number) {
  const normalized = label.trim();
  return normalized || `Видеоустройство ${index + 1}`;
}

export async function listVideoInputDevices(): Promise<DeviceOption[]> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return [];
  }

  let temporaryStream: MediaStream | null = null;
  try {
    const initialDevices = await navigator.mediaDevices.enumerateDevices();
    const hasHiddenLabels = initialDevices.some((item) => item.kind === "videoinput" && !item.label.trim());
    if (hasHiddenLabels && navigator.mediaDevices.getUserMedia) {
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
