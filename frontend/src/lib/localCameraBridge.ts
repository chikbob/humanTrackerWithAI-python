export type CameraBridgeProbe = {
  camera_index: number;
  width: number;
  height: number;
  ok: boolean;
  backend?: string;
  attempts?: string[];
};

export type CameraBridgeResolved = {
  probe: CameraBridgeProbe;
  probeText: string;
  sourceLabel: string;
  streamUrl: string;
};

const LOCALHOST_BRIDGE_ORIGIN = "http://127.0.0.1:8123";

export function isWindowsClientBrowser() {
  const navigatorWithUserAgentData = navigator as Navigator & { userAgentData?: { platform?: string } };
  const platformValue = navigatorWithUserAgentData.userAgentData?.platform || navigator.platform || navigator.userAgent;
  return /win/i.test(platformValue);
}

function isLocalPageOrigin() {
  const hostname = window.location.hostname;
  return hostname === "127.0.0.1" || hostname === "localhost" || hostname === "::1";
}

async function fetchProbe(url: string): Promise<CameraBridgeProbe> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request_failed:${response.status}`);
  }
  return response.json() as Promise<CameraBridgeProbe>;
}

export async function resolveWindowsCameraFallback(cameraIndex: number) {
  const embeddedProbeUrl = `/api/v1/local-camera/probe?camera_index=${cameraIndex}&width=640&height=480`;
  const embeddedStreamUrl = `/api/v1/local-camera/stream?camera_index=${cameraIndex}&width=640&height=480`;
  const localBridgeProbeUrl = `${LOCALHOST_BRIDGE_ORIGIN}/camera/probe?camera_index=${cameraIndex}&width=640&height=480`;
  const localBridgeStreamUrl = `${LOCALHOST_BRIDGE_ORIGIN}/camera/stream?camera_index=${cameraIndex}&width=640&height=480`;

  if (isLocalPageOrigin()) {
    try {
      const probe = await fetchProbe(embeddedProbeUrl);
      return {
        probe,
        probeText: JSON.stringify(probe, null, 2),
        sourceLabel: "Embedded API fallback",
        streamUrl: embeddedStreamUrl
      } satisfies CameraBridgeResolved;
    } catch (backendError) {
      const backendMessage = backendError instanceof Error ? backendError.message : String(backendError);
      throw new Error(`Embedded API fallback unavailable: ${backendMessage}`);
    }
  }

  if (window.isSecureContext) {
    throw new Error(
      "Удаленная HTTPS-страница не может открыть локальный Windows HTTP camera bridge. Запустите проект на Windows через scripts\\start_windows_local_api.ps1 и откройте http://127.0.0.1:8000."
    );
  }

  try {
    const probe = await fetchProbe(localBridgeProbeUrl);
    return {
      probe,
      probeText: JSON.stringify(probe, null, 2),
      sourceLabel: "Windows localhost bridge",
      streamUrl: localBridgeStreamUrl
    } satisfies CameraBridgeResolved;
  } catch (bridgeError) {
    try {
      const probe = await fetchProbe(embeddedProbeUrl);
      return {
        probe,
        probeText: JSON.stringify(probe, null, 2),
        sourceLabel: "Embedded API fallback",
        streamUrl: embeddedStreamUrl
      } satisfies CameraBridgeResolved;
    } catch (backendError) {
      const bridgeMessage = bridgeError instanceof Error ? bridgeError.message : String(bridgeError);
      const backendMessage = backendError instanceof Error ? backendError.message : String(backendError);
      throw new Error(`Windows localhost bridge unavailable: ${bridgeMessage}. Embedded API fallback unavailable: ${backendMessage}`);
    }
  }
}
