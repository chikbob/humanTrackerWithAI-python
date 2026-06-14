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

export function shouldPreferWindowsLocalCameraFallback() {
  return isWindowsClientBrowser() && isLocalPageOrigin();
}

async function fetchProbe(url: string): Promise<CameraBridgeProbe> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request_failed:${response.status}`);
  }
  return response.json() as Promise<CameraBridgeProbe>;
}

function buildCameraIndexCandidates(preferredIndex: number) {
  const ordered = [preferredIndex, 0, 1, 2, 3, 4, 5];
  return Array.from(new Set(ordered.filter((value) => Number.isInteger(value) && value >= 0 && value <= 10)));
}

async function resolveFirstAvailableProbe(
  cameraIndices: number[],
  buildProbeUrl: (cameraIndex: number) => string,
  buildStreamUrl: (cameraIndex: number) => string,
  sourceLabel: string
) {
  const failures: string[] = [];
  for (const cameraIndex of cameraIndices) {
    try {
      const probe = await fetchProbe(buildProbeUrl(cameraIndex));
      if (probe.ok) {
        return {
          probe,
          probeText: JSON.stringify(
            {
              ...probe,
              requested_camera_indices: cameraIndices,
              resolved_camera_index: cameraIndex
            },
            null,
            2
          ),
          sourceLabel,
          streamUrl: buildStreamUrl(cameraIndex)
        } satisfies CameraBridgeResolved;
      }
      failures.push(`camera_index=${cameraIndex}: ${JSON.stringify(probe)}`);
    } catch (error) {
      failures.push(`camera_index=${cameraIndex}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  throw new Error(failures.join(" | "));
}

export async function resolveWindowsCameraFallback(cameraIndex: number) {
  const cameraIndices = buildCameraIndexCandidates(cameraIndex);
  const embeddedProbeUrl = (resolvedIndex: number) => `/api/v1/local-camera/probe?camera_index=${resolvedIndex}&width=640&height=480`;
  const embeddedStreamUrl = (resolvedIndex: number) => `/api/v1/local-camera/stream?camera_index=${resolvedIndex}&width=640&height=480`;
  const localBridgeProbeUrl = (resolvedIndex: number) => `${LOCALHOST_BRIDGE_ORIGIN}/camera/probe?camera_index=${resolvedIndex}&width=640&height=480`;
  const localBridgeStreamUrl = (resolvedIndex: number) => `${LOCALHOST_BRIDGE_ORIGIN}/camera/stream?camera_index=${resolvedIndex}&width=640&height=480`;

  if (isLocalPageOrigin()) {
    try {
      return await resolveFirstAvailableProbe(cameraIndices, embeddedProbeUrl, embeddedStreamUrl, "Embedded API fallback");
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
    return await resolveFirstAvailableProbe(cameraIndices, localBridgeProbeUrl, localBridgeStreamUrl, "Windows localhost bridge");
  } catch (bridgeError) {
    try {
      return await resolveFirstAvailableProbe(cameraIndices, embeddedProbeUrl, embeddedStreamUrl, "Embedded API fallback");
    } catch (backendError) {
      const bridgeMessage = bridgeError instanceof Error ? bridgeError.message : String(bridgeError);
      const backendMessage = backendError instanceof Error ? backendError.message : String(backendError);
      throw new Error(`Windows localhost bridge unavailable: ${bridgeMessage}. Embedded API fallback unavailable: ${backendMessage}`);
    }
  }
}
