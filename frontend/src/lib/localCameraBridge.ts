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

type DesktopCompanionStatus = {
  session_id: string;
  connected: boolean;
  has_frame: boolean;
  updated_at?: number | null;
  age_ms?: number;
  camera_index?: number;
  width?: number;
  height?: number;
  source_label?: string;
  backend_label?: string;
  host_name?: string;
};

const LOCALHOST_BRIDGE_ORIGIN = "http://127.0.0.1:8123";
const DESKTOP_COMPANION_SESSION_STORAGE_KEY = "desktop-companion-session-id";

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

function createDesktopCompanionSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `desk-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `desk-${Math.random().toString(36).slice(2, 10)}`;
}

export function getDesktopCompanionSessionId() {
  const stored = window.localStorage.getItem(DESKTOP_COMPANION_SESSION_STORAGE_KEY);
  if (stored && stored.trim()) {
    return stored;
  }
  const generated = createDesktopCompanionSessionId();
  window.localStorage.setItem(DESKTOP_COMPANION_SESSION_STORAGE_KEY, generated);
  return generated;
}

async function fetchProbe(url: string): Promise<CameraBridgeProbe> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request_failed:${response.status}`);
  }
  return response.json() as Promise<CameraBridgeProbe>;
}

async function fetchDesktopCompanionStatus(sessionId: string): Promise<DesktopCompanionStatus> {
  const response = await fetch(`/api/v1/desktop-companion/status?session_id=${encodeURIComponent(sessionId)}`, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request_failed:${response.status}`);
  }
  return response.json() as Promise<DesktopCompanionStatus>;
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
    const sessionId = getDesktopCompanionSessionId();
    try {
      const status = await fetchDesktopCompanionStatus(sessionId);
      if (status.connected && status.has_frame) {
        return {
          probe: {
            camera_index: status.camera_index ?? cameraIndex,
            width: status.width ?? 640,
            height: status.height ?? 480,
            ok: true,
            backend: status.backend_label,
            attempts: []
          },
          probeText: JSON.stringify(status, null, 2),
          sourceLabel: status.source_label || "Windows desktop companion",
          streamUrl: `/api/v1/desktop-companion/stream?session_id=${encodeURIComponent(sessionId)}`
        } satisfies CameraBridgeResolved;
      }
    } catch (statusError) {
      throw new Error(
        `Windows desktop companion unavailable: ${
          statusError instanceof Error ? statusError.message : String(statusError)
        }. Session ID: ${sessionId}. Start scripts\\start_windows_remote_companion.ps1 -ServerUrl ${window.location.origin} -SessionId ${sessionId}`
      );
    }
    throw new Error(
      `Windows desktop companion is not connected. Session ID: ${sessionId}. Start scripts\\start_windows_remote_companion.ps1 -ServerUrl ${window.location.origin} -SessionId ${sessionId}`
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
