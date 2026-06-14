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

async function fetchProbe(url: string): Promise<CameraBridgeProbe> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `request_failed:${response.status}`);
  }
  return response.json() as Promise<CameraBridgeProbe>;
}

export async function resolveWindowsCameraFallback(cameraIndex: number) {
  const localBridgeProbeUrl = `${LOCALHOST_BRIDGE_ORIGIN}/camera/probe?camera_index=${cameraIndex}&width=640&height=480`;
  try {
    const probe = await fetchProbe(localBridgeProbeUrl);
    return {
      probe,
      probeText: JSON.stringify(probe, null, 2),
      sourceLabel: "Windows localhost bridge",
      streamUrl: `${LOCALHOST_BRIDGE_ORIGIN}/camera/stream?camera_index=${cameraIndex}&width=640&height=480`
    } satisfies CameraBridgeResolved;
  } catch (bridgeError) {
    const backendProbeUrl = `/api/v1/local-camera/probe?camera_index=${cameraIndex}&width=640&height=480`;
    try {
      const probe = await fetchProbe(backendProbeUrl);
      return {
        probe,
        probeText: JSON.stringify(probe, null, 2),
        sourceLabel: "Embedded API fallback",
        streamUrl: `/api/v1/local-camera/stream?camera_index=${cameraIndex}&width=640&height=480`
      } satisfies CameraBridgeResolved;
    } catch (backendError) {
      const bridgeMessage = bridgeError instanceof Error ? bridgeError.message : String(bridgeError);
      const backendMessage = backendError instanceof Error ? backendError.message : String(backendError);
      throw new Error(`Windows localhost bridge unavailable: ${bridgeMessage}. Embedded API fallback unavailable: ${backendMessage}`);
    }
  }
}
