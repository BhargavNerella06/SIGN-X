/**
 * Camera selection and MediaStream lifecycle for the Live Camera panel.
 *
 * Kept free of React so it can be unit-tested with mocked MediaDevices
 * (see frontend/tests/camera.test.ts). Guarantees:
 *   - at most one open MediaStream and one capture loop at any time;
 *   - the previous stream's tracks are stopped before another camera is
 *     requested, and its capture loop cannot deliver frames afterwards;
 *   - a getUserMedia() call that resolves after a newer selection (or after
 *     dispose) is discarded and its tracks stopped (generation counter).
 *
 * Device IDs never leave the browser; only the chosen ID is remembered in
 * localStorage so the same camera is preferred on the next visit.
 */

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export type CameraStatus =
  | "requesting" // first permission / stream request
  | "switching" // replacing the stream with another camera
  | "connected"
  | "denied"
  | "no-camera"
  | "error"
  | "unavailable"; // API unsupported, or the active camera was disconnected

export interface CameraState {
  status: CameraStatus;
  devices: CameraDevice[];
  /** Camera currently streaming (null when none). */
  activeDeviceId: string | null;
  /** Camera the user picked; differs from activeDeviceId while switching or after a failure. */
  selectedDeviceId: string | null;
  error: string;
}

export interface FrameCapture {
  grab: () => Promise<Blob | null>;
  onFrame: (blob: Blob) => void;
  intervalMs: number;
  /** Skip a tick (without encoding a frame) while this returns false, e.g. the server is still busy. */
  ready?: () => boolean;
}

export type MediaDevicesLike = Pick<MediaDevices, "getUserMedia" | "enumerateDevices"> &
  Partial<Pick<MediaDevices, "addEventListener" | "removeEventListener">>;

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export interface CameraControllerOptions {
  mediaDevices?: MediaDevicesLike;
  storage?: StorageLike | null;
  onStateChange: (state: CameraState) => void;
  /** Called whenever the video source is about to change, before any frame from the new source is sent. */
  onSourceChange?: () => void;
}

export const CAMERA_STORAGE_KEY = "signx.cameraDeviceId";

const VIDEO_CONSTRAINTS = { width: { ideal: 640 }, height: { ideal: 480 } };

export const initialCameraState: CameraState = {
  status: "requesting",
  devices: [],
  activeDeviceId: null,
  selectedDeviceId: null,
  error: "",
};

/** Video inputs only, with "Camera N" when the browser hides labels (no permission yet). */
export function toCameraDevices(devices: MediaDeviceInfo[]): CameraDevice[] {
  return devices
    .filter((d) => d.kind === "videoinput")
    .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
}

export function describeCameraError(err: unknown): { status: CameraStatus; message: string } {
  const name = typeof err === "object" && err !== null && "name" in err ? String((err as { name: unknown }).name) : "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return {
        status: "denied",
        message: "Camera permission was denied. Allow camera access in the browser's site settings, then retry.",
      };
    case "NotFoundError":
    case "OverconstrainedError":
      return { status: "error", message: "The selected camera is not available. It may have been disconnected." };
    case "NotReadableError":
    case "AbortError":
      return { status: "error", message: "The camera could not be started. It may be in use by another application." };
    default:
      return {
        status: "error",
        message: err instanceof Error && err.message ? err.message : "Could not access the camera.",
      };
  }
}

function stopTracks(stream: MediaStream): void {
  stream.getTracks().forEach((t) => t.stop());
}

type OpenResult = "ok" | "failed" | "superseded";

export class CameraController {
  private readonly media: MediaDevicesLike | undefined;
  private readonly storage: StorageLike | null;
  private readonly onStateChange: (state: CameraState) => void;
  private readonly onSourceChange: (() => void) | undefined;

  private state: CameraState = initialCameraState;
  private stream: MediaStream | null = null;
  private generation = 0;
  private disposed = false;
  private video: { srcObject: MediaProvider | null } | null = null;
  private capture: FrameCapture | null = null;
  private captureTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: CameraControllerOptions) {
    this.media = options.mediaDevices;
    this.storage = options.storage ?? null;
    this.onStateChange = options.onStateChange;
    this.onSourceChange = options.onSourceChange;
  }

  getState(): CameraState {
    return this.state;
  }

  get currentStream(): MediaStream | null {
    return this.stream;
  }

  get isCapturing(): boolean {
    return this.captureTimer !== null;
  }

  /** Request the remembered camera (or the browser default) and list devices. */
  async start(): Promise<void> {
    if (this.disposed) return;
    if (!this.media?.getUserMedia || !this.media.enumerateDevices) {
      this.update({
        status: "unavailable",
        error: "This browser does not support camera access (getUserMedia).",
      });
      return;
    }
    this.media.addEventListener?.("devicechange", this.handleDeviceChange);

    const devices = await this.listDevices();
    if (this.disposed) return;
    this.update({ devices });

    const saved = this.readSaved();
    const preferred = saved && devices.some((d) => d.deviceId === saved) ? saved : null;
    const result = await this.open(preferred, "requesting");
    // The remembered camera vanished between enumeration and opening it:
    // fall back to the browser default instead of leaving the user stuck.
    if (result === "failed" && preferred && this.state.status !== "denied") {
      await this.open(null, "requesting");
    }
  }

  /** Switch to another camera. Selecting the active camera again is a no-op. */
  async select(deviceId: string): Promise<void> {
    if (this.disposed) return;
    if (deviceId && deviceId === this.state.activeDeviceId && this.state.status === "connected") return;
    await this.open(deviceId || null, "switching");
  }

  /** Re-request the selected camera after a failure (e.g. permission granted later). */
  async retry(): Promise<void> {
    if (this.disposed || !this.media?.getUserMedia) return;
    await this.open(this.state.selectedDeviceId, "requesting");
  }

  attachVideo(video: { srcObject: MediaProvider | null } | null): void {
    this.video = video;
    if (video) video.srcObject = this.stream;
  }

  /** Set (or clear, e.g. while paused) the frame capture loop. */
  setCapture(capture: FrameCapture | null): void {
    this.capture = capture;
    this.syncCapture();
  }

  async refreshDevices(): Promise<void> {
    const devices = await this.listDevices();
    if (this.disposed) return;
    // Some browsers list no devices at all while permission is denied, so
    // only report "no camera" when nothing else explains the empty list.
    const noCameraCandidate = this.state.status === "error" || this.state.status === "unavailable";
    if (devices.length === 0 && noCameraCandidate) {
      this.update({ devices, status: "no-camera", error: "No cameras found. Connect a camera to continue." });
    } else {
      this.update({ devices });
    }
  }

  dispose(): void {
    this.disposed = true;
    this.generation += 1; // any in-flight getUserMedia() is now superseded
    this.capture = null;
    this.stopStream();
    this.media?.removeEventListener?.("devicechange", this.handleDeviceChange);
  }

  private readonly handleDeviceChange = (): void => {
    if (this.disposed) return;
    const wasWithoutCamera = this.state.status === "no-camera";
    void this.refreshDevices().then(() => {
      // A camera was plugged in while none was available: start it.
      if (wasWithoutCamera && !this.disposed && this.state.devices.length > 0) {
        void this.open(null, "requesting");
      }
    });
  };

  private async open(deviceId: string | null, status: "requesting" | "switching"): Promise<OpenResult> {
    const gen = ++this.generation;
    // Stop the old capture loop and camera before asking for the new one:
    // no frame from the old camera can be sent after this point, and many
    // webcams cannot be opened twice.
    this.stopStream();
    this.onSourceChange?.();
    this.update({ status, error: "", activeDeviceId: null, selectedDeviceId: deviceId });

    let stream: MediaStream;
    try {
      stream = await this.media!.getUserMedia({
        video: deviceId ? { ...VIDEO_CONSTRAINTS, deviceId: { exact: deviceId } } : VIDEO_CONSTRAINTS,
        audio: false,
      });
    } catch (err) {
      if (gen !== this.generation) return "superseded";
      const { status: failedStatus, message } = describeCameraError(err);
      this.update({ status: failedStatus, error: message });
      await this.refreshDevices();
      return "failed";
    }

    if (gen !== this.generation) {
      stopTracks(stream); // a newer selection (or unmount) won
      return "superseded";
    }

    const track = stream.getVideoTracks()[0];
    const activeId = track?.getSettings?.().deviceId || deviceId;
    // Unplugging the camera mid-session ends the track: react instead of
    // silently freezing on the last frame.
    track?.addEventListener("ended", () => this.handleEnded(stream));

    this.stream = stream;
    if (this.video) this.video.srcObject = stream;
    if (activeId) this.writeSaved(activeId);
    this.update({ status: "connected", error: "", activeDeviceId: activeId, selectedDeviceId: activeId });
    this.syncCapture();
    // Labels (and stable IDs) are only exposed once permission is granted.
    await this.refreshDevices();
    return "ok";
  }

  private handleEnded(stream: MediaStream): void {
    if (stream !== this.stream || this.disposed) return;
    this.generation += 1;
    this.stopStream();
    this.update({
      status: "unavailable",
      activeDeviceId: null,
      error: "Camera disconnected. Reconnect it or choose another camera.",
    });
    void this.refreshDevices();
  }

  private syncCapture(): void {
    this.stopCaptureLoop();
    const capture = this.capture;
    if (!capture || !this.stream || this.state.status !== "connected" || this.disposed) return;

    const gen = this.generation;
    this.captureTimer = setInterval(() => {
      if (capture.ready && !capture.ready()) return;
      capture.grab().then(
        (blob) => {
          // Encoding is async: drop a frame that finishes after the camera
          // changed, capture was paused/replaced, or the stream stopped.
          if (blob && gen === this.generation && capture === this.capture && this.state.status === "connected") {
            capture.onFrame(blob);
          }
        },
        () => {},
      );
    }, capture.intervalMs);
  }

  private stopCaptureLoop(): void {
    if (this.captureTimer !== null) {
      clearInterval(this.captureTimer);
      this.captureTimer = null;
    }
  }

  private stopStream(): void {
    this.stopCaptureLoop();
    if (this.stream) {
      stopTracks(this.stream);
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
  }

  private async listDevices(): Promise<CameraDevice[]> {
    try {
      return toCameraDevices(await this.media!.enumerateDevices());
    } catch {
      return [];
    }
  }

  private readSaved(): string | null {
    try {
      return this.storage?.getItem(CAMERA_STORAGE_KEY) ?? null;
    } catch {
      return null;
    }
  }

  private writeSaved(deviceId: string): void {
    try {
      this.storage?.setItem(CAMERA_STORAGE_KEY, deviceId);
    } catch {
      // Storage blocked (private mode etc.): the choice just isn't remembered.
    }
  }

  private update(partial: Partial<CameraState>): void {
    if (this.disposed) return;
    this.state = { ...this.state, ...partial };
    this.onStateChange(this.state);
  }
}
