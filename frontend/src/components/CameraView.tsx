import { useEffect, useRef, useState } from "react";
import { CameraController, initialCameraState, type CameraState, type StorageLike } from "../services/camera";
import type { PredictionMessage } from "../types";

interface CameraViewProps {
  onFrame: (blob: Blob) => void;
  /** Called when the video source changes (camera switch), before frames from the new camera are sent. */
  onSourceChange?: () => void;
  /** Return false to skip capturing while the previous frame is still being processed. */
  canSendFrame?: () => boolean;
  landmarks?: PredictionMessage["landmarks"];
  captureIntervalMs?: number;
  paused: boolean;
}

const POINT_COLORS: Record<string, string> = {
  left_hand: "#3763f4",
  right_hand: "#22c55e",
  pose: "#f59e0b",
};

const STATUS_TEXT: Record<CameraState["status"], string> = {
  requesting: "Requesting camera access...",
  switching: "Switching camera...",
  connected: "",
  denied: "Camera access denied.",
  "no-camera": "No camera found.",
  error: "Camera could not be started.",
  unavailable: "Camera unavailable.",
};

function safeLocalStorage(): StorageLike | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

/**
 * Owns the webcam (via CameraController), lets the user pick which camera
 * to use, streams JPEG frames to the backend at a throttled interval, and
 * draws a lightweight landmark overlay from the most recent prediction
 * message on a canvas positioned over the video.
 *
 * Handles "no webcam" / "permission denied" / "webcam disconnected" /
 * failed camera switches gracefully (Section 25) instead of leaving a
 * blank/broken video element.
 */
// ~15 fps. At 10 fps a 2-3 s sign yields too few frames: offline accuracy on
// browser-quality frames dropped from 69% (15 fps) to 59% (10 fps); the
// models were trained on 25 fps clips. See docs/performance.md.
const DEFAULT_CAPTURE_INTERVAL_MS = 67;

export default function CameraView({
  onFrame,
  onSourceChange,
  canSendFrame,
  landmarks,
  captureIntervalMs = DEFAULT_CAPTURE_INTERVAL_MS,
  paused,
}: CameraViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const controllerRef = useRef<CameraController | null>(null);
  const onSourceChangeRef = useRef(onSourceChange);
  const [camera, setCamera] = useState<CameraState>(initialCameraState);

  useEffect(() => {
    onSourceChangeRef.current = onSourceChange;
  }, [onSourceChange]);

  // One controller per mount: acquires the camera, tracks devices, and is
  // disposed (tracks stopped, listeners removed) on unmount.
  useEffect(() => {
    const controller = new CameraController({
      mediaDevices: navigator.mediaDevices,
      storage: safeLocalStorage(),
      onStateChange: setCamera,
      onSourceChange: () => onSourceChangeRef.current?.(),
    });
    controllerRef.current = controller;
    controller.attachVideo(videoRef.current);
    void controller.start();
    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, []);

  // Capture loop: draw the current video frame to a hidden canvas and send
  // it as a JPEG blob at a throttled interval (not every render). The
  // controller runs it only while a camera is connected.
  useEffect(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    if (paused) {
      controller.setCapture(null);
      return;
    }

    const grab = () =>
      new Promise<Blob | null>((resolve) => {
        const video = videoRef.current;
        const canvas = captureCanvasRef.current;
        if (!video || !canvas || video.videoWidth === 0) return resolve(null);
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");
        if (!ctx) return resolve(null);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(resolve, "image/jpeg", 0.8);
      });

    controller.setCapture({ grab, onFrame, intervalMs: captureIntervalMs, ready: canSendFrame });
    return () => controller.setCapture(null);
  }, [paused, captureIntervalMs, onFrame, canSendFrame]);

  // Draw the landmark overlay whenever new points arrive.
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!landmarks) return;

    for (const [group, points] of Object.entries(landmarks)) {
      if (!points) continue;
      ctx.fillStyle = POINT_COLORS[group] ?? "#ffffff";
      for (const [x, y] of points) {
        ctx.beginPath();
        ctx.arc(x * canvas.width, y * canvas.height, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [landmarks]);

  const { status, devices, activeDeviceId, selectedDeviceId, error } = camera;
  const busy = status === "requesting" || status === "switching";
  const selectValue = selectedDeviceId ?? activeDeviceId ?? devices[0]?.deviceId ?? "";
  const canRetry = status === "denied" || status === "error" || status === "unavailable";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <label
          htmlFor="camera-select"
          className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400"
        >
          Camera
        </label>
        <select
          id="camera-select"
          value={selectValue}
          disabled={busy || devices.length === 0}
          onChange={(e) => void controllerRef.current?.select(e.target.value)}
          className="min-w-0 flex-1 basis-48 truncate rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:opacity-60 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200"
        >
          {devices.length === 0 && <option value="">{busy ? "Looking for cameras…" : "No cameras found"}</option>}
          {devices.map((d, i) => (
            <option key={`${d.deviceId}-${i}`} value={d.deviceId}>
              {status === "connected" && d.deviceId === activeDeviceId ? `${d.label} (active)` : d.label}
            </option>
          ))}
        </select>
        {status === "switching" && <span className="text-xs text-slate-500 dark:text-slate-400">Switching…</span>}
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-slate-950">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="h-full w-full object-cover -scale-x-100"
        />
        <canvas ref={overlayCanvasRef} className="pointer-events-none absolute inset-0 -scale-x-100" />
        <canvas ref={captureCanvasRef} className="hidden" />

        {status !== "connected" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-slate-950/90 text-center text-slate-300 p-6">
            <span className="text-3xl">📷</span>
            <p className="text-sm font-medium">{STATUS_TEXT[status]}</p>
            {error && <p className="text-xs text-slate-500 max-w-xs">{error}</p>}
            {canRetry && (
              <button className="btn-secondary mt-2" onClick={() => void controllerRef.current?.retry()}>
                Retry
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
