import type { ConnectionStatus, PredictionMessage } from "../types";

interface ControlPanelProps {
  connectionStatus: ConnectionStatus;
  prediction: PredictionMessage | null;
}

const STATUS_META: Record<ConnectionStatus, { label: string; dot: string }> = {
  connected: { label: "Connected", dot: "bg-emerald-500" },
  connecting: { label: "Connecting…", dot: "bg-amber-500" },
  disconnected: { label: "Disconnected", dot: "bg-rose-500" },
  error: { label: "Connection error", dot: "bg-rose-500" },
};

export default function ControlPanel({ connectionStatus, prediction }: ControlPanelProps) {
  const meta = STATUS_META[connectionStatus];

  return (
    <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm">
      <div className="flex items-center gap-2">
        <span className={`h-2.5 w-2.5 rounded-full ${meta.dot} ${connectionStatus === "connected" ? "animate-pulse-dot" : ""}`} />
        <span className="text-slate-600 dark:text-slate-300">{meta.label}</span>
        {prediction && !prediction.hands_detected && connectionStatus === "connected" && (
          <span className="text-xs text-slate-400 dark:text-slate-500">· no hands detected</span>
        )}
      </div>
      <div className="flex gap-4 text-slate-500 dark:text-slate-400">
        <span>
          FPS <span className="font-semibold text-slate-800 dark:text-slate-200 tabular-nums">{prediction?.fps.toFixed(1) ?? "—"}</span>
        </span>
        <span>
          Latency <span className="font-semibold text-slate-800 dark:text-slate-200 tabular-nums">{prediction ? `${prediction.latency_ms.toFixed(0)} ms` : "—"}</span>
        </span>
      </div>
    </div>
  );
}
