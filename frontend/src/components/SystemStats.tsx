import { modelDisplayName } from "../services/modelInfo";
import type { ModelInfo, PredictionMessage } from "../types";

interface SystemStatsProps {
  modelInfo: ModelInfo | null;
  prediction: PredictionMessage | null;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-slate-500 dark:text-slate-400">{label}</span>
      <span className="font-semibold tabular-nums text-slate-800 dark:text-slate-200">{value}</span>
    </div>
  );
}

export default function SystemStats({ modelInfo, prediction }: SystemStatsProps) {
  return (
    <div className="card p-5">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        System Status
      </h2>
      <div className="mt-2 divide-y divide-slate-100 dark:divide-slate-800">
        <Stat label="Model" value={modelInfo ? modelDisplayName(modelInfo) : "—"} />
        <Stat label="Trained" value={modelInfo ? (modelInfo.is_trained ? "Yes" : "No (demo weights)") : "—"} />
        <Stat label="Device" value={modelInfo?.device.toUpperCase() ?? "—"} />
        <Stat label="FPS" value={prediction ? prediction.fps.toFixed(1) : "—"} />
        <Stat label="Latency" value={prediction ? `${prediction.latency_ms.toFixed(0)} ms` : "—"} />
        <Stat label="Frames" value={prediction ? String(prediction.frames) : "0"} />
      </div>
      {modelInfo && !modelInfo.is_trained && (
        <p className="mt-3 rounded-lg bg-amber-50 dark:bg-amber-950/40 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
          Running with untrained demo weights — predictions are not meaningful yet. Train a model (see docs/model.md) to get real predictions.
        </p>
      )}
    </div>
  );
}
