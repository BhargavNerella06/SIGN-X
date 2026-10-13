import ConfidenceBar from "./ConfidenceBar";
import type { PredictionMessage } from "../types";

interface PredictionPanelProps {
  prediction: PredictionMessage | null;
}

export default function PredictionPanel({ prediction }: PredictionPanelProps) {
  const sign = prediction?.sign ?? "—";
  const confidence = prediction?.confidence ?? 0;
  const isUnknown = prediction?.is_unknown ?? true;
  const topPredictions = prediction?.top_predictions ?? [];

  return (
    <div className="card p-5">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Current Sign
      </h2>

      <div className="mt-2 flex items-baseline gap-2">
        <span
          className={`text-4xl font-bold tracking-tight ${
            isUnknown ? "text-slate-400 dark:text-slate-600" : "text-slate-900 dark:text-white"
          }`}
        >
          {sign}
        </span>
        {isUnknown && (
          <span className="badge bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400">
            Low confidence
          </span>
        )}
      </div>

      <div className="mt-4">
        <ConfidenceBar value={confidence} isUnknown={isUnknown} />
      </div>

      {topPredictions.length > 0 && (
        <div className="mt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Top Predictions
          </h3>
          <ul className="mt-2 space-y-1.5">
            {topPredictions.map((p) => (
              <li key={p.label} className="flex items-center justify-between text-sm">
                <span className="text-slate-700 dark:text-slate-300">{p.label}</span>
                <span className="font-medium tabular-nums text-slate-500 dark:text-slate-400">
                  {Math.round(p.confidence * 100)}%
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
