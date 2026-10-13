import type { HistoryEntry } from "../types";

interface PredictionHistoryProps {
  history: HistoryEntry[];
  onClear: () => void;
}

export default function PredictionHistory({ history, onClear }: PredictionHistoryProps) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          Prediction History
        </h2>
        <button
          onClick={onClear}
          disabled={history.length === 0}
          className="text-xs font-medium text-slate-400 hover:text-slate-600 disabled:opacity-40 dark:hover:text-slate-300"
        >
          Clear
        </button>
      </div>

      {history.length === 0 ? (
        <p className="mt-3 text-sm text-slate-400 dark:text-slate-600">No committed signs yet.</p>
      ) : (
        <ul className="mt-2 max-h-48 space-y-1 overflow-y-auto text-sm">
          {history.map((entry, i) => (
            <li key={`${entry.time}-${i}`} className="flex items-center justify-between py-1 font-mono text-xs">
              <span className="text-slate-400 dark:text-slate-600">{entry.time}</span>
              <span className="font-sans font-medium text-slate-700 dark:text-slate-300">{entry.sign}</span>
              <span className="text-slate-500 dark:text-slate-400">{Math.round(entry.confidence * 100)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
