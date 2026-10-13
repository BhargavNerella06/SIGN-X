import { useEffect, useState } from "react";
import { api } from "../services/api";
import { modelDisplayName } from "../services/modelInfo";
import type { ModelInfo } from "../types";

export default function Model() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getModelInfo().then(setInfo).catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Model</h1>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Live status of the model currently loaded by the backend — see <code>docs/model.md</code>.
      </p>

      {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}

      {info && (
        <div className="mt-6 card p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
              {modelDisplayName(info)}
            </h2>
            <span
              className={`badge ${
                info.is_trained
                  ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300"
                  : "bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300"
              }`}
            >
              {info.is_trained ? "Trained checkpoint" : "Untrained demo weights"}
            </span>
          </div>

          {!info.is_trained && (
            <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
              No trained checkpoint was found, so the backend is running an untrained model with random weights
              purely to keep the pipeline runnable. Predictions from this model are not meaningful. Run{" "}
              <code>python -m training.train --config training/config.yaml</code> to train a real one.
            </p>
          )}

          <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-3">
            {info.dataset_name && <Field label="Dataset" value={info.dataset_name} />}
            <Field label="Device" value={info.device.toUpperCase()} />
            <Field label="Input dimension" value={String(info.input_dim)} />
            <Field label="Classes" value={String(info.num_classes)} />
            <Field label="Parameters" value={info.num_parameters.toLocaleString()} />
            {Object.entries(info.metrics).map(([k, v]) => (
              <Field key={k} label={k.replace(/_/g, " ")} value={typeof v === "number" ? v.toFixed(4) : String(v)} />
            ))}
          </div>

          <div className="mt-5">
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Class names</p>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {info.class_names.map((c) => (
                <span key={c} className="badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                  {c}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-800/60">
      <p className="text-xs capitalize text-slate-500 dark:text-slate-400">{label}</p>
      <p className="text-sm font-semibold text-slate-900 dark:text-white">{value}</p>
    </div>
  );
}
