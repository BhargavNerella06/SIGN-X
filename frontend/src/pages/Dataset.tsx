import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { DatasetInfoResponse } from "../types";

export default function Dataset() {
  const [info, setInfo] = useState<DatasetInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDatasetInfo().then(setInfo).catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Dataset</h1>
      <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
        Reflects whatever has actually been prepared under <code>backend/datasets/</code> — nothing here is fabricated.
      </p>

      {error && <p className="mt-4 text-sm text-rose-500">{error}</p>}

      {info?.status !== "ok" && info && (
        <div className="mt-6 card p-6">
          <p className="text-sm text-slate-600 dark:text-slate-300">{info.message}</p>
          <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Configured vocabulary</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {info.configured_vocabulary?.map((w) => (
              <span key={w} className="badge bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                {w}
              </span>
            ))}
          </div>
          <p className="mt-4 text-xs text-slate-400">See <code>docs/dataset.md</code> for recommended ISL datasets.</p>
        </div>
      )}

      {info?.status === "ok" &&
        info.datasets?.map((ds) => (
          <div key={ds.name} className="mt-6 card p-6">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">{ds.name}</h2>
            <p className="text-xs text-slate-400">{ds.path}</p>

            <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <Stat label="Classes" value={ds.num_classes} />
              <Stat label="Total samples" value={ds.total_samples} />
              <Stat label="Train" value={ds.splits.train ?? 0} />
              <Stat label="Validation" value={ds.splits.validation ?? 0} />
            </div>

            <div className="mt-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">Classes</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {ds.classes.map((c) => (
                  <span key={c} className="badge bg-brand-50 text-brand-700 dark:bg-brand-900/40 dark:text-brand-300">
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3 dark:bg-slate-800/60">
      <p className="text-xs text-slate-500 dark:text-slate-400">{label}</p>
      <p className="text-xl font-bold text-slate-900 dark:text-white">{value}</p>
    </div>
  );
}
