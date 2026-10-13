interface ConfidenceBarProps {
  value: number; // 0..1
  isUnknown?: boolean;
}

export default function ConfidenceBar({ value, isUnknown }: ConfidenceBarProps) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const color = isUnknown
    ? "bg-slate-400 dark:bg-slate-600"
    : pct >= 80
      ? "bg-emerald-500"
      : pct >= 60
        ? "bg-brand-500"
        : "bg-amber-500";

  return (
    <div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
        <div
          className={`h-full rounded-full transition-all duration-300 ease-out ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-500 dark:text-slate-400">
        <span>Confidence</span>
        <span className="font-semibold tabular-nums">{pct}%</span>
      </div>
    </div>
  );
}
