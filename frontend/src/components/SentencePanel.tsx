interface SentencePanelProps {
  sentence: string;
  onSpeak: () => void;
  onStop: () => void;
  onClear: () => void;
  onTogglePause: () => void;
  paused: boolean;
  isSpeaking: boolean;
}

export default function SentencePanel({
  sentence,
  onSpeak,
  onStop,
  onClear,
  onTogglePause,
  paused,
  isSpeaking,
}: SentencePanelProps) {
  return (
    <div className="card p-5">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        Translated Sentence
      </h2>
      <p className="mt-3 min-h-[2.5rem] text-xl font-medium text-slate-900 dark:text-white">
        {sentence ? `"${sentence}"` : <span className="text-slate-400 dark:text-slate-600">Sign something to build a sentence…</span>}
      </p>

      <div className="mt-5 flex flex-wrap gap-2">
        <button className="btn-primary" onClick={onSpeak} disabled={!sentence}>
          {isSpeaking ? "🔊 Speaking…" : "🔊 Speak"}
        </button>
        <button className="btn-secondary" onClick={onStop} disabled={!isSpeaking}>
          Stop
        </button>
        <button className="btn-secondary" onClick={onClear} disabled={!sentence}>
          Clear
        </button>
        <button className="btn-danger" onClick={onTogglePause}>
          {paused ? "▶ Resume" : "⏸ Pause"}
        </button>
      </div>
    </div>
  );
}
