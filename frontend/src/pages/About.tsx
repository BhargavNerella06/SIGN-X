import type { ReactNode } from "react";

export default function About() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
      <h1 className="text-2xl font-bold text-slate-900 dark:text-white">About SIGN-X</h1>

      <Section title="What it does">
        <p>
          SIGN-X captures webcam video in the browser, extracts hand/pose/face landmarks with
          MediaPipe, feeds a sliding window of landmark sequences into a temporal deep learning
          model (a Transformer or BiLSTM), and converts the recognized sign into a growing sentence
          that can be spoken aloud with text-to-speech.
        </p>
      </Section>

      <Section title="How it works">
        <p>
          Video is never sent to any external service — landmark extraction and inference run
          entirely on your own backend server, in memory, and are not stored to disk by default.
          See <code>docs/architecture.md</code> for the full pipeline diagram.
        </p>
      </Section>

      <Section title="Technology stack">
        <ul className="list-disc space-y-1 pl-5">
          <li>Backend: FastAPI, Uvicorn, WebSockets, OpenCV, MediaPipe, PyTorch, scikit-learn</li>
          <li>Frontend: React, TypeScript, Tailwind CSS, Vite</li>
          <li>Speech: pyttsx3 (local/offline text-to-speech)</li>
        </ul>
      </Section>

      <Section title="Limitations">
        <ul className="list-disc space-y-1 pl-5">
          <li>ISL grammar (topic-comment structure, spatial referencing, grammatical facial expressions) is far richer than this project's small template-based sentence builder.</li>
          <li>The default vocabulary is a small ~20-word starting set, not a claim of complete ISL coverage.</li>
          <li>Recognition quality depends on lighting, camera angle, and distance from the camera.</li>
          <li>Different signers sign the same word differently; generalization depends entirely on training data.</li>
          <li>Continuous sentence recognition is architecturally supported but is a substantially harder problem than isolated-sign recognition.</li>
          <li>No accuracy numbers are claimed here unless they come from an actual evaluation run against a real trained model — see the Model page.</li>
        </ul>
      </Section>

      <Section title="Dataset information">
        <p>
          No dataset is bundled with this project. See <code>docs/dataset.md</code> for recommended,
          real ISL datasets (INCLUDE by AI4Bharat, ISL-CSLTR) and their sources/licenses, and the
          Dataset page for what's currently prepared on this installation.
        </p>
      </Section>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="mt-6 card p-6">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-brand-600 dark:text-brand-400">{title}</h2>
      <div className="mt-2 space-y-2 text-sm leading-relaxed text-slate-600 dark:text-slate-300">{children}</div>
    </div>
  );
}
