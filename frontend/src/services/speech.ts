/**
 * Browser-side text-to-speech (Web Speech API).
 *
 * Speech must play on the user's device: a hosted backend has no speakers,
 * and the server's pyttsx3 engine only helps when browser and server run on
 * the same machine. The Dashboard uses this when the browser supports it and
 * falls back to the server's "speak" control message otherwise.
 *
 * Kept free of React (and of the global `window`, except in
 * createBrowserSpeech) so it can be unit-tested with a fake synthesizer.
 */

export interface UtteranceLike {
  onend: (() => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
}

export interface SynthesizerLike {
  speak(utterance: UtteranceLike): void;
  cancel(): void;
}

export interface SpeechCallbacks {
  onEnd: () => void;
  onError: (message: string) => void;
}

export class BrowserSpeech {
  private readonly synth: SynthesizerLike | null;
  private readonly makeUtterance: (text: string) => UtteranceLike;
  private current: UtteranceLike | null = null;

  constructor(synth: SynthesizerLike | null, makeUtterance: (text: string) => UtteranceLike) {
    this.synth = synth;
    this.makeUtterance = makeUtterance;
  }

  get available(): boolean {
    return this.synth !== null;
  }

  get speaking(): boolean {
    return this.current !== null;
  }

  /** Speak `text`, replacing anything still being spoken. Returns false if nothing was started. */
  speak(text: string, callbacks: SpeechCallbacks): boolean {
    if (!this.synth || !text.trim()) return false;
    this.stop();

    let utterance: UtteranceLike;
    try {
      utterance = this.makeUtterance(text);
    } catch {
      callbacks.onError("Speech is not available in this browser.");
      return false;
    }
    this.current = utterance;
    // Only the latest utterance may report back; a cancelled one is silent.
    utterance.onend = () => {
      if (this.current !== utterance) return;
      this.current = null;
      callbacks.onEnd();
    };
    utterance.onerror = (event) => {
      if (this.current !== utterance) return;
      this.current = null;
      if (event.error === "interrupted" || event.error === "canceled") callbacks.onEnd();
      else callbacks.onError(`Speech failed (${event.error ?? "unknown error"}).`);
    };

    try {
      this.synth.speak(utterance);
    } catch {
      this.current = null;
      callbacks.onError("Speech failed to start.");
      return false;
    }
    return true;
  }

  stop(): void {
    this.current = null;
    try {
      this.synth?.cancel();
    } catch {
      // Nothing useful to do: speech is being abandoned anyway.
    }
  }
}

export function createBrowserSpeech(): BrowserSpeech {
  const synth =
    typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined"
      ? window.speechSynthesis
      : null;
  return new BrowserSpeech(synth as SynthesizerLike | null, (text) => {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-IN";
    return utterance as unknown as UtteranceLike;
  });
}
