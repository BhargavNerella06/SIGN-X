// Run with: npm test
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { BrowserSpeech, type UtteranceLike } from "../src/services/speech.ts";

class FakeUtterance implements UtteranceLike {
  text: string;
  onend: (() => void) | null = null;
  onerror: ((event: { error?: string }) => void) | null = null;
  constructor(text: string) {
    this.text = text;
  }
}

class FakeSynth {
  spoken: FakeUtterance[] = [];
  cancels = 0;
  failSpeak = false;
  speak(u: FakeUtterance) {
    if (this.failSpeak) throw new Error("synthesis-failed");
    this.spoken.push(u);
  }
  cancel() {
    this.cancels += 1;
    // Browsers report a cancelled utterance as an "interrupted" error.
    this.spoken.forEach((u) => u.onerror?.({ error: "interrupted" }));
  }
}

function setup() {
  const synth = new FakeSynth();
  const speech = new BrowserSpeech(synth, (t) => new FakeUtterance(t));
  const events: string[] = [];
  const callbacks = { onEnd: () => events.push("end"), onError: (m: string) => events.push(`error:${m}`) };
  return { synth, speech, events, callbacks };
}

describe("BrowserSpeech", () => {
  it("is unavailable without a synthesizer and starts nothing", () => {
    const speech = new BrowserSpeech(null, (t) => new FakeUtterance(t));
    assert.equal(speech.available, false);
    assert.equal(speech.speak("Hello.", { onEnd() {}, onError() {} }), false);
    speech.stop(); // must not throw
  });

  it("speaks the sentence and reports the end once", () => {
    const { synth, speech, events, callbacks } = setup();
    assert.equal(speech.speak("Hello, how are you?", callbacks), true);
    assert.equal(speech.speaking, true);
    assert.equal(synth.spoken[0].text, "Hello, how are you?");
    synth.spoken[0].onend?.();
    synth.spoken[0].onend?.();
    assert.deepEqual(events, ["end"]);
    assert.equal(speech.speaking, false);
  });

  it("never speaks empty or whitespace-only text", () => {
    const { synth, speech, callbacks } = setup();
    assert.equal(speech.speak("", callbacks), false);
    assert.equal(speech.speak("   ", callbacks), false);
    assert.equal(synth.spoken.length, 0);
  });

  it("a new sentence replaces the one being spoken; the old one stays silent", () => {
    const { synth, speech, events, callbacks } = setup();
    speech.speak("Hello.", callbacks);
    speech.speak("Thank you.", callbacks);
    assert.equal(synth.cancels >= 1, true);
    assert.deepEqual(events, []); // the interrupted first utterance reports nothing
    synth.spoken[1].onend?.();
    assert.deepEqual(events, ["end"]);
  });

  it("stop() cancels without reporting an error", () => {
    const { synth, speech, events, callbacks } = setup();
    speech.speak("Hello.", callbacks);
    speech.stop();
    assert.equal(speech.speaking, false);
    assert.deepEqual(events, []);
    assert.ok(synth.cancels >= 1);
  });

  it("reports synthesis errors", () => {
    const { synth, speech, events, callbacks } = setup();
    speech.speak("Hello.", callbacks);
    synth.spoken[0].onerror?.({ error: "audio-busy" });
    assert.deepEqual(events, ["error:Speech failed (audio-busy)."]);
    assert.equal(speech.speaking, false);
  });

  it("reports a synthesizer that throws when asked to speak", () => {
    const { synth, speech, events, callbacks } = setup();
    synth.failSpeak = true;
    assert.equal(speech.speak("Hello.", callbacks), false);
    assert.deepEqual(events, ["error:Speech failed to start."]);
    assert.equal(speech.speaking, false);
  });
});
