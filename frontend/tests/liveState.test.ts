// Run with: npm test
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { initialLiveState, liveReducer, type LiveAction, type LiveState } from "../src/services/liveState.ts";
import type { PredictionMessage, ServerMessage } from "../src/types.ts";

function prediction(overrides: Partial<PredictionMessage> = {}): PredictionMessage {
  return {
    type: "prediction",
    sign: "UNKNOWN",
    confidence: 0,
    is_unknown: true,
    committed: false,
    commit_id: 0,
    sentence: "",
    top_predictions: [],
    fps: 10,
    latency_ms: 40,
    landmark_ms: 30,
    inference_ms: 10,
    frames: 1,
    hands_detected: true,
    ...overrides,
  };
}

const commit = (sign: string, commitId: number, sentence: string) =>
  prediction({
    sign,
    confidence: 0.9,
    is_unknown: false,
    committed: true,
    commit_id: commitId,
    sentence,
    top_predictions: [{ label: sign, confidence: 0.9 }],
    landmarks: { right_hand: [[0.5, 0.5]] },
  });

function run(state: LiveState, ...msgs: Array<ServerMessage | LiveAction>): LiveState {
  return msgs.reduce<LiveState>((s, m) => {
    const action: LiveAction = "kind" in m ? m : { kind: "message", msg: m, time: "12:00:00" };
    return liveReducer(s, action);
  }, state);
}

describe("liveReducer", () => {
  it("records each commit once, even if the commit event is delivered twice", () => {
    const hello = commit("HELLO", 1, "Hello.");
    const s = run(initialLiveState, hello, hello, prediction({ sign: "HELLO", commit_id: 1, sentence: "Hello." }));
    assert.deepEqual(
      s.history.map((h) => h.sign),
      ["HELLO"],
    );
    const s2 = run(s, commit("WATER", 2, "Hello water."));
    assert.deepEqual(
      s2.history.map((h) => h.sign),
      ["WATER", "HELLO"],
    );
  });

  it("keeps sentence and history when the current prediction becomes UNKNOWN", () => {
    const s = run(
      initialLiveState,
      commit("BLACK", 1, "Black."),
      prediction({ sign: "UNKNOWN", commit_id: 1, sentence: "Black.", hands_detected: false }),
    );
    assert.equal(s.prediction?.sign, "UNKNOWN");
    assert.equal(s.prediction?.confidence, 0);
    assert.equal(s.prediction?.sentence, "Black.");
    assert.deepEqual(
      s.history.map((h) => h.sign),
      ["BLACK"],
    );
  });

  it("camera switch clears the current prediction but keeps sentence and history", () => {
    const s = run(initialLiveState, commit("HELLO", 1, "Hello."), { kind: "source-change", resetSent: true });
    assert.equal(s.prediction?.sign, "UNKNOWN");
    assert.equal(s.prediction?.confidence, 0);
    assert.equal(s.prediction?.is_unknown, true);
    assert.deepEqual(s.prediction?.top_predictions, []);
    assert.equal(s.prediction?.landmarks, undefined);
    assert.equal(s.prediction?.sentence, "Hello.");
    assert.equal(s.history.length, 1);
    assert.equal(s.awaitingReset, true);
  });

  it("ignores in-flight predictions from the old camera until the server confirms the reset", () => {
    let s = run(initialLiveState, commit("HELLO", 1, "Hello."), { kind: "source-change", resetSent: true });
    // Old-camera frames processed before the reset reached the server:
    s = run(s, commit("WATER", 2, "Hello water."), { type: "error", message: "bad frame" });
    assert.equal(s.prediction?.sign, "UNKNOWN");
    assert.equal(s.history.length, 1);
    assert.equal(s.frameError, null);

    s = run(s, { type: "status", message: "stream_reset", sentence: "Hello water." });
    assert.equal(s.awaitingReset, false);
    assert.equal(s.prediction?.sign, "UNKNOWN");
    assert.equal(s.prediction?.sentence, "Hello water.");

    s = run(s, commit("GO", 3, "Hello water go."));
    assert.deepEqual(
      s.history.map((h) => h.sign),
      ["GO", "HELLO"],
    );
  });

  it("does not wait for a reset ack that was never sent", () => {
    const s = run(initialLiveState, { kind: "source-change", resetSent: false }, commit("HELLO", 1, "Hello."));
    assert.equal(s.prediction?.sign, "HELLO");
    assert.equal(s.history.length, 1);
  });

  it("a new connection restarts commit ids and drops a pending reset", () => {
    let s = run(initialLiveState, commit("HELLO", 3, "Hello."), { kind: "source-change", resetSent: true });
    s = run(s, {
      type: "connection",
      status: "connected",
      vision_available: true,
      model_info: {} as never,
    });
    assert.equal(s.awaitingReset, false);
    s = run(s, commit("WATER", 1, "Water."));
    assert.deepEqual(
      s.history.map((h) => h.sign),
      ["WATER", "HELLO"],
    );
  });

  it("clears a frame error once a frame is processed successfully", () => {
    let s = run(initialLiveState, { type: "error", message: "Landmark extraction failed on this frame" });
    assert.equal(s.frameError, "Landmark extraction failed on this frame");
    s = run(s, prediction());
    assert.equal(s.frameError, null);
  });

  it("never records an UNKNOWN commit", () => {
    const s = run(initialLiveState, prediction({ committed: true, commit_id: 1 }));
    assert.equal(s.history.length, 0);
  });

  it("applies a cleared sentence and clears history on request", () => {
    let s = run(initialLiveState, commit("HELLO", 1, "Hello."), { type: "status", message: "Sentence cleared", sentence: "" });
    assert.equal(s.prediction?.sentence, "");
    s = run(s, { kind: "clear-history" });
    assert.equal(s.history.length, 0);
  });

  it("an error reply never records a commit, even one that carries commit fields", () => {
    // The server re-sends its last prediction fields on an error frame.
    const err = { ...commit("HELLO", 2, "Hello."), type: "error", message: "Inference failed on this frame" } as ServerMessage;
    const s = run(initialLiveState, commit("HELLO", 1, "Hello."), err, err);
    assert.deepEqual(s.history.map((h) => h.sign), ["HELLO"]);
    assert.equal(s.frameError, "Inference failed on this frame");
  });

  it("a stale sign is replaced by UNKNOWN as soon as the server reports no hands", () => {
    const s = run(
      initialLiveState,
      commit("HELLO", 1, "Hello."),
      prediction({ sign: "UNKNOWN", is_unknown: true, hands_detected: false, commit_id: 1, sentence: "Hello." }),
    );
    assert.equal(s.prediction?.sign, "UNKNOWN");
    assert.equal(s.prediction?.is_unknown, true);
    assert.equal(s.prediction?.sentence, "Hello.");
    assert.equal(s.history.length, 1);
  });
});
