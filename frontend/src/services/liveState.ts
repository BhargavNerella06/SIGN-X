import type { HistoryEntry, PredictionMessage, ServerMessage } from "../types";

/**
 * Dashboard state driven by /ws/predict messages: the current prediction,
 * committed-sign history and the bookkeeping that keeps stale events out.
 *
 *   - A commit is recorded once: only when its commit_id is newer than the
 *     last one recorded on this connection.
 *   - After a camera switch, predictions from the old camera may still be in
 *     flight. They are ignored until the server acknowledges the reset.
 *   - Committed signs (sentence + history) survive both.
 */

export const UNKNOWN_SIGN = "UNKNOWN";

export interface LiveState {
  prediction: PredictionMessage | null;
  history: HistoryEntry[];
  lastCommitId: number;
  awaitingReset: boolean;
  frameError: string | null;
}

export type LiveAction =
  | { kind: "message"; msg: ServerMessage; time: string }
  | { kind: "source-change"; resetSent: boolean }
  | { kind: "clear-history" };

export const initialLiveState: LiveState = {
  prediction: null,
  history: [],
  lastCommitId: 0,
  awaitingReset: false,
  frameError: null,
};

const MAX_HISTORY = 50;

/** Same message, but describing "no sign right now"; the sentence is kept. */
export function toUnknown(prediction: PredictionMessage): PredictionMessage {
  return {
    ...prediction,
    sign: UNKNOWN_SIGN,
    confidence: 0,
    is_unknown: true,
    committed: false,
    top_predictions: [],
    hands_detected: false,
    landmarks: undefined,
  };
}

export function liveReducer(state: LiveState, action: LiveAction): LiveState {
  switch (action.kind) {
    case "clear-history":
      return { ...state, history: [] };

    case "source-change":
      return {
        ...state,
        prediction: state.prediction ? toUnknown(state.prediction) : null,
        awaitingReset: action.resetSent,
        frameError: null,
      };

    case "message":
      return applyMessage(state, action.msg, action.time);
  }
}

function applyMessage(state: LiveState, msg: ServerMessage, time: string): LiveState {
  switch (msg.type) {
    case "connection":
      // A (re)connection gets a fresh server-side predictor: commit ids
      // restart and no reset is pending on it.
      return { ...state, lastCommitId: 0, awaitingReset: false };

    case "prediction": {
      if (state.awaitingReset) return state; // from the previous camera
      const commitId = msg.commit_id ?? state.lastCommitId;
      const isNewCommit =
        msg.committed && msg.sign !== UNKNOWN_SIGN && (msg.commit_id === undefined || msg.commit_id > state.lastCommitId);
      return {
        ...state,
        prediction: msg,
        frameError: null,
        lastCommitId: Math.max(state.lastCommitId, commitId),
        history: isNewCommit
          ? [{ time, sign: msg.sign, confidence: msg.confidence }, ...state.history].slice(0, MAX_HISTORY)
          : state.history,
      };
    }

    case "error":
      if (state.awaitingReset) return state;
      return { ...state, frameError: msg.message ?? "An error occurred while processing the last frame." };

    case "status": {
      if (msg.message === "stream_reset") {
        return {
          ...state,
          awaitingReset: false,
          prediction: state.prediction ? { ...toUnknown(state.prediction), sentence: msg.sentence ?? "" } : null,
        };
      }
      if (msg.sentence !== undefined && state.prediction) {
        return { ...state, prediction: { ...state.prediction, sentence: msg.sentence } };
      }
      return state;
    }
  }
}
