import { test } from "node:test";
import assert from "node:assert/strict";
import { modelDisplayName } from "../src/services/modelInfo.ts";

test("single model shows its architecture", () => {
  assert.equal(modelDisplayName({ model_type: "transformer" }), "ISL-Transformer");
  assert.equal(modelDisplayName({ model_type: "bilstm", ensemble_members: [] }), "ISL-BiLSTM");
});

test("live ensemble lists every averaged architecture", () => {
  assert.equal(
    modelDisplayName({ model_type: "transformer", ensemble_members: ["transformer", "bilstm"] }),
    "ISL-Transformer + BiLSTM",
  );
});
