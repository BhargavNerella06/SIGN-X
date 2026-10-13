import type { ModelInfo } from "../types";

const ARCH_NAMES: Record<string, string> = { transformer: "Transformer", bilstm: "BiLSTM" };

/** "ISL-Transformer", or "ISL-Transformer + BiLSTM" when the server averages an ensemble. */
export function modelDisplayName(info: Pick<ModelInfo, "model_type" | "ensemble_members">): string {
  const members = info.ensemble_members?.length ? info.ensemble_members : [info.model_type];
  return `ISL-${members.map((m) => ARCH_NAMES[m] ?? m).join(" + ")}`;
}
