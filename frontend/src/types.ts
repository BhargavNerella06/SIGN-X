// Shared types mirroring the backend's WebSocket/REST payloads
// (see backend/app/inference/predictor.py and backend/app/api/*.py).

export interface TopPrediction {
  label: string;
  confidence: number;
}

export interface PredictionMessage {
  type: "prediction" | "error";
  sign: string;
  confidence: number;
  is_unknown: boolean;
  committed: boolean;
  /** Running commit count on this connection; a commit is recorded once per id. */
  commit_id?: number;
  sentence: string;
  top_predictions: TopPrediction[];
  fps: number;
  latency_ms: number;
  landmark_ms: number;
  inference_ms: number;
  frames: number;
  hands_detected: boolean;
  landmarks?: Partial<Record<"left_hand" | "right_hand" | "pose", [number, number][]>>;
  message?: string;
}

export interface ModelInfo {
  model_type: "transformer" | "bilstm";
  is_trained: boolean;
  checkpoint_path: string | null;
  device: string;
  input_dim: number;
  num_classes: number;
  num_parameters: number;
  class_names: string[];
  metrics: Record<string, number>;
  dataset_name?: string | null;
  /** Architectures averaged for live inference (empty/absent: model_type alone). */
  ensemble_members?: ("transformer" | "bilstm")[];
}

export interface ConnectionMessage {
  type: "connection";
  status: string;
  model_info: ModelInfo;
  vision_available: boolean;
}

export interface StatusMessage {
  type: "status";
  message: string;
  sentence?: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = PredictionMessage | ConnectionMessage | StatusMessage | ErrorMessage;

export interface HistoryEntry {
  time: string;
  sign: string;
  confidence: number;
}

export interface DatasetSplitInfo {
  name: string;
  path: string;
  num_classes: number;
  classes: string[];
  splits: Record<string, number>;
  total_samples: number;
  is_prepared: boolean;
}

export interface DatasetInfoResponse {
  status: "ok" | "not_found" | "not_prepared";
  message?: string;
  datasets?: DatasetSplitInfo[];
  configured_vocabulary?: string[];
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error";
