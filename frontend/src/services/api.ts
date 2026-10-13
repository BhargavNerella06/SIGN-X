import type { DatasetInfoResponse, ModelInfo } from "../types";

// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.ts).
// In production, set VITE_API_BASE_URL to the deployed backend origin.
const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`Request to ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export interface HealthResponse {
  status: string;
  app: string;
  environment: string;
  device: string;
}

export const api = {
  getHealth: () => getJson<HealthResponse>("/api/health"),
  getModelInfo: () => getJson<ModelInfo>("/api/model/info"),
  getDatasetInfo: () => getJson<DatasetInfoResponse>("/api/dataset/info"),
};
