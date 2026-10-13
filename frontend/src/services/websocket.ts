import type { ConnectionStatus, ServerMessage } from "../types";
import { InFlightLimiter } from "./frameFlow.ts";

const WS_PATH = "/ws/predict";

/**
 * WebSocket origin for the backend, in order of preference:
 *   1. VITE_WS_BASE_URL, verbatim;
 *   2. VITE_API_BASE_URL with http(s) swapped for ws(s), so a deployment
 *      only has to configure the backend URL once;
 *   3. port 8000 on the page's host (local development), using wss: when the
 *      page itself is served over https (browsers block ws: from https pages).
 */
export function resolveWsBase(
  env: { VITE_WS_BASE_URL?: string; VITE_API_BASE_URL?: string },
  location: { protocol: string; hostname: string },
): string {
  if (env.VITE_WS_BASE_URL) return env.VITE_WS_BASE_URL.replace(/\/+$/, "");
  if (env.VITE_API_BASE_URL) return env.VITE_API_BASE_URL.replace(/^http/i, "ws").replace(/\/+$/, "");
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.hostname}:8000`;
}

function defaultUrl(): string {
  return `${resolveWsBase(import.meta.env, window.location)}${WS_PATH}`;
}

type MessageHandler = (message: ServerMessage) => void;
type StatusHandler = (status: ConnectionStatus) => void;

export interface SignXSocketOptions {
  /** Full /ws/predict URL; defaults to resolveWsBase() for this page. */
  url?: string;
  /** WebSocket constructor (injectable for tests). */
  WebSocketImpl?: typeof WebSocket;
}

/**
 * Thin wrapper around the /ws/predict WebSocket protocol (see
 * backend/app/api/websocket.py). Handles reconnection with backoff so a
 * transient backend restart or network blip doesn't require a page reload
 * (Section 43: "backend/frontend can reconnect after WebSocket interruption").
 */
export class SignXSocket {
  private ws: WebSocket | null = null;
  private messageHandlers = new Set<MessageHandler>();
  private statusHandlers = new Set<StatusHandler>();
  private reconnectAttempts = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private closedByClient = false;
  private frames = new InFlightLimiter();
  private readonly url: string | undefined;
  private readonly WS: typeof WebSocket;

  constructor(options: SignXSocketOptions = {}) {
    this.url = options.url;
    this.WS = options.WebSocketImpl ?? WebSocket;
  }

  connect(): void {
    this.closedByClient = false;
    this.setStatus("connecting");

    const socket = new this.WS(this.url ?? defaultUrl());
    socket.binaryType = "blob";
    // Events from a socket that has since been replaced (e.g. a late close
    // after disconnect() + connect()) must not touch the current connection.
    const isCurrent = () => this.ws === socket;

    socket.onopen = () => {
      if (!isCurrent()) return;
      this.reconnectAttempts = 0;
      this.frames.reset();
      this.setStatus("connected");
    };

    socket.onmessage = (event) => {
      if (!isCurrent()) return;
      try {
        const data = JSON.parse(event.data) as ServerMessage;
        // Every binary frame gets exactly one prediction/error reply.
        if (data.type === "prediction" || data.type === "error") this.frames.received();
        this.messageHandlers.forEach((handler) => handler(data));
      } catch {
        // Malformed message from the server — ignore rather than crash the UI.
      }
    };

    socket.onerror = () => {
      if (!isCurrent()) return;
      this.setStatus("error");
    };

    socket.onclose = () => {
      if (!isCurrent()) return;
      this.setStatus("disconnected");
      if (!this.closedByClient) {
        this.scheduleReconnect();
      }
    };

    this.ws = socket;
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return;
    const delay = Math.min(1000 * 2 ** this.reconnectAttempts, 10000);
    this.reconnectAttempts += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }

  disconnect(): void {
    this.closedByClient = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    const socket = this.ws;
    this.ws = null;
    if (socket) {
      socket.close();
      this.setStatus("disconnected");
    }
  }

  /** True when a frame may be sent now (connected, previous frame answered). */
  canSendFrame(): boolean {
    return this.ws?.readyState === this.WS.OPEN && this.frames.canSend(performance.now());
  }

  sendFrame(blob: Blob): void {
    if (this.ws?.readyState === this.WS.OPEN) {
      this.frames.sent(performance.now());
      this.ws.send(blob);
    }
  }

  private sendControl(type: string): boolean {
    if (this.ws?.readyState === this.WS.OPEN) {
      this.ws.send(JSON.stringify({ type }));
      return true;
    }
    return false;
  }

  clearSentence(): void {
    this.sendControl("clear");
  }

  /** Tell the server the video source changed. Returns false if not sent
   * (not connected — the next connection starts with fresh state anyway). */
  resetStream(): boolean {
    return this.sendControl("reset");
  }

  speak(): void {
    this.sendControl("speak");
  }

  stopSpeech(): void {
    this.sendControl("stop_speech");
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onStatusChange(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    return () => this.statusHandlers.delete(handler);
  }

  private setStatus(status: ConnectionStatus): void {
    this.statusHandlers.forEach((handler) => handler(status));
  }
}
