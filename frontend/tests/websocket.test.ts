// Run with: npm test
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "node:test";
import { SignXSocket, resolveWsBase } from "../src/services/websocket.ts";
import type { ConnectionStatus, ServerMessage } from "../src/types.ts";

// --- WebSocket fake ----------------------------------------------------------

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readyState = FakeWebSocket.CONNECTING;
  binaryType = "";
  sent: unknown[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }
  send(data: unknown) {
    this.sent.push(data);
  }
  /** Client-initiated close: like browsers, the close event arrives later. */
  close() {
    this.readyState = FakeWebSocket.CLOSING;
  }
  // --- driven by the test ---
  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }
  receive(msg: unknown) {
    this.onmessage?.({ data: typeof msg === "string" ? msg : JSON.stringify(msg) });
  }
  dropped() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onerror?.();
    this.onclose?.();
  }
  finishClose() {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.();
  }
}

const URL = "ws://test/ws/predict";
const live = () => FakeWebSocket.instances.filter((s) => s.readyState <= FakeWebSocket.OPEN);

function makeSocket() {
  const socket = new SignXSocket({ url: URL, WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket });
  const statuses: ConnectionStatus[] = [];
  const messages: ServerMessage[] = [];
  socket.onStatusChange((s) => statuses.push(s));
  socket.onMessage((m) => messages.push(m));
  return { socket, statuses, messages };
}

const prediction = { type: "prediction", sign: "HELLO", confidence: 0.9 } as unknown as ServerMessage;
const blob = new Blob(["jpeg"]);

describe("resolveWsBase", () => {
  const http = { protocol: "http:", hostname: "localhost" };
  const https = { protocol: "https:", hostname: "signx.example.com" };

  it("uses VITE_WS_BASE_URL verbatim (minus a trailing slash)", () => {
    assert.equal(resolveWsBase({ VITE_WS_BASE_URL: "wss://api.example.com/" }, https), "wss://api.example.com");
  });

  it("derives a secure URL from VITE_API_BASE_URL", () => {
    assert.equal(resolveWsBase({ VITE_API_BASE_URL: "https://signx-api.onrender.com" }, https), "wss://signx-api.onrender.com");
    assert.equal(resolveWsBase({ VITE_API_BASE_URL: "http://192.168.1.5:8000/" }, http), "ws://192.168.1.5:8000");
  });

  it("falls back to port 8000 on the page host, matching the page's security", () => {
    assert.equal(resolveWsBase({}, http), "ws://localhost:8000");
    // An https page may not open ws:// (mixed content).
    assert.equal(resolveWsBase({}, https), "wss://signx.example.com:8000");
  });

  it("ignores empty variables", () => {
    assert.equal(resolveWsBase({ VITE_WS_BASE_URL: "", VITE_API_BASE_URL: "" }, http), "ws://localhost:8000");
  });
});

describe("SignXSocket", () => {
  beforeEach(() => {
    FakeWebSocket.instances = [];
    mock.timers.enable({ apis: ["setTimeout"] });
  });
  afterEach(() => mock.timers.reset());

  it("connects to the configured URL and reports status", () => {
    const { socket, statuses } = makeSocket();
    socket.connect();
    assert.equal(FakeWebSocket.instances.length, 1);
    assert.equal(FakeWebSocket.instances[0].url, URL);
    FakeWebSocket.instances[0].open();
    assert.deepEqual(statuses, ["connecting", "connected"]);
  });

  it("only sends frames and controls while open", () => {
    const { socket } = makeSocket();
    socket.connect();
    const ws = FakeWebSocket.instances[0];
    assert.equal(socket.canSendFrame(), false);
    socket.sendFrame(blob);
    assert.equal(socket.resetStream(), false);
    assert.deepEqual(ws.sent, []);

    ws.open();
    assert.equal(socket.canSendFrame(), true);
    socket.sendFrame(blob);
    socket.clearSentence();
    socket.speak();
    socket.stopSpeech();
    assert.equal(socket.resetStream(), true);
    assert.deepEqual(ws.sent, [
      blob,
      '{"type":"clear"}',
      '{"type":"speak"}',
      '{"type":"stop_speech"}',
      '{"type":"reset"}',
    ]);
  });

  it("allows one frame in flight; a prediction or error reply frees the slot", () => {
    const { socket } = makeSocket();
    socket.connect();
    const ws = FakeWebSocket.instances[0];
    ws.open();
    socket.sendFrame(blob);
    assert.equal(socket.canSendFrame(), false);
    ws.receive({ type: "status", message: "pong" }); // not a frame reply
    assert.equal(socket.canSendFrame(), false);
    ws.receive(prediction);
    assert.equal(socket.canSendFrame(), true);
    socket.sendFrame(blob);
    ws.receive({ type: "error", message: "Invalid or oversized frame" });
    assert.equal(socket.canSendFrame(), true);
  });

  it("ignores malformed server messages without breaking later ones", () => {
    const { socket, messages } = makeSocket();
    socket.connect();
    const ws = FakeWebSocket.instances[0];
    ws.open();
    ws.receive("{not json");
    ws.receive("NaN");
    ws.receive(prediction);
    assert.deepEqual(messages, [prediction]);
  });

  it("reconnects with exponential backoff after the connection drops", () => {
    const { socket, statuses } = makeSocket();
    socket.connect();
    FakeWebSocket.instances[0].open();
    FakeWebSocket.instances[0].dropped();
    assert.equal(statuses.at(-1), "disconnected");

    mock.timers.tick(999);
    assert.equal(FakeWebSocket.instances.length, 1);
    mock.timers.tick(1);
    assert.equal(FakeWebSocket.instances.length, 2); // after 1 s

    FakeWebSocket.instances[1].dropped(); // still down
    mock.timers.tick(1999);
    assert.equal(FakeWebSocket.instances.length, 2);
    mock.timers.tick(1);
    assert.equal(FakeWebSocket.instances.length, 3); // after 2 s

    FakeWebSocket.instances[2].open(); // back: backoff resets
    FakeWebSocket.instances[2].dropped();
    mock.timers.tick(1000);
    assert.equal(FakeWebSocket.instances.length, 4);
  });

  it("caps the backoff at 10 s", () => {
    const { socket } = makeSocket();
    socket.connect();
    for (let i = 0; i < 6; i++) {
      FakeWebSocket.instances.at(-1)!.dropped();
      mock.timers.tick(10_000);
    }
    const before = FakeWebSocket.instances.length;
    FakeWebSocket.instances.at(-1)!.dropped();
    mock.timers.tick(9_999);
    assert.equal(FakeWebSocket.instances.length, before);
    mock.timers.tick(1);
    assert.equal(FakeWebSocket.instances.length, before + 1);
  });

  it("a frame lost across a reconnect does not block the new connection", () => {
    const { socket } = makeSocket();
    socket.connect();
    FakeWebSocket.instances[0].open();
    socket.sendFrame(blob); // its reply never arrives
    FakeWebSocket.instances[0].dropped();
    mock.timers.tick(1000);
    FakeWebSocket.instances[1].open();
    assert.equal(socket.canSendFrame(), true);
  });

  it("does not reconnect after disconnect(), even when the close event arrives late", () => {
    const { socket } = makeSocket();
    socket.connect();
    const ws = FakeWebSocket.instances[0];
    ws.open();
    socket.disconnect();
    ws.finishClose();
    mock.timers.tick(60_000);
    assert.equal(FakeWebSocket.instances.length, 1);
    assert.equal(socket.canSendFrame(), false);
  });

  it("disconnect() cancels a pending reconnect", () => {
    const { socket } = makeSocket();
    socket.connect();
    FakeWebSocket.instances[0].dropped();
    socket.disconnect();
    mock.timers.tick(60_000);
    assert.equal(FakeWebSocket.instances.length, 1);
  });

  it("disconnect() then connect() leaves exactly one live connection", () => {
    // The old socket's close event arrives after the new socket was created:
    // it must neither flip the status to "disconnected" nor schedule a
    // second, duplicate connection.
    const { socket, statuses, messages } = makeSocket();
    socket.connect();
    const first = FakeWebSocket.instances[0];
    first.open();
    socket.disconnect();
    socket.connect();
    const second = FakeWebSocket.instances[1];
    second.open();
    first.finishClose();
    mock.timers.tick(60_000);

    assert.equal(FakeWebSocket.instances.length, 2);
    assert.deepEqual(live(), [second]);
    assert.equal(statuses.at(-1), "connected");

    first.receive(prediction); // a straggler from the old connection
    assert.deepEqual(messages, []);
  });

  it("unsubscribed handlers receive nothing", () => {
    const socket = new SignXSocket({ url: URL, WebSocketImpl: FakeWebSocket as unknown as typeof WebSocket });
    const got: ServerMessage[] = [];
    const unsubscribe = socket.onMessage((m) => got.push(m));
    socket.connect();
    FakeWebSocket.instances[0].open();
    unsubscribe();
    FakeWebSocket.instances[0].receive(prediction);
    assert.deepEqual(got, []);
  });
});
