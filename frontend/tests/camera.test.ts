// Run with: npm test  (Node's built-in test runner; Node >= 22.18 strips TS types natively)
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, it, mock } from "node:test";
import {
  CAMERA_STORAGE_KEY,
  CameraController,
  toCameraDevices,
  type CameraState,
} from "../src/services/camera.ts";

// --- MediaDevices / MediaStream fakes -------------------------------------

class FakeTrack {
  deviceId: string;
  stopped = false;
  private listeners: Record<string, Array<() => void>> = {};
  constructor(deviceId: string) {
    this.deviceId = deviceId;
  }
  stop() {
    this.stopped = true;
  }
  getSettings() {
    return { deviceId: this.deviceId };
  }
  addEventListener(type: string, fn: () => void) {
    (this.listeners[type] ??= []).push(fn);
  }
  fire(type: string) {
    this.stopped = true;
    (this.listeners[type] ?? []).forEach((fn) => fn());
  }
}

class FakeStream {
  track: FakeTrack;
  constructor(deviceId: string) {
    this.track = new FakeTrack(deviceId);
  }
  getTracks() {
    return [this.track];
  }
  getVideoTracks() {
    return [this.track];
  }
  get stopped() {
    return this.track.stopped;
  }
}

interface FakeDevice {
  deviceId: string;
  label: string;
}

function domError(name: string): Error {
  return Object.assign(new Error(name), { name });
}

class FakeMediaDevices {
  devices: FakeDevice[];
  granted = false;
  deny = false;
  /** When true, getUserMedia() waits until the test calls resolveNext()/resolveFor(). */
  manual = false;
  failFor = new Map<string, string>();
  requests: Array<string | null> = [];
  streams: FakeStream[] = [];
  /** Snapshot: were all previously opened streams stopped when getUserMedia() was called? */
  previousStoppedAtRequest: boolean[] = [];
  pending: Array<{ deviceId: string | null; resolve: () => void }> = [];
  listeners = new Set<() => void>();

  constructor(devices: FakeDevice[]) {
    this.devices = devices;
  }

  async enumerateDevices() {
    return [
      { kind: "audioinput", deviceId: "mic", label: this.granted ? "Microphone" : "", groupId: "" },
      ...this.devices.map((d) => ({ kind: "videoinput", deviceId: d.deviceId, label: this.granted ? d.label : "", groupId: "" })),
    ] as unknown as MediaDeviceInfo[];
  }

  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStream> {
    const video = constraints.video as { deviceId?: { exact: string } };
    const deviceId = video.deviceId?.exact ?? null;
    this.requests.push(deviceId);
    this.previousStoppedAtRequest.push(this.streams.every((s) => s.stopped));

    const settle = (resolve: (s: MediaStream) => void, reject: (e: Error) => void) => {
      if (this.deny) return reject(domError("NotAllowedError"));
      if (this.devices.length === 0) return reject(domError("NotFoundError"));
      const target = deviceId ? this.devices.find((d) => d.deviceId === deviceId) : this.devices[0];
      if (!target) return reject(domError("OverconstrainedError"));
      const failure = this.failFor.get(target.deviceId);
      if (failure) return reject(domError(failure));
      this.granted = true;
      const stream = new FakeStream(target.deviceId);
      this.streams.push(stream);
      resolve(stream as unknown as MediaStream);
    };

    return new Promise((resolve, reject) => {
      if (this.manual) this.pending.push({ deviceId, resolve: () => settle(resolve, reject) });
      else settle(resolve, reject);
    });
  }

  resolveFor(deviceId: string | null) {
    const i = this.pending.findIndex((p) => p.deviceId === deviceId);
    assert.ok(i >= 0, `no pending request for ${deviceId}`);
    this.pending.splice(i, 1)[0].resolve();
  }

  addEventListener(_type: string, fn: () => void) {
    this.listeners.add(fn);
  }
  removeEventListener(_type: string, fn: () => void) {
    this.listeners.delete(fn);
  }
  fireDeviceChange() {
    this.listeners.forEach((fn) => fn());
  }

  get liveStreams() {
    return this.streams.filter((s) => !s.stopped);
  }
}

class MemoryStorage {
  data = new Map<string, string>();
  getItem(k: string) {
    return this.data.get(k) ?? null;
  }
  setItem(k: string, v: string) {
    this.data.set(k, v);
  }
}

const flush = () => new Promise<void>((r) => setImmediate(r));

const CAMERAS: FakeDevice[] = [
  { deviceId: "builtin", label: "Integrated Webcam" },
  { deviceId: "usb", label: "USB Camera" },
  { deviceId: "virtual", label: "OBS Virtual Camera" },
];

function setup(devices: FakeDevice[] = CAMERAS, storage: MemoryStorage | null = new MemoryStorage()) {
  const media = new FakeMediaDevices(devices.map((d) => ({ ...d })));
  const states: CameraState[] = [];
  const video = { srcObject: null as MediaProvider | null };
  let sourceChanges = 0;
  const controller = new CameraController({
    mediaDevices: media as unknown as MediaDevices,
    storage,
    onStateChange: (s) => states.push(s),
    onSourceChange: () => {
      sourceChanges += 1;
    },
  });
  controller.attachVideo(video);
  return {
    media,
    controller,
    video,
    storage,
    states,
    get state() {
      return controller.getState();
    },
    get sourceChanges() {
      return sourceChanges;
    },
  };
}

// --- Tests -----------------------------------------------------------------

describe("toCameraDevices", () => {
  it("keeps video inputs only and falls back to 'Camera N' without labels", () => {
    const devices = toCameraDevices([
      { kind: "audioinput", deviceId: "mic", label: "" },
      { kind: "videoinput", deviceId: "a", label: "" },
      { kind: "videoinput", deviceId: "b", label: "USB Camera" },
    ] as unknown as MediaDeviceInfo[]);
    assert.deepEqual(devices, [
      { deviceId: "a", label: "Camera 1" },
      { deviceId: "b", label: "USB Camera" },
    ]);
  });
});

describe("CameraController", () => {
  beforeEach(() => mock.timers.enable({ apis: ["setInterval"] }));
  afterEach(() => mock.timers.reset());

  it("starts the default camera and lists human-readable labels once permitted", async () => {
    const t = setup();
    await t.controller.start();
    assert.deepEqual(t.media.requests, [null]); // browser default, no stored choice
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.activeDeviceId, "builtin");
    assert.deepEqual(
      t.state.devices.map((d) => d.label),
      ["Integrated Webcam", "USB Camera", "OBS Virtual Camera"],
    );
    assert.equal(t.video.srcObject, t.media.streams[0]);
    assert.equal(t.storage!.getItem(CAMERA_STORAGE_KEY), "builtin");
  });

  it("shows fallback labels while permission is not granted", async () => {
    const t = setup();
    t.media.deny = true;
    await t.controller.start();
    assert.equal(t.state.status, "denied");
    assert.match(t.state.error, /permission/i);
    assert.deepEqual(
      t.state.devices.map((d) => d.label),
      ["Camera 1", "Camera 2", "Camera 3"],
    );
    assert.equal(t.media.liveStreams.length, 0);

    t.media.deny = false; // user allows access, then presses Retry
    await t.controller.retry();
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.devices[0].label, "Integrated Webcam");
  });

  it("prefers the remembered camera and ignores a stale remembered id", async () => {
    const storage = new MemoryStorage();
    storage.setItem(CAMERA_STORAGE_KEY, "usb");
    const t = setup(CAMERAS, storage);
    await t.controller.start();
    assert.deepEqual(t.media.requests, ["usb"]);
    assert.equal(t.state.activeDeviceId, "usb");

    const storage2 = new MemoryStorage();
    storage2.setItem(CAMERA_STORAGE_KEY, "gone-device");
    const t2 = setup(CAMERAS, storage2);
    await t2.controller.start();
    assert.deepEqual(t2.media.requests, [null]);
    assert.equal(t2.state.activeDeviceId, "builtin");
  });

  it("falls back to the default camera if the remembered one disappears during startup", async () => {
    const storage = new MemoryStorage();
    storage.setItem(CAMERA_STORAGE_KEY, "usb");
    const t = setup(CAMERAS, storage);
    t.media.failFor.set("usb", "NotFoundError");
    await t.controller.start();
    assert.deepEqual(t.media.requests, ["usb", null]);
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.activeDeviceId, "builtin");
  });

  it("works without storage", async () => {
    const t = setup(CAMERAS, null);
    await t.controller.start();
    assert.equal(t.state.status, "connected");
  });

  it("switches camera: stops the old stream first, attaches the new one", async () => {
    const t = setup();
    await t.controller.start();
    const first = t.media.streams[0];
    const sourceChangesBefore = t.sourceChanges;

    const switching = t.controller.select("usb");
    assert.equal(t.state.status, "switching");
    assert.equal(first.stopped, true);
    await switching;

    assert.deepEqual(t.media.requests, [null, "usb"]);
    assert.equal(t.media.previousStoppedAtRequest[1], true, "old tracks stopped before requesting the new camera");
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.activeDeviceId, "usb");
    assert.equal(t.video.srcObject, t.media.streams[1]);
    assert.equal(t.media.liveStreams.length, 1);
    assert.equal(t.sourceChanges, sourceChangesBefore + 1);
    assert.equal(t.storage!.getItem(CAMERA_STORAGE_KEY), "usb");
  });

  it("switches back to the original camera", async () => {
    const t = setup();
    await t.controller.start();
    await t.controller.select("usb");
    await t.controller.select("builtin");
    assert.equal(t.state.activeDeviceId, "builtin");
    assert.equal(t.media.liveStreams.length, 1);
    assert.equal(t.media.liveStreams[0].track.deviceId, "builtin");
  });

  it("selecting the active camera again does nothing", async () => {
    const t = setup();
    await t.controller.start();
    const changes = t.sourceChanges;
    await t.controller.select("builtin");
    assert.deepEqual(t.media.requests, [null]);
    assert.equal(t.sourceChanges, changes);
  });

  it("runs exactly one capture loop and none while switching", async () => {
    const t = setup();
    const frames: string[] = [];
    let camera = "builtin";
    const capture = {
      grab: async () => new Blob([camera]),
      onFrame: (b: Blob) => frames.push(b.size ? camera : "?"),
      intervalMs: 100,
    };
    t.controller.setCapture(capture);
    t.controller.setCapture(capture); // re-set (e.g. React effect re-run) must not add a loop
    await t.controller.start();

    mock.timers.tick(100);
    await flush();
    assert.equal(frames.length, 1);

    // Switch while the new camera is still opening: no frames at all.
    t.media.manual = true;
    const switching = t.controller.select("usb");
    assert.equal(t.controller.isCapturing, false);
    mock.timers.tick(500);
    await flush();
    assert.equal(frames.length, 1);

    camera = "usb";
    t.media.resolveFor("usb");
    await switching;
    mock.timers.tick(100);
    await flush();
    assert.deepEqual(frames, ["builtin", "usb"]);
  });

  it("drops a frame from the old camera that finishes encoding after the switch", async () => {
    const t = setup();
    await t.controller.start();
    let release: (b: Blob) => void = () => {};
    const delivered: Blob[] = [];
    t.controller.setCapture({
      grab: () => new Promise<Blob>((r) => (release = r)),
      onFrame: (b) => delivered.push(b),
      intervalMs: 100,
    });
    mock.timers.tick(100); // old-camera frame starts encoding
    await t.controller.select("usb");
    release(new Blob(["old"]));
    await flush();
    assert.equal(delivered.length, 0);
  });

  it("skips ticks without encoding a frame while not ready (server busy)", async () => {
    const t = setup();
    await t.controller.start();
    let ready = false;
    let grabs = 0;
    let frames = 0;
    t.controller.setCapture({
      grab: async () => {
        grabs += 1;
        return new Blob(["x"]);
      },
      onFrame: () => frames++,
      intervalMs: 67,
      ready: () => ready,
    });
    mock.timers.tick(67 * 3);
    await flush();
    assert.equal(grabs, 0);
    assert.equal(frames, 0);

    ready = true;
    mock.timers.tick(67);
    await flush();
    assert.equal(grabs, 1);
    assert.equal(frames, 1);
  });

  it("stops capture while paused", async () => {
    const t = setup();
    await t.controller.start();
    let frames = 0;
    t.controller.setCapture({ grab: async () => new Blob(["x"]), onFrame: () => frames++, intervalMs: 100 });
    t.controller.setCapture(null);
    mock.timers.tick(300);
    await flush();
    assert.equal(frames, 0);
    assert.equal(t.controller.isCapturing, false);
  });

  it("only the latest of several rapid selections wins", async () => {
    const t = setup();
    await t.controller.start();
    t.media.manual = true;
    const a = t.controller.select("usb");
    const b = t.controller.select("virtual");
    // The older request resolves last: it must be discarded and stopped.
    t.media.resolveFor("virtual");
    t.media.resolveFor("usb");
    await Promise.all([a, b]);

    assert.equal(t.state.activeDeviceId, "virtual");
    assert.equal(t.state.status, "connected");
    assert.equal(t.media.liveStreams.length, 1);
    assert.equal(t.media.liveStreams[0].track.deviceId, "virtual");
    assert.equal(t.video.srcObject, t.media.liveStreams[0]);
  });

  it("reports a failed switch, leaves no stream open, and allows another choice", async () => {
    const t = setup();
    await t.controller.start();
    t.media.failFor.set("usb", "NotReadableError");
    let frames = 0;
    t.controller.setCapture({ grab: async () => new Blob(["x"]), onFrame: () => frames++, intervalMs: 100 });

    await t.controller.select("usb");
    assert.equal(t.state.status, "error");
    assert.match(t.state.error, /in use/);
    assert.equal(t.media.liveStreams.length, 0);
    assert.equal(t.video.srcObject, null);
    mock.timers.tick(300);
    await flush();
    assert.equal(frames, 0);

    await t.controller.select("virtual");
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.activeDeviceId, "virtual");
    assert.equal(t.state.error, "");
  });

  it("handles the active camera being unplugged", async () => {
    const t = setup();
    await t.controller.start();
    let frames = 0;
    t.controller.setCapture({ grab: async () => new Blob(["x"]), onFrame: () => frames++, intervalMs: 100 });

    t.media.devices = t.media.devices.filter((d) => d.deviceId !== "builtin");
    t.media.streams[0].track.fire("ended");
    t.media.fireDeviceChange();
    await flush();

    assert.equal(t.state.status, "unavailable");
    assert.match(t.state.error, /disconnected/i);
    assert.equal(t.state.activeDeviceId, null);
    assert.deepEqual(
      t.state.devices.map((d) => d.deviceId),
      ["usb", "virtual"],
    );
    mock.timers.tick(300);
    await flush();
    assert.equal(frames, 0);

    await t.controller.select("usb");
    assert.equal(t.state.status, "connected");
  });

  it("reports when no cameras exist and starts one when plugged in", async () => {
    const t = setup([]);
    await t.controller.start();
    assert.equal(t.state.status, "no-camera");
    assert.match(t.state.error, /No cameras found/);

    t.media.devices = [{ deviceId: "usb", label: "USB Camera" }];
    t.media.fireDeviceChange();
    await flush();
    await flush();
    assert.equal(t.state.status, "connected");
    assert.equal(t.state.activeDeviceId, "usb");
  });

  it("reports an unsupported browser", async () => {
    const states: CameraState[] = [];
    const controller = new CameraController({ mediaDevices: undefined, onStateChange: (s) => states.push(s) });
    await controller.start();
    assert.equal(controller.getState().status, "unavailable");
  });

  it("cleans up on unmount, including a request still in flight", async () => {
    const t = setup();
    await t.controller.start();
    assert.equal(t.media.listeners.size, 1);
    t.media.manual = true;
    const switching = t.controller.select("usb");

    t.controller.dispose();
    assert.equal(t.media.listeners.size, 0);
    assert.equal(t.media.liveStreams.length, 0);
    const updatesAtDispose = t.states.length;

    t.media.resolveFor("usb"); // resolves after unmount
    await switching;
    assert.equal(t.media.liveStreams.length, 0, "late stream is stopped");
    assert.equal(t.states.length, updatesAtDispose, "no state updates after unmount");
    assert.equal(t.controller.isCapturing, false);
  });
});
