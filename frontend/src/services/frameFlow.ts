/**
 * Backpressure for webcam frames: allow at most `maxInFlight` frames sent to
 * the server without a reply. The backend processes frames one at a time,
 * so sending faster than it answers only queues stale frames and makes the
 * displayed prediction lag further and further behind the camera.
 *
 * If a reply never arrives (e.g. a frame lost across a reconnect), sending
 * resumes after `staleMs`.
 */
export class InFlightLimiter {
  private readonly maxInFlight: number;
  private readonly staleMs: number;
  private inFlight = 0;
  private lastSentAt = 0;

  constructor(maxInFlight = 1, staleMs = 2000) {
    this.maxInFlight = maxInFlight;
    this.staleMs = staleMs;
  }

  canSend(now: number): boolean {
    return this.inFlight < this.maxInFlight || now - this.lastSentAt > this.staleMs;
  }

  sent(now: number): void {
    if (this.inFlight >= this.maxInFlight) this.inFlight = 0; // replies presumed lost
    this.inFlight += 1;
    this.lastSentAt = now;
  }

  received(): void {
    this.inFlight = Math.max(0, this.inFlight - 1);
  }

  reset(): void {
    this.inFlight = 0;
  }

  get pending(): number {
    return this.inFlight;
  }
}
