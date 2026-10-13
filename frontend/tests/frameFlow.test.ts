// Run with: npm test
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { InFlightLimiter } from "../src/services/frameFlow.ts";

describe("InFlightLimiter", () => {
  it("allows one frame in flight until its reply arrives", () => {
    const l = new InFlightLimiter(1, 2000);
    assert.equal(l.canSend(0), true);
    l.sent(0);
    assert.equal(l.canSend(10), false);
    assert.equal(l.canSend(1999), false);
    l.received();
    assert.equal(l.canSend(20), true);
  });

  it("resumes sending if a reply is lost", () => {
    const l = new InFlightLimiter(1, 2000);
    l.sent(0);
    assert.equal(l.canSend(2001), true);
    l.sent(2001);
    assert.equal(l.pending, 1);
    l.received(); // the late reply to the first frame, or the second one
    l.received(); // never goes negative
    assert.equal(l.pending, 0);
    assert.equal(l.canSend(2002), true);
  });

  it("reset clears pending frames (new connection)", () => {
    const l = new InFlightLimiter();
    l.sent(0);
    l.reset();
    assert.equal(l.canSend(1), true);
  });
});
