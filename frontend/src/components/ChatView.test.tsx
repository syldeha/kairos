import { describe, expect, it } from "vitest";

import { chatMessages } from "./ChatView";
import type { AtlasState } from "../protocol";

describe("Chat", () => {
  it("interleaves people and Kairos and measures Kairos's delay after the last human line", () => {
    const state = {
      identity_name: "Kairos",
      transcript: [
        { id: "L1", text: "Kairos, un vol pour Split ?", speaker: "Claude", committed_at: "2026-09-27T19:00:00.000Z" },
      ],
      speeches: [
        { id: "s1", text: "Je regarde.", status: "finished", created_at: "2026-09-27T19:00:00.800Z" },
        { id: "s2", text: "Le vol direct est à 71 euros.", status: "interrupted", created_at: "2026-09-27T19:00:03.100Z" },
      ],
    } as unknown as AtlasState;
    const messages = chatMessages(state);
    expect(messages.map((m) => m.who)).toEqual(["Claude", "Kairos", "Kairos"]);
    expect(messages[1].delay).toBeCloseTo(0.8);
    expect(messages[2].delay).toBeCloseTo(3.1);
    expect(messages[2].cut).toBe(true);
  });
});
