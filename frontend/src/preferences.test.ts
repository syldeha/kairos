import { describe, expect, it, vi } from "vitest";

import { promptBarVisibleByDefault, savePromptBarVisibility } from "./preferences";

describe("prompt bar preference", () => {
  it("is enabled by default and persists an explicit hidden state", () => {
    expect(promptBarVisibleByDefault({ getItem: () => null })).toBe(true);
    expect(promptBarVisibleByDefault({ getItem: () => "false" })).toBe(false);
    const setItem = vi.fn();
    savePromptBarVisibility(false, { setItem });
    expect(setItem).toHaveBeenCalledWith("atlas.promptBarVisible", "false");
  });
});
