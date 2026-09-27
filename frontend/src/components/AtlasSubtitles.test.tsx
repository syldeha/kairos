import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AtlasSubtitles } from "./AtlasSubtitles";

describe("AtlasSubtitles", () => {
  it("shows only Atlas spoken text while enabled", () => {
    const { rerender } = render(
      <AtlasSubtitles enabled text="I found the source." companionName="Atlas" />,
    );
    expect(screen.getByRole("status").textContent).toContain("I found the source.");
    rerender(<AtlasSubtitles enabled={false} text="Hidden" companionName="Atlas" />);
    expect(screen.queryByRole("status")).toBeNull();
    rerender(<AtlasSubtitles enabled text="Current spoken phrase" companionName="Atlas" />);
    expect(screen.getByRole("status").textContent).toContain("Current spoken phrase");
  });

  it("shows the current trailing clauses without ellipsis for long speech", () => {
    const longText =
      "The first sentence explains the background in detail. " +
      "The second sentence contains the current spoken point. " +
      "The final sentence is what participants should read now.";
    render(<AtlasSubtitles enabled text={longText} companionName="Atlas" />);
    const subtitle = screen.getByRole("status");
    expect(subtitle.textContent).toContain("The final sentence is what participants should read now.");
    expect(subtitle.textContent).not.toContain("...");
  });
});
