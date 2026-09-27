import { fireEvent, render, screen } from "@testing-library/react";
import { Circle } from "lucide-react";
import { describe, expect, it, vi } from "vitest";

import { WorkspaceNavigation } from "./WorkspaceNavigation";

describe("WorkspaceNavigation", () => {
  it("previews on hover and selects on click", () => {
    const preview = vi.fn();
    const cancel = vi.fn();
    const select = vi.fn();
    render(
      <WorkspaceNavigation
        active="flow"
        items={[
          { view: "flow", label: "Flow", icon: Circle },
          { view: "notes", label: "Notes", icon: Circle },
        ]}
        onPreview={preview}
        onCancelPreview={cancel}
        onSelect={select}
      />,
    );
    const notes = screen.getByTitle("Notes");
    fireEvent.mouseEnter(notes);
    fireEvent.mouseLeave(notes);
    fireEvent.click(notes);
    expect(preview).toHaveBeenCalledWith("notes");
    expect(cancel).toHaveBeenCalledOnce();
    expect(select).toHaveBeenCalledWith("notes");
  });
});
