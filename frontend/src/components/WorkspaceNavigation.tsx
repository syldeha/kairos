import type { ComponentType } from "react";

export type WorkspaceView = "flow" | "board" | "notes" | "transcript" | "monitor";

type Item = {
  view: WorkspaceView;
  label: string;
  icon: ComponentType<{ size?: number }>;
};

type Props = {
  active: WorkspaceView;
  items: Item[];
  onPreview: (view: WorkspaceView) => void;
  onCancelPreview: () => void;
  onSelect: (view: WorkspaceView) => void;
};

export function WorkspaceNavigation(props: Props) {
  return (
    <nav className="view-nav" aria-label="Workspace views">
      {props.items.map((item) => {
        const Icon = item.icon;
        return (
          <button
            className={props.active === item.view ? "active" : ""}
            onMouseEnter={() => props.onPreview(item.view)}
            onMouseLeave={props.onCancelPreview}
            onFocus={() => props.onPreview(item.view)}
            onClick={() => props.onSelect(item.view)}
            key={item.view}
            title={item.label}
            aria-pressed={props.active === item.view}
          >
            <Icon size={16} />
            <span>{item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
