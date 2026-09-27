import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AtlasState } from "../protocol";
import { MonitorView } from "./MonitorView";

const state = {
  project_id: "atlas",
  protocol_version: 12,
  topic: "l'anniversaire de Karim",
  transcript: [
    { id: "utt_1", text: "À 35 euros chacun, à six ça fait 180 euros.", speaker: "unknown" },
    { id: "utt_2", text: "Toi, tu es dispo samedi ?", speaker: "unknown" },
  ],
  trace: [
    { id: "t1", utterance_id: "utt_1", stage: "heard", ms: 0, detail: "À 35 euros chacun" },
    { id: "t2", utterance_id: "utt_1", stage: "checker", ms: 810, detail: "correction: 210, pas 180." },
    { id: "t3", utterance_id: "utt_1", stage: "policy", ms: 1080, detail: "speaks: correction (fit 0.71)" },
    { id: "t4", utterance_id: "utt_2", stage: "speaker", ms: 260, detail: "not for Atlas (another_participant)" },
  ],
  thoughts: [
    { id: "th_1", kind: "idea", utterance: "Le Massyl a un menu de groupe.", status: "ready", chosen: 0.12 },
  ],
  reservoir_log: [],
  policy_decisions: [
    { id: "p1", speak: true, reason: "correction (fit 0.71)" },
    { id: "p2", speak: false, reason: "Jev: nothing worth saying (best 0.12, none 0.80)" },
  ],
  agent_runs: [{ id: "a1", agent: "checker", summary: "Checking", status: "done", duration_ms: 810 }],
  speeches: [],
  cards: [],
  tasks: [],
  activities: [],
} as unknown as AtlasState;

describe("MonitorView", () => {
  it("shows each turn from hearing to the decision, and why Atlas spoke or stayed silent", () => {
    render(<MonitorView state={state} />);
    expect(screen.getByText("À 35 euros chacun, à six ça fait 180 euros.")).toBeTruthy();
    expect(screen.getByText("+810 ms")).toBeTruthy();
    expect(screen.getByText("Jev: nothing worth saying (best 0.12, none 0.80)")).toBeTruthy();
    expect(screen.getByText("Le Massyl a un menu de groupe.")).toBeTruthy();
    expect(screen.getAllByText("checker").length).toBe(2); // its step in the turn, its row in the latencies
  });

  it("filters the turns where Atlas stayed silent", () => {
    render(<MonitorView state={state} />);
    fireEvent.click(screen.getByText("Atlas stayed silent"));
    expect(screen.queryByText("À 35 euros chacun, à six ça fait 180 euros.")).toBeNull();
    expect(screen.getByText("Toi, tu es dispo samedi ?")).toBeTruthy();
  });
});
