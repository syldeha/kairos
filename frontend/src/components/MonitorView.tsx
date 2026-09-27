import { Download } from "lucide-react";
import { useState } from "react";
import type { AtlasState } from "../protocol";

type Trace = NonNullable<AtlasState["trace"]>[number];
type Filter = "all" | "spoke" | "silent";

/** How each turn was handled, end to end: the tool for understanding a behaviour and reporting a bug. */
export function MonitorView({ state }: { state: AtlasState }) {
  const [filter, setFilter] = useState<Filter>("all");
  const trace = state.trace ?? [];
  const byTurn = new Map<string, Trace[]>();
  for (const event of trace) {
    const events = byTurn.get(event.utterance_id) ?? [];
    events.push(event);
    byTurn.set(event.utterance_id, events);
  }
  const texts = new Map(state.transcript.map((item) => [item.id, item.text]));
  const turns = [...byTurn.entries()]
    .filter(([, events]) => {
      const spoke = events.some((event) => event.stage === "voice" || event.detail.startsWith("speaks"));
      return filter === "all" || (filter === "spoke" ? spoke : !spoke);
    })
    .reverse()
    .slice(0, 30);
  const thoughts = (state.thoughts ?? []).filter((item) => item.status === "ready" || item.status === "pending");
  const decisions = (state.policy_decisions ?? []).slice(-20).reverse();

  return <section className="monitor" aria-label="Monitor">
    <header className="monitor-heading">
      <div>
        <h2>Monitor</h2>
        <p>Every turn from hearing to speaking, with the reason for each step. Topic: {state.topic || "not named yet"}.</p>
      </div>
      <div className="monitor-actions">
        {(["all", "spoke", "silent"] as Filter[]).map((item) => <button key={item} className={filter === item ? "active" : ""} aria-pressed={filter === item} onClick={() => setFilter(item)}>{item === "all" ? "All turns" : item === "spoke" ? "Atlas spoke" : "Atlas stayed silent"}</button>)}
        <button onClick={() => exportTrace(state)}><Download size={13} /> Export JSON</button>
      </div>
    </header>
    <div className="monitor-grid">
      <div className="monitor-panel">
        <h3>Turns</h3>
        {turns.map(([utteranceId, events]) => <div className="turn" key={utteranceId}>
          <p>{texts.get(utteranceId) ?? events.find((event) => event.stage === "heard" || event.stage === "partial")?.detail ?? utteranceId}</p>
          <ol>{events.map((event) => <li key={event.id}><time>+{event.ms} ms</time><span className={`stage ${event.stage}`}>{event.stage}</span><span>{event.detail}</span></li>)}</ol>
        </div>)}
        {!turns.length && <p className="muted">No turn yet.</p>}
      </div>
      <div className="monitor-grid" style={{ gridTemplateColumns: "1fr" }}>
        <div className="monitor-panel">
          <h3>Reservoir ({thoughts.length} ready)</h3>
          {thoughts.slice().reverse().map((item) => <div className="thought-row" key={item.id}>
            <header><span>{item.kind}{item.owed ? " · owed" : ""}</span><span>Jev {item.chosen?.toFixed(2) ?? "-"}</span><span>said {item.already_said?.toFixed(2) ?? "-"}</span>{item.kind === "correction" && <span>right {item.fit?.toFixed(2) ?? "-"}</span>}</header>
            <p>{item.utterance}</p>
          </div>)}
          {!thoughts.length && <p className="muted">Nothing prepared.</p>}
          <h3 style={{ marginTop: 16 }}>Reservoir log</h3>
          {(state.reservoir_log ?? []).slice(-12).reverse().map((event) => <div className="silence" key={event.id}><b>{event.event}</b><span>{event.by}: {event.utterance}{event.why ? ` (${event.why})` : ""}</span></div>)}
        </div>
        <div className="monitor-panel">
          <h3>Why Atlas spoke or stayed silent</h3>
          {decisions.map((item) => <div className="silence" key={item.id}><b className={item.speak ? "speak" : ""}>{item.speak ? "speaks" : "silent"}</b><span>{item.reason}</span></div>)}
          {!decisions.length && <p className="muted">No decision yet.</p>}
        </div>
        <div className="monitor-panel">
          <h3>Latency per agent</h3>
          <LatencyTable state={state} />
        </div>
      </div>
    </div>
  </section>;
}

function LatencyTable({ state }: { state: AtlasState }) {
  const rows = new Map<string, number[]>();
  for (const run of state.agent_runs) {
    if (run.duration_ms == null) continue;
    const values = rows.get(run.agent) ?? [];
    values.push(run.duration_ms);
    rows.set(run.agent, values);
  }
  if (!rows.size) return <p className="muted">No completed run yet.</p>;
  return <table className="latency">
    <thead><tr><th>Agent</th><th>Runs</th><th>Mean</th><th>Last</th><th>Max</th></tr></thead>
    <tbody>{[...rows.entries()].map(([agent, values]) => <tr key={agent}>
      <td>{agent}</td><td>{values.length}</td><td>{seconds(values.reduce((a, b) => a + b, 0) / values.length)}</td><td>{seconds(values[values.length - 1])}</td><td>{seconds(Math.max(...values))}</td>
    </tr>)}</tbody>
  </table>;
}

function seconds(ms: number) {
  return `${(ms / 1000).toFixed(2)} s`;
}

function exportTrace(state: AtlasState) {
  const payload = {
    session_id: state.session_id,
    title: state.title,
    exported_at: new Date().toISOString(),
    topic: state.topic,
    transcript: state.transcript,
    trace: state.trace,
    thoughts: state.thoughts,
    reservoir_log: state.reservoir_log,
    policy_decisions: state.policy_decisions,
    decisions: state.decisions,
    speeches: state.speeches,
    agent_runs: state.agent_runs,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `atlas-trace-${state.session_id ?? "session"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
