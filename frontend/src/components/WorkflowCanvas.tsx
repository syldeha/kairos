import {
  AudioLines,
  Bot,
  BrainCircuit,
  Calculator,
  Compass,
  Database,
  FileText,
  Layers,
  Lightbulb,
  Mic2,
  Scale,
  Search,
  Sparkles,
  Volume2,
  X,
} from "lucide-react";
import { useState, type ComponentType } from "react";
import type { AtlasState } from "../protocol";

type AudioStatus = "idle" | "requesting" | "active" | "paused" | "error";
type NodeKey =
  | "room" | "stt" | "memory" | "jev" | "speaker" | "worker" | "notes" | "board" | "voice"
  | "checker" | "thinkers" | "topic" | "reservoir" | "policy";

type FlowNode = {
  key: NodeKey;
  label: string;
  detail: string;
  status: string;
  icon: ComponentType<{ size?: number }>;
  active: boolean;
};

const POSITIONS: Record<NodeKey, { x: number; y: number }> = {
  room: { x: 20, y: 250 },
  stt: { x: 215, y: 250 },
  jev: { x: 420, y: 70 },
  memory: { x: 420, y: 330 },
  board: { x: 215, y: 480 },
  notes: { x: 420, y: 500 },
  checker: { x: 640, y: 20 },
  thinkers: { x: 640, y: 160 },
  topic: { x: 640, y: 300 },
  worker: { x: 640, y: 440 },
  speaker: { x: 860, y: 40 },
  reservoir: { x: 860, y: 230 },
  policy: { x: 860, y: 420 },
  voice: { x: 1080, y: 230 },
};

const EDGES: Array<[NodeKey, NodeKey, string]> = [
  ["room", "stt", "audio"],
  ["stt", "memory", "heard"],
  ["stt", "jev", "sentences, even unfinished"],
  ["memory", "jev", "decision context"],
  ["jev", "speaker", "said to Atlas: direct response"],
  ["jev", "worker", "triggers searches"],
  ["jev", "topic", "subject changed"],
  ["memory", "checker", "figures to check"],
  ["memory", "thinkers", "what could be said"],
  ["checker", "reservoir", "corrections"],
  ["thinkers", "reservoir", "prepared thoughts"],
  ["topic", "reservoir", "retires old ideas"],
  ["worker", "reservoir", "results"],
  ["worker", "memory", "evidence"],
  ["reservoir", "policy", "Jev's pick"],
  ["policy", "voice", "speak or stay silent"],
  ["memory", "notes", "notes"],
  ["memory", "board", "board"],
  ["speaker", "voice", "voice"],
  ["voice", "room", "voice"],
];

export function WorkflowCanvas({ state, audioStatus }: { state: AtlasState; audioStatus: AudioStatus }) {
  const [selected, setSelected] = useState<NodeKey | null>(null);
  const agentRuns = state.agent_runs ?? [];
  const running = agentRuns.filter((run) => run.status === "running");
  const tasks = state.tasks ?? [];
  const runningTasks = tasks.filter((task) => task.status === "running" || task.status === "queued");
  const lastTask = tasks.at(-1);
  const speaker = running.find((run) => run.agent === "speaker");
  const worker = running.find((run) => run.agent === "worker" || run.agent === "coordinator");
  const notes = running.find((run) => run.agent === "notes");
  const naming = running.find((run) => run.agent === "naming");
  const speech = state.speeches.at(-1);
  const decision = state.decisions?.at(-1);
  const decisionActive = Boolean(
    decision?.decided_at && Date.now() - new Date(decision.decided_at).getTime() < 3000,
  );
  const recentBoardActivity = state.activities
    .slice()
    .reverse()
    .find((item) => item.kind.startsWith("board."));
  const recentBoardActivityAt = recentBoardActivity?.created_at;
  const boardActive = Boolean(
    worker?.agent === "coordinator"
    || (recentBoardActivityAt && Date.now() - new Date(recentBoardActivityAt).getTime() < 3000),
  );
  const pipeline = state.pipeline;
  const thoughts = (state.thoughts ?? []).filter((item) => item.status === "ready" || item.status === "pending");
  const best = thoughts.reduce<(typeof thoughts)[number] | undefined>(
    (top, item) => ((item.chosen ?? -1) > (top?.chosen ?? -1) ? item : top),
    undefined,
  );
  const lastPolicy = state.policy_decisions?.at(-1);
  const checker = running.find((run) => run.agent === "checker");
  const thinkers = running.find((run) => run.agent === "thinkers");
  const topicRun = running.find((run) => run.agent === "topic");
  const lastCorrection = (state.thoughts ?? []).filter((item) => item.kind === "correction").at(-1);
  const nodes: FlowNode[] = [
    { key: "room", label: "Room", detail: `${pipeline?.audio_frames ?? 0} audio frames`, status: audioStatus, icon: Mic2, active: audioStatus === "active" },
    { key: "stt", label: "Gradium STT", detail: `${pipeline?.stt_fragments ?? 0} fragments, ${pipeline?.stt_turns ?? 0} turns`, status: pipeline?.stt_last_event || "waiting", icon: AudioLines, active: state.session_status === "listening" },
    { key: "memory", label: "Shared memory", detail: `${state.transcript.length} turns, ${tasks.length} tasks`, status: "synced", icon: Database, active: running.length > 0 || runningTasks.length > 0 },
    { key: "jev", label: "Jev router", detail: decision ? `${decision.result.route} / ${decision.result.addressee} / ${decision.result.initiative}` : "Waiting for a turn", status: decisionActive ? "deciding" : "idle", icon: BrainCircuit, active: decisionActive },
    { key: "speaker", label: "Speaker", detail: speaker?.summary || "Ready when addressed", status: speaker?.status || (state.voice_mode === "muted" ? "muted" : "idle"), icon: Bot, active: Boolean(speaker) },
    { key: "worker", label: "Workers", detail: worker?.summary || lastTask?.summary || "No mission running", status: runningTasks.length ? `${runningTasks.length} running` : lastTask?.status || "idle", icon: Search, active: Boolean(worker || runningTasks.length) },
    { key: "notes", label: "Notes agent", detail: notes?.summary || `${state.notes_cursor ?? 0}/${state.transcript.length} turns integrated`, status: notes?.status || `v${state.notes_version ?? 0}`, icon: FileText, active: Boolean(notes) },
    { key: "board", label: "Board curator", detail: `${state.cards.length} durable concepts`, status: boardActive ? "curating" : naming ? "naming" : "synced", icon: Lightbulb, active: boardActive },
    { key: "voice", label: "Voice", detail: speech?.text || "Waiting for a useful moment", status: speech?.status || "idle", icon: Volume2, active: Boolean(speech && ["waiting_gap", "authorized", "playing"].includes(speech.status)) },
    { key: "checker", label: "Checker", detail: state.checking_utterance_id ? "Reading a figure" : lastCorrection?.utterance || "No figure to correct", status: checker ? "checking" : "idle", icon: Calculator, active: Boolean(checker) },
    { key: "thinkers", label: "Thinkers", detail: thinkers?.summary || `${thoughts.filter((item) => item.kind === "idea").length} ideas prepared`, status: thinkers ? "thinking" : "idle", icon: Sparkles, active: Boolean(thinkers) },
    { key: "topic", label: "Topic", detail: state.topic || "Not named yet", status: topicRun ? "following" : state.previous_topic ? "changed" : "stable", icon: Compass, active: Boolean(topicRun) },
    { key: "reservoir", label: "Reservoir", detail: best ? `${best.utterance.slice(0, 60)}` : "Nothing prepared", status: `${thoughts.length} ready${best?.chosen != null ? ` · Jev ${best.chosen.toFixed(2)}` : ""}`, icon: Layers, active: thoughts.length > 0 },
    { key: "policy", label: "Policy", detail: lastPolicy?.reason || "Silent by default", status: lastPolicy ? (lastPolicy.speak ? "speaks" : "silent") : "idle", icon: Scale, active: Boolean(lastPolicy?.speak && Date.now() - new Date(lastPolicy.at ?? 0).getTime() < 4000) },
  ];
  const active = new Set<NodeKey>(nodes.filter((node) => node.active).map((node) => node.key));

  return <section className="workflow-shell">
    <header className="workflow-heading">
      <div><BrainCircuit /><span>Agent workflow</span></div>
      <p>{running.length} agents and {runningTasks.length} workers active</p>
    </header>
    <div className="workflow-viewport">
      <div className="workflow-canvas">
        <svg viewBox="0 0 1300 640" aria-hidden="true">
          <defs><filter id="glow"><feGaussianBlur stdDeviation="3" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
          {EDGES.map(([from, to, channel]) => {
            const a = POSITIONS[from];
            const b = POSITIONS[to];
            const lit = active.has(from) && active.has(to);
            return <g key={`${from}-${to}`} className={`flow-edge ${lit ? "active" : ""}`}>
              <path d={`M ${a.x + 75} ${a.y + 48} C ${(a.x + b.x) / 2 + 75} ${a.y + 48}, ${(a.x + b.x) / 2 + 75} ${b.y + 48}, ${b.x + 75} ${b.y + 48}`} />
              <circle r="4" filter="url(#glow)"><animateMotion dur="1.4s" repeatCount="indefinite" path={`M ${a.x + 75} ${a.y + 48} C ${(a.x + b.x) / 2 + 75} ${a.y + 48}, ${(a.x + b.x) / 2 + 75} ${b.y + 48}, ${b.x + 75} ${b.y + 48}`} /></circle>
              <title>{channel}</title>
            </g>;
          })}
        </svg>
        {nodes.map((node) => {
          const Icon = node.icon;
          const position = POSITIONS[node.key];
          return <button className={`flow-node ${node.active ? "active" : ""}`} data-node={node.key} key={node.key} style={{ left: position.x, top: position.y }} onClick={() => setSelected(node.key)}>
            <div className="node-icon"><Icon size={18} /></div>
            <div className="node-copy"><strong>{node.label}</strong><span>{node.detail}</span></div>
            <small>{node.status}</small>
          </button>;
        })}
      </div>
    </div>
    {selected && <NodeInspector node={selected} state={state} close={() => setSelected(null)} />}
  </section>;
}

function NodeInspector({ node, state, close }: { node: NodeKey; state: AtlasState; close: () => void }) {
  const runs = (state.agent_runs ?? []).filter((run) => {
    if (node === "speaker") return run.agent === "speaker";
    if (node === "worker") return run.agent === "worker" || run.agent === "coordinator";
    return run.agent === node;
  }).slice(-8).reverse();
  const title = node === "jev" ? "Jev decision universe" : `${node[0].toUpperCase()}${node.slice(1)} universe`;
  const thoughts = state.thoughts ?? [];
  return <aside className="node-inspector">
    <header><div><span>Inspecting</span><h2>{title}</h2></div><button onClick={close}><X size={18} /></button></header>
    {node === "jev" && <div className="inspector-stack">
      {(state.decisions ?? []).slice(-6).reverse().map((decision) => <article className="inspection-card" key={decision.id}>
        <div><b>{decision.result.route}</b><time>{decision.decided_at ? new Date(decision.decided_at).toLocaleTimeString() : "now"}</time></div>
        <p>{decision.context.new_utterance as string}</p>
        <dl><dt>Addressee</dt><dd>{decision.result.addressee}</dd><dt>Initiative</dt><dd>{decision.result.initiative}</dd><dt>Memory</dt><dd>{decision.result.memory}</dd><dt>Timing</dt><dd>{decision.result.timing}</dd></dl>
        <details><summary>Context sent to Jev</summary><pre>{JSON.stringify(decision.context, null, 2)}</pre></details>
      </article>)}
      {!state.decisions?.length && <p className="muted">No Jev decision yet.</p>}
    </div>}
    {node === "worker" && <div className="inspector-stack">
      {state.tasks.slice().reverse().map((task) => <article className="inspection-card" key={task.id}><div><b>{task.tool}</b><span className={task.status}>{task.phase}</span></div><p>{task.summary}</p>{task.result && <details><summary>Result projection</summary><pre>{JSON.stringify(task.result, null, 2)}</pre></details>}</article>)}
      {!state.tasks.length && <p className="muted">No worker mission yet.</p>}
    </div>}
    {node === "memory" && <div className="memory-inspector"><Stat label="Transcript" value={state.transcript.length} /><Stat label="Cards" value={state.cards.length} /><Stat label="Tasks" value={state.tasks.length} /><Stat label="Notes version" value={state.notes_version ?? 0} /><Stat label="Speech turns" value={state.speeches.length} /><Stat label="Activities" value={state.activities.length} /></div>}
    {node === "room" && <div className="memory-inspector"><Stat label="Audio frames" value={state.pipeline?.audio_frames ?? 0} /><Stat label="Audio bytes" value={state.pipeline?.audio_bytes ?? 0} /><Stat label="Committed turns" value={state.pipeline?.stt_turns ?? 0} /></div>}
    {node === "stt" && <div className="memory-inspector"><Stat label="Messages" value={state.pipeline?.stt_messages ?? 0} /><Stat label="Fragments" value={state.pipeline?.stt_fragments ?? 0} /><Stat label="Final turns" value={state.pipeline?.stt_turns ?? 0} /><Stat label="Silence" value={`${Math.round((state.pipeline?.stt_inactivity_probability ?? 0) * 100)}%`} /></div>}
    {node === "notes" && <div className="memory-inspector"><Stat label="Version" value={state.notes_version ?? 0} /><Stat label="Integrated" value={`${state.notes_cursor ?? 0}/${state.transcript.length}`} /><Stat label="Characters" value={state.notes.length} /></div>}
    {node === "board" && <div className="inspector-stack">{state.activities.filter((item) => item.kind.startsWith("board.")).slice(-10).reverse().map((item) => <article className="inspection-card" key={item.id}><b>{item.kind}</b><p>{item.summary}</p></article>)}</div>}
    {node === "voice" && <div className="inspector-stack">{state.speeches.slice(-8).reverse().map((speech) => <article className="inspection-card" key={speech.id}><div><b>{speech.reason}</b><span className={speech.status}>{speech.status}</span></div><p>{speech.text}</p></article>)}</div>}
    {node === "reservoir" && <div className="inspector-stack">
      {thoughts.filter((item) => item.status === "ready" || item.status === "pending").slice().reverse().map((item) => <article className="inspection-card" key={item.id}>
        <div><b>{item.kind}{item.owed ? " · owed" : ""}</b><span className={item.status}>{item.status}</span></div>
        <p>{item.utterance}</p>
        <dl><dt>Jev pick</dt><dd>{item.chosen?.toFixed(2) ?? "not rated"}</dd><dt>Already said</dt><dd>{item.already_said?.toFixed(2) ?? "-"}</dd>{item.kind === "correction" && <><dt>Right now</dt><dd>{item.fit?.toFixed(2) ?? "-"}</dd></>}</dl>
      </article>)}
      <h3 className="inspector-subtitle">Log</h3>
      {(state.reservoir_log ?? []).slice(-15).reverse().map((event) => <article className="inspection-card" key={event.id}><div><b>{event.event}</b><span>{event.by}</span></div><p>{event.utterance}</p>{event.why && <code>{event.why}</code>}</article>)}
      {!thoughts.length && <p className="muted">Nothing prepared yet.</p>}
    </div>}
    {node === "policy" && <div className="inspector-stack">
      {(state.policy_decisions ?? []).slice(-12).reverse().map((item) => <article className="inspection-card" key={item.id}><div><b>{item.speak ? "speaks" : "silent"}</b><time>{item.at ? new Date(item.at).toLocaleTimeString() : ""}</time></div><p>{item.reason}</p></article>)}
      {!state.policy_decisions?.length && <p className="muted">No decision yet: Atlas is silent by default.</p>}
    </div>}
    {node === "topic" && <div className="inspector-stack">
      <article className="inspection-card"><div><b>Current</b><time>{state.topic_since ? new Date(state.topic_since).toLocaleTimeString() : ""}</time></div><p>{state.topic || "Not named yet"}</p></article>
      {state.previous_topic && <article className="inspection-card"><b>Before</b><p>{state.previous_topic}</p></article>}
      {(state.trace ?? []).filter((item) => item.stage === "topic").slice(-8).reverse().map((item) => <article className="inspection-card" key={item.id}><p>{item.detail}</p></article>)}
    </div>}
    {(node === "checker" || node === "thinkers") && <div className="inspector-stack">
      {thoughts.filter((item) => (node === "checker" ? item.kind === "correction" : item.kind === "idea")).slice(-8).reverse().map((item) => <article className="inspection-card" key={item.id}><div><b>{item.topic || item.kind}</b><span className={item.status}>{item.status}</span></div><p>{item.utterance}</p></article>)}
    </div>}
    {(node === "speaker" || node === "worker" || node === "notes" || node === "checker" || node === "thinkers") && <div className="inspector-stack">{runs.map((run) => <article className="inspection-card" key={run.id}><div><b>{run.summary}</b><span className={run.status}>{run.status}</span></div><p>{run.duration_ms ? `${(run.duration_ms / 1000).toFixed(1)} seconds` : "Running now"}</p>{run.error && <code>{run.error}</code>}</article>)}{!runs.length && <p className="muted">No run recorded.</p>}</div>}
  </aside>;
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return <div><span>{label}</span><b>{value}</b></div>;
}
