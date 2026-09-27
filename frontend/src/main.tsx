import { StrictMode, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Columns3,
  FileText,
  MessageSquareText,
  Send,
  Workflow,
} from "lucide-react";

import {
  bootstrap,
  deleteAtlasSession,
  exportAtlasSession,
  liveSocket,
  renameAtlasSession,
  send,
} from "./api";
import { AudioBridge, type CaptureMode } from "./audio";
import { AtlasSubtitles } from "./components/AtlasSubtitles";
import { promptBarVisibleByDefault, savePromptBarVisibility } from "./preferences";
import { MonitorView } from "./components/MonitorView";
import { NotesView } from "./components/NotesView";
import { SessionControls } from "./components/SessionControls";
import { WorkspaceNavigation, type WorkspaceView } from "./components/WorkspaceNavigation";
import { WorkflowCanvas } from "./components/WorkflowCanvas";
import { CLIENT_PROTOCOL_VERSION } from "./protocol";
import type { AtlasLanguage, AtlasState, ServerMessage, SessionSummary, SpeechAuthorized } from "./protocol";
import "./style.css";

type AudioStatus = "idle" | "requesting" | "active" | "paused" | "error";

const WAVE_SHAPE = [0.45, 0.7, 1, 0.6, 0.85, 0.5, 0.95, 0.65, 0.8, 0.4, 0.75, 0.55];
const LANGUAGES: Array<{ value: AtlasLanguage; label: string }> = [
  { value: "en", label: "English" },
  { value: "fr", label: "Français" },
  { value: "es", label: "Español" },
  { value: "de", label: "Deutsch" },
  { value: "pt", label: "Português" },
];

function App() {
  const [state, setState] = useState<AtlasState>();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [partial, setPartial] = useState("");
  const [connected, setConnected] = useState(false);
  const [view, setView] = useState<WorkspaceView>("flow");
  const [displayName, setDisplayName] = useState("Atlas");
  const [captureMode, setCaptureMode] = useState<CaptureMode>("mixed");
  const [language, setLanguage] = useState<AtlasLanguage>("en");
  const [manual, setManual] = useState("");
  const [error, setError] = useState("");
  const [audioStatus, setAudioStatus] = useState<AudioStatus>("idle");
  const [audioLevel, setAudioLevel] = useState(0);
  const [subtitlesEnabled, setSubtitlesEnabled] = useState(true);
  const [promptBarVisible, setPromptBarVisible] = useState(promptBarVisibleByDefault);
  const [subtitle, setSubtitle] = useState("");
  const socketRef = useRef<WebSocket | undefined>(undefined);
  const audioRef = useRef<AudioBridge | undefined>(undefined);
  const languageRef = useRef<AtlasLanguage>("en");
  const activeSpeechRef = useRef<string | null>(null);
  const streamStartedRef = useRef(false);
  const subtitlesEnabledRef = useRef(true);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    let active = true;
    const bootstrapRequest = bootstrap();
    void bootstrapRequest.then((payload) => {
      if (!active) return;
      if (payload.protocol_version !== CLIENT_PROTOCOL_VERSION) {
        throw new Error(`Frontend protocol ${CLIENT_PROTOCOL_VERSION} does not match backend protocol ${payload.protocol_version}. Restart the backend and reload.`);
      }
      document.title = payload.display_name;
      setDisplayName(payload.display_name);
      setState(payload.state);
      setSessions(payload.sessions);
      const initialLanguage = payload.state.language ?? "en";
      setLanguage(initialLanguage);
      languageRef.current = initialLanguage;
    }).catch((reason: unknown) => setError(String(reason)));
    const socket = liveSocket();
    socketRef.current = socket;
    const audio = new AudioBridge(
      (chunk) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(chunk);
      },
      (busy) => {
        send(socket, { type: "floor.changed", busy });
        if (busy) audio.stopCue();
      },
      setAudioLevel,
      () => {
        const speechId = activeSpeechRef.current;
        activeSpeechRef.current = null;
        streamStartedRef.current = false;
        if (speechId && socket.readyState === WebSocket.OPEN) {
          send(socket, { type: "playback.interrupted", speech_id: speechId });
        }
      },
    );
    audioRef.current = audio;
    const speechStartedAt = new Map<string, number>();
    const pendingSubtitles = new Map<string, Array<{ text: string; start: number }>>();
    const subtitleTimers = new Set<ReturnType<typeof setTimeout>>();
    const scheduleSubtitle = (speechId: string, text: string, startSeconds: number) => {
      const startedAt = speechStartedAt.get(speechId);
      if (startedAt === undefined) {
        const pending = pendingSubtitles.get(speechId) ?? [];
        pending.push({ text, start: startSeconds });
        pendingSubtitles.set(speechId, pending);
        return;
      }
      const elapsed = performance.now() - startedAt;
      const delay = Math.max(0, startSeconds * 1000 - elapsed);
      const timer = setTimeout(() => {
        subtitleTimers.delete(timer);
        if (activeSpeechRef.current === speechId) setSubtitle(text);
      }, delay);
      subtitleTimers.add(timer);
    };
    socket.onopen = async () => {
      try {
        const payload = await bootstrapRequest;
        if (payload.protocol_version !== CLIENT_PROTOCOL_VERSION) throw new Error("Backend restart required.");
        setConnected(true);
        send(socket, {
          type: "client.hello",
          protocol_version: CLIENT_PROTOCOL_VERSION,
          client_id: crypto.randomUUID(),
          capabilities: { audio_capture: true, audio_playback: true },
        });
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : String(reason));
        socket.close();
      }
    };
    socket.onclose = () => setConnected(false);
    socket.onmessage = (event) => {
      const message = JSON.parse(String(event.data)) as ServerMessage;
      if (message.type === "state.snapshot") {
        setState(message.state);
        setSessions(message.sessions);
        const currentLanguage = message.state.language ?? "en";
        setLanguage(currentLanguage);
        languageRef.current = currentLanguage;
      }
      else if (message.type === "transcript.partial") setPartial(message.text);
      else if (message.type === "speech.stop") {
        audio.stopPlayback();
        setSubtitle("");
        streamStartedRef.current = false;
        if (activeSpeechRef.current === message.speech_id) activeSpeechRef.current = null;
        speechStartedAt.delete(message.speech_id);
        pendingSubtitles.delete(message.speech_id);
      }
      else if (message.type === "speech.authorized") {
        void playSpeech(socket, audio, message, activeSpeechRef, () => setSubtitle(""));
      }
      else if (message.type === "speech.subtitle") {
        if (!message.final && message.text) {
          scheduleSubtitle(message.speech_id, message.text, message.start_s ?? 0);
        }
      }
      else if (message.type === "speech.audio.chunk") {
        if (activeSpeechRef.current === message.speech_id) {
          if (!streamStartedRef.current) {
            streamStartedRef.current = true;
            void audio.beginPcmStream();
            speechStartedAt.set(message.speech_id, performance.now());
            for (const pending of pendingSubtitles.get(message.speech_id) ?? []) {
              scheduleSubtitle(message.speech_id, pending.text, pending.start);
            }
            pendingSubtitles.delete(message.speech_id);
            send(socket, { type: "playback.started", speech_id: message.speech_id });
          }
          audio.pushPcmChunk(message.data_base64, message.sample_rate);
        }
      }
      else if (message.type === "speech.audio.end") {
        if (activeSpeechRef.current === message.speech_id) {
          void audio.endPcmStream().then(() => {
            if (activeSpeechRef.current !== message.speech_id) return;
            activeSpeechRef.current = null;
            streamStartedRef.current = false;
            setSubtitle("");
            speechStartedAt.delete(message.speech_id);
            pendingSubtitles.delete(message.speech_id);
            send(socket, { type: "playback.finished", speech_id: message.speech_id });
          });
        }
      }
      else if (message.type === "presence.cue") void audio.playCue(message.audio ?? undefined);
      else if (message.type === "protocol.error") setError(message.message);
    };
    return () => {
      active = false;
      socket.close();
      audio.stopPlayback();
      for (const timer of subtitleTimers) clearTimeout(timer);
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
      void audio.stopCapture();
    };
  }, []);

  async function start(): Promise<void> {
    setError("");
    try {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Backend connection is not ready.");
      interruptPlayback();
      await startCapture();
      send(socket, {
        type: "session.start",
        language,
        capture_mode: captureMode,
        output_mode: "local_only",
      });
    } catch (reason) {
      setAudioStatus("error");
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function command(value: "pause" | "resume" | "stop"): Promise<void> {
    setError("");
    try {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Backend connection is not ready.");
      if (value === "pause" || value === "stop") interruptPlayback();
      if (value === "resume") await startCapture();
      if (value === "pause" || value === "stop") {
        await audioRef.current?.stopCapture();
        setAudioStatus(value === "pause" ? "paused" : "idle");
      }
      send(socket, { type: "session.command", command: value });
    } catch (reason) {
      setAudioStatus("error");
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function startCapture(): Promise<void> {
    setAudioStatus("requesting");
    const result = await audioRef.current?.start(captureMode);
    setAudioStatus("active");
    if (captureMode === "mixed" && result && !result.system) {
      setError("System audio was not shared. Atlas is listening through the microphone only.");
    }
  }

  async function resumeSession(sessionId: string): Promise<void> {
    setError("");
    try {
      const socket = socketRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Backend connection is not ready.");
      interruptPlayback();
      await startCapture();
      send(socket, { type: "session.open", session_id: sessionId });
      send(socket, { type: "session.command", command: "resume" });
    } catch (reason) {
      setAudioStatus("error");
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function changeLanguage(value: AtlasLanguage): void {
    setLanguage(value);
    languageRef.current = value;
    if (state?.session_id && socketRef.current) {
      send(socketRef.current, { type: "session.language", language: value });
    }
  }

  function interruptPlayback(): void {
    audioRef.current?.stopPlayback();
    setSubtitle("");
    const speechId = activeSpeechRef.current;
    activeSpeechRef.current = null;
    const socket = socketRef.current;
    if (speechId && socket?.readyState === WebSocket.OPEN) {
      send(socket, { type: "playback.interrupted", speech_id: speechId });
    }
  }

  async function renameSession(sessionId: string, currentTitle: string): Promise<void> {
    const title = window.prompt("Session title", currentTitle)?.trim();
    if (!title || title === currentTitle) return;
    try {
      await renameAtlasSession(sessionId, title);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  async function deleteSession(sessionId: string, title: string): Promise<void> {
    if (!window.confirm(`Delete "${title}" permanently?`)) return;
    try {
      await deleteAtlasSession(sessionId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  }

  function inject(): void {
    const text = manual.trim();
    if (!text || !socketRef.current) return;
    send(socketRef.current, { type: "transcript.inject", text });
    setManual("");
  }

  function previewView(nextView: WorkspaceView): void {
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = setTimeout(() => setView(nextView), 140);
  }

  function cancelPreview(): void {
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
  }

  function toggleSubtitles(): void {
    const next = !subtitlesEnabledRef.current;
    subtitlesEnabledRef.current = next;
    setSubtitlesEnabled(next);
  }

  function togglePromptBar(): void {
    setPromptBarVisible((current) => {
      const next = !current;
      savePromptBarVisibility(next);
      return next;
    });
  }

  if (!state) return <main className="loading">{error || `Connecting to ${displayName}...`}</main>;
  const listening = state.session_status === "listening";
  const capturing = audioStatus === "active" || audioStatus === "requesting";
  const navigation: Array<{ view: WorkspaceView; label: string; icon: typeof Workflow }> = [
    { view: "flow", label: "Flow", icon: Workflow },
    { view: "board", label: "Board", icon: Columns3 },
    { view: "notes", label: "Notes", icon: FileText },
    { view: "transcript", label: "Transcript", icon: MessageSquareText },
    { view: "monitor", label: "Monitor", icon: Activity },
  ];

  return (
    <main>
      <header className={listening ? "session-header live" : "session-header"}>
        <div className="brand">
          <img src="/atlas-mark.svg" alt="" />
          <div><strong>{state.session_id ? state.title : displayName}</strong><small>{state.session_id ? `${state.identity_name} / ${state.session_status}` : "Your ambient collaborator"}</small></div>
        </div>
        {state.session_id && <WorkspaceNavigation active={view} items={navigation} onPreview={previewView} onCancelPreview={cancelPreview} onSelect={setView} />}
        <div className="header-actions">
          {state.session_status !== "idle" && state.session_status !== "closed" && <>
            <SessionControls
              language={language}
              languages={LANGUAGES}
              canCurate={state.cards.length > 0 || (state.notes_document?.topics?.length ?? 0) > 0}
              curatorRunning={state.agent_runs.some((run) => run.agent === "coordinator" && run.status === "running")}
              voiceMode={state.voice_mode}
              subtitlesEnabled={subtitlesEnabled}
              promptBarVisible={promptBarVisible}
              listening={listening}
              capturing={capturing}
              onLanguage={changeLanguage}
              onCurate={() => socketRef.current && send(socketRef.current, { type: "board.curate" })}
              onToggleVoice={() => socketRef.current && send(socketRef.current, { type: "voice.mode", mode: state.voice_mode === "muted" ? "active" : "muted" })}
              onToggleSubtitles={toggleSubtitles}
              onTogglePromptBar={togglePromptBar}
              onRename={() => state.session_id && void renameSession(state.session_id, state.title)}
              onExport={() => state.session_id && exportAtlasSession(state.session_id)}
              onPauseResume={() => void command(listening && capturing ? "pause" : "resume")}
              onEnd={() => void command("stop")}
            />
          </>}
          <div className={`connection ${connected ? "online" : ""}`}><i />{connected ? "live" : "offline"}</div>
        </div>
      </header>

      {state.session_status === "idle" || state.session_status === "closed" ? (
        <section className="launch">
           <div className="launch-mark"><img src="/atlas-mark.svg" alt="" /></div>
           <p className="eyebrow">ATLAS HOME</p>
           <h1>Hold the whole room.<br />Move at the right moment.</h1>
           <p>Atlas listens beside any conversation, keeps the shared picture coherent, and acts without joining the call.</p>
          <div className="launch-controls">
            <label>Language<select value={language} onChange={(event) => changeLanguage(event.target.value as AtlasLanguage)}>
              {LANGUAGES.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
            </select></label>
            <label>Audio source<select value={captureMode} onChange={(event) => setCaptureMode(event.target.value as CaptureMode)}>
              <option value="microphone">Microphone</option>
              <option value="mixed">Microphone + system audio</option>
              <option value="system">System audio</option>
            </select></label>
           <button className="start" onClick={() => void start()}>New session</button>
          </div>
          {error && <p className="error">{error}</p>}
          {sessions.length > 0 && <div className="session-library">
            <div className="session-library-heading"><span>Previous sessions</span><b>{sessions.length}</b></div>
            <div className="session-list">
              {sessions.map((session) => <article className="session-item" key={session.session_id}>
                <button className="session-resume" onClick={() => void resumeSession(session.session_id)}>
                  <span><strong>{session.title}</strong><small>{session.preview || "No transcript yet"}</small></span>
                  <span className="session-meta"><small>{session.utterance_count} turns</small><time>{new Date(session.updated_at).toLocaleDateString()}</time></span>
                </button>
                <div className="session-tools">
                  <button onClick={() => void renameSession(session.session_id, session.title)}>Rename</button>
                   <button onClick={() => exportAtlasSession(session.session_id)}>Export</button>
                  <button className="danger" onClick={() => void deleteSession(session.session_id, session.title)}>Delete</button>
                </div>
              </article>)}
            </div>
          </div>}
        </section>
      ) : (
        <>
          <section className="health-row compact">
            {Object.entries(state.health).map(([name, health]) => (
              <span className={`health ${health.status}`} key={name}><i />{name}<b>{health.status}</b></span>
            ))}
          </section>
          <section className="view-stage">
              {view === "flow" && <WorkflowCanvas state={state} audioStatus={audioStatus} />}
              {view === "board" && <Board state={state} />}
              {view === "notes" && <NotesView state={state} />}
              {view === "transcript" && <Transcript state={state} partial={partial} />}
              {view === "monitor" && <MonitorView state={state} />}
          </section>
          {promptBarVisible && <section className="manual command-bar">
            <input value={manual} onChange={(event) => setManual(event.target.value)} onKeyDown={(event) => event.key === "Enter" && inject()} placeholder="Inject a sentence when testing without audio" />
            <button onClick={inject} title="Send"><Send size={17} /></button>
          </section>}
          {error && <p className="error global-error">{error}</p>}
           <ActivityStrip state={state} partial={partial} audioStatus={audioStatus} audioLevel={audioLevel} />
           <AtlasSubtitles enabled={subtitlesEnabled} text={subtitle} companionName={state.identity_name} />
        </>
      )}
    </main>
  );
}

async function playSpeech(
  socket: WebSocket,
  audio: AudioBridge,
  message: SpeechAuthorized,
  activeSpeech: { current: string | null },
  onFinished: () => void,
): Promise<void> {
    activeSpeech.current = message.speech_id;
  try {
    if (message.audio) {
      send(socket, { type: "playback.started", speech_id: message.speech_id });
      await audio.playAudio(message.audio.data_base64, message.audio.format, message.audio.sample_rate);
      if (activeSpeech.current === message.speech_id) {
        activeSpeech.current = null;
        onFinished();
        send(socket, { type: "playback.finished", speech_id: message.speech_id });
      }
    }
  } catch {
    activeSpeech.current = null;
    onFinished();
    send(socket, { type: "playback.interrupted", speech_id: message.speech_id });
  }
}

function Board({ state }: { state: AtlasState }) {
  if (!state.cards.length) return <div className="empty"><span>01</span><h2>The board grows with the conversation.</h2><p>Ideas, decisions and sourced findings will appear here.</p></div>;
  return <div className="board">{state.cards.slice().reverse().map((card, index) => <article className={`card c${index % 4}`} key={card.id}><small>{card.kind}</small><h2>{card.title}</h2><p>{card.body}</p></article>)}</div>;
}

function Transcript({ state, partial }: { state: AtlasState; partial: string }) {
  return <div className="transcript">{state.transcript.slice().reverse().map((item) => <div className="line" key={item.id}><time>{item.committed_at ? new Date(item.committed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "now"}</time><p>{item.text}</p></div>)}{partial && <div className="line partial"><time>live</time><p>{partial}</p></div>}</div>;
}

function ActivityStrip({ state, partial, audioStatus, audioLevel }: { state: AtlasState; partial: string; audioStatus: AudioStatus; audioLevel: number }) {
  const latest = state.activities.at(-1);
  return <footer>
    <div className="audio-monitor" aria-label={`Microphone ${audioStatus}`}>
      <span>mic {audioStatus}</span>
      <div className="wave" aria-hidden="true">
        {WAVE_SHAPE.map((shape, index) => <i key={index} style={{ height: `${4 + audioLevel * shape * 24}px` }} />)}
      </div>
    </div>
    <p>{partial || latest?.summary || "Waiting for the room."}</p>
  </footer>;
}

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
