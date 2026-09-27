import {
  Captions,
  CaptionsOff,
  Download,
  MessageSquareText,
  Pause,
  Pencil,
  Play,
  Sparkles,
  Square,
  Volume2,
  VolumeX,
} from "lucide-react";

import type { AtlasLanguage } from "../protocol";

type Props = {
  language: AtlasLanguage;
  languages: Array<{ value: AtlasLanguage; label: string }>;
  canCurate: boolean;
  curatorRunning: boolean;
  voiceMode: "active" | "muted";
  subtitlesEnabled: boolean;
  promptBarVisible: boolean;
  listening: boolean;
  capturing: boolean;
  onLanguage: (language: AtlasLanguage) => void;
  onCurate: () => void;
  onToggleVoice: () => void;
  onToggleSubtitles: () => void;
  onTogglePromptBar: () => void;
  onRename: () => void;
  onExport: () => void;
  onPauseResume: () => void;
  onEnd: () => void;
};

export function SessionControls(props: Props) {
  const curateDisabled = !props.canCurate || props.curatorRunning;
  return (
    <>
      <select
        className="header-language"
        value={props.language}
        onChange={(event) => props.onLanguage(event.target.value as AtlasLanguage)}
        title="Session language"
        aria-label="Session language"
      >
        {props.languages.map((item) => (
          <option value={item.value} key={item.value}>{item.label}</option>
        ))}
      </select>
      <button
        className="icon-button"
        disabled={curateDisabled}
        onClick={props.onCurate}
        title={!props.canCurate ? "Nothing to curate yet" : "Consolidate the board"}
        aria-label="Consolidate board"
      >
        <Sparkles size={17} />
      </button>
      <button
        className={`icon-button ${props.voiceMode === "muted" ? "danger" : ""}`}
        onClick={props.onToggleVoice}
        title={props.voiceMode === "muted" ? "Enable Atlas voice" : "Mute Atlas voice"}
        aria-pressed={props.voiceMode === "active"}
      >
        {props.voiceMode === "muted" ? <VolumeX size={17} /> : <Volume2 size={17} />}
      </button>
      <button
        className={`icon-button ${props.subtitlesEnabled ? "active" : ""}`}
        onClick={props.onToggleSubtitles}
        title={props.subtitlesEnabled ? "Hide Atlas subtitles" : "Show Atlas subtitles"}
        aria-pressed={props.subtitlesEnabled}
      >
        {props.subtitlesEnabled ? <Captions size={17} /> : <CaptionsOff size={17} />}
      </button>
      <button
        className={`icon-button ${props.promptBarVisible ? "active" : ""}`}
        onClick={props.onTogglePromptBar}
        title={props.promptBarVisible ? "Hide prompt bar" : "Show prompt bar"}
        aria-pressed={props.promptBarVisible}
      >
        <MessageSquareText size={17} />
      </button>
      <button className="icon-button" onClick={props.onRename} title="Rename session">
        <Pencil size={17} />
      </button>
      <button className="icon-button" onClick={props.onExport} title="Export session">
        <Download size={17} />
      </button>
      <button
        className="icon-button primary"
        onClick={props.onPauseResume}
        title={props.listening && props.capturing ? "Pause listening" : "Resume listening"}
      >
        {props.listening && props.capturing ? <Pause size={17} /> : <Play size={17} />}
      </button>
      <button className="icon-button danger" onClick={props.onEnd} title="End session">
        <Square size={16} />
      </button>
    </>
  );
}
