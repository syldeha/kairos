import type { AtlasState } from "../protocol";

type Message = {
  id: string;
  who: string;
  text: string;
  at: number;
  kairos: boolean;
  delay?: number;
  cut?: boolean;
};

/** The conversation as a chat: people and Kairos, with how long Kairos took after the last human line. */
export function ChatView({ state, partial }: { state: AtlasState; partial: string }) {
  const messages = chatMessages(state);
  return (
    <div className="chat" aria-label="Conversation">
      {messages.length === 0 && !partial && <p className="chat-empty">Nothing said yet.</p>}
      {messages.map((message) => (
        <div className={`chat-row ${message.kairos ? "kairos" : "human"}`} key={message.id}>
          <div className="chat-bubble">
            <header>
              <strong>{message.who}</strong>
              <time>{new Date(message.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time>
              {message.delay !== undefined && <span className="chat-delay" title="After the last human line">+{message.delay.toFixed(1)} s</span>}
              {message.cut && <span className="chat-cut">cut off</span>}
            </header>
            <p>{message.text}</p>
          </div>
        </div>
      ))}
      {partial && (
        <div className="chat-row human">
          <div className="chat-bubble partial"><header><strong>…</strong></header><p>{partial}</p></div>
        </div>
      )}
    </div>
  );
}

export function chatMessages(state: AtlasState): Message[] {
  const humans: Message[] = state.transcript.map((line) => ({
    id: line.id ?? "",
    who: line.speaker && line.speaker !== "unknown" ? line.speaker : "Room",
    text: line.text,
    at: Date.parse(line.committed_at ?? ""),
    kairos: false,
  }));
  const said: Message[] = state.speeches
    .filter((speech) => speech.text)
    .map((speech) => ({
      id: speech.id ?? "",
      who: state.identity_name || "Kairos",
      text: speech.text,
      at: Date.parse(speech.created_at ?? ""),
      kairos: true,
      cut: speech.status === "interrupted",
    }));
  const messages = [...humans, ...said].sort((a, b) => a.at - b.at);
  let lastHuman: number | undefined;
  for (const message of messages) {
    if (!message.kairos) lastHuman = message.at;
    else if (lastHuman !== undefined) message.delay = Math.max(0, (message.at - lastHuman) / 1000);
  }
  return messages;
}
