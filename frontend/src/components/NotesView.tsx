import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { BookOpenText } from "lucide-react";
import type { AtlasState } from "../protocol";

export function NotesView({ state }: { state: AtlasState }) {
  if (!state.notes) {
    return <div className="empty notes-empty"><BookOpenText /><h2>Notes will grow with the room.</h2><p>The notes agent is waiting for a complete idea.</p></div>;
  }
  const readableNotes = state.notes.replace(/\s*\[(utt_[^\]]+)]/g, "");
  return <section className="notes-shell">
    <header className="notes-meta">
      <div><BookOpenText /><span>Living notes</span></div>
      <div><b>v{state.notes_version ?? 0}</b><span>{state.notes_cursor ?? 0}/{state.transcript.length} turns integrated</span></div>
    </header>
    <article className="notes-document">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{readableNotes}</ReactMarkdown>
    </article>
  </section>;
}
