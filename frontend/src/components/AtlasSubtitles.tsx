type Props = {
  enabled: boolean;
  text: string;
  companionName: string;
};

export function AtlasSubtitles({ enabled, text, companionName }: Props) {
  if (!enabled || !text) return null;
  // If text is very long (over 120 chars), display only the most recent chunk/clause smoothly
  let displayText = text.trim();
  if (displayText.length > 120) {
    const segments = displayText.split(/(?<=[.?!,;—\n])\s+/);
    if (segments.length > 1) {
      displayText = segments.slice(-2).join(" ");
    }
  }

  return (
    <div className="atlas-subtitles" role="status" aria-live="polite">
      <span>{companionName}</span>
      <p>{displayText}</p>
    </div>
  );
}
