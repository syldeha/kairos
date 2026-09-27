const PROMPT_BAR_KEY = "atlas.promptBarVisible";

export function promptBarVisibleByDefault(storage: Pick<Storage, "getItem"> = localStorage): boolean {
  return storage.getItem(PROMPT_BAR_KEY) !== "false";
}

export function savePromptBarVisibility(
  visible: boolean,
  storage: Pick<Storage, "setItem"> = localStorage,
): void {
  storage.setItem(PROMPT_BAR_KEY, String(visible));
}
