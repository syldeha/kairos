import type { Bootstrap, ClientMessage } from "./protocol";

export async function bootstrap(): Promise<Bootstrap> {
  const response = await fetch("/v1/bootstrap");
  if (!response.ok) throw new Error(`Bootstrap failed: ${response.status}`);
  return response.json() as Promise<Bootstrap>;
}

export function liveSocket(): WebSocket {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(`${scheme}://${location.host}/v1/live`);
}

export function send(socket: WebSocket, message: ClientMessage): void {
  if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

export async function renameAtlasSession(sessionId: string, title: string): Promise<void> {
  const response = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!response.ok) throw new Error(`Rename failed: ${response.status}`);
}

export async function deleteAtlasSession(sessionId: string): Promise<void> {
  const response = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(`Delete failed: ${response.status}`);
}

export function exportAtlasSession(sessionId: string): void {
  const link = document.createElement("a");
  link.href = `/v1/sessions/${encodeURIComponent(sessionId)}/export`;
  link.download = `${sessionId}.md`;
  link.click();
}
