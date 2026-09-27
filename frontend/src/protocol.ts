import type { components } from "./protocol.gen";

export type AtlasState = components["schemas"]["AtlasState"];
export type AtlasLanguage = Exclude<AtlasState["language"], undefined>;
export type Bootstrap = components["schemas"]["BootstrapResponse"];
export type ClientMessage = components["schemas"]["ProtocolDocument"]["client"];
export type ServerMessage = components["schemas"]["ProtocolDocument"]["server"];
export type SpeechAuthorized = components["schemas"]["SpeechAuthorized"];
export type SessionSummary = components["schemas"]["SessionSummary"];

export const CLIENT_PROTOCOL_VERSION = 11;
