"""The Atlas interface's data contract, copied from Atlas (github.com/KpihX/atlas, backend/src/atlas/core/
models.py, protocol 11) so that Kairos's state reaches the Atlas frontend in exactly the shape it reads.

Do not edit by hand: update it from Atlas when the frontend's protocol changes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

AtlasLanguage = Literal["en", "fr", "es", "de", "pt"]
ATLAS_LANGUAGES: tuple[AtlasLanguage, ...] = ("en", "fr", "es", "de", "pt")


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProcessStatus(StrEnum):
    BOOTING = "booting"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"


class SessionStatus(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    LISTENING = "listening"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    CLOSED = "closed"
    FAILED = "failed"


class Health(DomainModel):
    status: Literal["ok", "standby", "degraded", "down", "unconfigured"]
    detail: str = ""


class Utterance(DomainModel):
    id: str = Field(default_factory=lambda: new_id("utt"))
    text: str
    source: str = "audio"
    speaker: str = "unknown"
    language: str = "auto"
    confidence: float | None = None
    committed_at: str = Field(default_factory=now_iso)


class Card(DomainModel):
    id: str = Field(default_factory=lambda: new_id("card"))
    kind: Literal["idea", "question", "decision", "suggestion", "finding"] = "finding"
    title: str
    body: str
    source_ids: list[str] = Field(default_factory=list)
    concept_key: str = ""
    updated_at: str = Field(default_factory=now_iso)


class Task(DomainModel):
    id: str = Field(default_factory=lambda: new_id("task"))
    mission_key: str = ""
    tool: str
    summary: str
    status: Literal["queued", "running", "done", "failed", "canceled", "stale"] = "queued"
    phase: Literal[
        "queued", "planning", "executing", "synthesizing", "integrating", "reporting", "complete"
    ] = "queued"
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str = Field(default_factory=now_iso)
    completed_at: str | None = None

    @model_validator(mode="after")
    def align_phase_with_terminal_status(self) -> Task:
        if self.status == "running" and self.phase == "queued":
            self.phase = "executing"
        elif self.status in {"done", "failed", "canceled", "stale"}:
            self.phase = "complete"
        return self


class Speech(DomainModel):
    id: str = Field(default_factory=lambda: new_id("say"))
    text: str
    reason: Literal[
        "direct_address", "requested_result", "critical_finding", "reservoir_thought", "correction"
    ]
    status: Literal[
        "proposed",
        "waiting_gap",
        "authorized",
        "playing",
        "finished",
        "interrupted",
        "suppressed",
        "expired",
        "canceled",
        "failed",
    ] = "proposed"
    source_ids: list[str] = Field(default_factory=list)
    room_epoch: int = 0
    error: str | None = None
    created_at: str = Field(default_factory=now_iso)


class Activity(DomainModel):
    id: str = Field(default_factory=lambda: new_id("act"))
    kind: str
    summary: str
    created_at: str = Field(default_factory=now_iso)


class AgentRun(DomainModel):
    id: str = Field(default_factory=lambda: new_id("agent"))
    agent: Literal[
        "speaker",
        "worker",
        "coordinator",
        "notes",
        "naming",
        "thinkers",
        "checker",
        "topic",
        "rater",
        "anticipation",
    ]
    summary: str
    status: Literal["running", "done", "failed", "canceled"] = "running"
    error: str | None = None
    duration_ms: int | None = None
    started_at: str = Field(default_factory=now_iso)
    completed_at: str | None = None


class PipelineMetrics(DomainModel):
    audio_frames: int = 0
    audio_bytes: int = 0
    last_audio_at: str | None = None
    stt_messages: int = 0
    stt_fragments: int = 0
    stt_turns: int = 0
    stt_last_event: str = ""
    stt_last_fragment: str = ""
    stt_inactivity_probability: float = 0
    decisions: int = 0
    agent_runs: int = 0
    note_updates: int = 0


class NotesDocument(DomainModel):
    synthesis: list[str] = Field(default_factory=list)
    participants: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    ideas: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    current_work: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class Decision(DomainModel):
    route: Literal["ignore", "capture", "investigate", "respond", "act", "control"]
    addressee: Literal["atlas", "another_participant", "room", "uncertain"] = "uncertain"
    memory: Literal["ignore", "capture"] = "ignore"
    initiative: Literal["none", "assigned", "proactive"] = "none"
    speech_depth: Literal["silent", "brief", "normal", "deep"] = "silent"
    timing: Literal["silent", "next_gap", "later"] = "silent"
    rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_scores(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        source = cast(dict[object, object], value)
        migrated: dict[str, object] = {key: item for key, item in source.items() if isinstance(key, str)}
        legacy_fields = {
            "addressed_probability",
            "salience",
            "speech_value",
            "memory_value",
            "mission_probability",
            "intervention_urgency",
        }
        if not legacy_fields.intersection(migrated):
            return migrated
        route = migrated.get("route", "capture")
        migrated.setdefault("addressee", "uncertain")
        migrated.setdefault("memory", "ignore" if route == "ignore" else "capture")
        migrated.setdefault("initiative", "assigned" if route in {"investigate", "act"} else "none")
        for field in legacy_fields:
            migrated.pop(field, None)
        return migrated


class TurnSignals(DomainModel):
    """What Jev read in a turn beyond the routing decision: topic change and its choice in the reservoir."""

    new_topic: float = 0.0
    personal: float = 0.0
    addressed_atlas: float = 0.0
    pick: dict[str, float] = Field(default_factory=dict)
    pick_none: float = 1.0
    said: dict[str, float] = Field(default_factory=dict)
    fit: dict[str, float] = Field(default_factory=dict)


class Thought(DomainModel):
    """Something Atlas could say, prepared before it is time to say it."""

    id: str = Field(default_factory=lambda: new_id("thought"))
    kind: Literal["idea", "answer", "correction", "finding", "question"] = "idea"
    topic: str = ""
    utterance: str
    importance: float = Field(default=3.0, ge=1, le=5)
    status: Literal["ready", "pending", "spoken", "stale"] = "ready"
    owed: bool = False
    chosen: float | None = None
    fit: float | None = None
    already_said: float | None = None
    rated_utterance_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    task_id: str | None = None
    created_ts: float = 0.0
    created_at: str = Field(default_factory=now_iso)
    note: str = ""


class ReservoirEvent(DomainModel):
    id: str = Field(default_factory=lambda: new_id("rev"))
    thought_id: str
    event: str
    by: str
    why: str = ""
    utterance: str = ""
    at: str = Field(default_factory=now_iso)


class TraceEvent(DomainModel):
    """One step in the life of a room turn, for the Monitor: what happened, when, and why."""

    id: str = Field(default_factory=lambda: new_id("trace"))
    utterance_id: str
    stage: Literal[
        "heard",
        "partial",
        "jev",
        "checker",
        "thinkers",
        "reservoir",
        "rater",
        "topic",
        "policy",
        "speaker",
        "worker",
        "voice",
    ]
    ms: int = 0
    detail: str
    at: str = Field(default_factory=now_iso)


class PolicyDecision(DomainModel):
    id: str = Field(default_factory=lambda: new_id("policy"))
    utterance_id: str | None = None
    speak: bool
    reason: str
    thought_id: str | None = None
    at: str = Field(default_factory=now_iso)


class DecisionRecord(DomainModel):
    id: str = Field(default_factory=lambda: new_id("decision"))
    utterance_id: str
    context: dict[str, Any]
    result: Decision
    decided_at: str = Field(default_factory=now_iso)


class AtlasState(DomainModel):
    project_id: str
    protocol_version: int
    process_status: ProcessStatus = ProcessStatus.BOOTING
    session_status: SessionStatus = SessionStatus.IDLE
    session_id: str | None = None
    title: str = "Nouvelle session"
    identity_name: str = "Atlas"
    language: AtlasLanguage = "en"
    voice_mode: Literal["active", "muted"] = "active"
    capture_mode: Literal["microphone", "system", "mixed"] = "mixed"
    output_mode: Literal["local_only", "room_speaker", "meeting_injected"] = "local_only"
    floor_busy: bool = False
    room_epoch: int = 0
    partial: str = ""
    notes: str = ""
    notes_document: NotesDocument = Field(default_factory=NotesDocument)
    notes_version: int = 0
    notes_cursor: int = 0
    title_version: int = 0
    title_cursor: int = 0
    working: str = ""
    transcript: list[Utterance] = []
    cards: list[Card] = []
    tasks: list[Task] = []
    speeches: list[Speech] = []
    activities: list[Activity] = []
    agent_runs: list[AgentRun] = []
    decisions: list[DecisionRecord] = []
    thoughts: list[Thought] = []
    reservoir_log: list[ReservoirEvent] = []
    trace: list[TraceEvent] = []
    policy_decisions: list[PolicyDecision] = []
    topic: str = ""
    previous_topic: str = ""
    topic_since: str | None = None
    checking_utterance_id: str | None = None
    health: dict[str, Health] = {}
    pipeline: PipelineMetrics = Field(default_factory=PipelineMetrics)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class SessionSummary(DomainModel):
    session_id: str
    title: str
    status: SessionStatus
    preview: str = ""
    utterance_count: int = 0
    created_at: str
    updated_at: str


class ToolRequest(DomainModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class BoardOperation(DomainModel):
    action: Literal["create", "update", "merge", "delete"]
    card_id: str | None = None
    merge_ids: list[str] = Field(default_factory=list)
    kind: Literal["idea", "question", "decision", "suggestion", "finding"] = "idea"
    concept_key: str = Field(default="", max_length=120)
    title: str = Field(default="", max_length=120)
    body: str = Field(default="", max_length=600)
    reason: str = ""


class BoardMergeEvidence(DomainModel):
    same_resolution: bool
    mutually_substitutable: bool
    loses_independent_value: bool
    rationale: str = Field(min_length=1, max_length=400)

    @property
    def proves_equivalence(self) -> bool:
        return self.same_resolution and self.mutually_substitutable and not self.loses_independent_value


class BoardConcept(DomainModel):
    kind: Literal["idea", "question", "decision", "suggestion", "finding"]
    concept_key: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=600)
    source_card_ids: list[str] = Field(default_factory=list)
    merge_evidence: BoardMergeEvidence | None = None


class BoardRetirement(DomainModel):
    card_id: str
    reason: str = Field(min_length=1, max_length=400)


class BoardProposal(DomainModel):
    desired_cards: list[BoardConcept] = Field(default_factory=lambda: list[BoardConcept]())
    retirements: list[BoardRetirement] = Field(default_factory=lambda: list[BoardRetirement]())


class CoordinatorResult(DomainModel):
    working: str = ""
    board_ops: list[BoardOperation] = Field(default_factory=lambda: list[BoardOperation]())
    speech: str = ""
    control: Literal["none", "mute", "unmute", "end_session"] = "none"
    tool: ToolRequest | None = None


class SpeechPlan(DomainModel):
    speak: bool = False
    spoken_core: str = ""
    visual_detail: str = ""
    control: Literal["none", "mute", "unmute", "end_session"] = "none"


class AudioResult(DomainModel):
    data_base64: str
    format: str
    sample_rate: int


class LLMResult(DomainModel):
    content: str
    model: str
    provider: str
    raw: dict[str, Any] = Field(default_factory=dict)
