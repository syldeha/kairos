import { beforeEach, describe, expect, it, vi } from "vitest";

import { AudioBridge } from "./audio";

class FakeNode {
  gain = { value: 1 };
  onaudioprocess: ((event: AudioProcessingEvent) => void) | null = null;
  connect() { return this; }
  disconnect() {}
}

class FakeContext {
  static processors: FakeNode[] = [];
  state = "running";
  sampleRate = 48000;
  destination = new FakeNode();
  createGain() { return new FakeNode(); }
  createMediaStreamSource() { return new FakeNode(); }
  createScriptProcessor() {
    const node = new FakeNode();
    FakeContext.processors.push(node);
    return node;
  }
  async resume() {}
  async close() { this.state = "closed"; }
}

function stream(audioTracks: MediaStreamTrack[], videoTracks: MediaStreamTrack[] = []): MediaStream {
  return {
    getAudioTracks: () => audioTracks,
    getVideoTracks: () => videoTracks,
    getTracks: () => [...audioTracks, ...videoTracks],
  } as unknown as MediaStream;
}

describe("AudioBridge capture", () => {
  const microphoneTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
  const systemTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
  const videoTrack = { stop: vi.fn() } as unknown as MediaStreamTrack;
  const getUserMedia = vi.fn();
  const getDisplayMedia = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    FakeContext.processors = [];
    vi.stubGlobal("AudioContext", FakeContext);
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia, getDisplayMedia },
    });
    getUserMedia.mockResolvedValue(stream([microphoneTrack]));
    getDisplayMedia.mockResolvedValue(stream([systemTrack], [videoTrack]));
  });

  it("captures microphone and shared system audio in mixed mode", async () => {
    const bridge = new AudioBridge(vi.fn(), vi.fn(), vi.fn(), vi.fn());
    const result = await bridge.start("mixed");
    expect(result).toEqual({ microphone: true, system: true });
    expect(getUserMedia).toHaveBeenCalledOnce();
    expect(getDisplayMedia).toHaveBeenCalledOnce();
    expect(videoTrack.stop).toHaveBeenCalledOnce();
    await bridge.stopCapture();
    expect(microphoneTrack.stop).toHaveBeenCalled();
    expect(systemTrack.stop).toHaveBeenCalled();
  });

  it("keeps microphone capture when a shared surface has no audio", async () => {
    getDisplayMedia.mockResolvedValue(stream([], [videoTrack]));
    const bridge = new AudioBridge(vi.fn(), vi.fn(), vi.fn(), vi.fn());
    await expect(bridge.start("mixed")).resolves.toEqual({ microphone: true, system: false });
    await bridge.stopCapture();
  });

  it("rejects system-only capture when the selected surface has no audio", async () => {
    getDisplayMedia.mockResolvedValue(stream([], [videoTrack]));
    const bridge = new AudioBridge(vi.fn(), vi.fn(), vi.fn(), vi.fn());
    await expect(bridge.start("system")).rejects.toThrow("No audio track was shared");
  });

  it("uses microphone energy for barge-in while mixed system audio remains STT-only", async () => {
    const onFloor = vi.fn();
    const bridge = new AudioBridge(vi.fn(), onFloor, vi.fn(), vi.fn());
    await bridge.start("mixed");
    const [mixProcessor, floorProcessor] = FakeContext.processors;
    const loud = new Float32Array(2048).fill(0.2);
    const event = { inputBuffer: { getChannelData: () => loud } } as unknown as AudioProcessingEvent;

    mixProcessor.onaudioprocess?.(event);
    expect(onFloor).not.toHaveBeenCalled();
    floorProcessor.onaudioprocess?.(event);
    floorProcessor.onaudioprocess?.(event);
    expect(onFloor).toHaveBeenCalledWith(true);
    await bridge.stopCapture();
  });

  it("does not treat mixed output energy as barge-in during Atlas playback", async () => {
    const onFloor = vi.fn();
    const onBargeIn = vi.fn();
    const bridge = new AudioBridge(vi.fn(), onFloor, vi.fn(), onBargeIn);
    await bridge.start("mixed");
    Object.defineProperty(bridge, "playbackActive", { value: true, writable: true });
    const [mixProcessor, floorProcessor] = FakeContext.processors;
    const loud = new Float32Array(2048).fill(0.2);
    const event = { inputBuffer: { getChannelData: () => loud } } as unknown as AudioProcessingEvent;

    for (let index = 0; index < 4; index += 1) mixProcessor.onaudioprocess?.(event);
    expect(onBargeIn).not.toHaveBeenCalled();
    for (let index = 0; index < 6; index += 1) floorProcessor.onaudioprocess?.(event);
    expect(onBargeIn).toHaveBeenCalledOnce();
    await bridge.stopCapture();
  });

  it("does not interrupt playback for moderate microphone leakage", async () => {
    const onBargeIn = vi.fn();
    const bridge = new AudioBridge(vi.fn(), vi.fn(), vi.fn(), onBargeIn);
    await bridge.start("mixed");
    Object.defineProperty(bridge, "playbackActive", { value: true, writable: true });
    const floorProcessor = FakeContext.processors[1];
    const leakage = new Float32Array(2048).fill(0.055);
    const event = { inputBuffer: { getChannelData: () => leakage } } as unknown as AudioProcessingEvent;

    for (let index = 0; index < 10; index += 1) floorProcessor.onaudioprocess?.(event);
    expect(onBargeIn).not.toHaveBeenCalled();
    await bridge.stopCapture();
  });
});

describe("Kairos barge-in", () => {
  it("stops Kairos after two loud 80 ms microphone frames, not after one", () => {
    const onBargeIn = vi.fn();
    const bridge = new AudioBridge(vi.fn(), vi.fn(), vi.fn(), onBargeIn);
    Object.defineProperty(bridge, "playbackActive", { value: true, writable: true });
    const frame = (value: number) => new Int16Array(1920).fill(value);
    const process = (bridge as unknown as { processMicFrame: (pcm: Int16Array) => void }).processMicFrame.bind(bridge);
    process(frame(3000));
    process(frame(200));
    process(frame(3000));
    expect(onBargeIn).not.toHaveBeenCalled();
    process(frame(3000));
    expect(onBargeIn).toHaveBeenCalledOnce();
  });
});
