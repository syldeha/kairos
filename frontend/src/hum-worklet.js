class HumWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.attack = null;
    this.sustain = null;
    this.attackPosition = 0;
    this.sustainPosition = 0;
    this.transitionStart = 0;
    this.transitionLength = 1;
    this.fadeSamples = 0;
    this.fadeRemaining = 0;
    this.ready = false;
    this.port.onmessage = (event) => {
      if (event.data.type === "load") {
        this.attack = new Float32Array(event.data.attack);
        this.sustain = new Float32Array(event.data.sustain);
        this.transitionStart = event.data.transitionStart;
        this.transitionLength = Math.max(1, event.data.transitionLength);
        this.attackPosition = 0;
        this.sustainPosition = 0;
        this.ready = true;
      } else if (event.data.type === "stop") {
        this.fadeSamples = Math.max(1, Math.round(sampleRate * event.data.fadeSeconds));
        this.fadeRemaining = this.fadeSamples;
      }
    };
  }

  process(_inputs, outputs) {
    const channel = outputs[0]?.[0];
    if (!channel) return true;
    for (let index = 0; index < channel.length; index += 1) {
      let sample = 0;
      if (this.ready && this.attack && this.sustain && this.sustain.length > 0) {
        sample = this.nextSample();
      }
      if (this.fadeRemaining > 0) {
        sample *= this.fadeRemaining / this.fadeSamples;
        this.fadeRemaining -= 1;
        if (this.fadeRemaining === 0) {
          this.ready = false;
          this.port.postMessage({ type: "ended" });
        }
      }
      channel[index] = sample;
    }
    return true;
  }

  nextSample() {
    const sustainSample = this.sustain[this.sustainPosition];
    this.sustainPosition = (this.sustainPosition + 1) % this.sustain.length;
    if (this.attackPosition >= this.attack.length) return sustainSample;

    const attackSample = this.attack[this.attackPosition];
    if (this.attackPosition < this.transitionStart) {
      this.attackPosition += 1;
      return attackSample;
    }
    const progress = Math.min(
      1,
      (this.attackPosition - this.transitionStart) / this.transitionLength,
    );
    const attackGain = Math.cos(progress * Math.PI * 0.5);
    const sustainGain = Math.sin(progress * Math.PI * 0.5);
    this.attackPosition += 1;
    return attackSample * attackGain + sustainSample * sustainGain;
  }
}

registerProcessor("hum-sustain", HumWorkletProcessor);
