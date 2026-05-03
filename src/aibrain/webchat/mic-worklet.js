class AIBrainMicCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options?.processorOptions || {};
    this.outputSampleRate = processorOptions.outputSampleRate || 16000;
    this.frameSampleCount = processorOptions.frameSampleCount || 320;
    this.pending = new Float32Array(this.frameSampleCount);
    this.pendingLength = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (output) {
      output.fill(0);
    }
    if (!input || !input.length) {
      return true;
    }

    const downsampled = downsample(input, sampleRate, this.outputSampleRate);
    this.queueSamples(downsampled);
    return true;
  }

  queueSamples(samples) {
    let offset = 0;
    while (offset < samples.length) {
      const available = this.frameSampleCount - this.pendingLength;
      const count = Math.min(available, samples.length - offset);
      this.pending.set(samples.subarray(offset, offset + count), this.pendingLength);
      this.pendingLength += count;
      offset += count;

      if (this.pendingLength === this.frameSampleCount) {
        const frame = this.pending.slice();
        const pcm = floatToPcm16(frame);
        this.port.postMessage(
          {
            type: "pcm",
            level: levelOf(frame),
            pcm,
          },
          [pcm]
        );
        this.pendingLength = 0;
      }
    }
  }
}

function levelOf(samples) {
  if (!samples.length) return 0;
  let sum = 0;
  for (const sample of samples) sum += sample * sample;
  return Math.min(1, Math.sqrt(sum / samples.length) * 4);
}

function downsample(samples, inputRate, outputRate) {
  if (inputRate === outputRate) return samples;
  const ratio = inputRate / outputRate;
  const newLength = Math.round(samples.length / ratio);
  const result = new Float32Array(newLength);
  for (let index = 0; index < newLength; index += 1) {
    const start = Math.floor(index * ratio);
    const end = Math.min(Math.floor((index + 1) * ratio), samples.length);
    let sum = 0;
    for (let cursor = start; cursor < end; cursor += 1) sum += samples[cursor];
    result[index] = sum / Math.max(1, end - start);
  }
  return result;
}

function floatToPcm16(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(index * 2, sample < 0 ? sample * 32768 : sample * 32767, true);
  }
  return buffer;
}

registerProcessor("aibrain-mic-capture", AIBrainMicCaptureProcessor);
