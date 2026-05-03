// Browser-side example for the /brain WebSocket TTS event contract.
// It queues each TTS segment and plays segments sequentially, so audio never
// overlaps even when text and audio events arrive while the model is still
// generating.

export class BrainAudioPlaylist {
  constructor(audioContext = new AudioContext()) {
    this.audioContext = audioContext;
    this.pendingSegments = new Map();
    this.playQueue = [];
    this.playing = false;
  }

  handleEvent(event) {
    if (event.type === 'tts.playlist.start') {
      this.pendingSegments.clear();
      this.playQueue.length = 0;
      return;
    }

    if (event.type === 'tts.audio') {
      const segmentId = event.segment_id ?? `${event.playback_id}:${event.segment_index}`;
      const segment = this.pendingSegments.get(segmentId) ?? {
        id: segmentId,
        index: event.segment_index,
        sampleRate: event.sample_rate,
        encoding: event.encoding,
        chunks: [],
      };
      segment.chunks.push(base64ToBytes(event.audio));
      this.pendingSegments.set(segmentId, segment);
      return;
    }

    if (event.type === 'tts.done') {
      const segmentId = event.segment_id ?? `${event.playback_id}:${event.segment_index}`;
      const segment = this.pendingSegments.get(segmentId);
      if (!segment) return;
      this.pendingSegments.delete(segmentId);
      this.enqueue(segment);
    }
  }

  enqueue(segment) {
    this.playQueue.push(segment);
    this.playQueue.sort((left, right) => left.index - right.index);
    void this.pump();
  }

  async pump() {
    if (this.playing) return;
    this.playing = true;
    try {
      while (this.playQueue.length) {
        const segment = this.playQueue.shift();
        await this.playPcmSegment(segment);
      }
    } finally {
      this.playing = false;
    }
  }

  async playPcmSegment(segment) {
    if (segment.encoding !== 'pcm_s16le') {
      throw new Error(`Unsupported encoding: ${segment.encoding}`);
    }

    const pcm = concatBytes(segment.chunks);
    const samples = new Float32Array(pcm.length / 2);
    const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let i = 0; i < samples.length; i += 1) {
      samples[i] = Math.max(-1, Math.min(1, view.getInt16(i * 2, true) / 32768));
    }

    const buffer = this.audioContext.createBuffer(1, samples.length, segment.sampleRate);
    buffer.copyToChannel(samples, 0);

    await this.audioContext.resume();
    await new Promise((resolve) => {
      const source = this.audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(this.audioContext.destination);
      source.onended = resolve;
      source.start();
    });
  }
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function concatBytes(chunks) {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
