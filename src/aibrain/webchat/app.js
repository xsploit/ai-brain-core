const state = {
  chatSocket: null,
  voiceSocket: null,
  mic: null,
  sttRecorder: null,
  assistantMessage: null,
  playlist: null,
  chatUsesTts: false,
  responseDone: false,
  ttsPlaylistDone: false,
  pendingAssistantText: "",
  assistantFlushScheduled: false,
  textDeltaCount: 0,
  ttsAudioCount: 0,
  ttsAudioBytes: 0,
  turnStartedAt: 0,
  firstTextAt: 0,
  firstAudioAt: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

class AudioPlaylist {
  constructor() {
    this.audioContext = null;
    this.pendingSegments = new Map();
    this.queue = [];
    this.playing = false;
    this.sources = new Set();
    this.nextPlayTime = 0;
    this.scheduleChain = Promise.resolve();
    this.generation = 0;
  }

  context() {
    if (!this.audioContext) {
      this.audioContext = new AudioContext();
    }
    return this.audioContext;
  }

  async unlock() {
    try {
      await this.context().resume();
    } catch (error) {
      logEvent("audio.unlock.error", { message: error.message });
    }
  }

  reset() {
    this.generation += 1;
    this.pendingSegments.clear();
    this.queue.length = 0;
    this.playing = false;
    this.nextPlayTime = 0;
    this.scheduleChain = Promise.resolve();
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        // Source may have already ended.
      }
    }
    this.sources.clear();
  }

  handle(event) {
    if (event.type === "tts.playlist.start") {
      this.reset();
      return;
    }
    if (event.type === "tts.audio") {
      const generation = this.generation;
      this.scheduleChain = this.scheduleChain
        .then(() => yieldToBrowser())
        .then(() => {
          if (generation !== this.generation) return null;
          return this.scheduleAudioEvent(event, generation);
        })
        .catch((error) => logEvent("audio.error", { message: error.message }));
      return;
    }
    if (event.type === "tts.done") {
      return;
    }
  }

  async scheduleAudioEvent(event, generation) {
    if (event.encoding !== "pcm_s16le") {
      logEvent("audio.error", { message: `Unsupported encoding: ${event.encoding}` });
      return;
    }
    if (!event.audio) return;
    const audioContext = this.context();
    await audioContext.resume();
    if (generation !== this.generation) return;
    const buffer = this.audioBufferFromPcm(base64ToBytes(event.audio), event.sample_rate || 22050);
    if (!buffer.length) return;
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    source.onended = () => this.sources.delete(source);
    const startAt = Math.max(audioContext.currentTime + 0.02, this.nextPlayTime || 0);
    this.nextPlayTime = startAt + buffer.duration;
    this.sources.add(source);
    source.start(startAt);
  }

  audioBufferFromPcm(pcm, sampleRate) {
    const audioContext = this.context();
    const samples = new Float32Array(Math.floor(pcm.length / 2));
    const view = new DataView(pcm.buffer, pcm.byteOffset, pcm.byteLength);
    for (let index = 0; index < samples.length; index += 1) {
      samples[index] = Math.max(-1, Math.min(1, view.getInt16(index * 2, true) / 32768));
    }
    const buffer = audioContext.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    return buffer;
  }

  handleBuffered(event) {
    if (event.type === "tts.audio") {
      const segmentId = event.segment_id || `${event.playback_id}:${event.segment_index}`;
      const segment = this.pendingSegments.get(segmentId) || {
        id: segmentId,
        index: event.segment_index || 0,
        sampleRate: event.sample_rate || 22050,
        encoding: event.encoding || "pcm_s16le",
        chunks: [],
      };
      segment.chunks.push(base64ToBytes(event.audio));
      this.pendingSegments.set(segmentId, segment);
      return;
    }
    if (event.type === "tts.done") {
      const segmentId = event.segment_id || `${event.playback_id}:${event.segment_index}`;
      const segment = this.pendingSegments.get(segmentId);
      if (!segment) return;
      this.pendingSegments.delete(segmentId);
      this.queue.push(segment);
      this.queue.sort((left, right) => left.index - right.index);
      void this.pump();
    }
  }

  async pump() {
    if (this.playing) return;
    this.playing = true;
    try {
      while (this.queue.length) {
        await this.playSegment(this.queue.shift());
      }
    } finally {
      this.playing = false;
    }
  }

  async playSegment(segment) {
    if (segment.encoding !== "pcm_s16le") {
      logEvent("audio.error", { message: `Unsupported encoding: ${segment.encoding}` });
      return;
    }
    const audioContext = this.context();
    const buffer = this.audioBufferFromPcm(concatBytes(segment.chunks), segment.sampleRate);
    await audioContext.resume();
    await new Promise((resolve) => {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);
      source.onended = () => {
        this.sources.delete(source);
        resolve();
      };
      this.sources.add(source);
      source.start();
    });
  }
}

function init() {
  state.playlist = new AudioPlaylist();
  bindTabs();
  bindChat();
  bindTts();
  bindStt();
  bindVoice();
  bindHeartbeat();
  bindGlobalButtons();
  void checkHealth();
  void loadVoices();
  void loadModels();
}

function bindTabs() {
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((item) => item.classList.remove("active"));
      $$(".main-panel").forEach((panel) => panel.classList.remove("active"));
      tab.classList.add("active");
      $(`[data-panel="${tab.dataset.tab}"]`).classList.add("active");
    });
  });
}

function bindGlobalButtons() {
  $("#newThreadButton").addEventListener("click", () => {
    $("#threadId").value = `webchat:${crypto.randomUUID()}`;
    addMessage("system", `thread ${$("#threadId").value}`);
  });
  $("#clearEventsButton").addEventListener("click", () => {
    $("#eventLog").textContent = "";
  });
  $("#refreshModelsButton").addEventListener("click", () => void loadModels());
}

function bindChat() {
  $("#chatForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("#chatInput").value.trim();
    if (!text) return;
    $("#chatInput").value = "";
    state.pendingAssistantText = "";
    state.assistantFlushScheduled = false;
    state.textDeltaCount = 0;
    state.ttsAudioCount = 0;
    state.ttsAudioBytes = 0;
    state.turnStartedAt = performance.now();
    state.firstTextAt = 0;
    state.firstAudioAt = 0;
    addMessage("user", text);
    state.assistantMessage = addMessage("assistant", "");
    const wantsTts = $("#ttsMode").checked;
    if (wantsTts) {
      $("#streamMode").checked = true;
      await state.playlist.unlock();
    }
    if ($("#streamMode").checked || wantsTts) {
      await sendStreamingChat(text);
    } else {
      await sendHttpChat(text);
    }
  });
  $("#clearChatButton").addEventListener("click", () => {
    $("#messages").textContent = "";
    state.assistantMessage = null;
  });
  $("#stopChatButton").addEventListener("click", () => {
    if (state.chatSocket) {
      state.chatSocket.close();
      state.chatSocket = null;
    }
    state.playlist.reset();
    setStatus("#chatStatus", "stopped");
  });
}

async function sendStreamingChat(text) {
  const useTts = $("#ttsMode").checked;
  const socket = new WebSocket(wsUrl(useTts ? "/brain" : "/stream"));
  state.chatSocket = socket;
  state.chatUsesTts = useTts;
  state.responseDone = false;
  state.ttsPlaylistDone = !useTts;
  setStatus("#chatStatus", "connecting");
  socket.onopen = async () => {
    setStatus("#chatStatus", "streaming");
    socket.send(JSON.stringify(await buildAskPayload(text, useTts)));
  };
  socket.onmessage = (message) => handleBrainEvent(JSON.parse(message.data), "#chatStatus");
  socket.onerror = () => setStatus("#chatStatus", "socket error");
  socket.onclose = () => {
    if (state.chatSocket === socket) state.chatSocket = null;
    setStatus("#chatStatus", "idle");
  };
}

async function sendHttpChat(text) {
  setStatus("#chatStatus", "requesting");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(await buildAskPayload(text, $("#ttsMode").checked)),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(data));
    appendAssistant(data.text || "");
    if (data.audio) {
      await playAudioEvent(data.audio);
    }
    logEvent("ask.done", data);
  } catch (error) {
    appendAssistant(`\n${error.message}`);
    logEvent("ask.error", { message: error.message });
  } finally {
    setStatus("#chatStatus", "idle");
  }
}

async function buildAskPayload(text, tts) {
  const persona = buildPersona();
  const images = await buildImages();
  const options = parseJsonField("#responseOptions", {});
  const tools = $("#toolNames").value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  return {
    type: "ask",
    text,
    thread_id: $("#threadId").value.trim() || "webchat:default",
    persona,
    images,
    use_memory: $("#memoryMode").checked,
    tool_names: tools.length ? tools : null,
    tts,
    tts_options: buildTtsOptions(),
    options,
  };
}

function handleBrainEvent(event, statusSelector) {
  if (event.type === "text.delta") {
    appendAssistant(event.text || "");
  } else if (event.type === "stt.final") {
    addMessage("user", event.text || "");
  } else if (event.type === "memory.hit") {
    setStatus(statusSelector, "memory hit");
  } else if (event.type === "tool.call") {
    setStatus(statusSelector, `tool ${event.name}`);
  } else if (event.type === "error") {
    appendAssistant(`\n${event.message || "error"}`);
    setStatus(statusSelector, "error");
  } else if (event.type === "response.done") {
    state.responseDone = true;
    setStatus(statusSelector, "done");
    closeChatSocketIfComplete();
  } else if (event.type === "tts.playlist.done") {
    state.ttsPlaylistDone = true;
    closeChatSocketIfComplete();
  }
  if (event.type.startsWith("tts.")) {
    state.playlist.handle(event);
  }
  recordEvent(event.type, event);
}

function closeChatSocketIfComplete() {
  if (!state.chatSocket || !state.responseDone || !state.ttsPlaylistDone) return;
  state.chatSocket.close();
}

function bindTts() {
  $("#refreshVoicesButton").addEventListener("click", () => void loadVoices());
  $("#ttsButton").addEventListener("click", async () => {
    setStatus("#ttsStatus", "rendering");
    try {
      await state.playlist.unlock();
      const response = await fetch("/tts", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: $("#ttsText").value, options: buildTtsOptions() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      await playAudioEvent(data);
      logEvent("tts.done", { sample_rate: data.sample_rate, voice: data.voice });
      setStatus("#ttsStatus", "done");
    } catch (error) {
      setStatus("#ttsStatus", "error");
      logEvent("tts.error", { message: error.message });
    }
  });
}

function bindStt() {
  $("#sttFileButton").addEventListener("click", async () => {
    const file = $("#sttFile").files[0];
    if (!file) return;
    setStatus("#sttStatus", "transcribing");
    try {
      const audio = await fileToBase64(file);
      const response = await fetch("/stt", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          audio,
          format: $("#sttFormat").value,
          sample_rate: 16000,
          channels: 1,
          language: $("#sttLanguage").value.trim() || null,
          options: {},
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(JSON.stringify(data));
      $("#sttOutput").textContent = JSON.stringify(data, null, 2);
      logEvent("stt.final", data);
      setStatus("#sttStatus", "done");
    } catch (error) {
      setStatus("#sttStatus", "error");
      logEvent("stt.error", { message: error.message });
    }
  });
  $("#sttRecordButton").addEventListener("click", async () => {
    const chunks = [];
    state.sttRecorder = await startMic((pcm) => chunks.push(pcm), updateMeter);
    state.sttRecorder.chunks = chunks;
    setStatus("#sttStatus", "recording");
  });
  $("#sttStopButton").addEventListener("click", async () => {
    if (!state.sttRecorder) return;
    const recorder = state.sttRecorder;
    state.sttRecorder = null;
    recorder.stop();
    const audio = concatBytes(recorder.chunks);
    $("#sttOutput").textContent = "";
    setStatus("#sttStatus", "transcribing");
    const response = await fetch("/stt", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        audio: bytesToBase64(audio),
        format: "pcm_s16le",
        sample_rate: 16000,
        channels: 1,
        language: $("#sttLanguage").value.trim() || null,
        options: {},
      }),
    });
    const data = await response.json();
    $("#sttOutput").textContent = JSON.stringify(data, null, 2);
    logEvent(response.ok ? "stt.final" : "stt.error", data);
    setStatus("#sttStatus", response.ok ? "done" : "error");
  });
}

function bindVoice() {
  $("#voiceStartButton").addEventListener("click", async () => {
    if (state.voiceSocket) return;
    const socket = new WebSocket(wsUrl("/voice"));
    state.voiceSocket = socket;
    setStatus("#voiceStatus", "connecting");
    socket.onopen = async () => {
      socket.send(
        JSON.stringify({
          type: "audio.start",
          thread_id: $("#threadId").value.trim() || "webchat:voice",
          persona: buildPersona(),
          encoding: "pcm_s16le",
          sample_rate: 16000,
          channels: 1,
          tts: $("#voiceTtsMode").checked,
          use_memory: $("#memoryMode").checked,
          tool_names: $("#toolNames").value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean),
          vad: {
            provider: $("#voiceVad").value,
            threshold: Number($("#voiceThreshold").value),
            end_silence_ms: Number($("#voiceSilence").value),
          },
          tts_options: buildTtsOptions(),
          options: parseJsonField("#responseOptions", {}),
        })
      );
      state.mic = await startMic((pcm) => {
        if (socket.readyState === WebSocket.OPEN) socket.send(pcm);
      }, updateMeter);
    };
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data);
      handleBrainEvent(event, "#voiceStatus");
      if (event.type === "audio.started") setStatus("#voiceStatus", "listening");
      if (event.type === "vad.speech.start") setStatus("#voiceStatus", "speech");
      if (event.type === "vad.speech.end") setStatus("#voiceStatus", "thinking");
    };
    socket.onclose = () => {
      if (state.mic) state.mic.stop();
      state.mic = null;
      state.voiceSocket = null;
      setStatus("#voiceStatus", "idle");
    };
    socket.onerror = () => setStatus("#voiceStatus", "socket error");
  });
  $("#voiceStopButton").addEventListener("click", () => {
    if (state.mic) {
      state.mic.stop();
      state.mic = null;
    }
    if (state.voiceSocket?.readyState === WebSocket.OPEN) {
      state.voiceSocket.send(JSON.stringify({ type: "audio.stop" }));
    }
  });
  $("#voiceCancelButton").addEventListener("click", () => {
    if (state.voiceSocket?.readyState === WebSocket.OPEN) {
      state.voiceSocket.send(JSON.stringify({ type: "audio.cancel" }));
      state.voiceSocket.close();
    }
  });
}

function bindHeartbeat() {
  $("#heartbeatButton").addEventListener("click", async () => {
    setStatus("#heartbeatStatus", "running");
    try {
      const response = await fetch("/heartbeat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          thread_id: $("#threadId").value.trim() || "webchat:heartbeat",
          persona: buildPersona(),
          context: parseJsonField("#heartbeatContext", {}),
          actions: [
            {
              name: $("#heartbeatAction").value.trim() || "message",
              description: "Return a proactive message for this webchat session.",
            },
          ],
          config: {
            run_probability: Number($("#heartbeatProbability").value),
            heartbeat_path: "HEARTBEAT.md",
          },
          options: { use_memory: $("#memoryMode").checked },
        }),
      });
      const data = await response.json();
      $("#heartbeatOutput").textContent = JSON.stringify(data, null, 2);
      logEvent(response.ok ? "heartbeat.done" : "heartbeat.error", data);
      setStatus("#heartbeatStatus", response.ok ? "done" : "error");
    } catch (error) {
      setStatus("#heartbeatStatus", "error");
      logEvent("heartbeat.error", { message: error.message });
    }
  });
}

async function startMic(onPcm, onLevel) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const silentGain = audioContext.createGain();
  silentGain.gain.value = 0;
  processor.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    const downsampled = downsample(input, audioContext.sampleRate, 16000);
    onLevel(levelOf(downsampled));
    onPcm(floatToPcm16(downsampled));
  };
  source.connect(processor);
  processor.connect(silentGain);
  silentGain.connect(audioContext.destination);
  return {
    stop() {
      processor.disconnect();
      source.disconnect();
      silentGain.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      void audioContext.close();
      updateMeter(0);
    },
  };
}

function buildPersona() {
  const persona = {
    id: $("#personaId").value.trim() || "webchat",
    name: $("#personaName").value.trim() || "Webchat",
    instructions: $("#personaInstructions").value.trim(),
  };
  const model = $("#personaModel").value.trim();
  if (model) persona.model = model;
  return persona;
}

async function buildImages() {
  const images = [];
  const detail = $("#imageDetail").value;
  const url = $("#imageUrl").value.trim();
  if (url) images.push({ url, detail });
  const file = $("#imageFile").files[0];
  if (file) {
    const dataUrl = await fileToDataUrl(file);
    images.push({ url: dataUrl, detail });
  }
  return images;
}

function buildTtsOptions() {
  const voice = $("#voiceSelect").value;
  return voice ? { voice } : {};
}

async function loadVoices() {
  const select = $("#voiceSelect");
  const previous = select.value;
  select.textContent = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "default";
  select.append(defaultOption);
  try {
    const response = await fetch("/tts/voices");
    const voices = await response.json();
    for (const voice of voices) {
      const option = document.createElement("option");
      option.value = voice.slug;
      option.textContent = voice.label || voice.slug;
      select.append(option);
    }
    if (previous && Array.from(select.options).some((option) => option.value === previous)) {
      select.value = previous;
    }
    logEvent("tts.voices", { count: voices.length });
  } catch (error) {
    logEvent("tts.voices.error", { message: error.message });
  }
}

async function loadModels() {
  const select = $("#personaModel");
  const previous = select.value;
  select.textContent = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "default";
  select.append(defaultOption);
  try {
    const response = await fetch("/models");
    const models = await response.json();
    if (!response.ok) throw new Error(JSON.stringify(models));
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.default ? `${model.label} default` : model.label;
      select.append(option);
    }
    if (previous && Array.from(select.options).some((option) => option.value === previous)) {
      select.value = previous;
    }
    logEvent("models.loaded", { count: models.length });
  } catch (error) {
    for (const model of ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4.1-mini"]) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      select.append(option);
    }
    logEvent("models.error", { message: error.message });
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    $("#healthStatus").textContent = data.status || "ok";
    $("#healthStatus").classList.add("ok");
  } catch {
    $("#healthStatus").textContent = "offline";
    $("#healthStatus").classList.add("bad");
  }
}

function addMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = text;
  $("#messages").append(node);
  $("#messages").scrollTop = $("#messages").scrollHeight;
  return node;
}

function appendAssistant(text) {
  if (!text) return;
  state.pendingAssistantText += text;
  if (state.assistantFlushScheduled) return;
  state.assistantFlushScheduled = true;
  requestAnimationFrame(flushAssistantText);
}

function flushAssistantText() {
  state.assistantFlushScheduled = false;
  const text = state.pendingAssistantText;
  state.pendingAssistantText = "";
  if (!text) return;
  if (!state.assistantMessage) {
    state.assistantMessage = addMessage("assistant", "");
  }
  state.assistantMessage.append(document.createTextNode(text));
  $("#messages").scrollTop = $("#messages").scrollHeight;
}

function recordEvent(type, data) {
  if (type === "text.delta") {
    state.textDeltaCount += 1;
    if (!state.firstTextAt) {
      state.firstTextAt = performance.now();
      logEvent("stream.first_text", {
        ms: Math.round(state.firstTextAt - state.turnStartedAt),
        text: data.text || "",
      });
    }
    return;
  }
  if (type === "tts.audio") {
    state.ttsAudioCount += 1;
    if (typeof data.audio === "string") {
      state.ttsAudioBytes += Math.floor((data.audio.length * 3) / 4);
    }
    if (!state.firstAudioAt) {
      state.firstAudioAt = performance.now();
      logEvent("stream.first_audio", {
        ms: Math.round(state.firstAudioAt - state.turnStartedAt),
        segment_index: data.segment_index,
        bytes_estimate: typeof data.audio === "string" ? Math.floor((data.audio.length * 3) / 4) : 0,
      });
    }
    return;
  }
  if (type === "tts.start" || type === "tts.done") {
    return;
  }
  if (type === "response.done" && state.textDeltaCount) {
    logEvent("text.delta.summary", { chunks: state.textDeltaCount });
    state.textDeltaCount = 0;
  }
  if (type === "tts.playlist.done" && state.ttsAudioCount) {
    logEvent("tts.audio.summary", {
      chunks: state.ttsAudioCount,
      bytes_estimate: state.ttsAudioBytes,
    });
    state.ttsAudioCount = 0;
    state.ttsAudioBytes = 0;
  }
  logEvent(type, data);
}

function logEvent(type, data) {
  const row = document.createElement("div");
  row.className = "event-row";
  row.textContent = `${new Date().toLocaleTimeString()} ${type}\n${JSON.stringify(
    sanitizeEventForLog(data),
    null,
    2
  )}`;
  $("#eventLog").prepend(row);
  while ($("#eventLog").children.length > 80) {
    $("#eventLog").lastElementChild.remove();
  }
}

function yieldToBrowser() {
  return new Promise((resolve) => {
    if (document.visibilityState === "visible") {
      requestAnimationFrame(() => resolve());
    } else {
      setTimeout(resolve, 0);
    }
  });
}

function sanitizeEventForLog(data) {
  if (!data || typeof data !== "object") return data;
  const copy = { ...data };
  if (typeof copy.audio === "string") {
    copy.audio_bytes_estimate = Math.floor((copy.audio.length * 3) / 4);
    copy.audio = `<base64 ${copy.audio.length} chars>`;
  }
  if (typeof copy.text === "string" && copy.text.length > 220) {
    copy.text = `${copy.text.slice(0, 220)}...`;
  }
  return copy;
}

function setStatus(selector, text) {
  $(selector).textContent = text;
}

function parseJsonField(selector, fallback) {
  const text = $(selector).value.trim();
  if (!text) return fallback;
  try {
    return JSON.parse(text);
  } catch (error) {
    logEvent("json.error", { field: selector, message: error.message });
    return fallback;
  }
}

function wsUrl(path) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}${path}`;
}

function updateMeter(level) {
  const value = Math.max(0, Math.min(1, level));
  $("#voiceLevel").textContent = `${Math.round(value * 100)}%`;
  const canvas = $("#voiceMeter");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0f1419";
  context.fillRect(0, 0, width, height);
  context.fillStyle = value > 0.6 ? "#ff8a6b" : "#4fc3b1";
  context.fillRect(0, height - height * value, width, height * value);
  context.strokeStyle = "#343d49";
  context.lineWidth = 2;
  context.strokeRect(1, 1, width - 2, height - 2);
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
  return new Uint8Array(buffer);
}

function base64ToBytes(value) {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function bytesToBase64(bytes) {
  let binary = "";
  for (let index = 0; index < bytes.length; index += 1) {
    binary += String.fromCharCode(bytes[index]);
  }
  return btoa(binary);
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

async function fileToBase64(file) {
  const buffer = await file.arrayBuffer();
  return bytesToBase64(new Uint8Array(buffer));
}

async function fileToDataUrl(file) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function playAudioEvent(data) {
  const playlist = state.playlist || new AudioPlaylist();
  playlist.reset();
  playlist.queue.push({
    index: 0,
    sampleRate: data.sample_rate,
    encoding: data.encoding,
    chunks: [base64ToBytes(data.audio)],
  });
  await playlist.pump();
}

init();
