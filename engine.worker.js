// Pocketframe engine. Runs in a worker so the page stays responsive.
// It downloads the model once into this browser's private storage, loads it
// onto the graphics chip with ONNX Runtime Web, and turns prompts into images.

import { ClipTokenizer } from './clip-tokenizer.js';
import { encodePrompt, denoise, decode } from './pipeline.js';
import { externalizeWeights } from './onnx-external.js';

const ORT_VERSION = '1.31.1';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.bundle.min.mjs`;

// SD-Turbo converted for browsers (half-precision weights). Pinned to one
// revision so the files can never change underneath a partial download.
const MODEL_REVISION = 'ace89b7d2cd849f9a73914cdbb8a3ea60c853dd1';
const MODEL_CDN = `https://huggingface.co/schmuell/sd-turbo-ort-web/resolve/${MODEL_REVISION}`;

const FILES = [
  { key: 'unet', path: 'unet/model.onnx', bytes: 1733430199, label: 'Image model' },
  { key: 'text_encoder', path: 'text_encoder/model.onnx', bytes: 681393168, label: 'Prompt reader' },
  { key: 'vae_decoder', path: 'vae_decoder/model.onnx', bytes: 99094314, label: 'Image decoder' },
  { key: 'vocab', path: 'tokenizer/vocab.json', bytes: 1059962, label: 'Vocabulary' },
  { key: 'merges', path: 'tokenizer/merges.txt', bytes: 524619, label: 'Vocabulary' },
];
let totalBytes = FILES.reduce((sum, f) => sum + f.bytes, 0);

const SIZE = 512; // pixels; the model's native size
const LATENT = SIZE / 8;
const STORE_DIR = 'pocketframe-models';
const STORE_MODEL = 'sd-turbo';
const FLUSH_EVERY = 32 * 1024 * 1024;
const READ_SLICE = 64 * 1024 * 1024;

let config = { local: false, backend: 'webgpu' };
let ort = null;
let sessions = null;
let tokenizer = null;
let downloadAbort = null;
let cancelGeneration = false;
const promptCache = new Map();

class EngineError extends Error {
  constructor(code, message, detail) {
    super(message);
    this.code = code;
    this.detail = detail;
  }
}

function emit(event, data) {
  self.postMessage({ event, ...data });
}

// ---------------------------------------------------------------- storage

function hasPrivateStorage() {
  return !!(navigator.storage && navigator.storage.getDirectory) &&
    typeof FileSystemFileHandle !== 'undefined' &&
    'createSyncAccessHandle' in FileSystemFileHandle.prototype;
}

async function modelDir(create) {
  const root = await navigator.storage.getDirectory();
  const top = await root.getDirectoryHandle(STORE_DIR, { create });
  return top.getDirectoryHandle(STORE_MODEL, { create });
}

async function openFile(file, create) {
  const dir = await modelDir(create);
  const handle = await dir.getFileHandle(`${file.key}.bin`, { create });
  try {
    return await handle.createSyncAccessHandle();
  } catch (err) {
    if (err && (err.name === 'NoModificationAllowedError' || err.name === 'InvalidStateError')) {
      throw new EngineError('busy_other_tab', 'The model files are in use by another Pocketframe tab.');
    }
    throw err;
  }
}

async function storedBytes(file) {
  if (!hasPrivateStorage()) return 0;
  let access;
  try {
    access = await openFile(file, false);
  } catch (err) {
    if (err instanceof EngineError) throw err;
    return 0; // not there yet
  }
  try {
    return access.getSize();
  } finally {
    access.close();
  }
}

async function downloadStatus() {
  const files = [];
  let have = 0;
  for (const file of FILES) {
    const stored = Math.min(await storedBytes(file), file.bytes);
    have += stored;
    files.push({ key: file.key, label: file.label, bytes: file.bytes, stored });
  }
  return { files, have, total: totalBytes, complete: have === totalBytes };
}

async function storageEstimate() {
  try {
    const est = await navigator.storage.estimate();
    return { quota: est.quota || 0, usage: est.usage || 0 };
  } catch {
    return { quota: 0, usage: 0 };
  }
}

function fileUrl(file) {
  if (config.local) return new URL(`./models/sd-turbo/${file.path}`, import.meta.url).href;
  return `${MODEL_CDN}/${file.path}`;
}

// Storage calls fail with a quota error when the device (or the browser's
// allowance for this site) is full.
function storageWrite(work) {
  try {
    return work();
  } catch (err) {
    if (err && (err.name === 'QuotaExceededError' || /quota/i.test(String(err.message)))) {
      throw new EngineError('quota', 'This device ran out of free space for the model.');
    }
    throw err;
  }
}

// Streams one file to storage, picking up where a previous attempt stopped.
async function downloadFile(file, onBytes, signal) {
  const access = await openFile(file, true);
  try {
    let have = access.getSize();
    if (have > file.bytes) {
      access.truncate(0);
      have = 0;
    }
    if (have === file.bytes) return;

    const headers = have > 0 ? { Range: `bytes=${have}-` } : {};
    let res;
    try {
      res = await fetch(fileUrl(file), { headers, signal, cache: 'no-store' });
    } catch (err) {
      if (signal.aborted) throw new EngineError('paused', 'Download paused.');
      throw new EngineError('offline', 'The connection dropped.', String(err && err.message || err));
    }
    if (have > 0 && res.status === 200) {
      // The server sent the whole file again instead of the rest of it.
      access.truncate(0);
      onBytes(-have);
      have = 0;
    } else if (have > 0 && res.status !== 206) {
      throw new EngineError('http', `The model host answered ${res.status}.`);
    } else if (!res.ok) {
      throw new EngineError('http', `The model host answered ${res.status}.`);
    }

    const reader = res.body.getReader();
    let unflushed = 0;
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (err) {
        try { access.flush(); } catch { /* keep whatever was written */ }
        if (signal.aborted) throw new EngineError('paused', 'Download paused.');
        throw new EngineError('offline', 'The connection dropped.', String(err && err.message || err));
      }
      if (chunk.done) break;
      const data = chunk.value;
      if (have + data.byteLength > file.bytes) {
        access.truncate(0);
        throw new EngineError('size_mismatch', `${file.label} came back larger than expected.`);
      }
      const written = storageWrite(() => access.write(data, { at: have }));
      if (written !== data.byteLength) {
        throw new EngineError('quota', 'This device ran out of free space for the model.');
      }
      have += data.byteLength;
      unflushed += data.byteLength;
      if (unflushed >= FLUSH_EVERY) {
        storageWrite(() => access.flush());
        unflushed = 0;
      }
      onBytes(data.byteLength);
    }
    storageWrite(() => access.flush());
    if (have !== file.bytes) {
      throw new EngineError('offline', 'The download stopped early.');
    }
  } finally {
    access.close();
  }
}

async function downloadModel() {
  if (!hasPrivateStorage()) {
    throw new EngineError('no_storage', 'This browser cannot keep large files for a website.');
  }
  if (downloadAbort) throw new EngineError('busy', 'A download is already running.');

  const before = await downloadStatus();
  if (before.complete) return before;

  const { quota, usage } = await storageEstimate();
  const needed = before.total - before.have;
  if (quota && quota - usage < needed + 64 * 1024 * 1024) {
    throw new EngineError('quota', 'Not enough free space for the model.', {
      needed, free: Math.max(0, quota - usage),
    });
  }

  downloadAbort = new AbortController();
  const signal = downloadAbort.signal;
  let have = before.have;
  let lastEmit = 0;
  let windowStart = performance.now();
  let windowBytes = 0;
  let speed = 0;

  const report = (label, force) => {
    const now = performance.now();
    if (!force && now - lastEmit < 250) return;
    if (now - windowStart > 1500) {
      speed = (windowBytes * 1000) / (now - windowStart);
      windowStart = now;
      windowBytes = 0;
    }
    lastEmit = now;
    emit('download-progress', { have, total: totalBytes, label, speed });
  };

  try {
    for (const file of FILES) {
      let stalls = 0;
      for (;;) {
        const startedAt = have;
        try {
          await downloadFile(file, (n) => {
            have += n;
            if (n > 0) windowBytes += n;
            report(file.label, false);
          }, signal);
          break;
        } catch (err) {
          if (!(err instanceof EngineError) || err.code !== 'offline') throw err;
          // Keep going while each attempt makes progress; give up after a few dead ones.
          stalls = have > startedAt ? 0 : stalls + 1;
          if (stalls >= 4) throw err;
          emit('download-progress', { have, total: totalBytes, label: file.label, speed: 0, retrying: true });
          await new Promise((resolve) => setTimeout(resolve, 1500 * (stalls + 1)));
          if (signal.aborted) throw new EngineError('paused', 'Download paused.');
        }
      }
      report(file.label, true);
    }
  } finally {
    downloadAbort = null;
  }
  return downloadStatus();
}

async function readStored(file) {
  const access = await openFile(file, false);
  try {
    const size = access.getSize();
    if (size !== file.bytes) {
      throw new EngineError('not_downloaded', 'The model download is incomplete.');
    }
    let buffer;
    try {
      buffer = new Uint8Array(size);
    } catch {
      throw new EngineError('memory', 'Not enough free memory to load the model.');
    }
    for (let at = 0; at < size; at += READ_SLICE) {
      const end = Math.min(size, at + READ_SLICE);
      const got = access.read(buffer.subarray(at, end), { at });
      if (got !== end - at) throw new EngineError('engine_load', 'A model file could not be read back.');
    }
    return buffer;
  } finally {
    access.close();
  }
}

async function deleteModel() {
  await unload();
  if (!hasPrivateStorage()) return;
  const root = await navigator.storage.getDirectory();
  try {
    await root.removeEntry(STORE_DIR, { recursive: true });
  } catch (err) {
    if (err && err.name === 'NoModificationAllowedError') {
      throw new EngineError('busy_other_tab', 'The model files are in use by another Pocketframe tab.');
    }
    if (!err || err.name !== 'NotFoundError') throw err;
  }
}

// ------------------------------------------------------------- capability

async function probe() {
  const caps = {
    backend: config.backend,
    webgpu: typeof navigator !== 'undefined' && !!navigator.gpu,
    adapter: false,
    f16: false,
    gpu: '',
    storage: hasPrivateStorage(),
    memoryGb: navigator.deviceMemory || null,
    quota: 0,
    usage: 0,
  };
  if (caps.webgpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        caps.adapter = true;
        caps.f16 = adapter.features.has('shader-f16');
        const info = adapter.info || {};
        caps.gpu = [info.vendor, info.architecture || info.device].filter(Boolean).join(' ');
      }
    } catch {
      // leave adapter false
    }
  }
  Object.assign(caps, await storageEstimate());
  caps.ready = config.backend === 'wasm' ? true : caps.webgpu && caps.adapter && caps.f16;
  return caps;
}

// ----------------------------------------------------------------- engine

function sessionOptions(overrides) {
  return {
    executionProviders: [config.backend],
    enableMemPattern: false,
    enableCpuMemArena: false,
    extra: {
      session: {
        disable_prepacking: '1',
        use_device_allocator_for_initializers: '1',
        use_ort_model_bytes_directly: '1',
        use_ort_model_bytes_for_initializers: '1',
      },
    },
    freeDimensionOverrides: overrides,
  };
}

const SESSION_PLAN = [
  // Largest first: peak memory is lowest when nothing else is loaded yet.
  { key: 'unet', overrides: { batch_size: 1, num_channels: 4, height: LATENT, width: LATENT, sequence_length: 77, steps: 1 } },
  { key: 'text_encoder', overrides: { batch_size: 1, sequence_length: 77 } },
  { key: 'vae_decoder', overrides: { batch_size: 1, num_channels_latent: 4, height_latent: LATENT, width_latent: LATENT } },
];

async function loadBytes(key) {
  return readStored(FILES.find((f) => f.key === key));
}

// Reads one stored model as { graph, weights }: a small graph whose weights
// point into the original file, plus that file's bytes. Loading it this way
// keeps the engine's own working memory small (see onnx-external.js).
async function loadModelParts(key) {
  const file = FILES.find((f) => f.key === key);
  const location = `${key}.bin`;
  const access = await openFile(file, false);
  try {
    const size = access.getSize();
    if (size !== file.bytes) {
      throw new EngineError('not_downloaded', 'The model download is incomplete.');
    }
    let graph;
    try {
      graph = externalizeWeights((at, length) => {
        const chunk = new Uint8Array(length);
        access.read(chunk, { at });
        return chunk;
      }, size, location).graph;
    } catch (err) {
      throw new EngineError('engine_load', 'A model file could not be read.', String(err && err.message || err));
    }
    let weights;
    try {
      weights = new Uint8Array(size);
    } catch {
      throw new EngineError('memory', 'Not enough free memory to load the model.');
    }
    for (let at = 0; at < size; at += READ_SLICE) {
      const end = Math.min(size, at + READ_SLICE);
      const got = access.read(weights.subarray(at, end), { at });
      if (got !== end - at) throw new EngineError('engine_load', 'A model file could not be read back.');
    }
    return { graph, weights, location };
  } finally {
    access.close();
  }
}

async function load() {
  if (sessions) return { loaded: true };
  if (!hasPrivateStorage()) {
    throw new EngineError('no_storage', 'This browser cannot keep large files for a website.');
  }
  const status = await downloadStatus();
  if (!status.complete) throw new EngineError('not_downloaded', 'Download the model first.');

  emit('load-progress', { stage: 'runtime', done: 0, of: SESSION_PLAN.length });
  if (!ort) {
    const url = config.local
      ? new URL('./ortdist/ort.webgpu.bundle.min.mjs', import.meta.url).href
      : ORT_CDN;
    try {
      ort = await import(url);
    } catch (err) {
      throw new EngineError('offline', 'The engine code could not be fetched. Check the connection and try again.', String(err && err.message || err));
    }
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
  }

  if (!tokenizer) {
    const decoder = new TextDecoder();
    const vocab = JSON.parse(decoder.decode(await loadBytes('vocab')));
    const merges = decoder.decode(await loadBytes('merges'));
    tokenizer = new ClipTokenizer(vocab, merges);
  }

  const built = {};
  try {
    for (let i = 0; i < SESSION_PLAN.length; i++) {
      const plan = SESSION_PLAN[i];
      emit('load-progress', { stage: plan.key, done: i, of: SESSION_PLAN.length });
      let parts = await loadModelParts(plan.key);
      built[plan.key] = await ort.InferenceSession.create(parts.graph, {
        ...sessionOptions(plan.overrides),
        externalData: [{ path: parts.location, data: parts.weights }],
      });
      parts = null; // let the copy in page memory go as soon as the engine has it
    }
  } catch (err) {
    for (const s of Object.values(built)) {
      try { await s.release(); } catch { /* already gone */ }
    }
    if (err instanceof EngineError) throw err;
    const text = String(err && err.message || err);
    if (/f16/i.test(text)) {
      throw new EngineError('no_f16', 'This graphics chip lacks a feature the model needs (16-bit shader maths).', text);
    }
    if (/memory|alloc|oom/i.test(text)) {
      throw new EngineError('memory', 'The phone ran out of memory while loading the model.', text);
    }
    throw new EngineError('engine_load', 'The model could not be loaded onto the graphics chip.', text);
  }
  sessions = built;
  emit('load-progress', { stage: 'done', done: SESSION_PLAN.length, of: SESSION_PLAN.length });
  return { loaded: true };
}

async function unload() {
  const current = sessions;
  sessions = null;
  promptCache.clear();
  if (!current) return;
  for (const s of Object.values(current)) {
    try { await s.release(); } catch { /* already gone */ }
  }
}

function checkCancelled() {
  if (cancelGeneration) throw new EngineError('cancelled', 'Stopped.');
}

async function cachedPrompt(prompt) {
  const cached = promptCache.get(prompt);
  if (cached) return cached;
  const encoded = await encodePrompt(ort, sessions.text_encoder, tokenizer, prompt);
  promptCache.set(prompt, encoded);
  if (promptCache.size > 6) promptCache.delete(promptCache.keys().next().value);
  return encoded;
}

async function generate({ prompt, steps, seed }) {
  if (!sessions) throw new EngineError('not_loaded', 'The engine is not loaded.');
  const text = String(prompt || '').trim();
  if (!text) throw new EngineError('empty_prompt', 'Write a prompt first.');
  cancelGeneration = false;

  const started = performance.now();
  const timing = {};
  try {
    emit('gen-progress', { stage: 'prompt', step: 0, steps });
    const encoded = await cachedPrompt(text);
    timing.prompt = performance.now() - started;
    checkCancelled();

    const denoiseStart = performance.now();
    const latents = await denoise(ort, sessions.unet, encoded, {
      steps,
      seed,
      latentSize: LATENT,
      onStep: (step, of) => emit('gen-progress', { stage: 'denoise', step, steps: of }),
      shouldStop: () => cancelGeneration,
    });
    if (!latents) throw new EngineError('cancelled', 'Stopped.');
    timing.denoise = performance.now() - denoiseStart;

    emit('gen-progress', { stage: 'decode', step: steps, steps });
    const decodeStart = performance.now();
    const { rgba, width, height } = await decode(ort, sessions.vae_decoder, latents, LATENT);
    const canvas = new OffscreenCanvas(width, height);
    canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    const blob = await canvas.convertToBlob({ type: 'image/png' });
    timing.decode = performance.now() - decodeStart;

    return {
      blob,
      width,
      height,
      seed,
      steps,
      ms: performance.now() - started,
      timing,
      promptTokens: encoded.used,
      promptTruncated: encoded.truncated,
    };
  } catch (err) {
    if (err instanceof EngineError) throw err;
    const textErr = String(err && err.message || err);
    if (/lost|destroyed|memory|alloc/i.test(textErr)) {
      // The graphics device is gone; the sessions cannot be reused.
      sessions = null;
      promptCache.clear();
      throw new EngineError('device_lost', 'The graphics chip ran out of memory or was reset.', textErr);
    }
    throw new EngineError('engine_run', 'The image could not be generated.', textErr);
  }
}

// --------------------------------------------------------------- messages

const commands = {
  async init(payload) {
    config = { local: !!payload.local, backend: payload.backend === 'wasm' ? 'wasm' : 'webgpu' };
    if (config.local) {
      // Development only: a local model folder may hold stand-in files of other sizes.
      try {
        const sizes = await (await fetch(new URL('./models/sd-turbo/sizes.json', import.meta.url))).json();
        for (const file of FILES) if (sizes[file.key]) file.bytes = sizes[file.key];
        totalBytes = FILES.reduce((sum, f) => sum + f.bytes, 0);
      } catch { /* use the real sizes */ }
    }
    const caps = await probe();
    let status = { files: [], have: 0, total: totalBytes, complete: false };
    try {
      status = await downloadStatus();
    } catch (err) {
      if (!(err instanceof EngineError)) throw err;
      status.blocked = err.code;
    }
    return { caps, status, size: SIZE, totalBytes: totalBytes };
  },
  status: () => downloadStatus(),
  download: () => downloadModel(),
  pause() {
    if (downloadAbort) downloadAbort.abort();
    return {};
  },
  load: () => load(),
  unload: async () => { await unload(); return {}; },
  generate: (payload) => generate(payload),
  cancel() {
    cancelGeneration = true;
    return {};
  },
  deleteModel: async () => { await deleteModel(); return downloadStatus(); },
};

self.onmessage = async (message) => {
  const { id, cmd, payload } = message.data || {};
  const run = commands[cmd];
  if (!run) {
    self.postMessage({ id, ok: false, error: { code: 'bad_command', message: `Unknown command ${cmd}` } });
    return;
  }
  try {
    const result = await run(payload || {});
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    const known = err instanceof EngineError;
    self.postMessage({
      id,
      ok: false,
      error: {
        code: known ? err.code : 'unexpected',
        message: known ? err.message : 'Something unexpected went wrong.',
        detail: known ? err.detail : String(err && (err.stack || err.message) || err),
      },
    });
  }
};
