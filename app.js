// Pocketframe page logic: talks to the engine worker, keeps the photo library
// in this browser, and draws the interface.

import { randomSeed } from './sampler.js';

const STYLES = [
  { id: 'none', name: 'No style', suffix: '' },
  { id: 'portrait', name: 'Portrait', suffix: '85mm portrait photo, soft window light, shallow depth of field, natural skin texture' },
  { id: 'film', name: '35mm film', suffix: '35mm film photo, warm tones, soft grain, natural light' },
  { id: 'cinematic', name: 'Cinematic', suffix: 'cinematic still, anamorphic lens, moody lighting, shallow depth of field' },
  { id: 'street', name: 'Street', suffix: 'candid street photography, natural light, 35mm lens' },
  { id: 'studio', name: 'Studio', suffix: 'studio photo, softbox lighting, seamless backdrop, sharp focus' },
  { id: 'product', name: 'Product', suffix: 'product photo, clean background, soft shadows, sharp detail' },
  { id: 'landscape', name: 'Landscape', suffix: 'wide landscape photo, golden hour, crisp detail' },
  { id: 'mono', name: 'Black and white', suffix: 'black and white photo, high contrast, fine grain' },
  { id: 'illustration', name: 'Illustration', suffix: 'detailed digital illustration, clean lines, rich color' },
];

const EXAMPLES = [
  'Portrait of an old fisherman in a yellow raincoat, harbor at dawn',
  'A red vintage car on a rainy neon street at night',
  'A bowl of ramen on a wooden counter, steam rising, window light',
];

const BUILD = '3';
const SETTINGS_KEY = 'pocketframe.settings';
const STAGE_KEY = 'pocketframe.stage';

const $ = (id) => document.getElementById(id);
const el = {
  engineButton: $('engineButton'), engineLabel: $('engineLabel'),
  printWindow: $('printWindow'), printCaption: $('printCaption'),
  printActions: $('printActions'), usedPrompt: $('usedPrompt'),
  saveButton: $('saveButton'), shareButton: $('shareButton'), reuseButton: $('reuseButton'), deleteButton: $('deleteButton'),
  stripSection: $('stripSection'), strip: $('strip'), stripCount: $('stripCount'),
  dock: $('dock'), styleSelect: $('styleSelect'), stepsSeg: $('stepsSeg'), countSeg: $('countSeg'),
  seedToggle: $('seedToggle'), seedInput: $('seedInput'),
  prompt: $('prompt'), goButton: $('goButton'), hint: $('hint'),
  enginePanel: $('enginePanel'), closePanel: $('closePanel'), facts: $('facts'), modelLine: $('modelLine'),
  unloadButton: $('unloadButton'), reportButton: $('reportButton'), deleteModelButton: $('deleteModelButton'),
  toast: $('toast'),
};

const state = {
  engine: 'checking', // checking | unsupported | needs-model | downloading | paused | downloaded | loading | ready | error
  caps: null,
  status: null,
  totalBytes: 0,
  download: { have: 0, total: 0, speed: 0, retrying: false },
  engineError: null, // { code, message, detail, retry }
  crashInfo: null, // what the page was doing if it closed unexpectedly last time
  lastCrash: null,
  working: null, // { index, count, stage, step, steps, model }
  stopRequested: false,
  genError: null,
  frames: [],
  currentId: null,
  freshId: null,
  peek: false,
  truncatedWarning: false,
  firstSinceLoad: false,
  lastTiming: null,
  settings: { style: 'none', steps: 1, count: 1, seedLocked: false, seed: 1234 },
};

// ------------------------------------------------------------ small helpers

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function formatBytes(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(0, Math.round(n / 1e3))} KB`;
}

function formatSeconds(ms) {
  const s = ms / 1000;
  return s >= 10 ? `${Math.round(s)} s` : `${s.toFixed(1)} s`;
}

function timeLeft(bytes, speed) {
  if (!speed || speed < 1) return '';
  const seconds = bytes / speed;
  if (seconds < 60) return 'less than a minute left';
  const minutes = Math.round(seconds / 60);
  return `about ${minutes} min left`;
}

function readStorage(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function writeStorage(key, value) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch { /* private mode: carry on without it */ }
}

// A note of what the page is doing right now, kept where it survives a crash.
// If the browser closes the page mid-task, the next visit can say where.
function markStage(info) {
  writeStorage(STAGE_KEY, info ? JSON.stringify({ ...info, build: BUILD, at: Date.now() }) : null);
}

const MODEL_NAMES = { text_encoder: 'prompt reader', unet: 'image model', vae_decoder: 'image decoder' };

function describeStage(info) {
  if (!info) return '';
  if (info.stage === 'engine') return 'starting the engine';
  if (info.stage === 'prompt') return 'reading the prompt';
  if (info.stage === 'model') {
    const name = MODEL_NAMES[info.key] || 'model';
    if (info.phase === 'read') return `preparing the ${name}`;
    if (info.phase === 'done') return `just after loading the ${name}`;
    return `loading the ${name} (${formatBytes(info.sent || 0)} of ${formatBytes(info.total || 0)} sent to the graphics chip)`;
  }
  if (info.stage === 'denoise') return `forming the image (step ${info.step} of ${info.steps})`;
  if (info.stage === 'decode') return 'turning the image into pixels';
  return 'working';
}

let toastTimer = 0;
function toast(text) {
  el.toast.textContent = text;
  el.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.toast.hidden = true; }, 2200);
}

// Two-tap confirmation for destructive buttons.
function confirmTap(button, question, action) {
  const original = button.textContent;
  let armed = false;
  let timer = 0;
  button.addEventListener('click', async () => {
    if (!armed) {
      armed = true;
      button.textContent = question;
      timer = setTimeout(() => { armed = false; button.textContent = original; }, 3500);
      return;
    }
    clearTimeout(timer);
    armed = false;
    button.textContent = original;
    await action();
  });
}

const wake = {
  lock: null,
  wanted: false,
  async acquire() {
    this.wanted = true;
    try {
      if ('wakeLock' in navigator && !this.lock) {
        this.lock = await navigator.wakeLock.request('screen');
        this.lock.addEventListener('release', () => { this.lock = null; });
      }
    } catch { /* not allowed right now */ }
  },
  release() {
    this.wanted = false;
    try { this.lock?.release(); } catch { /* already released */ }
    this.lock = null;
  },
};
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible' && wake.wanted) wake.acquire();
});

// ---------------------------------------------------------- photo library

const library = {
  db: null,
  urls: new Map(),
  memoryId: 0,
  async open() {
    try {
      this.db = await new Promise((resolve, reject) => {
        const request = indexedDB.open('pocketframe', 1);
        request.onupgradeneeded = () => request.result.createObjectStore('frames', { keyPath: 'id', autoIncrement: true });
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
    } catch {
      this.db = null; // photos will last for this visit only
    }
  },
  tx(mode, work) {
    return new Promise((resolve, reject) => {
      const store = this.db.transaction('frames', mode).objectStore('frames');
      const request = work(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async all() {
    if (!this.db) return [];
    const rows = await this.tx('readonly', (s) => s.getAll());
    return rows.sort((a, b) => b.id - a.id);
  },
  async add(frame) {
    if (this.db) {
      try {
        frame.id = await this.tx('readwrite', (s) => s.add(frame));
        return frame;
      } catch { /* storage full or blocked: keep it for this visit */ }
    }
    this.memoryId = Math.max(this.memoryId, ...state.frames.map((f) => f.id), 0) + 1;
    frame.id = this.memoryId;
    return frame;
  },
  async remove(id) {
    const url = this.urls.get(id);
    if (url) { URL.revokeObjectURL(url); this.urls.delete(id); }
    if (this.db) {
      try { await this.tx('readwrite', (s) => s.delete(id)); } catch { /* nothing to remove */ }
    }
  },
  url(frame) {
    let url = this.urls.get(frame.id);
    if (!url) {
      url = URL.createObjectURL(frame.blob);
      this.urls.set(frame.id, url);
    }
    return url;
  },
};

function currentFrame() {
  return state.frames.find((f) => f.id === state.currentId) || null;
}

function fileName(frame) {
  return `pocketframe-${String(frame.id).padStart(4, '0')}.png`;
}

// ------------------------------------------------------------------ engine

const params = new URLSearchParams(location.search);
const engineConfig = {
  local: params.get('local') === '1',
  backend: params.get('backend') === 'wasm' ? 'wasm' : 'webgpu',
};

let worker = null;
let callSeq = 0;
const pending = new Map();

function startWorker() {
  worker = new Worker(new URL('./engine.worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (message) => {
    const data = message.data;
    if (data.event) { onEngineEvent(data); return; }
    const waiter = pending.get(data.id);
    if (!waiter) return;
    pending.delete(data.id);
    if (data.ok) waiter.resolve(data.result);
    else waiter.reject(data.error);
  };
  worker.onerror = (event) => {
    const error = { code: 'worker_crash', message: 'The engine stopped unexpectedly.', detail: event.message || '' };
    for (const waiter of pending.values()) waiter.reject(error);
    pending.clear();
    state.working = null;
    setEngineError(error, 'reload');
  };
}

function call(cmd, payload) {
  return new Promise((resolve, reject) => {
    const id = ++callSeq;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, payload });
  });
}

function onEngineEvent(data) {
  if (data.event === 'download-progress') {
    state.download = { have: data.have, total: data.total, speed: data.speed || 0, retrying: !!data.retrying };
    updateProgress();
    renderEngineButton();
  } else if (data.event === 'model-progress' && state.working) {
    state.working.model = data.phase === 'done' ? null : { key: data.key, phase: data.phase, sent: data.sent, total: data.total };
    markStage({ stage: 'model', key: data.key, phase: data.phase, sent: data.sent, total: data.total });
    updateProgress();
  } else if (data.event === 'gen-progress' && state.working) {
    state.working.stage = data.stage;
    state.working.step = data.step;
    state.working.steps = data.steps;
    state.working.model = null;
    markStage({ stage: data.stage, step: data.step, steps: data.steps });
    updateProgress();
  }
}

function setEngine(next) {
  state.engine = next;
  if (next !== 'error') state.engineError = null;
  render();
}

function setEngineError(error, retry) {
  state.engineError = { code: error.code || 'unexpected', message: error.message || 'Something went wrong.', detail: error.detail, retry };
  state.engine = 'error';
  render();
}

function unsupportedReasons(caps) {
  const reasons = [];
  if (!caps.webgpu) reasons.push('webgpu');
  else if (!caps.adapter) reasons.push('adapter');
  else if (!caps.f16) reasons.push('f16');
  if (!caps.storage) reasons.push('storage');
  return reasons;
}

async function boot() {
  startWorker();
  await library.open();
  state.frames = await library.all();
  state.currentId = state.frames[0]?.id ?? null;

  try {
    const left = JSON.parse(readStorage(STAGE_KEY) || 'null');
    if (left) {
      state.crashInfo = left;
      state.lastCrash = left;
    }
  } catch { /* unreadable note: ignore it */ }
  markStage(null);

  let init;
  try {
    init = await call('init', engineConfig);
  } catch (error) {
    setEngineError(error, 'reload');
    return;
  }
  state.caps = init.caps;
  state.status = init.status;
  state.totalBytes = init.totalBytes;

  const backendOk = engineConfig.backend === 'wasm' || (init.caps.webgpu && init.caps.adapter && init.caps.f16);
  if (!backendOk || !init.caps.storage) {
    setEngine('unsupported');
  } else if (init.status.blocked) {
    setEngineError({ code: init.status.blocked, message: 'The model files are in use by another Pocketframe tab.' }, 'reload');
  } else if (init.status.complete) {
    setEngine('downloaded');
  } else {
    setEngine(init.status.have > 0 ? 'paused' : 'needs-model');
  }
}

async function refreshStatus() {
  try { state.status = await call('status'); } catch { /* keep the last known status */ }
}

async function startDownload() {
  state.download = { have: state.status?.have || 0, total: state.totalBytes, speed: 0, retrying: false };
  state.crashInfo = null;
  setEngine('downloading');
  try { navigator.storage?.persist?.(); } catch { /* optional */ }
  await wake.acquire();
  try {
    state.status = await call('download');
  } catch (error) {
    await refreshStatus();
    wake.release();
    if (error.code === 'paused') {
      setEngine(state.status?.have > 0 ? 'paused' : 'needs-model');
    } else {
      setEngineError(error, 'download');
    }
    return;
  }
  wake.release();
  await startEngine();
}

async function pauseDownload() {
  try { await call('pause'); } catch { /* the download call reports the outcome */ }
}

async function startEngine() {
  state.crashInfo = null;
  setEngine('loading');
  markStage({ stage: 'engine' });
  await wake.acquire();
  try {
    await call('load');
    state.firstSinceLoad = true;
    markStage(null);
    wake.release();
    setEngine('ready');
    return true;
  } catch (error) {
    markStage(null);
    wake.release();
    await refreshStatus();
    setEngineError(error, error.code === 'not_downloaded' ? 'download' : 'load');
    return false;
  }
}

async function unloadEngine() {
  try { await call('unload'); } catch { /* nothing loaded */ }
  state.firstSinceLoad = true;
  render();
}

async function deleteModel() {
  try {
    state.status = await call('deleteModel');
    setEngine('needs-model');
    toast('Model deleted');
  } catch (error) {
    toast(error.message || 'The model could not be deleted');
  }
  renderPanel();
}

// --------------------------------------------------------------- generate

function fullPrompt(text) {
  const style = STYLES.find((s) => s.id === state.settings.style) || STYLES[0];
  return style.suffix ? `${text}, ${style.suffix}` : text;
}

function lockedSeed() {
  const value = Math.floor(Number(el.seedInput.value));
  if (!Number.isFinite(value) || value < 0) return 0;
  return Math.min(value, 4294967295);
}

async function runBatch() {
  const text = el.prompt.value.trim();
  if (!text) {
    setHint('Write a prompt first.', true);
    el.prompt.focus();
    return;
  }
  state.genError = null;
  state.crashInfo = null;
  state.peek = false;
  state.truncatedWarning = false;

  if (state.engine === 'downloaded') {
    const ok = await startEngine();
    if (!ok) return;
  }
  if (state.engine !== 'ready') return;

  const { steps, count, seedLocked } = state.settings;
  const prompt = fullPrompt(text);
  const baseSeed = seedLocked ? lockedSeed() : null;
  state.stopRequested = false;
  await wake.acquire();

  for (let i = 0; i < count; i++) {
    if (state.stopRequested) break;
    const seed = baseSeed === null ? randomSeed() : (baseSeed + i) >>> 0;
    state.working = { index: i + 1, count, stage: 'prompt', step: 0, steps, model: null };
    markStage({ stage: 'prompt' });
    render();
    try {
      const result = await call('generate', { prompt, steps, seed });
      const frame = await library.add({
        blob: result.blob,
        prompt: text,
        style: state.settings.style,
        seed: result.seed,
        steps: result.steps,
        ms: Math.round(result.ms),
        at: Date.now(),
      });
      state.frames.unshift(frame);
      state.currentId = frame.id;
      state.freshId = frame.id;
      state.firstSinceLoad = false;
      state.lastTiming = result.timing;
      if (result.promptTruncated) state.truncatedWarning = true;
    } catch (error) {
      if (error.code !== 'cancelled') {
        state.genError = error;
        if (error.code === 'device_lost' || error.code === 'memory') state.firstSinceLoad = true;
        if (error.code === 'not_loaded') state.engine = 'downloaded';
      }
      break;
    }
  }

  state.working = null;
  markStage(null);
  wake.release();
  render();
}

async function stopBatch() {
  state.stopRequested = true;
  el.goButton.disabled = true;
  try { await call('cancel'); } catch { /* nothing running */ }
}

// ----------------------------------------------------------------- render

const ERROR_COPY = {
  offline: ['Download interrupted', 'The connection dropped. What you already have is kept, so you can carry on from here.'],
  http: ['The model host is not answering', 'Try again in a few minutes.'],
  quota: ['Not enough free space', 'The model needs 2.5 GB. Free up some storage on this phone, then try again.'],
  size_mismatch: ['The download did not match', 'A file came back with the wrong size. Try the download again.'],
  busy_other_tab: ['Open in another tab', 'Pocketframe is already open in another tab. Close that tab, then reload this one.'],
  busy: ['Already downloading', 'A download is already running.'],
  memory: ['Out of memory', 'The phone ran out of memory while loading the model. Close other apps and browser tabs, then try again.'],
  no_f16: ['Graphics chip not supported', 'Chrome on this device does not offer a graphics feature the model needs.'],
  no_storage: ['Storage unavailable', 'This browser cannot keep the model. Leave Incognito mode and open the page in Chrome.'],
  engine_load: ['The model would not load', 'The graphics chip refused the model. Reload the page and try again; if it keeps happening, copy the device report from the Engine panel.'],
  not_downloaded: ['The model is incomplete', 'Some model files are missing. Download them again.'],
  worker_crash: ['The engine stopped', 'Reload the page to start it again.'],
  unexpected: ['Something went wrong', 'Reload the page and try again.'],
};

const RETRY_LABEL = { download: 'Resume download', load: 'Try loading again', reload: 'Reload page' };

function checksHtml() {
  const caps = state.caps;
  if (!caps) return '';
  const line = (ok, text) => `<li class="${ok ? 'ok' : 'bad'}"><span class="mark" aria-hidden="true">${ok ? '✓' : '✕'}</span><span>${text}</span></li>`;
  const free = caps.quota ? Math.max(0, caps.quota - caps.usage) : 0;
  const rows = [];
  if (engineConfig.backend === 'wasm') {
    rows.push(line(true, 'Test mode: running on the processor'));
  } else {
    rows.push(line(caps.webgpu, caps.webgpu ? 'WebGPU is available' : 'WebGPU is missing in this browser'));
    if (caps.webgpu) rows.push(line(caps.adapter, caps.adapter ? `Graphics chip found${caps.gpu ? ` (${escapeHtml(caps.gpu)})` : ''}` : 'Chrome did not offer a graphics chip'));
    if (caps.adapter) rows.push(line(caps.f16, caps.f16 ? '16-bit shader maths supported' : '16-bit shader maths not supported'));
  }
  rows.push(line(caps.storage, caps.storage ? (free ? `Storage available (${formatBytes(free)} free for this site)` : 'Storage available') : 'This browser cannot store large files'));
  return `<ul class="checks">${rows.join('')}</ul>`;
}

function crashHintHtml() {
  return state.crashInfo
    ? `<p><strong>Last time the page closed while ${escapeHtml(describeStage(state.crashInfo))}.</strong> That usually means the device ran short of memory. Close other apps and browser tabs before trying again.</p>`
    : '';
}

function detailsHtml(detail) {
  if (!detail) return '';
  const text = typeof detail === 'string' ? detail : JSON.stringify(detail);
  return `<details><summary>Technical details</summary><pre>${escapeHtml(text)}</pre></details>`;
}

// Each view returns { key, html }. The print window is rebuilt only when the
// key changes, so progress updates never restart an animation.
function printView() {
  const total = state.totalBytes || 0;
  const frame = currentFrame();

  if (state.working) {
    const title = state.working.count > 1 ? `Photo ${state.working.index} of ${state.working.count}` : 'Making your photo';
    return {
      key: `work:${state.working.index}`,
      html: `<div class="print-note"><h2>${title}</h2><p id="workStage"></p>
        <div class="meter"><span id="workMeter"></span></div>
        ${state.firstSinceLoad ? '<p>The first photo takes the longest: the model has to load onto the graphics chip first.</p>' : ''}</div>`,
    };
  }

  if (state.crashInfo && !state.peek && state.engine !== 'checking' && state.engine !== 'unsupported') {
    return {
      key: 'crash',
      html: `<div class="print-note"><h2>The page closed unexpectedly last time</h2>
        <p>It happened while <strong>${escapeHtml(describeStage(state.crashInfo))}</strong>.</p>
        <p>That usually means the device ran short of memory. Close other apps and browser tabs, then generate again.</p>
        <button class="btn btn-paper-outline" type="button" data-action="dismiss-crash">OK</button></div>`,
    };
  }

  if (state.genError) {
    const err = state.genError;
    const lost = err.code === 'device_lost' || err.code === 'memory';
    return {
      key: `generr:${err.code}`,
      html: `<div class="print-note"><h2>${lost ? 'The graphics chip ran out of memory' : 'That photo failed'}</h2>
        <p>${lost ? 'Close other apps and tabs, then generate again. The model will reload first.' : 'Try generating again. If it keeps failing, copy the device report from the Engine panel.'}</p>
        ${detailsHtml(err.detail || err.message)}</div>`,
    };
  }

  if (state.peek && frame) return imageView(frame);

  switch (state.engine) {
    case 'checking':
      return { key: 'checking', html: '<div class="print-note"><h2>Checking this device</h2><p>One moment.</p></div>' };
    case 'unsupported': {
      const reasons = unsupportedReasons(state.caps);
      const advice = reasons.includes('webgpu') || reasons.includes('adapter')
        ? 'Open this page in an up-to-date Chrome on Android 12 or newer, or in Chrome or Edge on a computer.'
        : reasons.includes('f16')
          ? 'Updating Chrome sometimes adds the missing feature. Otherwise this device cannot run the model in a browser.'
          : 'Leave Incognito mode and open this page in a regular Chrome tab.';
      return {
        key: 'unsupported',
        html: `<div class="print-note"><h2>This browser cannot run Pocketframe</h2>${checksHtml()}<p>${advice}</p></div>`,
      };
    }
    case 'needs-model':
      return {
        key: 'needs-model',
        html: `<div class="print-note"><h2>Download the model once</h2>
          <p>${formatBytes(total)}, saved on this device. Use Wi-Fi. After that, photos are made right here and your prompts never leave the device.</p>
          ${checksHtml()}${crashHintHtml()}
          <button class="btn btn-ink" type="button" data-action="download">Download model (${formatBytes(total)})</button></div>`,
      };
    case 'paused':
      return {
        key: 'paused',
        html: `<div class="print-note"><h2>Download paused</h2>
          <p>${formatBytes(state.status?.have || 0)} of ${formatBytes(total)} is saved on this device.</p>
          <button class="btn btn-ink" type="button" data-action="download">Resume download</button></div>`,
      };
    case 'downloading':
      return {
        key: 'downloading',
        html: `<div class="print-note"><h2>Downloading the model</h2>
          <p>Keep this page open. If the connection drops, the download carries on from where it stopped.</p>
          <div class="meter"><span id="dlMeter"></span></div>
          <div class="meter-line"><span id="dlAmount"></span><span id="dlTime"></span></div>
          <p style="margin-top:14px"><button class="btn btn-paper-outline" type="button" data-action="pause">Pause</button></p></div>`,
      };
    case 'loading':
      return {
        key: 'loading',
        html: `<div class="print-note"><h2>Starting the engine</h2>
          <div class="meter indeterminate"><span></span></div>
          <p>The first time, this fetches about 30 MB of engine code.</p></div>`,
      };
    case 'error': {
      const err = state.engineError;
      const [title, body] = ERROR_COPY[err.code] || ERROR_COPY.unexpected;
      return {
        key: `error:${err.code}:${err.retry}`,
        html: `<div class="print-note"><h2>${title}</h2><p>${body}</p>
          <button class="btn btn-ink" type="button" data-action="retry">${RETRY_LABEL[err.retry] || 'Try again'}</button>
          ${detailsHtml(err.detail || err.message)}</div>`,
      };
    }
    default:
      break;
  }

  if (frame) return imageView(frame);

  return {
    key: 'empty',
    html: `<div class="print-note"><h2>Describe a photo to make your first one</h2>${crashHintHtml()}
      <p>Or start from one of these:</p>
      <div class="examples">${EXAMPLES.map((text, i) => `<button class="example" type="button" data-example="${i}">${escapeHtml(text)}</button>`).join('')}</div></div>`,
  };
}

function imageView(frame) {
  const fresh = state.freshId === frame.id;
  return {
    key: `image:${frame.id}`,
    html: `<img src="${library.url(frame)}" alt="${escapeHtml(frame.prompt)}" width="512" height="512"${fresh ? ' class="developing"' : ''}>`,
  };
}

let printKey = '';

function renderPrint() {
  const view = printView();
  if (view.key !== printKey) {
    printKey = view.key;
    el.printWindow.innerHTML = view.html;
    if (view.key.startsWith('image:')) state.freshId = null;
  }
  updateProgress();

  const showingImage = printKey.startsWith('image:');
  const frame = showingImage ? currentFrame() : null;
  el.printActions.hidden = !frame;
  el.usedPrompt.hidden = !frame;
  if (frame) {
    const style = STYLES.find((s) => s.id === frame.style);
    el.printCaption.innerHTML = `<span><strong>No. ${frame.id}</strong></span><span>seed ${frame.seed}, ${frame.steps} ${frame.steps === 1 ? 'step' : 'steps'}, ${formatSeconds(frame.ms)}</span>`;
    el.usedPrompt.textContent = style && style.id !== 'none' ? `${frame.prompt} (${style.name} style)` : frame.prompt;
  } else {
    el.printCaption.innerHTML = '<span>512 × 512</span><span>made on this device</span>';
  }
}

function updateProgress() {
  if (state.working) {
    const stageLine = $('workStage');
    const meter = $('workMeter');
    if (stageLine && meter) {
      const w = state.working;
      let text = 'Reading the prompt';
      let fraction = 0.04;
      if (w.model) {
        const name = MODEL_NAMES[w.model.key] || 'model';
        const share = w.model.total ? w.model.sent / w.model.total : 0;
        text = w.model.phase === 'read'
          ? `Preparing the ${name}`
          : `Loading the ${name}: ${formatBytes(w.model.sent)} of ${formatBytes(w.model.total)}`;
        fraction = Math.max(0.02, share);
      } else if (w.stage === 'denoise') {
        text = w.steps > 1 ? `Step ${w.step} of ${w.steps}` : 'Forming the image';
        fraction = 0.1 + (0.7 * (w.step - 1)) / w.steps;
      } else if (w.stage === 'decode') {
        text = 'Turning it into pixels';
        fraction = 0.82;
      }
      stageLine.textContent = text;
      meter.style.width = `${Math.round(fraction * 100)}%`;
    }
    return;
  }
  if (state.engine === 'downloading') {
    const { have, total, speed, retrying } = state.download;
    const meter = $('dlMeter');
    if (meter) {
      meter.style.width = `${total ? ((have / total) * 100).toFixed(1) : 0}%`;
      $('dlAmount').textContent = `${formatBytes(have)} of ${formatBytes(total)}`;
      $('dlTime').textContent = retrying ? 'Reconnecting' : timeLeft(total - have, speed);
    }
  }
}

function renderEngineButton() {
  let label = 'Checking';
  let lamp = 'idle';
  if (state.working) { label = 'Working'; lamp = 'busy'; }
  else {
    switch (state.engine) {
      case 'unsupported': label = 'Not supported'; lamp = 'problem'; break;
      case 'needs-model': label = 'Set up'; break;
      case 'paused': label = 'Paused'; break;
      case 'downloading': {
        const { have, total } = state.download;
        label = `Downloading ${total ? Math.floor((have / total) * 100) : 0}%`;
        lamp = 'busy';
        break;
      }
      case 'downloaded': label = 'Standing by'; break;
      case 'loading': label = 'Loading'; lamp = 'busy'; break;
      case 'ready': label = 'Ready'; lamp = 'ready'; break;
      case 'error': label = 'Problem'; lamp = 'problem'; break;
      default: break;
    }
  }
  el.engineLabel.textContent = label;
  el.engineButton.dataset.state = lamp;
}

function setHint(text, warn) {
  el.hint.textContent = text;
  el.hint.classList.toggle('warn', !!warn);
}

function renderDock() {
  const canGenerate = state.engine === 'ready' || state.engine === 'downloaded';
  if (state.working) {
    el.goButton.textContent = 'Stop';
    el.goButton.dataset.mode = 'stop';
    el.goButton.disabled = state.stopRequested;
  } else {
    el.goButton.textContent = 'Generate';
    el.goButton.dataset.mode = 'go';
    el.goButton.disabled = !canGenerate;
  }

  for (const button of el.stepsSeg.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset.value) === state.settings.steps));
  }
  for (const button of el.countSeg.querySelectorAll('button')) {
    button.setAttribute('aria-pressed', String(Number(button.dataset.value) === state.settings.count));
  }
  el.styleSelect.value = state.settings.style;
  el.seedToggle.setAttribute('aria-pressed', String(state.settings.seedLocked));
  el.seedToggle.textContent = state.settings.seedLocked ? 'Seed kept' : 'Random seed';
  el.seedInput.hidden = !state.settings.seedLocked;

  const style = STYLES.find((s) => s.id === state.settings.style);
  if (state.truncatedWarning) {
    setHint('That prompt was longer than the model can read, so the end was ignored.', true);
  } else if (state.engine === 'needs-model' || state.engine === 'paused') {
    setHint('Download the model to start generating.', false);
  } else if (state.engine === 'downloaded' && !state.working) {
    setHint('The model loads when you generate. That takes a moment.', false);
  } else if (style && style.suffix) {
    setHint(`${style.name} adds: ${style.suffix}`, false);
  } else if (state.settings.steps > 1) {
    setHint('More steps add detail and take longer.', false);
  } else {
    setHint('', false);
  }
}

function renderStrip() {
  el.stripSection.hidden = state.frames.length === 0;
  el.stripCount.textContent = state.frames.length === 1 ? '1 photo' : `${state.frames.length} photos`;
  const wanted = state.frames.map((f) => f.id).join(',');
  if (el.strip.dataset.ids !== wanted) {
    el.strip.dataset.ids = wanted;
    el.strip.innerHTML = state.frames.map((frame) =>
      `<button class="frame" type="button" data-frame="${frame.id}"><img src="${library.url(frame)}" alt="${escapeHtml(frame.prompt)}" loading="lazy" width="76" height="76"><span>No. ${frame.id}</span></button>`
    ).join('');
  }
  for (const button of el.strip.querySelectorAll('.frame')) {
    button.setAttribute('aria-current', String(Number(button.dataset.frame) === state.currentId && printKey.startsWith('image:')));
  }
}

function renderPanel() {
  const caps = state.caps || {};
  const free = caps.quota ? Math.max(0, caps.quota - caps.usage) : 0;
  const facts = [
    ['Runs on', engineConfig.backend === 'wasm' ? 'Processor (test mode)' : (caps.gpu || 'Graphics chip')],
    ['Engine', { ready: 'Started', loading: 'Starting', downloaded: 'Not loaded', downloading: 'Downloading model', paused: 'Download paused', 'needs-model': 'No model yet', unsupported: 'Not supported', error: 'Problem', checking: 'Checking' }[state.engine]],
    ['Photo size', '512 × 512'],
    ['Photos saved here', String(state.frames.length)],
    ['Build', BUILD],
  ];
  if (caps.memoryGb) facts.push(['Device memory', caps.memoryGb >= 8 ? '8 GB or more' : `${caps.memoryGb} GB`]);
  if (free) facts.push(['Free space for this site', formatBytes(free)]);
  if (state.lastTiming) {
    const t = state.lastTiming;
    const loading = t.load > 500 ? `model load ${formatSeconds(t.load)}, ` : '';
    facts.push(['Last photo', `prompt ${formatSeconds(t.prompt)}, ${loading}image ${formatSeconds(t.denoise)}, pixels ${formatSeconds(t.decode)}`]);
  }
  el.facts.innerHTML = facts.map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('');

  const have = state.status?.have || 0;
  const total = state.totalBytes || 0;
  el.modelLine.textContent = have === 0 ? 'Not downloaded.' : have >= total ? `${formatBytes(total)} saved.` : `${formatBytes(have)} of ${formatBytes(total)} downloaded.`;
  el.deleteModelButton.disabled = have === 0 || state.engine === 'downloading' || state.engine === 'loading' || !!state.working;
  el.unloadButton.disabled = state.engine !== 'ready' || !!state.working;
}

function render() {
  renderPrint();
  renderEngineButton();
  renderDock();
  renderStrip();
  if (el.enginePanel.open) renderPanel();
}

// ----------------------------------------------------------------- events

function saveSettings() {
  writeStorage(SETTINGS_KEY, JSON.stringify(state.settings));
}

function loadSettings() {
  try {
    const saved = JSON.parse(readStorage(SETTINGS_KEY) || 'null');
    if (!saved) return;
    if (STYLES.some((s) => s.id === saved.style)) state.settings.style = saved.style;
    if ([1, 2, 4].includes(saved.steps)) state.settings.steps = saved.steps;
    if ([1, 2, 4].includes(saved.count)) state.settings.count = saved.count;
    state.settings.seedLocked = !!saved.seedLocked;
    if (Number.isFinite(saved.seed)) state.settings.seed = saved.seed;
  } catch { /* start from defaults */ }
}

function autoGrow() {
  if (window.matchMedia('(min-width: 940px)').matches) return;
  el.prompt.style.height = 'auto';
  el.prompt.style.height = `${Math.min(el.prompt.scrollHeight + 2, window.innerHeight * 0.3)}px`;
}

function wireEvents() {
  el.styleSelect.innerHTML = STYLES.map((s) => `<option value="${s.id}">${escapeHtml(s.name)}</option>`).join('');

  el.dock.addEventListener('submit', (event) => {
    event.preventDefault();
    if (state.working) stopBatch();
    else runBatch();
  });
  el.prompt.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!state.working && !el.goButton.disabled) runBatch();
    }
  });
  el.prompt.addEventListener('input', () => {
    autoGrow();
    if (state.truncatedWarning) { state.truncatedWarning = false; renderDock(); }
  });

  el.styleSelect.addEventListener('change', () => { state.settings.style = el.styleSelect.value; saveSettings(); renderDock(); });
  el.stepsSeg.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    state.settings.steps = Number(button.dataset.value); saveSettings(); renderDock();
  });
  el.countSeg.addEventListener('click', (event) => {
    const button = event.target.closest('button');
    if (!button) return;
    state.settings.count = Number(button.dataset.value); saveSettings(); renderDock();
  });
  el.seedToggle.addEventListener('click', () => {
    state.settings.seedLocked = !state.settings.seedLocked;
    if (state.settings.seedLocked) {
      const frame = currentFrame();
      if (frame && !el.seedInput.value) el.seedInput.value = String(frame.seed);
      if (!el.seedInput.value) el.seedInput.value = String(state.settings.seed);
      state.settings.seed = lockedSeed();
    }
    saveSettings(); renderDock();
  });
  el.seedInput.addEventListener('change', () => {
    el.seedInput.value = String(lockedSeed());
    state.settings.seed = lockedSeed(); saveSettings();
  });

  el.printWindow.addEventListener('click', (event) => {
    const action = event.target.closest('[data-action]')?.dataset.action;
    const example = event.target.closest('[data-example]');
    if (example) {
      el.prompt.value = EXAMPLES[Number(example.dataset.example)];
      autoGrow();
      el.prompt.focus();
      return;
    }
    if (action === 'dismiss-crash') { state.crashInfo = null; render(); return; }
    if (action === 'download') startDownload();
    else if (action === 'pause') pauseDownload();
    else if (action === 'retry') {
      const retry = state.engineError?.retry;
      if (retry === 'download') startDownload();
      else if (retry === 'load') startEngine();
      else location.reload();
    }
  });

  el.strip.addEventListener('click', (event) => {
    const button = event.target.closest('[data-frame]');
    if (!button || state.working) return;
    state.currentId = Number(button.dataset.frame);
    state.peek = true;
    state.genError = null;
    render();
    el.printWindow.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });

  el.saveButton.addEventListener('click', () => {
    const frame = currentFrame();
    if (!frame) return;
    const link = document.createElement('a');
    link.href = library.url(frame);
    link.download = fileName(frame);
    document.body.append(link);
    link.click();
    link.remove();
    toast('Saved to Downloads');
  });

  const canShareFiles = (() => {
    try {
      return !!navigator.canShare && navigator.canShare({ files: [new File([new Blob(['x'])], 'x.png', { type: 'image/png' })] });
    } catch { return false; }
  })();
  el.shareButton.hidden = !canShareFiles;
  el.shareButton.addEventListener('click', async () => {
    const frame = currentFrame();
    if (!frame) return;
    try {
      await navigator.share({ files: [new File([frame.blob], fileName(frame), { type: 'image/png' })] });
    } catch { /* closed the share sheet */ }
  });

  el.reuseButton.addEventListener('click', () => {
    const frame = currentFrame();
    if (!frame) return;
    el.prompt.value = frame.prompt;
    state.settings.style = STYLES.some((s) => s.id === frame.style) ? frame.style : 'none';
    state.settings.steps = [1, 2, 4].includes(frame.steps) ? frame.steps : 1;
    state.settings.seedLocked = true;
    state.settings.seed = frame.seed;
    el.seedInput.value = String(frame.seed);
    saveSettings();
    autoGrow();
    renderDock();
    toast('Prompt, style and seed restored');
  });

  confirmTap(el.deleteButton, 'Delete photo?', async () => {
    const frame = currentFrame();
    if (!frame) return;
    await library.remove(frame.id);
    state.frames = state.frames.filter((f) => f.id !== frame.id);
    state.currentId = state.frames[0]?.id ?? null;
    if (!state.currentId) state.peek = false;
    render();
  });

  el.engineButton.addEventListener('click', async () => {
    await refreshStatus();
    renderPanel();
    el.enginePanel.showModal();
  });
  el.closePanel.addEventListener('click', () => el.enginePanel.close());
  el.enginePanel.addEventListener('click', (event) => {
    if (event.target === el.enginePanel) el.enginePanel.close();
  });
  el.unloadButton.addEventListener('click', async () => { await unloadEngine(); renderPanel(); toast('Memory freed'); });
  confirmTap(el.deleteModelButton, 'Delete 2.5 GB model?', deleteModel);
  el.reportButton.addEventListener('click', async () => {
    const report = {
      build: BUILD,
      page: location.origin + location.pathname,
      userAgent: navigator.userAgent,
      engine: state.engine,
      caps: state.caps,
      model: state.status ? { have: state.status.have, total: state.status.total } : null,
      lastTiming: state.lastTiming,
      engineError: state.engineError,
      generateError: state.genError,
      closedWhile: state.lastCrash ? { ...state.lastCrash, summary: describeStage(state.lastCrash) } : null,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(report, null, 2));
      toast('Device report copied');
    } catch {
      toast('Copying was blocked');
    }
  });
}

loadSettings();
wireEvents();
el.seedInput.value = String(state.settings.seed);
render();
boot();
