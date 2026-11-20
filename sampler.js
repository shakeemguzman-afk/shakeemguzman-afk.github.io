// Sampling maths for SD-Turbo: the noise schedule the model was trained on,
// "trailing" timestep spacing, and seeded starting noise.

const TRAIN_STEPS = 1000;
const BETA_START = 0.00085;
const BETA_END = 0.012;

/** Latent scale the image decoder expects. */
export const VAE_SCALE = 0.18215;

let sigmaTable = null;

function allSigmas() {
  if (sigmaTable) return sigmaTable;
  const table = new Float64Array(TRAIN_STEPS);
  const lo = Math.sqrt(BETA_START);
  const hi = Math.sqrt(BETA_END);
  let alphaProduct = 1;
  for (let i = 0; i < TRAIN_STEPS; i++) {
    const root = lo + ((hi - lo) * i) / (TRAIN_STEPS - 1);
    alphaProduct *= 1 - root * root;
    table[i] = Math.sqrt((1 - alphaProduct) / alphaProduct);
  }
  sigmaTable = table;
  return table;
}

/**
 * @param {number} steps 1 to 4
 * @returns {{ timesteps: number[], sigmas: number[] }} sigmas has one extra
 * entry: the final 0 the last step lands on.
 */
export function buildSchedule(steps) {
  const count = Math.max(1, Math.min(4, Math.round(steps)));
  const table = allSigmas();
  const timesteps = [];
  const sigmas = [];
  for (let i = 0; i < count; i++) {
    const t = Math.round(TRAIN_STEPS - (i * TRAIN_STEPS) / count) - 1;
    timesteps.push(t);
    sigmas.push(table[t]);
  }
  sigmas.push(0);
  return { timesteps, sigmas };
}

/** Small fast seeded generator, uniform in [0, 1). */
export function mulberry32(seed) {
  let t = seed >>> 0;
  return function next() {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal noise; the same seed always gives the same values. */
export function seededNoise(length, seed) {
  const uniform = mulberry32(seed);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const u = Math.max(uniform(), 1e-12);
    const v = uniform();
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  return out;
}

export function randomSeed() {
  const box = new Uint32Array(1);
  crypto.getRandomValues(box);
  return box[0];
}
