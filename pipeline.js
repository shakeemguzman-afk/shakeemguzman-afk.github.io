// The three model passes behind every image: read the prompt, remove noise
// from a random latent, decode the latent into pixels. Kept free of browser
// APIs so the same code can be exercised outside a phone.

import { buildSchedule, seededNoise, VAE_SCALE } from './sampler.js';

function dispose(outputs) {
  for (const tensor of Object.values(outputs)) tensor.dispose?.();
}

/** @returns {{ data: Float32Array, dims: number[], used: number, truncated: boolean }} */
export async function encodePrompt(ort, session, tokenizer, prompt) {
  const { ids, used, truncated } = tokenizer.encode(prompt);
  const out = await session.run({ input_ids: new ort.Tensor('int32', ids, [1, ids.length]) });
  const hidden = out.last_hidden_state;
  const encoded = { data: Float32Array.from(hidden.data), dims: Array.from(hidden.dims), used, truncated };
  dispose(out);
  return encoded;
}

/**
 * Ancestral Euler sampling from pure noise down to a clean latent. With more
 * than one step, fresh noise is mixed back in between steps, which is how this
 * model was trained to be refined (plain Euler over-sharpens its output).
 * @param {object} options
 * @param {number} options.steps        1 to 4
 * @param {number} options.seed         32-bit seed; fixes every noise draw
 * @param {number} options.latentSize   image size / 8
 * @param {(step: number, steps: number) => void} [options.onStep]
 * @param {() => boolean} [options.shouldStop]
 * @returns {Promise<Float32Array|null>} latent ready for the decoder, or null if stopped
 */
export async function denoise(ort, session, encoded, options) {
  const { steps, seed, latentSize, onStep, shouldStop } = options;
  const { timesteps, sigmas } = buildSchedule(steps);
  const shape = [1, 4, latentSize, latentSize];
  const count = 4 * latentSize * latentSize;
  // One long seeded stream: the first slice is the starting noise, each later
  // slice is the noise mixed in after a step.
  const noise = seededNoise(count * timesteps.length, seed);
  const latents = new Float32Array(count);
  for (let i = 0; i < count; i++) latents[i] = noise[i] * sigmas[0];

  for (let s = 0; s < timesteps.length; s++) {
    if (onStep) onStep(s + 1, timesteps.length);
    const from = sigmas[s];
    const to = sigmas[s + 1];
    const scale = 1 / Math.sqrt(from * from + 1);
    const input = new Float32Array(count);
    for (let i = 0; i < count; i++) input[i] = latents[i] * scale;

    const out = await session.run({
      sample: new ort.Tensor('float32', input, shape),
      timestep: new ort.Tensor('int64', BigInt64Array.from([BigInt(timesteps[s])]), [1]),
      encoder_hidden_states: new ort.Tensor('float32', encoded.data, encoded.dims),
    });
    // The model predicts the noise in the latent. Step down to a slightly
    // lower noise level than the target, then add fresh noise to reach it.
    const predicted = out.out_sample.data;
    const up = Math.sqrt((to * to * (from * from - to * to)) / (from * from));
    const down = Math.sqrt(to * to - up * up);
    const dt = down - from;
    const fresh = (s + 1) * count;
    for (let i = 0; i < count; i++) {
      latents[i] += predicted[i] * dt;
      if (up > 0) latents[i] += noise[fresh + i] * up;
    }
    dispose(out);
    if (shouldStop && shouldStop()) return null;
  }

  for (let i = 0; i < count; i++) latents[i] /= VAE_SCALE;
  return latents;
}

/** @returns {Promise<{ rgba: Uint8ClampedArray, width: number, height: number }>} */
export async function decode(ort, session, latents, latentSize) {
  const out = await session.run({
    latent_sample: new ort.Tensor('float32', latents, [1, 4, latentSize, latentSize]),
  });
  const [, , height, width] = out.sample.dims;
  const src = out.sample.data;
  const plane = height * width;
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    rgba[i * 4] = (src[i] / 2 + 0.5) * 255;
    rgba[i * 4 + 1] = (src[plane + i] / 2 + 0.5) * 255;
    rgba[i * 4 + 2] = (src[2 * plane + i] / 2 + 0.5) * 255;
    rgba[i * 4 + 3] = 255;
  }
  dispose(out);
  return { rgba, width, height };
}
