# Pocketframe

A small photo generator that runs on your own device, in the browser. The model
is downloaded once (2.5 GB) and kept in the browser's private storage. After
that, prompts and photos never leave the device.

## What it needs

- Chrome on Android 12 or newer, or Chrome or Edge on a computer
- WebGPU with 16-bit shader support (the page checks this and tells you)
- About 2.5 GB of free storage and a phone with plenty of memory (12 GB is a sensible minimum)

## Putting it online

Pocketframe is plain static files with no build step. Upload this folder to any
host that serves files over HTTPS (GitHub Pages, Netlify, Cloudflare Pages) and
open `index.html`. It will not run from a file opened directly on a phone.

On GitHub Pages: create a public repository named `<your-username>.github.io`,
add these files at the top level, and open `https://<your-username>.github.io/`.

## Files

| File | Job |
| --- | --- |
| `index.html` | The page and its styles |
| `app.js` | Interface logic and the photo library (IndexedDB) |
| `engine.worker.js` | Model download with resume, loading, generation |
| `pipeline.js` | The three model passes: prompt, denoise, decode |
| `sampler.js` | Noise schedule and seeded noise |
| `clip-tokenizer.js` | Prompt tokenizer |

## Model and licence

Pocketframe runs [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo) by
Stability AI, using the browser conversion at
`schmuell/sd-turbo-ort-web`. The model's licence covers personal and research
use only, not commercial use. Photos are 512 × 512.

The engine is [ONNX Runtime Web](https://onnxruntime.ai/) 1.31.1, loaded from
jsDelivr.

## Testing without a graphics chip

`?local=1&backend=wasm` runs the engine on the processor against a local
`models/sd-turbo/` folder and a local `ortdist/` copy of ONNX Runtime Web. It is
slow and only meant for checking the page flow.
