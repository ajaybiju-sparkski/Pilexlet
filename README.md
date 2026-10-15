# PixelChat

A simple chatbot that generates images from text prompts, running **100% locally**
on a MacBook Air M3 (8GB RAM). No external image-generation APIs — ever.

Right now it uses a pre-trained open model, **SD-Turbo** (via Hugging Face
`diffusers`), downloaded once and run on your Mac's GPU (`mps`). Later, this gets
swapped for a diffusion model trained from scratch, by changing a single setting.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> If your shell aliases `python` to something else (e.g. Xcode's Python), use
> `venv/bin/python` explicitly instead of relying on `source venv/bin/activate`.

## Run

```bash
venv/bin/python app.py
```

Then open **http://127.0.0.1:7860**. The first launch downloads SD-Turbo
(~2.5GB) from Hugging Face and shows "Loading model…" until it's ready; after
that it's cached locally and loads in a few seconds.

The first image generation after startup is slower (GPU warm-up). After that,
a 512×512 image takes a few seconds with the default 2 steps.

## Example prompts

- "a red apple on a wooden table"
- "a cat sleeping on a sofa, cartoon style"
- "a cyberpunk city at night, neon lights"

Then try a follow-up:

- "make it blue"
- "now at night"
- "add a hat"
- "bigger"

## Chat commands

- Any normal message → a **new** image prompt.
- A follow-up that looks like an edit (starts with "make it", "now", "add",
  "change", "but", "more", "less", "bigger", "smaller", or is a lone
  color/style word) → combined with the **previous** prompt.
- `/new` → clears the conversation context.
- `/help` → shows usage.
- **Regenerate** → same prompt, new random seed.
- **Steps** slider (1–4) and optional **Seed** box control each generation.

Every image is saved to `outputs/` as `YYYYmmdd-HHMMSS_seed.png`.

## Tests

```bash
venv/bin/python -m pytest -q
```

Tests use a fake, solid-color generator — they never download or run the real
model.

## Hardware notes

- Runs on PyTorch's `mps` backend (falls back to `cpu` if unavailable). No CUDA.
- The pipeline loads once at startup and is reused for every request.
- Uses `float16` + attention slicing on MPS to keep memory low. If an image
  comes out black or all-NaN (a known MPS/fp16 quirk), the VAE is upcast to
  `float32` and the generation is retried once automatically.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so any op MPS doesn't support yet
  falls back to CPU instead of crashing.

## SD-Turbo license

`stabilityai/sd-turbo` is licensed for **non-commercial / research use**. That's
fine for this personal prototype — if PixelChat is ever used beyond that, swap
in a model with a license that permits it.

## Swapping in our own model later

Everything in the app talks to the `ImageGenerator` interface
(`pixelchat/generators/base.py`), never to `diffusers` directly. The pretrained
SD-Turbo backend lives in `pixelchat/generators/pretrained.py`; our own model
(trained in later phases) lives in `pixelchat/generators/custom.py`.

Once our model is trained, switching to it is **one change**:

```python
# pixelchat/config.py
GENERATOR = "pixelchat"  # was "sd-turbo"
```

No other code — the UI, chat logic, and saving/logging — needs to change.
