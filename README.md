# Iris-3B Studio (Pinokio)

A one-click [Pinokio](https://pinokio.co) launcher and local Gradio UI for
[speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b), a 3B-parameter diffusion transformer that
works directly in **pixel space** (no VAE). One backbone, three tasks:

| Tab | What it does | Weights (downloaded on first use) |
|---|---|---|
| **Text to Image** | Prompt → ~1 MP image, 7 preset sizes or any custom aspect ratio, CFG, seeds, batches, 24 official presets. Optional *experimental* image-to-image. | `model.safetensors` (~12 GB) + Qwen3-VL-4B text encoder (~9 GB) |
| **Monocular Depth** | Photo → relative depth: before/after slider, interactive 3D point cloud, colorized PNG, 16-bit PNG, raw `.npy` | `depth/` (~12 GB) |
| **Restore & Upscale** | Degraded / low-res image → restored 4× image (tiled, wavelet colour fix), optional 1×/2×/3× output | `upscaler/` (~12 GB) |

Everything runs on `127.0.0.1`. The official [iris-3b code](https://github.com/speridlabs/iris-3b) is cloned
into `app/iris-3b` and used unmodified; this repo only adds the launcher and the UI in `app/app.py`.
The image-to-image approach and the "restore only" output sizing follow
[ComfyUI-iris](https://github.com/ScryptHunter/ComfyUI-iris).

## Requirements

- **NVIDIA GPU** (Windows / Linux) recommended. Approximate VRAM:
  - 24 GB+: everything at reference FP32 precision, one task loaded at a time.
  - 12–16 GB: switch to *BF16* weights (Settings tab, or leave on *Auto*); the text encoder is parked in RAM between prompts.
  - 48 GB+: *Auto* keeps all three models on the GPU for instant tab switching.
- **Apple Silicon**: runs on MPS in FP32 (needs a lot of unified memory; slow). **AMD Linux**: ROCm 6.4 build.
  **CPU-only**: works but is very slow. **Intel Macs are not supported** (no PyTorch ≥ 2.7.1 build).
- Disk: ~5 GB for the environment, plus up to ~45 GB of weights if you use all three tabs.

## Install & use

1. In Pinokio, open this repo's URL (or **Download from URL**) → **Install**. The installer clones Iris-3B,
   creates a Python 3.11 env with `uv`, installs PyTorch 2.8 for your GPU and the inference dependencies.
2. Click **Start**, then **Open Web UI**.
3. Pick a tab. The first run of each tab downloads its weights; watch progress in the **Terminal**.
4. Results are saved to `app/outputs/<task>/` (sidebar → **Outputs**). Text-to-image PNGs carry the prompt,
   seed, and settings in their metadata.

### Tips

- **Text to Image**: the official gallery uses 100 steps / CFG 3; 50 steps is a good default. Click a preset to
  reproduce it exactly.
- **Custom sizes**: pick *Custom…* and type any width × height (e.g. 2560×1080). Iris was trained at ~1 MP
  with aspect ratios from 1:4 to 4:1; above ~1 MP its output breaks up into a visible 16 px block grid. So the
  app renders the **largest size for your aspect ratio that fits ~1 MP** (2560×1080 → 1568×656,
  1920×1080 → 1360×768) and shows it below the fields. For a bigger final image, use **Send to Upscale**.
- **Image to image** is pixel-space noise-and-denoise, not trained image conditioning. It uses the same
  prompt, size, steps, CFG and seed as text to image and always runs the full step count. Strength 0.25–0.5
  keeps the composition. **Keep init image's aspect ratio** (on by default) generates at the largest clean size
  with the init image's shape, ignoring the Size setting, so nothing is cropped. Turn it off to cover-crop the
  image to the chosen size.
- **Depth** is *relative* log depth (affine-invariant), not metric. The 16-bit PNG (near = white) is ready for
  Blender displacement, ControlNet, or compositing.
- **Restore & Upscale** is trained for 256 → 1024 (4×). Leave *Limit input size* on for big photos;
  "1× (restore only)" cleans an image without changing its size.
- **Settings** tab: memory mode (Auto / Keep all on GPU / Swap to RAM / Unload), FP32 vs BF16 weights,
  text-encoder parking, and an *Unload all models* button.

Set `IRIS_MODEL_REPO` to a local folder (with `config.yaml`, `model.safetensors`, `depth/`, `upscaler/`) to use
pre-downloaded weights, and `HF_TOKEN` if you need authenticated downloads.

## API

The UI is a Gradio app, so every tab is also an HTTP API. Replace the URL with the one shown in Pinokio.
Full live schema: open the UI and click **Use via API** at the bottom.

| Endpoint | Inputs (in order) | Outputs |
|---|---|---|
| `/generate` | prompt, negative_prompt, size (`"1024×1024"`, `"832×1280"`, …, or `"Custom…"`), steps, cfg, seed, random_seed (bool), images (1–8), init_image (image or `null`), strength, width, height (used when size is `"Custom…"`), keep_aspect (bool, image to image only) | gallery, seed used, info (includes the saved PNG path) |
| `/depth` | image, max_side (0 = native), palette (`inferno`, `magma`, `viridis`, `plasma`, `turbo`, `Spectral`, `gray`), near_bright (bool), point_cloud (bool) | slider HTML, info HTML, point-cloud HTML, colorized depth image, [colorized PNG, 16-bit PNG, `.npy`] |
| `/upscale` | image, limit_input (bool), max_short_side, max_long_side, output_size (`"4× (native)"`, `"3×"`, `"2×"`, `"1× (restore only)"`), color_fix (bool) | slider HTML, info HTML, result image, saved PNG path |

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860/")

# text to image
gallery, seed, info = client.predict(
    "a red fox sleeping in fresh snow, golden hour", "", "1024×1024", 50, 3.0, 0, True, 1, None, 0.5, 1024, 1024, True,
    api_name="/generate")
print(gallery[0]["image"], "seed:", seed)

# custom aspect ratio (21:9) → rendered at 1568×656
gallery, seed, info = client.predict(
    "a misty mountain lake at sunrise", "", "Custom…", 50, 3.0, 0, True, 1, None, 0.5, 2560, 1080, True,
    api_name="/generate")

# depth
*_, depth_image, files = client.predict(
    handle_file("photo.jpg"), 1024, "inferno", True, False, api_name="/depth")
print(files)  # colorized PNG, 16-bit PNG, raw log-depth .npy

# restore & upscale
*_, result, path = client.predict(
    handle_file("small.png"), True, 512, 1024, "4× (native)", True, api_name="/upscale")
print("saved:", path)
```

### JavaScript

```js
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860/");

const gen = await client.predict("/generate", [
  "a red fox sleeping in fresh snow, golden hour", "", "1024×1024", 50, 3.0, 0, true, 1, null, 0.5, 1024, 1024, true,
]);
console.log(gen.data[0][0].image.url, gen.data[2]); // first image, info with the saved PNG path

const photo = await (await fetch("https://example.com/photo.jpg")).blob();
const depth = await client.predict("/depth", [handle_file(photo), 1024, "inferno", true, false]);
console.log(depth.data[4]); // output files

const up = await client.predict("/upscale", [handle_file(photo), true, 512, 1024, "4× (native)", true]);
console.log(up.data[3]);
```

### curl

Gradio's REST API is two calls: POST to get an `event_id`, then GET the result stream.

```bash
URL=http://127.0.0.1:7860

# text to image
EVENT=$(curl -s -X POST $URL/gradio_api/call/generate -H "Content-Type: application/json" \
  -d '{"data": ["a red fox sleeping in fresh snow, golden hour", "", "1024×1024", 50, 3.0, 0, true, 1, null, 0.5, 1024, 1024, true]}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['event_id'])")
curl -sN $URL/gradio_api/call/generate/$EVENT

# depth / upscale take an image as {"path": "<url or uploaded path>"}
EVENT=$(curl -s -X POST $URL/gradio_api/call/depth -H "Content-Type: application/json" \
  -d '{"data": [{"path": "https://example.com/photo.jpg"}, 1024, "inferno", true, false]}' \
  | python -c "import sys, json; print(json.load(sys.stdin)['event_id'])")
curl -sN $URL/gradio_api/call/depth/$EVENT
```

## Files

```
install.js   clone Iris-3B → Python 3.11 env (uv) → PyTorch (torch.js) → requirements
start.js     python app.py; captures the local URL for "Open Web UI"
update.js    git pull this launcher and app/iris-3b, refresh dependencies
reset.js     delete app/env and app/iris-3b (weights stay in the Hugging Face cache)
torch.js     PyTorch 2.8 per platform (CUDA 12.8 / ROCm 6.4 / MPS / CPU)
app/app.py   the Gradio UI
```

## Credits & license

Iris-3B model and code © [Speridlabs](https://speridlabs.com), Apache-2.0
([paper](https://arxiv.org/abs/2610.09450)). Text encoder: [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
(Apache-2.0). Image-to-image and restoration sizing ideas from
[ComfyUI-iris](https://github.com/ScryptHunter/ComfyUI-iris) (Apache-2.0). This launcher is Apache-2.0.
