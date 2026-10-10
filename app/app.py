"""Iris-3B Studio: text-to-image, monocular depth, and image restoration / 4x upscaling.

A local Gradio UI around the official Iris-3B code (cloned to ./iris-3b, used unmodified).
Weights are fetched from the Hugging Face Hub the first time each task is used, so
someone who only wants depth never downloads the text-to-image model.
"""

import gc
import json
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IRIS = ROOT / "iris-3b"
DEMO = IRIS / "demo"
OUTPUTS = ROOT / "outputs"
SETTINGS_FILE = ROOT / "settings.json"
OUTPUTS.mkdir(exist_ok=True)
os.chdir(ROOT)  # the text encoder caches its null embedding under ./output
sys.path.append(str(DEMO))  # the official demo's before/after slider and point-cloud viewer

import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*removed in Gradio 6.0.*")  # we pin gradio<6

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402
from viewers import CSS as VIEWER_CSS  # noqa: E402
from viewers import HEAD_JS, cloud_html, slider_html  # noqa: E402

import iris3b.text  # noqa: E402,F401  (registers the text encoders)
from iris3b.config import inference_config  # noqa: E402
from iris3b.downstream.depth import DepthPredictor, colorize  # noqa: E402
from iris3b.downstream.restoration import Restorer, fit_budget, tile_positions  # noqa: E402
from iris3b.flow.solver import FlowDPMSolver  # noqa: E402
from iris3b.models.dit import IrisDiT  # noqa: E402
from iris3b.registry import TEXT_ENCODERS  # noqa: E402
from iris3b.sampling import load_for_inference  # noqa: E402

MODEL_REPO = os.environ.get("IRIS_MODEL_REPO", "speridlabs/iris-3b")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
CUDA = DEVICE.type == "cuda"
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if CUDA else 0.0
RAM_GB = psutil.virtual_memory().total / 2**30

# (height, width) buckets with real training mass; rarer training buckets show artifacts
SIZES = {f"{w}×{h}": (h, w) for h, w in [(768, 1344), (832, 1280), (896, 1152), (1024, 1024),
                                          (1152, 896), (1280, 832), (1344, 768)]}
PALETTES = ["inferno", "magma", "viridis", "plasma", "turbo", "Spectral", "gray"]

# ---------------------------------------------------------------------------------------------
# Settings (persisted to settings.json so they survive restarts)
# ---------------------------------------------------------------------------------------------

MEMORY_MODES = {
    "Auto": "Keep everything on the GPU if it has 48 GB+, otherwise one task at a time.",
    "Keep all on GPU": "Fastest switching between tabs. Needs ~45 GB VRAM at FP32 (~25 GB at BF16).",
    "Swap to RAM": "Only the active task lives on the GPU; others wait in system RAM (~45 GB RAM at FP32).",
    "Unload": "Only the active task is loaded at all; switching tabs reloads from disk. Lowest RAM use.",
}
PRECISIONS = {
    "Auto": "FP32 weights on GPUs with 24 GB+ (reference quality), BF16 below that.",
    "FP32": "Reference: FP32 weights with BF16 autocast, exactly like the official code. ~12 GB per model.",
    "BF16": "Half the memory (~6 GB per model). Tiny numerical differences from the reference.",
}
DEFAULT_SETTINGS = {"memory": "Auto", "precision": "Auto", "low_vram_text_encoder": "Auto"}


def load_settings() -> dict:
    try:
        return {**DEFAULT_SETTINGS, **json.loads(SETTINGS_FILE.read_text())}
    except (OSError, ValueError):
        return dict(DEFAULT_SETTINGS)


SETTINGS = load_settings()


def memory_mode() -> str:
    mode = SETTINGS["memory"]
    if mode == "Auto":
        return "Keep all on GPU" if VRAM_GB >= 48 else "Unload"
    return mode


def weight_dtype() -> torch.dtype:
    if not CUDA:  # the official code only autocasts on CUDA, so other devices stay FP32
        return torch.float32
    choice = SETTINGS["precision"]
    if choice == "Auto":
        return torch.float32 if VRAM_GB >= 23 else torch.bfloat16
    return torch.bfloat16 if choice == "BF16" else torch.float32


def text_encoder_offload() -> bool:
    """Park the 4B text encoder in RAM between prompts on small GPUs."""
    choice = SETTINGS["low_vram_text_encoder"]
    if choice == "Auto":
        return CUDA and VRAM_GB < 24
    return choice == "On"


# ---------------------------------------------------------------------------------------------
# Model manager: lazy download + load, one GPU lock, memory policy between tasks
# ---------------------------------------------------------------------------------------------

GPU_LOCK = threading.RLock()
STOP = threading.Event()


def autocast():
    return torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=CUDA)


def fetch(patterns: list[str]) -> Path:
    if os.path.isdir(MODEL_REPO):
        return Path(MODEL_REPO)
    print(f"[iris] fetching {patterns} from {MODEL_REPO} (first use only, ~12 GB per model)...", flush=True)
    return Path(snapshot_download(MODEL_REPO, allow_patterns=patterns, token=os.environ.get("HF_TOKEN")))


class T2I:
    """Base Iris-3B text-to-image model plus its Qwen3-VL-4B text encoder."""

    def __init__(self, dtype: torch.dtype):
        source = fetch(["config.yaml", "model.safetensors"])
        raw, weights = load_for_inference(source)
        self.cfg = inference_config(raw, [])
        with torch.device("meta"):
            model = IrisDiT(self.cfg.model)
        model.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.model = model.eval().requires_grad_(False).to(DEVICE, dtype)
        print("[iris] loading text encoder (Qwen/Qwen3-VL-4B-Instruct, downloaded on first use)...", flush=True)
        self.te_home = torch.device("cpu") if text_encoder_offload() else DEVICE
        self.text_encoder = TEXT_ENCODERS.build(self.cfg.text_encoder.name, self.cfg.text_encoder,
                                                device=self.te_home)
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def to(self, device):
        self.model.to(device)
        self.text_encoder.decoder.to(device if self.te_home != torch.device("cpu") else "cpu")
        self.text_encoder.device = torch.device(device if self.te_home != torch.device("cpu") else "cpu")
        return self

    @torch.no_grad()
    def encode(self, text: str, negative: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Prompt embeddings, cached on the CPU so re-rolling a seed skips the text encoder."""
        key = ("neg:" if negative else "pos:") + text
        if key not in self.cache:
            te = self.text_encoder
            if self.te_home.type == "cpu" and CUDA:
                te.decoder.to(DEVICE)
                te.device = DEVICE
            try:
                enc = te.null(text) if negative else te.encode([text])
                self.cache[key] = (enc.embeddings.cpu(), enc.mask.cpu())
            finally:
                if self.te_home.type == "cpu" and CUDA:
                    te.decoder.to("cpu")
                    te.device = torch.device("cpu")
                    torch.cuda.empty_cache()
            if len(self.cache) > 32:
                self.cache.pop(next(iter(self.cache)))
        emb, mask = self.cache[key]
        return emb.to(DEVICE), mask.to(DEVICE)


class Manager:
    def __init__(self):
        self.models: dict[str, object] = {}
        self.where: dict[str, str] = {}

    def _build(self, task: str):
        dtype = weight_dtype()
        if task == "t2i":
            return T2I(dtype)
        sub = "depth" if task == "depth" else "upscaler"
        path = fetch([f"{sub}/*"]) / sub
        obj = (DepthPredictor if task == "depth" else Restorer)(str(path), device="cpu")
        obj.model.to(dtype=dtype)
        return obj.to(DEVICE)

    def _evict(self, task: str, mode: str):
        if mode == "Swap to RAM":
            if self.where.get(task) != "cpu":
                self.models[task].to("cpu")
                self.where[task] = "cpu"
        else:
            del self.models[task]
            self.where.pop(task, None)
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()

    def get(self, task: str):
        mode = memory_mode()
        if mode != "Keep all on GPU":
            for other in [t for t in self.models if t != task]:
                print(f"[iris] {'moving' if mode == 'Swap to RAM' else 'unloading'} {other} to free VRAM", flush=True)
                self._evict(other, mode)
        if task not in self.models:
            t0 = time.time()
            print(f"[iris] loading {task} model...", flush=True)
            self.models[task] = self._build(task)
            print(f"[iris] {task} ready in {time.time() - t0:.0f}s", flush=True)
        elif self.where.get(task) == "cpu":
            self.models[task].to(DEVICE)
        self.where[task] = DEVICE.type
        return self.models[task]

    def unload_all(self):
        for task in list(self.models):
            del self.models[task]
        self.where.clear()
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()


MANAGER = Manager()


def status_md() -> str:
    lines = [f"**Device:** {DEVICE.type.upper()}"
             + (f" · {torch.cuda.get_device_name(0)} · {VRAM_GB:.0f} GB VRAM" if CUDA else ""),
             f"**System RAM:** {RAM_GB:.0f} GB",
             f"**Memory mode in effect:** {memory_mode()} · **weights:** "
             f"{'BF16' if weight_dtype() == torch.bfloat16 else 'FP32'} · **text encoder parked in RAM:** "
             f"{'yes' if text_encoder_offload() else 'no'}"]
    if CUDA:
        lines.append(f"**VRAM in use by Iris:** {torch.cuda.memory_allocated() / 2**30:.1f} GB "
                     f"(reserved {torch.cuda.memory_reserved() / 2**30:.1f} GB)")
    loaded = ", ".join(f"{t} ({w})" for t, w in MANAGER.where.items()) or "none yet: models load on first use"
    lines.append(f"**Loaded models:** {loaded}")
    return "  \n".join(lines)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def save_png(image: Image.Image, task: str, stem: str, meta: dict | None = None) -> str:
    folder = OUTPUTS / task
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{stem}.png"
    info = None
    if meta:
        info = PngInfo()
        info.add_text("parameters", json.dumps(meta, ensure_ascii=False))
    image.save(path, pnginfo=info)
    return str(path)


def to_pil(x: torch.Tensor) -> Image.Image:
    """[3, H, W] in [-1, 1] -> PIL."""
    pixels = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels)


def center_crop_16(image: Image.Image, height: int, width: int) -> torch.Tensor:
    """Cover-resize and center-crop the init image to the generation size, as [1, 3, H, W] in [-1, 1]."""
    image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (width, height), Image.Resampling.LANCZOS)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 127.5 - 1
    return x.to(DEVICE)


# ---------------------------------------------------------------------------------------------
# Text to image (and experimental image-to-image)
# ---------------------------------------------------------------------------------------------


@torch.no_grad()
def sample_one(t2i: T2I, cond, cond_mask, uncond, null_mask, height, width, steps, cfg_scale, seed,
               init=None, strength=1.0, on_step=None):
    """The official FlowDPM-Solver++ loop (iris3b.sampling.generate), unrolled so it can report
    progress, stop early, and start part-way down the schedule for image-to-image."""
    cfg, model = t2i.cfg, t2i.model
    model_dtype = next(model.parameters()).dtype
    cfg_mask = torch.cat([null_mask, cond_mask], dim=0) if uncond is not None else None

    def model_fn(x, t_model, y):
        return model(x.to(model_dtype), t_model, y, y_mask=cond_mask if y is cond else cfg_mask).x.float()

    solver = FlowDPMSolver(model_fn, num_timesteps=cfg.flow.num_train_timesteps, cfg_scale=cfg_scale,
                           cfg_interval=tuple(cfg.sample.cfg_interval), prediction=cfg.flow.prediction)
    total = steps
    if init is not None:
        # pixel-space img2img: run exactly `steps` steps over the last `strength` share of a stretched
        # schedule, so the step count you set is the step count you get
        total = max(steps, math.floor(steps / max(strength, 1e-3) + 0.5))
    grid = solver.time_grid(total, cfg.flow.shift)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    noise = torch.randn(1, model.cfg.in_channels, height, width, generator=generator, device=DEVICE)
    start = total - steps
    if init is not None:
        sigma = grid[start]
        x = (1 - sigma) * init.float() + sigma * noise
    else:
        x = noise
    history = []
    for i in range(start + 1, total + 1):
        if STOP.is_set():
            break
        s, t = grid[i - 1], grid[i]
        x0 = solver._pred_x0(x, s, cond, uncond)
        history.append((s, x0))
        step_order = min(i - start, cfg.sample.order, total + 1 - i)
        x = solver._first_order(x, s, t, x0) if step_order == 1 else solver._second_order(x, history[-2], history[-1], t)
        if len(history) > 2:
            history.pop(0)
        if on_step:
            on_step(i - start, steps, x0)
    return x.clamp(-1, 1)[0]


def run_t2i(prompt, negative, size, steps, cfg_scale, seed, randomize, count, init_image, strength,
            progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Write a prompt first.")
    STOP.clear()
    height, width = SIZES[size]
    seed = random.randint(0, 2**31 - 1) if randomize else int(seed)
    images, paths = [], []
    with GPU_LOCK:
        progress(0, desc="Loading text-to-image model (first run downloads ~21 GB)...")
        t2i = MANAGER.get("t2i")
        progress(0, desc="Encoding prompt...")
        cond, cond_mask = t2i.encode(prompt, negative=False)
        uncond = null_mask = None
        if cfg_scale != 1.0:
            uncond, null_mask = t2i.encode(negative or "", negative=True)
        init = center_crop_16(init_image, height, width) if init_image is not None else None
        for n in range(int(count)):
            if STOP.is_set():
                break
            this_seed = seed + n

            def on_step(i, total, x0, n=n):
                progress((n * total + i) / (int(count) * total), desc=f"Image {n + 1}/{int(count)} · step {i}/{total}")

            t0 = time.time()
            with autocast():
                x = sample_one(t2i, cond, cond_mask, uncond, null_mask, height, width, int(steps), cfg_scale,
                               this_seed, init, strength, on_step)
            if STOP.is_set():
                break
            image = to_pil(x)
            meta = {"prompt": prompt, "negative_prompt": negative, "size": size, "steps": int(steps),
                    "cfg": cfg_scale, "seed": this_seed, "model": "speridlabs/iris-3b"}
            if init is not None:
                meta["img2img_strength"] = strength
            paths.append(save_png(image, "txt2img", str(this_seed), meta))
            images.append((image, f"seed {this_seed} · {time.time() - t0:.1f}s"))
            yield images, seed, f"Saved to `{paths[-1]}`", paths[-1]
    if STOP.is_set():
        gr.Info("Stopped.")
    if not images:
        yield images, seed, "Stopped before the first image finished.", None


# ---------------------------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------------------------


def color_depth(depth: np.ndarray, palette: str, near_bright: bool) -> Image.Image:
    if palette == "gray":
        low, high = np.percentile(depth, [2, 98])
        near = np.clip((high - depth) / max(high - low, 1e-6), 0, 1)
        if not near_bright:
            near = 1 - near
        return Image.fromarray((near * 255).round().astype(np.uint8)).convert("RGB")
    image = colorize(depth if near_bright else -depth, palette)
    return image


def depth16(depth: np.ndarray) -> Image.Image:
    """16-bit grayscale, near = white, 2nd-98th percentile stretch (for Blender, ControlNet, etc.)."""
    low, high = np.percentile(depth, [2, 98])
    near = np.clip((high - depth) / max(high - low, 1e-6), 0, 1)
    return Image.fromarray((near * 65535).round().astype(np.uint16))


def run_depth(image, max_side, palette, near_bright, show_cloud, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    with GPU_LOCK:
        progress(0.1, desc="Loading depth model (first run downloads ~12 GB)...")
        model = MANAGER.get("depth")
        progress(0.6, desc="Estimating depth...")
        t0 = time.time()
        depth = model(image, max_side=int(max_side))
    colored = color_depth(depth, palette, near_bright)
    stem = f"depth_{palette}"
    color_path = save_png(colored, "depth", stem)
    gray_path = color_path.replace(f"_{stem}.png", "_depth16.png")
    depth16(depth).save(gray_path)
    npy_path = color_path.replace(f"_{stem}.png", "_logdepth.npy")
    np.save(npy_path, depth.astype(np.float32))
    note = (f'<div class="sl-meta"><span>Depth · {time.time() - t0:.1f}s</span><span>{image.width}×{image.height} · '
            f'relative (not metric) · {"near = bright" if near_bright else "far = bright"}</span></div>')
    cloud = cloud_html(depth, image) if show_cloud else ""
    return (slider_html(image, colored, "Photo", "Depth"), note, cloud, colored,
            [color_path, gray_path, npy_path])


# ---------------------------------------------------------------------------------------------
# Restoration and upscaling
# ---------------------------------------------------------------------------------------------


def plan_upscale(image, limit, short_side, long_side):
    if image is None:
        return ""
    small = fit_budget(image, int(short_side), int(long_side)) if limit else image
    w, h = small.width * 4, small.height * 4
    tile = 1024
    if min(w, h) <= tile:
        r = tile / min(w, h)
        w, h = max(tile, round(w * r)), max(tile, round(h * r))
    tiles = len(tile_positions(h, tile, 512)) * len(tile_positions(w, tile, 512))
    return (f"Input {image.width}×{image.height} → model input {small.width}×{small.height} → "
            f"4× output {small.width * 4}×{small.height * 4} · **{tiles} tile{'s' if tiles != 1 else ''}** of 1024²")


def run_upscale(image, limit, short_side, long_side, out_scale, color_fix, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    small = fit_budget(image, int(short_side), int(long_side)) if limit else image
    with GPU_LOCK:
        progress(0.02, desc="Loading restoration model (first run downloads ~12 GB)...")
        restorer = MANAGER.get("upscale")
        done = [0]
        out_w, out_h = small.width * 4, small.height * 4
        if min(out_w, out_h) <= restorer.tile:
            r = restorer.tile / min(out_w, out_h)
            out_w, out_h = max(restorer.tile, round(out_w * r)), max(restorer.tile, round(out_h * r))
        total = len(tile_positions(out_h, restorer.tile, restorer.tile // 2)) * len(
            tile_positions(out_w, restorer.tile, restorer.tile // 2))
        original = Restorer.restore_tile.__get__(restorer)

        def counted(x):
            out = original(x)
            done[0] += 1
            progress(done[0] / total, desc=f"Restoring tile {done[0]}/{total}")
            return out

        restorer.restore_tile = counted
        t0 = time.time()
        try:
            result = restorer(small, color_fix=color_fix)
        finally:
            del restorer.restore_tile
    factor = {"4× (native)": 4, "2×": 2, "3×": 3, "1× (restore only)": 1}[out_scale]
    if factor != 4:
        target = (round(small.width * factor), round(small.height * factor))
        result = result.resize(target, Image.Resampling.LANCZOS)
    path = save_png(result, "upscale", f"x{factor}")
    note = (f'<div class="sl-meta"><span>Restored · {time.time() - t0:.1f}s</span><span>{small.width}×{small.height} → '
            f"{result.width}×{result.height}</span></div>")
    return slider_html(small, result, "Input", "Iris-3B"), note, result, path


# ---------------------------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------------------------


def apply_settings(memory, precision, te):
    changed_precision = precision != SETTINGS["precision"] or te != SETTINGS["low_vram_text_encoder"]
    SETTINGS.update(memory=memory, precision=precision, low_vram_text_encoder=te)
    SETTINGS_FILE.write_text(json.dumps(SETTINGS, indent=2))
    with GPU_LOCK:
        if changed_precision:
            MANAGER.unload_all()  # takes effect on the next load
        elif memory_mode() != "Keep all on GPU":
            for task in list(MANAGER.models):
                MANAGER._evict(task, memory_mode())
    gr.Info("Settings saved.")
    return status_md()


def unload():
    with GPU_LOCK:
        MANAGER.unload_all()
    return status_md()


# ---------------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------------

PRESETS = json.loads((DEMO / "examples/presets.json").read_text())
DEPTH_EXAMPLES = ["1", "1048", "1006", "57", "70", "128", "10", "46"]
UPSCALE_EXAMPLES = ["1", "57", "20", "157", "152", "79", "71", "116"]

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
footer { display: none !important; }
.sl-meta { display: flex; justify-content: space-between; gap: 16px; padding-top: 6px;
  font-family: ui-monospace, monospace; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; opacity: .75; }
.sl-meta span:first-child { color: #d94a1f; }
:root { --sl-rule: rgba(127,127,127,.3); }
#hero h1 { margin-bottom: 0; }
#hero p { margin-top: 4px; opacity: .8; }
.presets .caption-label { display: none; }
"""

HERO = """
<div id="hero">
<h1>Iris-3B Studio</h1>
<p><b>Pixel-space diffusion, no VAE.</b> One 3B backbone: text-to-image, monocular depth, and image restoration / 4× upscaling.
Each tab downloads its own weights (~12 GB) the first time you use it.
<a href="https://huggingface.co/speridlabs/iris-3b" target="_blank">Model</a> ·
<a href="https://github.com/speridlabs/iris-3b" target="_blank">Code</a> ·
<a href="https://arxiv.org/abs/2610.09450" target="_blank">Paper</a></p>
</div>
"""

first = PRESETS[0]
with gr.Blocks(title="Iris-3B Studio", css=CSS + VIEWER_CSS, head=HEAD_JS,
               theme=gr.themes.Soft(primary_hue="orange")) as demo:
    gr.HTML(HERO)
    with gr.Tabs() as tabs:
        # ---------------- Text to image ----------------
        with gr.Tab("Text to Image", id="t2i"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    prompt = gr.Textbox(label="Prompt", lines=4, value=first["prompt"])
                    with gr.Row():
                        gen_btn = gr.Button("Generate", variant="primary", scale=3)
                        stop_btn = gr.Button("Stop", variant="stop", scale=1)
                    with gr.Row():
                        size = gr.Dropdown(list(SIZES), value=first["size"], label="Size (width × height)")
                        count = gr.Slider(1, 8, value=1, step=1, label="Images")
                    with gr.Accordion("Settings", open=True):
                        negative = gr.Textbox(label="Negative prompt", value="")
                        steps = gr.Slider(10, 150, value=50, step=1, label="Steps",
                                          info="Official samples use 100; 50 is a good speed/quality balance.")
                        cfg_scale = gr.Slider(1.0, 10.0, value=first["cfg"], step=0.1, label="Guidance (CFG)")
                        with gr.Row():
                            seed = gr.Number(value=first["seed"], precision=0, label="Seed")
                            randomize = gr.Checkbox(value=True, label="Random seed")
                    with gr.Accordion("Image to image (experimental)", open=False):
                        gr.Markdown("Starts sampling from your image instead of pure noise. Iris has **no trained "
                                    "image conditioning**; this is noise-and-denoise in pixel space, the same "
                                    "approach as ComfyUI-iris. The image is cover-cropped to the chosen size.")
                        init_image = gr.Image(type="pil", label="Init image (leave empty for text-to-image)", height=240)
                        strength = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Strength",
                                             info="How much of the image is re-noised. 0.25 = subtle, 1.0 = ignore image. "
                                                  "Always runs the full step count above.")
                with gr.Column(scale=7):
                    gallery = gr.Gallery(value=[(str(DEMO / first["image"]), "preset")], label="Result",
                                         columns=2, height=640, object_fit="contain", format="png")
                    gen_info = gr.Markdown("")
                    last_path = gr.State(None)
                    with gr.Row():
                        to_depth = gr.Button("Send to Depth")
                        to_upscale = gr.Button("Send to Upscale")
            gr.Markdown("### Presets: click to load the prompt and the exact settings used for the official gallery")
            presets = gr.Gallery([str(DEMO / p["image"]) for p in PRESETS], columns=8, height="auto",
                                 object_fit="cover", allow_preview=False, show_label=False, elem_classes="presets")

        # ---------------- Depth ----------------
        with gr.Tab("Monocular Depth", id="depth"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    depth_in = gr.Image(type="pil", label="Input image", height=360)
                    depth_btn = gr.Button("Estimate depth", variant="primary")
                    with gr.Accordion("Settings", open=True):
                        max_side = gr.Slider(0, 2048, value=1024, step=64, label="Max side (px)",
                                             info="Long side is downscaled to this for the model; 0 = native. "
                                                  "The depth map is returned at the input resolution.")
                        palette = gr.Dropdown(PALETTES, value="inferno", label="Palette")
                        near_bright = gr.Checkbox(value=True, label="Near = bright")
                        show_cloud = gr.Checkbox(value=True, label="Show 3D point cloud")
                    gr.Examples([[str(DEMO / f"examples/{n}.webp")] for n in DEPTH_EXAMPLES], inputs=[depth_in],
                                label="Examples")
                with gr.Column(scale=7):
                    depth_slider = gr.HTML()
                    depth_note = gr.HTML()
                    depth_cloud = gr.HTML()
                    with gr.Accordion("Depth map & downloads", open=False):
                        depth_img = gr.Image(type="pil", label="Colorized depth", format="png")
                        depth_files = gr.File(label="Colorized PNG · 16-bit PNG (near = white) · raw log-depth .npy",
                                              file_count="multiple")
            gr.Markdown("Output is **relative log depth** (affine-invariant, not metric): good for 3D parallax, "
                        "ControlNet, compositing and relighting; for metric depth fit a scale and shift.")

        # ---------------- Restoration / upscaling ----------------
        with gr.Tab("Restore & Upscale", id="upscale"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=5):
                    up_in = gr.Image(type="pil", label="Low-quality / low-resolution image", height=360)
                    up_btn = gr.Button("Restore & upscale", variant="primary")
                    up_plan = gr.Markdown()
                    with gr.Accordion("Settings", open=True):
                        limit = gr.Checkbox(value=True, label="Limit input size (recommended)",
                                            info="The model was trained on 256→1024 crops. Big inputs are "
                                                 "downscaled first so the job stays within ~21 tiles.")
                        with gr.Row():
                            short_side = gr.Slider(128, 1024, value=512, step=32, label="Max short side (px)")
                            long_side = gr.Slider(256, 2048, value=1024, step=32, label="Max long side (px)")
                        out_scale = gr.Radio(["4× (native)", "3×", "2×", "1× (restore only)"], value="4× (native)",
                                             label="Output size",
                                             info="The model always works at 4×; other sizes resample its result.")
                        color_fix = gr.Checkbox(value=True, label="Wavelet colour fix",
                                                info="Keeps the input's colours; disable only if colours look wrong.")
                    gr.Examples([[str(DEMO / f"examples/lowres/{n}.png")] for n in UPSCALE_EXAMPLES],
                                inputs=[up_in], label="Examples (degraded 256px images)")
                with gr.Column(scale=7):
                    up_slider = gr.HTML()
                    up_note = gr.HTML()
                    with gr.Accordion("Full-resolution result", open=False):
                        up_img = gr.Image(type="pil", label="Result", format="png")
                        up_file = gr.File(label="PNG")

        # ---------------- Settings ----------------
        with gr.Tab("Settings", id="settings"):
            status = gr.Markdown(status_md())
            with gr.Row():
                mem = gr.Radio(list(MEMORY_MODES), value=SETTINGS["memory"], label="Memory mode",
                               info=" · ".join(f"{k}: {v}" for k, v in MEMORY_MODES.items()))
            with gr.Row():
                prec = gr.Radio(list(PRECISIONS), value=SETTINGS["precision"], label="Weight precision",
                                info=" · ".join(f"{k}: {v}" for k, v in PRECISIONS.items()))
                te = gr.Radio(["Auto", "On", "Off"], value=SETTINGS["low_vram_text_encoder"],
                              label="Park text encoder in RAM between prompts",
                              info="Saves ~8 GB VRAM during sampling; costs ~1 s per new prompt. Auto = on below 24 GB.")
            with gr.Row():
                save_btn = gr.Button("Save settings", variant="primary")
                unload_btn = gr.Button("Unload all models")
                refresh_btn = gr.Button("Refresh status")
            gr.Markdown(f"Outputs are saved to `{OUTPUTS}`. Weights are cached by Hugging Face Hub "
                        "(Pinokio's shared cache) and are not re-downloaded on reinstall.")

    # ---------------- events ----------------
    t2i_inputs = [prompt, negative, size, steps, cfg_scale, seed, randomize, count, init_image, strength]
    gen_evt = gen_btn.click(run_t2i, t2i_inputs, [gallery, seed, gen_info, last_path], api_name="generate")
    sub_evt = prompt.submit(run_t2i, t2i_inputs, [gallery, seed, gen_info, last_path], api_name=False)
    stop_btn.click(lambda: STOP.set(), None, None, queue=False, api_name=False)

    def pick(evt: gr.SelectData):
        p = PRESETS[evt.index]
        return (p["prompt"], p["size"], p["steps"], p["cfg"], p["seed"], False,
                [(str(DEMO / p["image"]), f"preset · seed {p['seed']} · {p['steps']} steps · CFG {p['cfg']}")])

    presets.select(pick, None, [prompt, size, steps, cfg_scale, seed, randomize, gallery], api_name=False)

    def send(path):
        if not path:
            raise gr.Error("Generate an image first.")
        return Image.open(path)

    to_depth.click(send, last_path, depth_in, api_name=False).then(lambda: gr.Tabs(selected="depth"), None, tabs, api_name=False)
    to_upscale.click(send, last_path, up_in, api_name=False).then(lambda: gr.Tabs(selected="upscale"), None, tabs, api_name=False)

    depth_btn.click(run_depth, [depth_in, max_side, palette, near_bright, show_cloud],
                    [depth_slider, depth_note, depth_cloud, depth_img, depth_files], api_name="depth")

    up_inputs = [up_in, limit, short_side, long_side]
    for comp in up_inputs:
        comp.change(plan_upscale, up_inputs, up_plan, api_name=False, show_progress="hidden")
    up_btn.click(run_upscale, up_inputs + [out_scale, color_fix], [up_slider, up_note, up_img, up_file],
                 api_name="upscale")

    save_btn.click(apply_settings, [mem, prec, te], status, api_name=False)
    unload_btn.click(unload, None, status, api_name=False)
    refresh_btn.click(status_md, None, status, api_name=False)
    tabs.select(status_md, None, status, api_name=False)

if __name__ == "__main__":
    print(f"[iris] device={DEVICE} vram={VRAM_GB:.0f}GB ram={RAM_GB:.0f}GB memory={memory_mode()} "
          f"weights={'bf16' if weight_dtype() == torch.bfloat16 else 'fp32'}", flush=True)
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", allowed_paths=[str(OUTPUTS)])
