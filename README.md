# PanelForge

A local, offline anime image-generation workbench built directly on
[diffusers](https://github.com/huggingface/diffusers). Text-to-image,
image-to-image and mask-based inpainting, with LoRA stacking and prompt
weighting, behind a small web UI and a CLI.

Nothing leaves the machine. Models are read from disk, images are written to
disk, and the server binds to loopback.

## Why it is built this way

The hardware-specific decisions live in exactly one module, `device.py`. It
resolves a `DevicePolicy` — backend, dtype, VAE dtype and which memory measures
to enable — and the rest of the codebase consults that policy instead of
branching on `torch.backends.mps` itself. Moving from Apple Silicon to an
NVIDIA box is a config change, not a rewrite.

## Desktop app (.dmg)

Build a native macOS app with everything inside it (Python, every dependency,
the UI and all shipped models) and wrap it in a disk image:

```bash
packaging/build_dmg.sh
```

That produces `dist/PanelForge-<version>.dmg` (about 9 GB: model weights
barely compress) holding `PanelForge.app` (about 11 GB). Open the image
and drag the app to Applications. The build takes about four minutes on an M2
and needs roughly the models' size plus 8 GB of free disk.

**What runs.** The same UI, hosted in a native WebKit window rather than a
browser tab. The API server still exists, but only in-process, on a random
loopback port chosen at launch. Exporting a page, saving an image and choosing
a training folder use native macOS panels.

**Offline by construction.** The app ships Animagine XL 4.0, the SDXL-Lightning
LoRA, ControlNet-openpose and the face detector, plus the SDXL component configs
that loading a single-file checkpoint needs, and it never contacts the network.
The built-in configs cover SDXL; a user-added SD 1.5 checkpoint would need its
config folder added to the bundle.

**Where things live.** The signed bundle is never written to. Outputs,
characters, projects, trained LoRAs and `config.toml` live in
`~/Library/Application Support/panelforge`. Checkpoints or LoRAs dropped
into its `models/` folder appear next to the shipped ones, and a file with the
same name overrides the shipped copy. Logs go to
`~/Library/Logs/panelforge/panelforge.log`.

**First launch on another Mac.** The app is ad-hoc signed (this machine has no
Apple Developer ID), so a copy that arrives by download or AirDrop is blocked
the first time. Open **System Settings → Privacy & Security** and choose
**Open Anyway**. A copy built on the same Mac opens normally.

**Checking a build.**

```bash
PANELFORGE_DESKTOP_SELFTEST=1 "build/stage/PanelForge.app/Contents/MacOS/panelforge"
```

This launches the real app, confirms it sees the bundled models, that the UI
loaded all four workspaces and that the native bridge is connected, then quits.
The log line starts with `SELFTEST PASS` or `SELFTEST FAIL`.

Packaging details that matter if you change the build:

- The interpreter is copied into `Contents/MacOS` and started with `PYTHONHOME`
  pointing at the bundled runtime, so macOS attributes the process to the app
  (name, Dock icon, bundle id) instead of to "python".
- Dependencies are installed with `--link-mode=copy`: signing rewrites binaries
  in place, and hardlinks would carry those edits back into uv's cache.
- uv marks its interpreters externally managed; the build removes that marker
  from its private copy only.
- pywebview silently ignores script-driven `window.open` and disables downloads
  by default, which is why exports go through the native bridge.

## Install

```bash
uv sync
```

## Quick start

```bash
uv run panelforge info
```

Shows the resolved hardware policy and lists what is in `models/`. Then:

```bash
uv run panelforge serve
```

and open <http://127.0.0.1:8188>.

## Models

Drop files into these directories — `.safetensors` from Civitai or Hugging Face
work as-is, and diffusers-format directories are also accepted:

```
models/
├── checkpoints/   full models (SD 1.5 or SDXL)
├── loras/         LoRA adapters
├── vae/           standalone VAEs
└── embeddings/    textual inversions
```

Architecture is detected by reading the safetensors header only, so the model
list stays instant regardless of how many multi-gigabyte checkpoints you have.
LoRAs are matched to the loaded checkpoint's architecture; an SD 1.5 LoRA on an
SDXL model produces noise rather than an error, so the UI hides incompatible
ones.

**First run needs network access once.** Loading a single-file checkpoint
requires the component configs (small JSON files) from a reference repo. They
are cached afterwards. Set `offline = true` in `config.toml` once that has
happened to guarantee nothing reaches the network.

## CLI

```bash
uv run panelforge generate "1girl, silver hair, (masterpiece:1.2)" --count 4
```

```bash
uv run panelforge generate "cyberpunk street, neon" --lora sdxl_lightning_8step:1.0 --steps 8 --cfg 1.5
```

Every PNG carries its generation parameters in an A1111-compatible `parameters`
text chunk, so the images drop into other tooling and remain self-describing.

## Prompt weighting

`(tag:1.2)` emphasis and prompts longer than CLIP's 77-token window are handled
by [compel](https://github.com/damian0815/compel). If a transformers release
breaks compel's use of the text-encoder internals, generation falls back to
plain prompts with a logged warning rather than failing the job.

## Distilled checkpoints (Lightning / Turbo / LCM)

These need very different settings from a normal model:

| | steps | CFG | sampler |
|---|---|---|---|
| standard SDXL | 25–30 | 5–7 | `dpmpp_2m_karras` |
| SDXL-Lightning 8-step | 8 | 1.0–2.0 | `euler_trailing` |
| SDXL-Turbo | 1–4 | 0.0 | `euler_trailing` |

Leaving CFG at 6 with a Lightning LoRA produces burnt, oversaturated output.
The `euler_trailing` sampler exists specifically for these — distilled models
are trained on a trailing timestep schedule and look washed out without it.

## Hardware notes

`config.toml` can force any of this, but the defaults are:

- **Apple Silicon (MPS)** — fp16 UNet and text encoders. The SDXL VAE overflows
  in fp16, so decodes are upcast to fp32 for the decode step only. VAE tiling is
  on under 24 GB; attention slicing only under 12 GB, because on an M2 it cost
  21% per step and the SDPA kernels are already memory-efficient. CPU offload is
  used only while a ControlNet is resident, where it keeps weights out of the MPS
  allocator pool (see Hardware findings).
- **CUDA** — bf16 on Ampere and newer, which has fp32's exponent range and
  sidesteps the fp16 overflow entirely. Below 12 GB VRAM, model CPU offload;
  below 8 GB, sequential offload.
- **Seeds are portable.** Latents are always sampled on a CPU generator, so a
  seed means the same thing on MPS and CUDA.

## Layout

```
src/panelforge/
├── device.py      hardware policy — the only place that knows about MPS/CUDA
├── config.py      config.toml + env
├── models.py      model discovery, header-only architecture detection
├── samplers.py    scheduler presets, LCM, v-prediction handling
├── prompting.py   compel weighting with a safe fallback
├── pipelines.py   pipeline cache and variants, LoRA stacking, hi-res, face pass
├── enhance.py     face detection, masks, degenerate-output guard
├── pose.py        OpenPose skeleton rendering and pose presets
├── comic.py       page layouts, panels, speech bubbles, lettering
├── characters.py  character library and prompt composition
├── sheets.py      character reference sheets
├── styles.py      style presets (extensible via styles.json)
├── training.py    LoRA training with cached latents and embeddings
├── jobs.py        single-worker queue, progress, cancellation
├── server.py      FastAPI + static UI
├── cli.py         typer CLI (info, serve, generate, train)
└── web/           vanilla HTML/CSS/JS, no build step
```

## Security

The UI has no authentication. `panelforge serve` binds to `127.0.0.1`; binding
anywhere else warns, because anyone who can reach the port can drive the model
and read every image in `outputs/`.

## Feature comparison with anifusion.ai

PanelForge is an independent project. It is not affiliated with, endorsed by or
connected to anifusion.ai; the table compares features only.

Built and verified locally:

| anifusion.ai | PanelForge |
|---|---|
| Text-to-image | yes — SDXL / SD 1.5, LoRA stacking, prompt weighting |
| Inpaint | yes — mask-painting canvas |
| Image-to-image | yes |
| Panel layouts (presets + custom) | yes — 11 presets, LTR and RTL (manga) reading order |
| Text-to-manga (per-panel prompts) | yes — one queued generation per panel, auto-sized to the panel's aspect |
| Speech bubbles | yes — speech, thought, shout, whisper, narration, SFX, with draggable tails |
| Vertical text + manga fonts | yes — vertical CJK columns right-to-left |
| Style selector | yes — 10 presets, extensible via `styles.json` |
| Characters | yes — reusable cast with tags, LoRAs and a locked seed |
| High-resolution export | yes — hi-res second pass |
| Canvas editing | yes — click panels, drag bubbles, live server-rendered preview |
| Projects | yes — save/load page JSON, export page PNG |
| Pose control | yes — ControlNet-openpose with 10 editable skeleton presets |
| Face enhance | yes — YOLO face detection then per-face re-inpainting |
| Character sheets | yes — turnaround / expressions / full / outfit grids from one locked seed |
| LoRA training | yes — train character and style LoRAs on your own images |
| Model variety (LCM, v-prediction) | yes — LCM scheduler, automatic v-pred zero-SNR handling |
| Credits, gallery, contests, referrals | deliberately omitted — this is on-prem |

Not built: FLUX models (22GB, impractical on 16GB unified memory), video and
animation, and the infinite-canvas editor.

## Pose control

`models/controlnet/openpose-sdxl/` holds the ControlNet. Poses are rendered
from keypoints rather than estimated from a photo, so no pose *estimator* is
needed and the pose stays editable:

```bash
curl -s -X POST localhost:8188/api/pose/render \
  -H 'Content-Type: application/json' \
  -d '{"preset":"running","width":832,"height":1216,"mirror":true}'
```

Presets: standing, arms-up, waving, pointing, arms-crossed, running, sitting,
fighting, walking, bust. Custom keypoints can be posted directly.

## LoRA training

```bash
uv run panelforge train ~/datasets/my-character \
  --name my-character --trigger mychar --steps 1000 --rank 16 --resolution 768
```

20-30 images, optional `.txt` captions beside each image (kohya convention).
Latents and text embeddings are cached before training so the VAE and both text
encoders can be freed — that is what makes SDXL LoRA training fit in 16GB.
Measured at ~4 s/step at 512px on an M2, so a 1000-step run is roughly 70
minutes. Output lands in `models/loras/` and is picked up immediately.

## Character sheets

`POST /api/character-sheet` renders one image per view from the character's
locked seed, then `POST /api/character-sheet/compose` lays them out as a
labelled grid (also available in the Cast workspace). Presets: `turnaround`
(front, 3/4, side, back), `expressions`, `full`, `outfit`.

What decides the quality, measured on Animagine XL 4.0:

- **Each cell is prompted as a single view.** Putting `character sheet` or
  `reference sheet` in a cell prompt made the model draw a whole multi-view sheet
  inside every cell; those tags are now negated instead.
- **Describe the character completely.** With only "silver hair, blue eyes,
  school uniform", hair length, skirt colour and socks changed between views.
  Adding `short hair, bob cut, navy pleated skirt, black kneehighs, brown
  loafers` made all four views match.
- **Back views fight the model's bias toward showing the face.** Even with a
  weighted `(from behind:1.4)` and `looking back` negated, the back cell still
  tends to turn and look over the shoulder. For a true back view, use pose
  control with a skeleton seen from behind, or train a character LoRA.

## Face enhance

Detects faces with `models/detectors/face_yolov8m.pt`, crops each with padding,
re-renders it, and composites it back through a feathered mask.

- **Mode** (`face_mode`, default `inpaint`). `inpaint` regenerates a feathered
  ellipse over the face and keeps the hair and background in the padded margin
  verbatim, so the crop edge cannot drift. `img2img` refines the whole crop.
- **Work size** (`face_work_size`, default `0` = the model's native resolution:
  1024 for SDXL, 512 for SD 1.5). Measured on an M2 with an 85px face: at 512px
  SDXL fragmented the face into zig-zag strokes in every mode and strength; at
  768 and 1024 the passes were clean, 1024 the clearest. Cost per face on that
  machine: ~28 s at 512, ~80 s at 768, ~200 s at 1024. Pass
  `"face_work_size": 768` to trade some quality for speed.
- **Strength** defaults to 0.4. Denoising runs `steps x strength` real steps, so
  the nominal step count is scaled by `1/strength`.
- Framing tags (`full body`, `wide shot`, ...) are stripped from the face prompt.
- **Honest reporting.** A face whose re-render raises, or comes back collapsed,
  white or black relative to its input, is left untouched and reported in the
  job's `face_result`. Failed GPU passes on MPS do not always raise.

## Hardware findings (measured on an M2 / 16 GB)

Numbers from this machine, not estimates. They are why the defaults are what
they are.

| Change / workload | Result |
|---|---|
| Attention slicing on vs off, SDXL 1024² | 17.04 vs 14.08 s/step — slicing costs **21%** |
| LoRA fused vs unfused | 19.06 vs 12.76 s/step — fusing saves **33%** |
| SDXL txt2img 768², 12 steps | 44 s |
| SDXL-Lightning 832×1216, 8 steps | ~95 s |
| ControlNet + Lightning, 832×1216 | ~114 s |
| Hi-res pass 512×768 → 768×1152, Lightning | 151 s |
| Face enhance, per face | ~28 s @512 · ~80 s @768 · ~200 s @1024 |
| LoRA training, 512px, rank 8 | ~4 s/step (≈70 min for 1000 steps) |

What cost real debugging time:

- **`Pipeline.from_pipe()` recasts the shared model in place.** Building the
  img2img variant with `from_pipe` flipped the *base* pipeline's UNet, VAE and
  both text encoders from fp16 to fp32 and took MPS allocation from 6.48 GB to
  12.99 GB. Past that point MPS returned all-zero tensors without raising —
  compel embeddings and VAE encodes were zeros — so face enhance and hi-res
  decoded to solid white, and later txt2img jobs inherited the fp32 model and ran
  in swap. Every variant (img2img, inpaint, ControlNet) is now built with the
  pipeline constructor from `base.components`, and a guard restores and logs any
  precision change.
- **The shared VAE's dtype must be handed back after a custom decode.** txt2img's
  own decode only upcasts safely when it finds the VAE in fp16; left in fp32,
  MPS casts it down, and an fp16 SDXL decode is NaN in every element.
- **CPU offload helps on MPS, for a non-obvious reason.** Offloaded weights stay
  on the same physical chips but leave the MPS allocator pool, whose watermark
  (20.13 GB here) is what runs out. Used only while a ControlNet is resident.
- **Sustained generation drives the machine into swap**, not thermal throttling,
  so only back-to-back A/B timings are trustworthy.
- **Stale UI JavaScript.** The UI is plain files with no content hashes, and a
  cached `studio.js` once hid a new button's handler. UI assets are now served
  with `Cache-Control: no-cache` (cheap 304 revalidation). If you are upgrading
  from a build without it, hard-refresh the page once.

## Licence

The PanelForge source code is released under the MIT licence, copyright (c) 2026
Sovereign Edge Inc. See [LICENSE](LICENSE).

Model weights are not part of this repository and are not covered by that
licence. The models the app is built and tested with carry their own terms:

| Model | Licence |
|---|---|
| Animagine XL 4.0 (`cagliostrolab/animagine-xl-4.0`) | CreativeML Open RAIL++-M |
| SDXL-Lightning LoRA (`ByteDance/SDXL-Lightning`) | CreativeML Open RAIL++-M |
| ControlNet openpose SDXL (`xinsir/controlnet-openpose-sdxl-1.0`) | Apache-2.0 |
| Face detector (`Bingsu/adetailer`, `face_yolov8m.pt`) | Apache-2.0 |

A `.dmg` built with `packaging/build_dmg.sh` bundles those weights, so anyone
distributing that image must also comply with each model's licence.
