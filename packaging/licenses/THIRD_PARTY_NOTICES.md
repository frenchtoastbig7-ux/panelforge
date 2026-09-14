# Third-party notices

The PanelForge source code is licensed under the MIT licence, copyright (c) 2026
Sovereign Edge Inc. (see `PanelForge-LICENSE.txt`).

The macOS app built by `packaging/build_dmg.sh` also redistributes the model
weights below. They are included **unmodified**, exactly as published by their
authors, and remain under their own licences, not under the MIT licence.

| Component | Source | Licence | Licence text |
|---|---|---|---|
| Animagine XL 4.0 (`animagine-xl-4.0.safetensors`) | https://huggingface.co/cagliostrolab/animagine-xl-4.0 | CreativeML Open RAIL++-M | `CreativeML-Open-RAIL++-M.md` |
| SDXL-Lightning 8-step LoRA (`sdxl_lightning_8step_lora.safetensors`) | https://huggingface.co/ByteDance/SDXL-Lightning | CreativeML Open RAIL++-M | `CreativeML-Open-RAIL++-M.md` |
| ControlNet openpose SDXL (`controlnet/openpose-sdxl`) | https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0 | Apache-2.0 | `Apache-2.0.txt` |
| YOLOv8 face detector (`face_yolov8m.pt`) | https://huggingface.co/Bingsu/adetailer | Apache-2.0 | `Apache-2.0.txt` |
| SDXL component configs (`configs/sdxl`) | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | CreativeML Open RAIL++-M | `CreativeML-Open-RAIL++-M.md` |

`CreativeML-Open-RAIL++-M.md` is the licence text as published with
SDXL-Lightning, so its header names ByteDance Inc. as licensor for that model.
Animagine XL 4.0 (by Cagliostro Research Lab) and the SDXL configs (by Stability
AI) are distributed by their own authors under the same CreativeML Open
RAIL++-M terms; each model's licensor is the author listed at its source.

The Open RAIL++-M licence includes use-based restrictions (its Attachment A).
Anyone using or redistributing these models must follow them.

The app also bundles a Python runtime and third-party Python packages. Each
package's own licence files are kept in its `*.dist-info` folder inside the app
(`Contents/Resources/python/lib/python3.12/site-packages`).
