#!/bin/bash
# Build "PanelForge.app" and a .dmg containing it: the Python runtime,
# every dependency, the app, and the models, fully offline.
#
#   packaging/build_dmg.sh
#
# Output: dist/PanelForge-<version>.dmg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="PanelForge"
EXEC_NAME="panelforge"
BUNDLE_ID="inc.sovereignedge.panelforge"
VERSION="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$ROOT/pyproject.toml" | head -1)"
BUILD="$ROOT/build"
STAGE="$BUILD/stage"
APP="$STAGE/$APP_NAME.app"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
DIST="$ROOT/dist"
DMG="$DIST/PanelForge-$VERSION.dmg"
T0=$SECONDS

step() { printf '\n==> %s  (+%ss)\n' "$*" "$((SECONDS - T0))"; }
die()  { printf '\nerror: %s\n' "$*" >&2; exit 1; }

# ------------------------------------------------------------------ preflight
command -v uv >/dev/null || die "uv is required"
[ "$(uname -m)" = arm64 ] || die "build on Apple Silicon: the bundled torch is arm64-only"
MODELS=(checkpoints/animagine-xl-4.0.safetensors
        loras/sdxl_lightning_8step_lora.safetensors
        controlnet/openpose-sdxl/config.json
        controlnet/openpose-sdxl/diffusion_pytorch_model.safetensors
        detectors/face_yolov8m.pt)
for f in "${MODELS[@]}"; do [ -f "$ROOT/models/$f" ] || die "missing models/$f"; done
SDXL_CONFIG="${PANELFORGE_SDXL_CONFIG:-$(ls -d "$HOME"/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/*/ 2>/dev/null | head -1)}"
[ -n "$SDXL_CONFIG" ] && [ -f "$SDXL_CONFIG/model_index.json" ] \
  || die "SDXL config folder not found; run the app once from source, or set PANELFORGE_SDXL_CONFIG"
MODELS_GB=$(( $(du -sk "$ROOT/models" | cut -f1) / 1024 / 1024 ))
NEED_GB=$(( MODELS_GB + 8 ))      # the .dmg holds a full copy; app staging uses APFS clones
FREE_GB=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
[ "$FREE_GB" -ge "$NEED_GB" ] || die "need ~${NEED_GB} GB free, have ${FREE_GB} GB"

# ---------------------------------------------------------------------- build
step "Clean"
rm -rf "$BUILD" "$DMG"
mkdir -p "$CONTENTS/MacOS" "$RES" "$DIST"

step "Python runtime"
PYBASE="$(cd "$(dirname "$(readlink -f "$ROOT/.venv/bin/python3")")/.." && pwd)"
cp -R "$PYBASE" "$RES/python"
# uv marks its interpreters externally managed; this private copy is ours to fill.
rm -f "$RES/python/lib/python3.12/EXTERNALLY-MANAGED"
PY="$RES/python/bin/python3.12"
env -u PYTHONPATH -u PYTHONHOME "$PY" -s -B -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

step "Dependencies, pinned from uv.lock"
( cd "$ROOT" && uv export --frozen --no-dev --no-hashes --no-emit-project --format requirements-txt ) > "$BUILD/requirements.txt"
# --link-mode=copy is essential: codesign rewrites Mach-O files in place, and
# hardlinks back into uv's cache would carry those edits into the cache.
uv pip install --python "$PY" --break-system-packages --link-mode=copy --no-progress -r "$BUILD/requirements.txt"
( cd "$ROOT" && uv build --wheel -o "$BUILD/wheel" >/dev/null )
uv pip install --python "$PY" --break-system-packages --link-mode=copy --no-deps --no-progress "$BUILD"/wheel/*.whl
env -u PYTHONPATH -u PYTHONHOME "$PY" -s -B -c \
  'import panelforge.desktop, panelforge.server, webview, torch, diffusers; print("  imports ok: torch", torch.__version__)'

step "Bytecode (the signed bundle is read-only at runtime)"
find "$RES/python" -name __pycache__ -type d -prune -exec rm -rf {} +
# A few vendored test files don't compile; that doesn't affect the app.
env -u PYTHONPATH -u PYTHONHOME "$PY" -s -m compileall -q -j 0 "$RES/python/lib/python3.12" >/dev/null 2>&1 || true

step "Models (${MODELS_GB} GB, APFS clones)"
mkdir -p "$RES/models"
for sub in checkpoints loras controlnet detectors; do
  cp -cR "$ROOT/models/$sub" "$RES/models/$sub"
done
find "$RES/models" -name .DS_Store -delete

step "Checkpoint configs"
mkdir -p "$RES/configs"
cp -RL "$SDXL_CONFIG" "$RES/configs/sdxl"
[ -f "$RES/configs/sdxl/model_index.json" ] || die "SDXL config copy failed"

step "Icon"
[ -f "$ROOT/packaging/AppIcon.icns" ] || ( cd "$ROOT" && uv run python packaging/make_icon.py )
cp "$ROOT/packaging/AppIcon.icns" "$RES/AppIcon.icns"

step "Launcher and Info.plist"
# The interpreter sits in Contents/MacOS so macOS attributes the process to the
# app bundle (menu-bar name, Dock icon, bundle id) rather than to "python".
cp "$PY" "$CONTENTS/MacOS/python3"
cat > "$CONTENTS/MacOS/$EXEC_NAME" <<'LAUNCH'
#!/bin/bash
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
unset PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE VIRTUAL_ENV CONDA_PREFIX
export PYTHONHOME="$CONTENTS/Resources/python"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PANELFORGE_RESOURCES="$CONTENTS/Resources"
# Models and configs ship inside the app; it never contacts the network.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 YOLO_OFFLINE=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec "$CONTENTS/MacOS/python3" -s -B -m panelforge.desktop "$@"
LAUNCH
chmod +x "$CONTENTS/MacOS/$EXEC_NAME"

MIN_OS="$(sed -n 's/^Tag: .*macosx_\([0-9]*\)_\([0-9]*\)_arm64.*/\1.\2/p' \
  "$RES"/python/lib/python3.12/site-packages/torch-*.dist-info/WHEEL 2>/dev/null | sort -V | tail -1)"
MIN_OS="${MIN_OS:-14.0}"
cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleExecutable</key><string>$EXEC_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>$MIN_OS</string>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>Local build. Models are subject to their own licences.</string>
</dict>
</plist>
PLIST
plutil -lint "$CONTENTS/Info.plist" >/dev/null || die "Info.plist is invalid"

step "Sign (ad-hoc: no Developer ID on this Mac)"
# Inside out: every Mach-O first, then the bundle as a whole.
count=0
while IFS= read -r -d '' f; do
  if file -b "$f" | grep -q 'Mach-O'; then
    codesign --force --sign - --timestamp=none "$f" 2>/dev/null || die "could not sign $f"
    count=$((count + 1))
  fi
done < <(find "$RES/python" "$CONTENTS/MacOS" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) -print0)
echo "  signed $count Mach-O files"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --strict --verbose=1 "$APP" 2>&1 | sed 's/^/  /'

step "Disk image (this compresses ~$((MODELS_GB + 2)) GB)"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "$APP_NAME" -srcfolder "$STAGE" -fs APFS -format ULFO -ov "$DMG" >/dev/null
hdiutil verify "$DMG" >/dev/null && echo "  checksum verified"

step "Done"
printf '  app : %s  (%s)\n' "$APP" "$(du -sh "$APP" | cut -f1)"
printf '  dmg : %s  (%s)\n' "$DMG" "$(du -sh "$DMG" | cut -f1)"
printf '  selftest: PANELFORGE_DESKTOP_SELFTEST=1 "%s/MacOS/%s"\n' "$CONTENTS" "$EXEC_NAME"
