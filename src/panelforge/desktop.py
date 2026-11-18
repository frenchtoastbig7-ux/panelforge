"""Native macOS window for the workbench.

The UI is the same HTML/JS the server serves; this module hosts it in a
WKWebView window (pywebview) instead of a browser tab. The HTTP server still
runs, but only in-process, on a random loopback port chosen at launch.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("panelforge.desktop")

APP_TITLE = "PanelForge"
SELFTEST_ENV = "PANELFORGE_DESKTOP_SELFTEST"

SPLASH = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{height:100%;margin:0;background:#0f1115;color:#e6e8ee;
font:14px -apple-system,system-ui,sans-serif;display:grid;place-items:center}
.t{font-size:22px;font-weight:600;margin-bottom:8px;text-align:center}.t span{color:#949bab;font-weight:400}
.s{color:#949bab;text-align:center}.bar{width:220px;height:4px;background:#1e222b;border-radius:2px;margin:18px auto 0;overflow:hidden}
.bar i{display:block;width:40%;height:100%;background:linear-gradient(90deg,#7c5cff,#4ad0c0);animation:m 1.1s ease-in-out infinite}
@keyframes m{0%{margin-left:-40%}100%{margin-left:100%}}
</style></head><body><div><div class="t">Panel<span>Forge</span></div>
<div class="s">Starting the generation engine…</div><div class="bar"><i></i></div></div></body></html>"""

ERROR_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;padding:48px;background:#0f1115;color:#e6e8ee;font:14px -apple-system,system-ui,sans-serif}
h1{font-size:18px}pre{background:#171a21;padding:14px;border-radius:8px;white-space:pre-wrap;color:#ff9b9b}
p{color:#949bab}</style></head><body><h1>PanelForge could not start</h1>
<pre>{error}</pre><p>Details are in {log}</p></body></html>"""


def log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "panelforge" / "panelforge.log"


def setup_logging() -> Path:
    """A packaged app has no terminal, so everything goes to a rotating log."""
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    if sys.stderr is not None and sys.stderr.isatty():
        root.addHandler(logging.StreamHandler())
    return path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class DesktopAPI:
    """Exposed to page JavaScript as window.pywebview.api.*

    pywebview builds the bridge from public attributes, so the window reference
    is kept private.
    """

    def __init__(self, outputs_dir: Path):
        self._outputs = Path(outputs_dir)
        self._window = None

    def _attach(self, window) -> None:
        self._window = window

    def _resolve_output(self, name) -> Path | None:
        # The name comes from page JavaScript: one plain file in outputs, no paths.
        if not isinstance(name, str) or not name or name.startswith(".") \
                or "/" in name or "\\" in name:
            return None
        path = (self._outputs / name).resolve()
        if not path.is_file() or self._outputs.resolve() not in path.parents:
            return None
        return path

    def save_output(self, name, suggested=None) -> dict:
        """Copy a generated file to a location chosen in a native Save panel."""
        source = self._resolve_output(name)
        if source is None:
            return {"error": "no such output"}
        if self._window is None:
            return {"error": "window not ready"}
        import webview

        pictures = Path.home() / "Pictures"
        chosen = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            directory=str(pictures if pictures.is_dir() else Path.home()),
            save_filename=suggested if isinstance(suggested, str) and suggested else source.name,
        )
        if not chosen:
            return {"cancelled": True}
        target = Path(chosen if isinstance(chosen, str) else chosen[0])
        if target.suffix.lower() != source.suffix.lower():
            target = target.with_suffix(source.suffix)
        shutil.copyfile(source, target)
        log.info("saved %s -> %s", source.name, target)
        return {"saved": str(target)}

    def pick_folder(self):
        """Native folder picker; returns the path or None."""
        if self._window is None:
            return None
        import webview

        chosen = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if not chosen:
            return None
        return str(chosen if isinstance(chosen, str) else chosen[0])

    def reveal_output(self, name=None) -> bool:
        """Show a generated file (or the outputs folder) in Finder."""
        target = self._resolve_output(name) if name else None
        args = ["open", "-R", str(target)] if target else ["open", str(self._outputs)]
        subprocess.run(args, check=False)
        return True


def start_server(settings, port: int):
    """Run the API in a background thread and wait until it accepts requests."""
    import uvicorn

    from .server import create_app

    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", log_config=None))
    thread = threading.Thread(target=server.run, name="panelforge-http", daemon=True)
    thread.start()
    deadline = time.monotonic() + 180
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("the local API server exited during startup")
        if time.monotonic() > deadline:
            raise RuntimeError("the local API server did not start within 180 s")
        time.sleep(0.05)
    return server, thread


def _selftest(window, state: dict, settings) -> None:
    """Launch check for packaged builds: report what the app sees, then quit."""
    import urllib.request

    report: dict = {}
    ok = False
    try:
        base = state["url"].rstrip("/")

        def get(path):
            with urllib.request.urlopen(base + path, timeout=60) as resp:
                return json.load(resp)

        models = get("/api/models")
        nets = get("/api/controlnets")
        report["checkpoints"] = [(m["name"], m["architecture"], m["path"]) for m in models["checkpoints"]]
        report["loras"] = [m["name"] for m in models["loras"]]
        report["controlnets"] = [n["name"] for n in nets["controlnets"]]
        report["face_detector"] = nets["face_detector"]
        report["device"] = get("/api/status")["summary"]

        dom = {}
        for _ in range(40):
            raw = window.evaluate_js(
                "JSON.stringify({title: document.title,"
                " workspaces: document.querySelectorAll('.ws').length,"
                " bridge: !!(window.pywebview && window.pywebview.api"
                " && window.pywebview.api.save_output)})")
            dom = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if dom.get("bridge"):
                break
            time.sleep(0.25)
        report["dom"] = dom
        try:
            from Foundation import NSBundle

            bundle = NSBundle.mainBundle()
            report["bundle_id"] = str(bundle.bundleIdentifier())
            report["bundle_path"] = str(bundle.bundlePath())
        except Exception as exc:
            report["bundle_id"] = f"unavailable: {exc}"
        report["data_dir"] = str(settings.root)
        report["offline"] = settings.offline
        report["single_file_configs"] = settings.single_file_configs
        ok = bool(report["checkpoints"]) and dom.get("workspaces") == 4 and bool(dom.get("bridge"))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("selftest error")
    finally:
        log.info("SELFTEST %s %s", "PASS" if ok else "FAIL", json.dumps(report, default=str))
        state["selftest_failed"] = not ok
        window.destroy()


def main() -> int:
    path = setup_logging()
    selftest = os.environ.get(SELFTEST_ENV) == "1"

    import webview

    from .config import load_settings

    settings = load_settings()
    log.info("starting %s: data=%s bundled_models=%s offline=%s",
             APP_TITLE, settings.root, settings.bundled_models_dir, settings.offline)

    api = DesktopAPI(settings.outputs_dir)
    window = webview.create_window(
        APP_TITLE, html=SPLASH, js_api=api, width=1480, height=940,
        min_size=(1100, 720), background_color="#0f1115", text_select=True)
    api._attach(window)
    state: dict = {"server": None, "thread": None, "url": None, "error": None}

    def boot() -> None:
        # Heavy imports (torch, diffusers) happen here, behind the splash.
        try:
            port = free_port()
            state["server"], state["thread"] = start_server(settings, port)
            state["url"] = f"http://127.0.0.1:{port}/"
            log.info("UI served at %s", state["url"])
            window.load_url(state["url"])
        except Exception as exc:
            log.exception("startup failed")
            state["error"] = exc
            window.load_html(ERROR_PAGE.replace("{error}", f"{type(exc).__name__}: {exc}")
                             .replace("{log}", str(path)))
            if selftest:
                log.info("SELFTEST FAIL startup: %s", exc)
                state["selftest_failed"] = True
                window.destroy()

    def on_loaded() -> None:
        url = window.get_current_url() or ""
        if not state["url"] or not url.startswith(state["url"]):
            return
        log.info("UI loaded")
        if selftest and not state.get("selftest_started"):
            state["selftest_started"] = True
            threading.Thread(target=_selftest, args=(window, state, settings), daemon=True).start()

    window.events.loaded += on_loaded
    webview.start(boot)

    server = state["server"]
    if server is not None:
        server.should_exit = True
        state["thread"].join(timeout=10)
    failed = state["error"] is not None or state.get("selftest_failed", False)
    log.info("exiting (%s)", "error" if failed else "ok")
    logging.shutdown()
    # Job and training threads are daemons holding GPU state; skip interpreter
    # teardown rather than wait on them.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
