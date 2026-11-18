"use strict";
// Pose control, face enhance, character sheets and LoRA training.

(function () {
  const $ = (id) => document.getElementById(id);
  const api = async (path, opts) => {
    const res = await fetch(path, opts);
    if (!res.ok) {
      let d = res.statusText;
      try { d = (await res.json()).detail || d; } catch {}
      throw new Error(d);
    }
    return res.json();
  };
  const post = (p, b) => api(p, { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
  const fill = (el, items, sel) => {
    if (!el) return;
    el.innerHTML = "";
    for (const it of items) {
      const o = document.createElement("option");
      o.value = typeof it === "string" ? it : it.value;
      o.textContent = typeof it === "string" ? it : it.label;
      if (o.value === sel) o.selected = true;
      el.appendChild(o);
    }
  };

  const E = { poseImage: null, controlnet: null, faceDetector: false, timer: null };

  // ------------------------------------------------------------ pose control

  async function initExtras() {
    try {
      const cn = await api("/api/controlnets");
      E.controlnet = cn.controlnets[0]?.name || null;
      E.faceDetector = cn.face_detector;
      const note = $("cn-note");
      if (note) {
        note.textContent = E.controlnet ? `(${E.controlnet})` : "(no controlnet installed)";
      }
      if (!E.controlnet && $("pose-on")) {
        $("pose-on").disabled = true;
        $("pose-preview").disabled = true;
      }
      if (!E.faceDetector && $("face-on")) {
        $("face-on").disabled = true;
        $("face-on").parentElement.title = "models/detectors/face_yolov8m.pt is missing";
      }
      const poses = await api("/api/poses");
      fill($("pose-preset"), poses.poses, "standing");
    } catch {}

    try {
      const sheets = await api("/api/sheets/presets");
      fill($("c-sheet-preset"), Object.keys(sheets.presets), "full");
    } catch {}

    try {
      const models = await api("/api/models");
      fill($("t-checkpoint"),
           models.checkpoints.map((m) => ({ value: m.path, label: m.name })));
    } catch {}
  }

  async function previewPose() {
    const [w, h] = ($("size")?.value || "1024x1024").split("x").map(Number);
    try {
      const out = await post("/api/pose/render", {
        preset: $("pose-preset").value, width: w, height: h,
        mirror: $("pose-mirror").checked,
      });
      const img = $("pose-thumb");
      img.src = `/outputs/${out.image}?t=${Date.now()}`;
      img.classList.remove("hidden");
      // Fetch it back as a data URL so it can ride along with the request.
      const blob = await (await fetch(`/outputs/${out.image}`)).blob();
      E.poseImage = await new Promise((r) => {
        const fr = new FileReader();
        fr.onload = () => r(fr.result);
        fr.readAsDataURL(blob);
      });
    } catch (err) {
      console.warn("pose preview failed", err);
    }
  }

  // The Image workspace builds its own request body; extend it here so app.js
  // stays responsible for only the core generation fields.
  window.panelforgeExtras = {
    async decorate(body) {
      if ($("hires-on")?.checked) body.hires = true;
      if ($("face-on")?.checked && E.faceDetector) {
        body.face_enhance = true;
        body.face_mode = $("face-mode")?.value || "inpaint";
      }
      if ($("pose-on")?.checked && E.controlnet) {
        if (!E.poseImage) await previewPose();
        body.controlnet = E.controlnet;
        body.control_image = E.poseImage;
        body.controlnet_scale = Number($("cn-scale").value);
      }
      return body;
    },
  };

  // -------------------------------------------------------- character sheets

  async function generateSheet() {
    const id = $("c-id").value;
    if (!id) {
      $("c-sheet-out").innerHTML =
        '<p class="warn">Save (or edit) a character first, then generate its sheet.</p>';
      return;
    }
    const preset = $("c-sheet-preset").value;
    $("c-sheet-go").disabled = true;
    $("c-sheet-progress").classList.remove("hidden");
    $("c-sheet-out").innerHTML = "";
    try {
      const { queued } = await post("/api/character-sheet", {
        character_id: id, preset,
        checkpoint: $("t-checkpoint")?.value || undefined,
      });
      const images = new Array(queued.length).fill(null);
      let done = 0;
      await Promise.all(queued.map((q) => new Promise((resolve) => {
        const timer = setInterval(async () => {
          let job;
          try { job = await api(`/api/jobs/${q.job_id}`); }
          catch { clearInterval(timer); resolve(); return; }
          if (["done", "error", "cancelled"].includes(job.state)) {
            clearInterval(timer);
            if (job.state === "done" && job.images.length) images[q.cell] = job.images[0];
            done += 1;
            $("c-sheet-bar").style.width = `${Math.round(done / queued.length * 100)}%`;
            $("c-sheet-text").textContent = `${done}/${queued.length} views`;
            resolve();
          }
        }, 700);
      })));
      const sheet = await post("/api/character-sheet/compose", {
        character_id: id, preset, images,
      });
      $("c-sheet-out").innerHTML =
        `<img src="/outputs/${sheet.image}?t=${Date.now()}" style="width:100%;border-radius:8px">`;
    } catch (err) {
      $("c-sheet-out").innerHTML = `<p class="err">${err.message}</p>`;
    } finally {
      $("c-sheet-go").disabled = false;
      $("c-sheet-progress").classList.add("hidden");
      $("c-sheet-bar").style.width = "0%";
    }
  }

  // --------------------------------------------------------------- training

  async function startTraining() {
    const body = {
      dataset_dir: $("t-dataset").value.trim(),
      output_name: $("t-name").value.trim(),
      trigger_word: $("t-trigger").value.trim(),
      checkpoint: $("t-checkpoint").value,
      steps: Number($("t-steps").value),
      rank: Number($("t-rank").value),
      learning_rate: Number($("t-lr").value),
      resolution: Number($("t-res").value),
    };
    if (!body.dataset_dir || !body.output_name) {
      $("t-warnings").innerHTML =
        '<p class="err">Dataset folder and output name are required.</p>';
      return;
    }
    try {
      const state = await post("/api/train", body);
      $("t-warnings").innerHTML =
        (state.warnings || []).map((w) => `<p class="warn">! ${w}</p>`).join("");
      $("t-log").textContent =
        `started: ${state.images} images -> ${state.output_name}\n`;
      $("t-start").disabled = true;
      $("t-cancel").classList.remove("hidden");
      $("t-progress").classList.remove("hidden");
      pollTraining();
    } catch (err) {
      $("t-warnings").innerHTML = `<p class="err">${err.message}</p>`;
    }
  }

  function pollTraining() {
    clearInterval(E.timer);
    E.timer = setInterval(async () => {
      let s;
      try { s = await api("/api/train/status"); } catch { return; }
      const pct = s.total ? Math.round(s.step / s.total * 100) : 0;
      $("t-bar").style.width = `${pct}%`;
      $("t-text").textContent =
        `${s.phase || ""} ${s.step}/${s.total}` + (s.loss != null ? ` · loss ${s.loss}` : "");
      if (["done", "error", "idle"].includes(s.state)) {
        clearInterval(E.timer);
        $("t-start").disabled = false;
        $("t-cancel").classList.add("hidden");
        if (s.state === "done" && s.result) {
          $("t-log").textContent +=
            `\ndone: ${s.result.output}\n${s.result.size_mb} MB · ` +
            `${s.result.steps_completed} steps · loss ${s.result.final_loss} · ` +
            `${s.result.elapsed_s}s\n`;
        } else if (s.state === "error") {
          $("t-log").textContent += `\nfailed: ${s.error}\n`;
        }
      }
    }, 1500);
  }

  // ----------------------------------------------------------------- wiring

  function wire() {
    $("pose-preview")?.addEventListener("click", previewPose);
    $("pose-preset")?.addEventListener("change", () => { E.poseImage = null; });
    $("pose-mirror")?.addEventListener("change", () => { E.poseImage = null; });
    $("cn-scale")?.addEventListener("input", () =>
      ($("cn-scale-v").textContent = Number($("cn-scale").value).toFixed(2)));
    $("c-sheet-go")?.addEventListener("click", generateSheet);
    $("t-steps")?.addEventListener("input", () =>
      ($("t-steps-v").textContent = $("t-steps").value));
    $("t-rank")?.addEventListener("input", () =>
      ($("t-rank-v").textContent = $("t-rank").value));
    // Desktop app: a native folder picker instead of typing a path.
    const showPicker = () => $("t-pick")?.classList.remove("hidden");
    if (window.pywebview && window.pywebview.api) showPicker();
    window.addEventListener("pywebviewready", showPicker);
    $("t-pick")?.addEventListener("click", async () => {
      const folder = await window.pywebview.api.pick_folder();
      if (folder) $("t-dataset").value = folder;
    });
    $("t-start")?.addEventListener("click", startTraining);
    $("t-cancel")?.addEventListener("click", () =>
      post("/api/train/cancel", {}).catch(() => {}));
    initExtras();
  }

  wire();
})();
