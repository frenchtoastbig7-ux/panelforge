"use strict";

const $ = (id) => document.getElementById(id);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
};

const state = {
  mode: "txt2img",
  models: { checkpoints: [], loras: [], samplers: [], buckets: {} },
  job: null,
  poll: null,
  initImage: null,   // { dataUrl, width, height }
  images: [],
  arch: "sdxl",
};

// ---------------------------------------------------------------- bootstrap

async function boot() {
  wireControls();
  try {
    const models = await api("/api/models");
    state.models = models;
    fillSelect($("sampler"), models.samplers, models.defaults.sampler);
    fillSelect($("style"),
               [{ value: "none", label: "(no style)" }].concat(
                 (models.styles || []).map((s) => ({ value: s.name, label: s.label }))),
               "none");
    fillCheckpoints(models.checkpoints);
    $("negative").value = models.defaults.negative_prompt || "";
    $("steps").value = models.defaults.steps; syncRange();
    $("cfg").value = models.defaults.cfg; syncRange();
  } catch (err) {
    $("status").innerHTML = `<span class="err">${err.message}</span>`;
    return;
  }
  await refreshStatus();
  await refreshHistory();
}

function fillSelect(el, items, selected) {
  el.innerHTML = "";
  for (const it of items) {
    const opt = document.createElement("option");
    opt.value = typeof it === "string" ? it : it.value;
    opt.textContent = typeof it === "string" ? it : it.label;
    if (opt.value === selected) opt.selected = true;
    el.appendChild(opt);
  }
}

function fillCheckpoints(list) {
  const el = $("checkpoint");
  el.innerHTML = "";
  if (!list.length) {
    const opt = document.createElement("option");
    opt.textContent = "— no checkpoints found —";
    opt.value = "";
    el.appendChild(opt);
    $("go").disabled = true;
    return;
  }
  $("go").disabled = false;
  for (const m of list) {
    const opt = document.createElement("option");
    opt.value = m.path;
    opt.textContent = `${m.name}  ·  ${m.architecture}  ·  ${m.size_gb}GB`;
    opt.dataset.arch = m.architecture;
    el.appendChild(opt);
  }
  onCheckpointChange();
}

function onCheckpointChange() {
  const opt = $("checkpoint").selectedOptions[0];
  state.arch = opt?.dataset.arch === "sd15" ? "sd15" : "sdxl";
  const buckets = state.models.buckets[state.arch] || [];
  fillSelect(
    $("size"),
    buckets.map(([w, h]) => ({ value: `${w}x${h}`, label: `${w} × ${h}` })),
    `${buckets[0]?.[0]}x${buckets[0]?.[1]}`
  );
  refreshLoraOptions();
}

// LoRAs trained for one architecture silently produce noise on the other,
// so only offer the ones matching the selected checkpoint.
function compatibleLoras() {
  return state.models.loras.filter(
    (l) => l.architecture === state.arch || l.architecture === "unknown"
  );
}

// ----------------------------------------------------------------- controls

function wireControls() {
  $("modes").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === btn));
    state.mode = btn.dataset.mode;
    applyMode();
  });

  for (const id of ["steps", "cfg", "strength"]) $(id).addEventListener("input", syncRange);
  $("checkpoint").addEventListener("change", onCheckpointChange);
  $("add-lora").addEventListener("click", addLoraRow);
  $("go").addEventListener("click", generate);
  $("cancel").addEventListener("click", cancelJob);
  $("init-file").addEventListener("change", loadInitImage);
  $("clear-mask").addEventListener("click", clearMask);
  setupMaskPainting();

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
  });
}

function syncRange() {
  $("steps-v").textContent = $("steps").value;
  $("cfg-v").textContent = Number($("cfg").value).toFixed(1);
  $("strength-v").textContent = Number($("strength").value).toFixed(2);
}

function applyMode() {
  const needsImage = state.mode !== "txt2img";
  document.querySelectorAll(".img-only").forEach((el) => el.classList.toggle("hidden", !needsImage));
  document.querySelectorAll(".inpaint-only").forEach((el) =>
    el.classList.toggle("hidden", state.mode !== "inpaint"));
  $("canvas-wrap").classList.toggle("hidden", !needsImage || !state.initImage);
  $("viewer").classList.toggle("hidden", needsImage && !!state.initImage);
}

function addLoraRow() {
  const options = compatibleLoras();
  if (!options.length) return;
  const row = document.createElement("div");
  row.className = "lora-row";
  const sel = document.createElement("select");
  fillSelect(sel, options.map((l) => ({ value: l.path, label: l.name })));
  const weight = document.createElement("input");
  weight.type = "number"; weight.step = "0.05"; weight.value = "0.8";
  weight.min = "-2"; weight.max = "2";
  const rm = document.createElement("button");
  rm.className = "rm"; rm.textContent = "×"; rm.type = "button";
  rm.onclick = () => row.remove();
  row.append(sel, weight, rm);
  $("loras").appendChild(row);
}

function refreshLoraOptions() {
  $("loras").innerHTML = "";
  const has = compatibleLoras().length > 0;
  $("add-lora").disabled = !has;
  $("add-lora").textContent = has ? "+ add LoRA" : "no compatible LoRAs";
}

function collectLoras() {
  return [...$("loras").querySelectorAll(".lora-row")].map((row) => ({
    name: row.querySelector("select").value,
    weight: parseFloat(row.querySelector("input").value) || 0,
  }));
}

// -------------------------------------------------------------- init image

function loadInitImage(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      state.initImage = { dataUrl: reader.result, width: img.width, height: img.height };
      drawBase(img);
      applyMode();
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function drawBase(img) {
  // Cap the working canvas so a large photo does not make painting sluggish;
  // the full-resolution source is still what gets sent to the model.
  const maxSide = 768;
  const scale = Math.min(1, maxSide / Math.max(img.width, img.height));
  const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
  for (const id of ["base", "mask"]) {
    const c = $(id); c.width = w; c.height = h;
    c.style.width = w + "px"; c.style.height = h + "px";
  }
  $("base").getContext("2d").drawImage(img, 0, 0, w, h);
  clearMask();
}

function clearMask() {
  const c = $("mask");
  c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

function setupMaskPainting() {
  const c = $("mask");
  const ctx = c.getContext("2d");
  let painting = false;

  const pos = (e) => {
    const r = c.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / r.width) * c.width,
      ((e.clientY - r.top) / r.height) * c.height,
    ];
  };
  const stroke = (e) => {
    if (!painting) return;
    const [x, y] = pos(e);
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(x, y, Number($("brush").value) / 2, 0, Math.PI * 2);
    ctx.fill();
  };
  c.addEventListener("pointerdown", (e) => { painting = true; c.setPointerCapture(e.pointerId); stroke(e); });
  c.addEventListener("pointermove", stroke);
  c.addEventListener("pointerup", () => { painting = false; });
  c.addEventListener("pointerleave", () => { painting = false; });
}

// The mask the model wants is white-on-black at the source resolution, but the
// canvas holds transparent-on-white strokes at display resolution.
function maskDataUrl() {
  const src = $("mask");
  const out = document.createElement("canvas");
  out.width = state.initImage.width;
  out.height = state.initImage.height;
  const ctx = out.getContext("2d");
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(src, 0, 0, out.width, out.height);
  return out.toDataURL("image/png");
}

function hasMask() {
  const c = $("mask");
  const data = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) return true;
  return false;
}

// ---------------------------------------------------------------- generate

async function generate() {
  if ($("go").disabled) return;
  const [w, h] = $("size").value.split("x").map(Number);
  const body = {
    prompt: $("prompt").value.trim(),
    negative_prompt: $("negative").value.trim(),
    steps: Number($("steps").value),
    cfg: Number($("cfg").value),
    width: w, height: h,
    seed: Number($("seed").value),
    sampler: $("sampler").value,
    batch_size: Number($("batch").value),
    checkpoint: $("checkpoint").value,
    loras: collectLoras(),
    mode: state.mode,
    strength: Number($("strength").value),
    clip_skip: 2,
    style: $("style").value === "none" ? null : $("style").value,
  };
  if (!body.prompt) { $("meta").innerHTML = `<span class="err">Prompt is empty.</span>`; return; }

  if (state.mode !== "txt2img") {
    if (!state.initImage) { $("meta").innerHTML = `<span class="err">Pick an input image first.</span>`; return; }
    body.init_image = state.initImage.dataUrl;
    body.width = state.initImage.width;
    body.height = state.initImage.height;
    if (state.mode === "inpaint") {
      if (!hasMask()) { $("meta").innerHTML = `<span class="err">Paint a mask area first.</span>`; return; }
      body.mask_image = maskDataUrl();
    }
  }

  if (window.panelforgeExtras) {
    try { await window.panelforgeExtras.decorate(body); }
    catch (err) { console.warn("extras.decorate failed", err); }
  }

  setBusy(true);
  try {
    state.job = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("meta").textContent = "queued…";
    pollJob();
  } catch (err) {
    $("meta").innerHTML = `<span class="err">${err.message}</span>`;
    setBusy(false);
  }
}

function setBusy(busy) {
  $("go").disabled = busy;
  $("cancel").classList.toggle("hidden", !busy);
  $("progress").classList.toggle("hidden", !busy);
  if (!busy) { $("bar").style.width = "0%"; $("ptext").textContent = ""; }
}

function pollJob() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    if (!state.job) return;
    let job;
    try {
      job = await api(`/api/jobs/${state.job.id}`);
    } catch (err) {
      clearInterval(state.poll); setBusy(false);
      $("meta").innerHTML = `<span class="err">${err.message}</span>`;
      return;
    }
    $("bar").style.width = `${Math.round(job.progress * 100)}%`;
    $("ptext").textContent =
      job.state === "running"
        ? `${job.step}/${job.total_steps} · ${job.elapsed_s}s`
        : job.state;

    if (["done", "error", "cancelled"].includes(job.state)) {
      clearInterval(state.poll);
      setBusy(false);
      state.job = null;
      if (job.state === "done") { showResult(job); refreshHistory(); refreshStatus(); }
      else if (job.state === "error") $("meta").innerHTML = `<span class="err">${job.error}</span>`;
      else $("meta").textContent = "cancelled";
    }
  }, 400);
}

async function cancelJob() {
  if (!state.job) return;
  try { await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }); } catch {}
}

function showResult(job) {
  state.images = job.images;
  const viewer = $("viewer");
  viewer.classList.remove("hidden");
  $("canvas-wrap").classList.add("hidden");
  viewer.innerHTML = "";
  const main = document.createElement("img");
  main.src = `/outputs/${job.images[0]}`;
  viewer.appendChild(main);

  if (job.images.length > 1) {
    const strip = document.createElement("div");
    strip.className = "strip";
    job.images.forEach((name, i) => {
      const t = document.createElement("img");
      t.src = `/outputs/${name}`;
      t.className = i === 0 ? "on" : "";
      t.onclick = () => {
        main.src = t.src;
        strip.querySelectorAll("img").forEach((x) => x.classList.toggle("on", x === t));
      };
      strip.appendChild(t);
    });
    viewer.appendChild(strip);
  }

  const m = job.metadata || {};
  const loras = (m.loras || []).map((l) => `${l.name}@${l.weight}`).join(", ") || "none";
  $("meta").innerHTML =
    `<b>${m.checkpoint || ""}</b> · ${m.sampler} · ${m.steps} steps · cfg ${m.cfg} · ` +
    `${m.width}×${m.height} · seed ${(m.seeds || []).join(", ")} · loras: ${loras} · ` +
    `<b>${m.elapsed_s}s</b> on ${m.device}/${m.dtype}`;

  const native = window.pywebview && window.pywebview.api;
  if (native && job.images.length) {
    const save = document.createElement("button");
    save.className = "ghost"; save.type = "button"; save.textContent = "Save image…";
    save.style.marginLeft = "8px";
    save.onclick = () => {
      const shown = viewer.querySelector("img")?.getAttribute("src") || "";
      const name = decodeURIComponent(shown.split("/outputs/")[1]?.split("?")[0] || job.images[0]);
      native.save_output(name);
    };
    $("meta").appendChild(save);
  }
}

async function refreshHistory() {
  try {
    const { jobs } = await api("/api/jobs?limit=40");
    const el = $("history");
    el.innerHTML = "";
    for (const job of jobs) {
      for (const name of job.images) {
        const img = document.createElement("img");
        img.src = `/outputs/${name}`;
        img.title = job.prompt;
        img.onclick = () => showResult(job);
        el.appendChild(img);
      }
    }
  } catch {}
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    $("status").innerHTML =
      `<b>${s.device}</b>/${s.dtype} · ${s.memory_gb}GB${s.unified_memory ? " unified" : ""} · ` +
      `${s.checkpoint_name || "no model loaded"}` +
      (s.queue_depth ? ` · queue ${s.queue_depth}` : "");
  } catch (err) {
    $("status").innerHTML = `<span class="err">${err.message}</span>`;
  }
}

boot();
