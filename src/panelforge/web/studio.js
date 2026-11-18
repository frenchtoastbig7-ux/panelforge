"use strict";
// Comic Studio: page layout, panel prompts, lettering and project files.
// The page preview is rendered server-side by PIL; this file keeps an editable
// overlay in sync with it so panels and bubbles can be selected and dragged.

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
  const post = (path, body) =>
    api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body) });

  const S = {
    page: null,
    sel: null,          // {kind:"panel"|"bubble", index:number}
    meta: null,
    cast: [],
    selectedCast: new Set(),
    busyPanels: new Map(),
    renderSeq: 0,
  };

  // ------------------------------------------------------------ workspaces

  document.querySelector(".workspaces").addEventListener("click", (e) => {
    const btn = e.target.closest(".ws");
    if (!btn) return;
    document.querySelectorAll(".ws").forEach((b) => b.classList.toggle("active", b === btn));
    for (const name of ["image", "studio", "cast", "train"]) {
      $(`ws-${name}`).classList.toggle("hidden", name !== btn.dataset.ws);
    }
    if (btn.dataset.ws === "studio" && !S.page) initStudio();
    if (btn.dataset.ws === "cast") refreshCast();
  });

  // ---------------------------------------------------------------- setup

  async function initStudio() {
    try {
      S.meta = await api("/api/layouts");
      const models = await api("/api/models");
      fill($("s-layout"), S.meta.layouts, "manga-classic");
      fill($("s-preset"), Object.keys(S.meta.page_presets), "manga-b5");
      fill($("s-checkpoint"),
           models.checkpoints.map((m) => ({ value: m.path, label: m.name })));
      fill($("s-style"),
           [{ value: "none", label: "(no style)" }].concat(
             (models.styles || []).map((st) => ({ value: st.name, label: st.label }))),
           "anime-color");
      await refreshCast();
      await refreshProjects();
      await newPage();
    } catch (err) {
      $("s-insp-title").innerHTML = `<span class="err">${err.message}</span>`;
    }
  }

  function fill(el, items, selected) {
    el.innerHTML = "";
    for (const it of items) {
      const o = document.createElement("option");
      o.value = typeof it === "string" ? it : it.value;
      o.textContent = typeof it === "string" ? it : it.label;
      if (o.value === selected) o.selected = true;
      el.appendChild(o);
    }
  }

  async function newPage() {
    S.page = await post("/api/page/new", {
      layout: $("s-layout").value,
      page_preset: $("s-preset").value,
      reading_order: $("s-order").value,
    });
    S.page.gutter = Number($("s-gutter").value);
    S.sel = null;
    await renderPage();
  }

  // --------------------------------------------------------------- render

  async function renderPage() {
    if (!S.page) return;
    // Renders can overlap; only the newest result may touch the DOM.
    const seq = ++S.renderSeq;
    try {
      const out = await post("/api/page/render", { page: S.page, name: "_preview" });
      if (seq !== S.renderSeq) return;
      $("s-preview").src = `/outputs/${out.image}?t=${Date.now()}`;
      $("s-preview").onload = drawOverlay;
    } catch (err) {
      $("s-insp-title").innerHTML = `<span class="err">${err.message}</span>`;
    }
    drawInspector();
  }

  function drawOverlay() {
    const ov = $("s-overlay");
    const img = $("s-preview");
    ov.style.width = img.clientWidth + "px";
    ov.style.height = img.clientHeight + "px";
    ov.innerHTML = "";
    if (!S.page) return;

    const W = img.clientWidth, H = img.clientHeight;
    const m = S.page.margin, g = S.page.gutter / 2;
    const short = Math.min(S.page.width, S.page.height) / S.page.width;

    S.page.panels.forEach((p, i) => {
      // Mirror the server's margin/gutter maths so the hit area matches the art.
      const mx = m * Math.min(S.page.width, S.page.height) / S.page.width;
      const my = m * Math.min(S.page.width, S.page.height) / S.page.height;
      const gx = g * short, gy = g * Math.min(S.page.width, S.page.height) / S.page.height;
      const el = document.createElement("div");
      el.className = "ov-panel" + (isSel("panel", i) ? " sel" : "");
      el.style.left   = ((mx + p.x * (1 - 2 * mx) + gx) * W) + "px";
      el.style.top    = ((my + p.y * (1 - 2 * my) + gy) * H) + "px";
      el.style.width  = ((p.w * (1 - 2 * mx) - 2 * gx) * W) + "px";
      el.style.height = ((p.h * (1 - 2 * my) - 2 * gy) * H) + "px";
      el.innerHTML = `<span class="idx">${i + 1}</span>`;
      const busy = S.busyPanels.get(i);
      if (busy) el.innerHTML += `<div class="busy">${busy}</div>`;
      el.onclick = (e) => { e.stopPropagation(); select("panel", i); };
      ov.appendChild(el);
    });

    S.page.bubbles.forEach((b, i) => {
      const el = document.createElement("div");
      el.className = "ov-bubble" + (isSel("bubble", i) ? " sel" : "");
      el.style.left = (b.x * W) + "px";
      el.style.top = (b.y * H) + "px";
      el.style.width = (b.w * W) + "px";
      el.style.height = (b.h * H) + "px";
      el.onclick = (e) => { e.stopPropagation(); select("bubble", i); };
      dragMove(el, b, i, W, H);
      const rs = document.createElement("div");
      rs.className = "rs";
      dragResize(rs, b, i, W, H);
      el.appendChild(rs);
      ov.appendChild(el);
      if (b.tail_x != null && b.tail_y != null) {
        const tl = document.createElement("div");
        tl.className = "tl";
        tl.style.left = (b.tail_x * W - b.x * W) + "px";
        tl.style.top = (b.tail_y * H - b.y * H) + "px";
        dragTail(tl, b, i, W, H);
        el.appendChild(tl);
      }
    });
    ov.onclick = () => select(null);
  }

  const isSel = (kind, i) => S.sel && S.sel.kind === kind && S.sel.index === i;

  function select(kind, index) {
    S.sel = kind ? { kind, index } : null;
    drawOverlay();
    drawInspector();
  }

  // --------------------------------------------------------------- drags

  function pointerDrag(handle, onMove) {
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      const start = { x: e.clientX, y: e.clientY };
      handle.setPointerCapture(e.pointerId);
      const move = (ev) => onMove(ev.clientX - start.x, ev.clientY - start.y);
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        renderPage();
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
  }

  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function dragMove(el, b, i, W, H) {
    const o = { x: b.x, y: b.y };
    pointerDrag(el, (dx, dy) => {
      b.x = clamp01(o.x + dx / W); b.y = clamp01(o.y + dy / H);
      el.style.left = (b.x * W) + "px"; el.style.top = (b.y * H) + "px";
    });
  }
  function dragResize(handle, b, i, W, H) {
    const o = { w: b.w, h: b.h };
    pointerDrag(handle, (dx, dy) => {
      b.w = Math.max(0.04, o.w + dx / W); b.h = Math.max(0.03, o.h + dy / H);
      handle.parentElement.style.width = (b.w * W) + "px";
      handle.parentElement.style.height = (b.h * H) + "px";
    });
  }
  function dragTail(handle, b, i, W, H) {
    const o = { x: b.tail_x, y: b.tail_y };
    pointerDrag(handle, (dx, dy) => {
      b.tail_x = clamp01(o.x + dx / W); b.tail_y = clamp01(o.y + dy / H);
      handle.style.left = (b.tail_x * W - b.x * W) + "px";
      handle.style.top = (b.tail_y * H - b.y * H) + "px";
    });
  }

  // ----------------------------------------------------------- inspector

  function drawInspector() {
    const box = $("s-inspector");
    const title = $("s-insp-title");
    box.innerHTML = "";
    if (!S.sel) { title.textContent = "Nothing selected"; return; }

    if (S.sel.kind === "panel") {
      const p = S.page.panels[S.sel.index];
      title.textContent = `Panel ${S.sel.index + 1}`;
      box.append(
        fieldArea("Prompt", p.prompt, (v) => (p.prompt = v), 5),
        fieldArea("Negative", p.negative_prompt, (v) => (p.negative_prompt = v), 2),
        fieldNum("Seed (−1 random)", p.seed, (v) => (p.seed = v)),
        fieldRange("Focus X", p.focus_x, 0, 1, .01, (v) => (p.focus_x = v)),
        fieldRange("Focus Y", p.focus_y, 0, 1, .01, (v) => (p.focus_y = v)),
        button("Generate this panel", () => generate([S.sel.index]), "primary"),
        button("Clear art", () => { p.image = null; renderPage(); })
      );
    } else {
      const b = S.page.bubbles[S.sel.index];
      title.textContent = `Bubble ${S.sel.index + 1}`;
      box.append(
        fieldArea("Text", b.text, (v) => (b.text = v), 3),
        fieldSelect("Style", S.meta.bubble_styles, b.style, (v) => (b.style = v)),
        fieldSelect("Font", ["(auto)"].concat(S.meta.fonts), b.font || "(auto)",
                    (v) => (b.font = v === "(auto)" ? null : v)),
        fieldCheck("Vertical (CJK)", b.vertical, (v) => (b.vertical = v)),
        fieldNum("Font size (0 = auto)", b.font_size || 0,
                 (v) => (b.font_size = v > 0 ? v : null)),
        fieldRange("Rotation", b.rotation, -45, 45, 1, (v) => (b.rotation = v)),
        fieldCheck("Tail", b.tail_x != null, (v) => {
          if (v) { b.tail_x = clamp01(b.x + b.w / 2); b.tail_y = clamp01(b.y + b.h + .05); }
          else { b.tail_x = b.tail_y = null; }
        }),
        button("Delete bubble", () => {
          S.page.bubbles.splice(S.sel.index, 1); S.sel = null; renderPage();
        })
      );
    }
  }

  function wrap(label, node) {
    const d = document.createElement("label");
    d.className = "field";
    const s = document.createElement("span"); s.textContent = label;
    d.append(s, node);
    return d;
  }
  function fieldArea(label, value, set, rows) {
    const t = document.createElement("textarea");
    t.rows = rows; t.value = value || "";
    t.onchange = () => { set(t.value); renderPage(); };
    return wrap(label, t);
  }
  function fieldNum(label, value, set) {
    const i = document.createElement("input");
    i.type = "number"; i.value = value ?? 0;
    i.onchange = () => { set(Number(i.value)); renderPage(); };
    return wrap(label, i);
  }
  function fieldRange(label, value, min, max, step, set) {
    const i = document.createElement("input");
    i.type = "range"; i.min = min; i.max = max; i.step = step; i.value = value ?? 0;
    i.onchange = () => { set(Number(i.value)); renderPage(); };
    return wrap(label, i);
  }
  function fieldSelect(label, options, value, set) {
    const s = document.createElement("select");
    fill(s, options, value);
    s.onchange = () => { set(s.value); renderPage(); };
    return wrap(label, s);
  }
  function fieldCheck(label, value, set) {
    const l = document.createElement("label");
    l.className = "check";
    const i = document.createElement("input");
    i.type = "checkbox"; i.checked = !!value;
    i.onchange = () => { set(i.checked); renderPage(); };
    l.append(i, document.createTextNode(" " + label));
    return l;
  }
  function button(label, fn, cls) {
    const b = document.createElement("button");
    b.className = cls || "ghost"; b.textContent = label; b.onclick = fn;
    b.style.width = "100%"; b.style.marginTop = "6px";
    return b;
  }

  // ---------------------------------------------------------- generation

  async function generate(indices) {
    if (!S.page) return;
    const body = {
      page: S.page,
      checkpoint: $("s-checkpoint").value,
      steps: Number($("s-steps").value),
      cfg: Number($("s-cfg").value),
      characters: [...S.selectedCast],
      hires: $("s-hires").checked,
      style: $("s-style").value === "none" ? null : $("s-style").value,
      only_empty: indices ? false : $("s-onlyempty").checked,
    };
    if (indices) body.panel_indices = indices;

    let queued;
    try {
      queued = (await post("/api/page/generate", body)).queued;
    } catch (err) {
      $("s-insp-title").innerHTML = `<span class="err">${err.message}</span>`;
      return;
    }

    $("s-progress").classList.remove("hidden");
    $("s-generate").disabled = true;
    let done = 0;
    queued.forEach((q) => S.busyPanels.set(q.panel, "queued"));
    drawOverlay();

    await Promise.all(queued.map((q) => trackJob(q, () => {
      done += 1;
      $("s-bar").style.width = `${Math.round((done / queued.length) * 100)}%`;
      $("s-ptext").textContent = `${done}/${queued.length} panels`;
    })));

    $("s-generate").disabled = false;
    $("s-progress").classList.add("hidden");
    $("s-bar").style.width = "0%";
    S.busyPanels.clear();
    await renderPage();
  }

  function trackJob(q, onDone) {
    return new Promise((resolve) => {
      const timer = setInterval(async () => {
        let job;
        try { job = await api(`/api/jobs/${q.job_id}`); }
        catch { clearInterval(timer); onDone(); resolve(); return; }

        if (job.state === "running") {
          S.busyPanels.set(q.panel, `${Math.round(job.progress * 100)}%`);
          drawOverlay();
        }
        if (["done", "error", "cancelled"].includes(job.state)) {
          clearInterval(timer);
          if (job.state === "done" && job.images.length) {
            S.page.panels[q.panel].image = job.images[0];
            S.page.panels[q.panel].seed = (job.metadata.seeds || [-1])[0];
          }
          S.busyPanels.delete(q.panel);
          onDone(); resolve();
        }
      }, 500);
    });
  }

  // ---------------------------------------------------------------- cast

  async function refreshCast() {
    try {
      S.cast = (await api("/api/characters")).characters;
    } catch { S.cast = []; }
    const box = $("s-cast");
    if (box) {
      box.innerHTML = "";
      if (!S.cast.length) box.innerHTML = '<span class="hint">No characters yet — add them in Cast.</span>';
      for (const c of S.cast) {
        const l = document.createElement("label");
        const i = document.createElement("input");
        i.type = "checkbox"; i.checked = S.selectedCast.has(c.id);
        i.onchange = () => {
          i.checked ? S.selectedCast.add(c.id) : S.selectedCast.delete(c.id);
        };
        l.append(i, document.createTextNode(" " + c.name));
        box.appendChild(l);
      }
    }
    drawCastGrid();
  }

  function drawCastGrid() {
    const grid = $("c-list");
    if (!grid) return;
    grid.innerHTML = "";
    if (!S.cast.length) grid.innerHTML = '<p class="hint">No characters yet.</p>';
    for (const c of S.cast) {
      const card = document.createElement("div");
      card.className = "cast-card";
      card.innerHTML = `<h3></h3><p></p><div class="row"></div>`;
      card.querySelector("h3").textContent = c.name;
      card.querySelector("p").textContent =
        (c.prompt || "(no tags)") + (c.seed >= 0 ? `  · seed ${c.seed}` : "");
      const edit = document.createElement("button");
      edit.className = "ghost"; edit.textContent = "Edit";
      edit.onclick = () => loadCharForm(c);
      const del = document.createElement("button");
      del.className = "ghost"; del.textContent = "Delete";
      del.onclick = async () => {
        await api(`/api/characters/${c.id}`, { method: "DELETE" });
        S.selectedCast.delete(c.id);
        refreshCast();
      };
      card.querySelector(".row").append(edit, del);
      grid.appendChild(card);
    }
  }

  // ------------------------------------------------------------ cast LoRAs

  async function ensureLoraList() {
    if (S.loraList) return S.loraList;
    try {
      S.loraList = (await api("/api/models")).loras || [];
    } catch {
      S.loraList = [];
    }
    return S.loraList;
  }

  async function addCastLoraRow(preset) {
    const list = await ensureLoraList();
    const box = $("c-loras");
    const options = list.map((l) => ({ value: l.path, label: `${l.name} · ${l.architecture}` }));
    // A LoRA the character references but whose file has gone must stay
    // selected as-is; otherwise the select falls back to the first option and
    // the next save silently swaps in a different LoRA.
    if (preset?.name && !list.some((l) => l.path === preset.name)) {
      options.unshift({ value: preset.name, label: `(missing) ${preset.name.split("/").pop()}` });
    }
    if (!options.length) {
      box.innerHTML = '<span class="hint">No LoRAs in models/loras yet.</span>';
      return;
    }
    const row = document.createElement("div");
    row.className = "lora-row";
    const sel = document.createElement("select");
    fill(sel, options, preset?.name);
    const weight = document.createElement("input");
    weight.type = "number"; weight.step = "0.05"; weight.min = "-2"; weight.max = "2";
    weight.value = preset?.weight ?? 0.8;
    const rm = document.createElement("button");
    rm.className = "rm"; rm.type = "button"; rm.textContent = "×";
    rm.onclick = () => row.remove();
    row.append(sel, weight, rm);
    box.appendChild(row);
  }

  function collectCastLoras() {
    return [...$("c-loras").querySelectorAll(".lora-row")].map((row) => ({
      name: row.querySelector("select").value,
      weight: parseFloat(row.querySelector("input").value) || 0,
    }));
  }

  async function loadCharForm(c) {
    $("c-id").value = c.id || "";
    $("c-name").value = c.name || "";
    $("c-prompt").value = c.prompt || "";
    $("c-negative").value = c.negative_prompt || "";
    $("c-seed").value = c.seed ?? -1;
    // Rebuild the character's LoRA rows. Without this, editing and re-saving a
    // character sent an empty list and wiped the LoRAs it already had.
    $("c-loras").innerHTML = "";
    for (const lora of c.loras || []) await addCastLoraRow(lora);
  }

  // ------------------------------------------------------------ projects

  async function refreshProjects() {
    try {
      const { projects } = await api("/api/projects");
      fill($("s-projects"), projects.map((p) => ({ value: p.name, label: p.name })));
    } catch {}
  }

  // -------------------------------------------------------------- wiring

  function wire() {
    if (!$("s-new")) return;
    $("s-new").onclick = newPage;
    $("s-gutter").oninput = () => {
      $("s-gutter-v").textContent = Number($("s-gutter").value).toFixed(3);
      if (S.page) { S.page.gutter = Number($("s-gutter").value); renderPage(); }
    };
    $("s-steps").oninput = () => ($("s-steps-v").textContent = $("s-steps").value);
    $("s-cfg").oninput = () =>
      ($("s-cfg-v").textContent = Number($("s-cfg").value).toFixed(1));
    $("s-generate").onclick = () => generate(null);
    $("s-addbubble").onclick = () => {
      S.page.bubbles.push({
        x: .10, y: .10, w: .26, h: .10, text: "New line", style: "speech",
        font: null, font_size: null, vertical: false,
        tail_x: null, tail_y: null, rotation: 0,
      });
      select("bubble", S.page.bubbles.length - 1);
      renderPage();
    };
    $("s-save").onclick = async () => {
      const name = $("s-name").value.trim() || "untitled";
      await post("/api/projects", { name, page: S.page });
      await refreshProjects();
    };
    $("s-load").onclick = async () => {
      const name = $("s-projects").value;
      if (!name) return;
      const data = await api(`/api/projects/${encodeURIComponent(name)}`);
      S.page = data.page; S.sel = null; $("s-name").value = data.name;
      await renderPage();
    };
    $("s-export").onclick = async () => {
      const name = ($("s-name").value.trim() || "page") + "-export";
      const out = await post("/api/page/render", { page: S.page, name });
      const native = window.pywebview && window.pywebview.api;
      if (native) {
        // The desktop app has no browser tab to open: save via a native panel.
        await native.save_output(out.image, `${name}.png`);
      } else {
        window.open(`/outputs/${out.image}`, "_blank");
      }
    };
    $("c-save").onclick = async () => {
      const body = {
        id: $("c-id").value || null,
        name: $("c-name").value.trim() || "unnamed",
        prompt: $("c-prompt").value.trim(),
        negative_prompt: $("c-negative").value.trim(),
        seed: Number($("c-seed").value),
        loras: collectCastLoras(),
      };
      await post("/api/characters", body);
      $("c-id").value = "";
      await refreshCast();
    };
    $("c-clear").onclick = () =>
      loadCharForm({ id: "", name: "", prompt: "", negative_prompt: "", seed: -1 });
    $("c-addlora").onclick = () => addCastLoraRow();
    window.addEventListener("resize", () => drawOverlay());
  }

  wire();
})();
