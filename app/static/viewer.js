(() => {
  const $ = (id) => document.getElementById(id);
  const canvas = $("canvas");
  const ctx = canvas.getContext("2d");
  const stage = $("stage");

  // Colour = BI-RADS group: green 1-2, yellow 3 (and 0), red 4-5.
  const COLORS = { normal: "#34c77b", low: "#34c77b", medium: "#f5b83d", high: "#ff4d5e" };
  const LEVEL_RU = { low: "доброкачественное", medium: "вероятно доброкачественное, контроль", high: "подозрение на рак" };
  const BIRADS_RU = {
    1: "норма",
    2: "доброкачественные изменения",
    3: "вероятно доброкачественное, контроль",
    4: "подозрение на рак",
    5: "высокое подозрение на рак",
  };
  const TYPE_RU = {
    mass: "Образование",
    calcifications: "Кальцинаты",
    asymmetry: "Асимметрия",
    architectural_distortion: "Нарушение архитектоники",
    other: "Другое",
  };

  const view = { scale: 1, x: 0, y: 0 };
  let image = null;
  let result = null;
  let lastFile = null;
  let showOverlay = true;
  let invert = false;
  let hovered = -1;
  let drag = null;

  // ---------- rendering ----------
  function resize() {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = stage.clientWidth * dpr;
    canvas.height = stage.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function fit() {
    if (!image) return;
    const w = stage.clientWidth, h = stage.clientHeight;
    view.scale = Math.min(w / image.width, h / image.height) * 0.95;
    view.x = (w - image.width * view.scale) / 2;
    view.y = (h - image.height * view.scale) / 2;
    draw();
  }

  function filterString() {
    return `brightness(${$("brightness").value}%) contrast(${$("contrast").value}%)` + (invert ? " invert(1)" : "");
  }

  function drawImageAndOverlay(c, scale, ox, oy, highlight) {
    c.save();
    c.filter = filterString();
    c.imageSmoothingQuality = "high";
    c.drawImage(image, ox, oy, image.width * scale, image.height * scale);
    c.restore();
    if (!showOverlay || !result) return;
    drawBreastOutline(c, scale, ox, oy);

    result.findings.forEach((f, i) => {
      const [x1, y1, x2, y2] = f.bbox_xyxy;
      const x = ox + x1 * scale, y = oy + y1 * scale;
      const w = (x2 - x1) * scale, h = (y2 - y1) * scale;
      const color = COLORS[f.level];
      const active = i === highlight;

      c.save();
      c.strokeStyle = color;
      c.lineWidth = active ? 3 : 2;
      c.shadowColor = "rgba(0,0,0,.8)";
      c.shadowBlur = 4;
      c.strokeRect(x, y, w, h);
      if (active) {
        c.fillStyle = color + "22";
        c.fillRect(x, y, w, h);
      }
      c.restore();

      const label = `${i + 1}. ${TYPE_RU[f.type] || f.type} · BI-RADS ${f.birads} · ${Math.round(f.malignancy_prob * 100)}%`;
      c.save();
      c.font = "600 12px system-ui, sans-serif";
      const tw = c.measureText(label).width + 10;
      // Keep the label inside the visible canvas (narrow screens, zoomed-in views).
      const cw = c.canvas.width / (c === ctx ? window.devicePixelRatio || 1 : 1);
      const lx = Math.max(0, Math.min(x, cw - tw));
      const ly = y - 20 < 0 ? y + h + 2 : y - 20;
      c.fillStyle = color;
      c.fillRect(lx, ly, tw, 18);
      c.fillStyle = "#0b0d10";
      c.fillText(label, lx + 5, ly + 13);
      c.restore();
    });
  }

  // Outline of the whole breast in the colour of its BI-RADS category: BI-RADS 1-2 have no lesion
  // boxes, so this is how they are shown on the image.
  function drawBreastOutline(c, scale, ox, oy) {
    const pts = result.breast_contour || [];
    if (pts.length < 3 || !result.overall_birads) return;
    const color = COLORS[result.overall_level] || COLORS.medium;
    c.save();
    c.beginPath();
    pts.forEach(([x, y], i) => (i ? c.lineTo : c.moveTo).call(c, ox + x * scale, oy + y * scale));
    c.closePath();
    c.strokeStyle = color;
    c.lineWidth = 2;
    c.setLineDash([8, 6]);
    c.shadowColor = "rgba(0,0,0,.8)";
    c.shadowBlur = 4;
    c.stroke();
    c.restore();

    const label = `BI-RADS ${result.overall_birads} — ${BIRADS_RU[result.overall_birads]}`;
    const top = pts.reduce((a, p) => (p[1] < a[1] ? p : a));
    const xs = pts.map((p) => p[0]);
    const cx = ox + ((Math.min(...xs) + Math.max(...xs)) / 2) * scale;
    c.save();
    c.font = "600 13px system-ui, sans-serif";
    const tw = c.measureText(label).width + 12;
    const cw = c.canvas.width / (c === ctx ? window.devicePixelRatio || 1 : 1);
    const lx = Math.max(0, Math.min(cx - tw / 2, cw - tw));
    const ly = Math.max(0, oy + top[1] * scale + 6);
    c.fillStyle = color;
    c.fillRect(lx, ly, tw, 20);
    c.fillStyle = "#0b0d10";
    c.fillText(label, lx + 6, ly + 14);
    c.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) return;
    drawImageAndOverlay(ctx, view.scale, view.x, view.y, hovered);
  }

  // ---------- interaction ----------
  function toImage(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    return [(clientX - r.left - view.x) / view.scale, (clientY - r.top - view.y) / view.scale];
  }

  canvas.addEventListener("wheel", (e) => {
    if (!image) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015);
    const next = Math.min(Math.max(view.scale * k, 0.05), 20);
    view.x = mx - (mx - view.x) * (next / view.scale);
    view.y = my - (my - view.y) * (next / view.scale);
    view.scale = next;
    draw();
  }, { passive: false });

  canvas.addEventListener("pointerdown", (e) => {
    if (!image) return;
    drag = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    canvas.setPointerCapture(e.pointerId);
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointerup", () => { drag = null; canvas.classList.remove("dragging"); });
  canvas.addEventListener("pointermove", (e) => {
    if (!image) return;
    if (drag) {
      view.x = drag.vx + e.clientX - drag.x;
      view.y = drag.vy + e.clientY - drag.y;
      draw();
      return;
    }
    const [ix, iy] = toImage(e.clientX, e.clientY);
    $("coords").textContent = ix >= 0 && iy >= 0 && ix < image.width && iy < image.height
      ? `x ${Math.round(ix)}  y ${Math.round(iy)}  ·  ${Math.round(view.scale * 100)}%` : "";
    let hit = -1;
    if (result) {
      result.findings.forEach((f, i) => {
        const [x1, y1, x2, y2] = f.bbox_xyxy;
        if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) hit = i;
      });
    }
    setHover(hit);
  });

  function setHover(i) {
    if (i === hovered) return;
    hovered = i;
    document.querySelectorAll("#findings li").forEach((li, j) => li.classList.toggle("hl", j === i));
    draw();
  }

  function focusFinding(i) {
    const [x1, y1, x2, y2] = result.findings[i].bbox_xyxy;
    const w = stage.clientWidth, h = stage.clientHeight;
    view.scale = Math.min(Math.min(w / (x2 - x1), h / (y2 - y1)) * 0.35, 8);
    view.x = w / 2 - ((x1 + x2) / 2) * view.scale;
    view.y = h / 2 - ((y1 + y2) / 2) * view.scale;
    setHover(i);
    draw();
  }

  // ---------- results panel ----------
  function renderPanel() {
    const list = $("findings");
    list.innerHTML = "";
    const level = result ? result.overall_level : "none";
    const overall = $("overall");
    overall.className = `overall level-${result ? level : "none"}`;
    overall.textContent = !result ? "Нет снимка"
      : result.overall_birads ? `BI-RADS ${result.overall_birads} — ${BIRADS_RU[result.overall_birads]}`
      : "Находок нет";
    $("filename").textContent = result ? `${result.filename} · ${result.width}×${result.height}` : "";
    const sc = result && result.screening;
    $("screening").hidden = !sc;
    if (sc) {
      $("screening").className = `screening ${sc.verdict}`;
      const bars = Object.entries(sc.birads_probs).map(([b, p]) =>
        `<span class="bp"><i style="height:${Math.max(2, Math.round(p * 40))}px" class="b${b}"></i>${b}</span>`).join("");
      const group = sc.birads <= 2 ? "low" : sc.birads === 3 ? "medium" : "high";
      $("screening").innerHTML =
        `Вся грудь: <b style="color:${COLORS[group]}">BI-RADS ${sc.birads}</b> — ${BIRADS_RU[sc.birads]}` +
        `<div class="small muted">BI-RADS ≥3: ${Math.round(sc.p_suspicious * 100)}% ` +
        `(${sc.verdict === "normal" ? "ниже порога" : "требует внимания"}) · ` +
        `BI-RADS 4–5: ${Math.round(sc.p_cancer * 100)}%</div>` +
        `<div class="bprobs">${bars}</div>`;
    }

    if (result && !result.findings.length) {
      list.innerHTML = '<li class="muted" style="border-left-color:transparent">Очагов не найдено при текущем пороге</li>';
      return;
    }
    (result ? result.findings : []).forEach((f, i) => {
      const li = document.createElement("li");
      li.className = `level-${f.level}`;
      const pct = Math.round(f.malignancy_prob * 100);
      li.innerHTML = `
        <div class="title"><span>${i + 1}. ${TYPE_RU[f.type] || f.type}</span><span>${pct}%</span></div>
        <div class="meta">BI-RADS ${f.birads} · ${LEVEL_RU[f.level]} · подозрение на рак ${pct}% · детекция ${Math.round(f.type_conf * 100)}%</div>
        <div class="bar"><i class="level-${f.level}" style="width:${pct}%"></i></div>`;
      li.addEventListener("mouseenter", () => setHover(i));
      li.addEventListener("mouseleave", () => setHover(-1));
      li.addEventListener("click", () => focusFinding(i));
      list.appendChild(li);
    });
  }

  // ---------- upload ----------
  async function analyze(file) {
    lastFile = file;
    $("busy").hidden = false;
    $("busy").textContent = "Анализ…";
    try {
      const fd = new FormData();
      fd.append("file", file);
      const resp = await fetch(`/api/analyze?conf=${$("conf").value}`, { method: "POST", body: fd });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || resp.statusText);
      const img = new Image();
      img.onload = () => {
        image = img;
        result = data;
        hovered = -1;
        $("drop").classList.add("hidden");
        renderPanel();
        fit();
        $("busy").hidden = true;
      };
      img.src = `data:image/png;base64,${data.image_png_base64}`;
    } catch (err) {
      $("busy").textContent = `Ошибка: ${err.message}`;
      setTimeout(() => { $("busy").hidden = true; }, 4000);
    }
  }

  $("file").addEventListener("change", (e) => e.target.files[0] && analyze(e.target.files[0]));
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("over"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("over"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault();
    stage.classList.remove("over");
    if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
  });

  // ---------- toolbar ----------
  $("fit").onclick = fit;
  $("zoom1").onclick = () => {
    if (!image) return;
    view.scale = 1;
    view.x = (stage.clientWidth - image.width) / 2;
    view.y = (stage.clientHeight - image.height) / 2;
    draw();
  };
  $("toggle").onclick = () => { showOverlay = !showOverlay; $("toggle").classList.toggle("active", !showOverlay); draw(); };
  $("invert").onclick = () => { invert = !invert; $("invert").classList.toggle("active", invert); draw(); };
  $("brightness").oninput = draw;
  $("contrast").oninput = draw;
  $("reset").onclick = () => { $("brightness").value = 100; $("contrast").value = 100; draw(); };
  $("conf").oninput = () => { $("confOut").textContent = Number($("conf").value).toFixed(2); };
  $("conf").onchange = () => lastFile && analyze(lastFile);

  $("export").onclick = () => {
    if (!image) return;
    const off = document.createElement("canvas");
    off.width = image.width;
    off.height = image.height;
    drawImageAndOverlay(off.getContext("2d"), 1, 0, 0, -1);
    const a = document.createElement("a");
    a.download = (result?.filename || "mammogram").replace(/\.[^.]+$/, "") + "_mammoai.png";
    a.href = off.toDataURL("image/png");
    a.click();
  };

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.key === "f" || e.key === "F") fit();
    if (e.key === "o" || e.key === "O") $("toggle").click();
    if (e.key === "i" || e.key === "I") $("invert").click();
  });

  new ResizeObserver(resize).observe(stage);
  renderPanel();
})();
