// 手机上标注羽毛球的位置（#/label/<录像编号>）。
// Mac 节点把模型最没把握的几个回合切成图片（每个回合一个文件，state.json 记每帧在文件里的位置和模型的预测），
// 这里用 Range 请求一次取几帧。每一帧：模型猜对了点「对」，猜错了点画面上真正的球（拖动可以微调，放大镜跟着看），
// 没有球点「没有球」，看不清点「看不清」。每一下都存到网站，Mac 节点再同步回录像旁边的 .labels.csv。
"use strict";

const Label = (() => {
  const CHUNK = 8;            // 一次请求取几帧
  const AHEAD = 3;            // 往后预取几个块
  const KEEP_BLOBS = 320;     // 压缩的图最多留多少张（约 30 MB）
  const SNAP_PX = 16;         // 点在候选点这么近（屏幕像素）就吸到候选点上
  const DRAG_PX = 8;          // 手指移动超过这么多算拖动，不算点
  const FINE = 0.3;           // 拖动时点移动的比例：手指走 10 像素，点走 3 像素
  const LOUPE_ZOOM = 4;       // 放大镜相对屏幕上画面的倍数
  const TRAIL = 6;            // 画出前面几帧的轨迹

  let L = null;               // 当前在标的录像
  const $l = (id) => document.getElementById(id);

  // ---------- 读数据 ----------

  async function open(id) {
    if (L && L.id === id) { draw(); return; }
    close();
    L = { id, state: null, labels: new Map(), r: 0, i: 0, pending: null, blobs: new Map(), loading: new Map(), bitmap: null, bitmapKey: "",
          queue: new Map(), saving: false, saveError: "", armed: false, msg: "" };
    $l("lMsg").textContent = "读取中";
    $l("lStage").hidden = true;
    try {
      const [state, labels] = await Promise.all([api(`/videos/${id}/annot/files/state.json`), api(`/videos/${id}/labels`)]);
      if (!L || L.id !== id) return;
      L.state = state;
      for (const [f, lab] of Object.entries(labels.labels || {})) L.labels.set(Number(f), lab);
      $l("lStage").hidden = false;
      $l("lMsg").textContent = "";
      // 从第一个还没标完的回合、第一张没标的帧开始
      const start = firstTodo(0) || { r: 0, i: 0 };
      go(start.r, start.i);
      renderRallies();
    } catch (e) {
      $l("lMsg").textContent = e.status === 404 ? "标注用的图片还没准备好：回到录像的「处理」里点「准备标注」。" : "读取失败：" + e.message;
    }
  }

  function close() {
    if (L && L.bitmap) L.bitmap.close?.();
    L = null;
  }

  const rally = () => L.state.rallies[L.r];
  const frameOf = (r, i) => L.state.rallies[r].start + i;
  const done = (lab) => lab && lab[3] !== "machine";

  function firstTodo(fromRally) {
    const n = L.state.rallies.length;
    for (let k = 0; k < n; k++) {
      const r = (fromRally + k) % n, R = L.state.rallies[r];
      for (let i = 0; i < R.offsets.length; i++) if (!done(L.labels.get(R.start + i))) return { r, i };
    }
    return null;
  }

  async function chunk(r, c) {
    const key = r + ":" + c;
    if (L.loading.has(key)) return L.loading.get(key);
    const R = L.state.rallies[r];
    const a = c * CHUNK, b = Math.min(R.offsets.length, a + CHUNK) - 1;
    if (a > b) return;
    if (L.blobs.has(r + ":" + a) && L.blobs.has(r + ":" + b)) return;
    const from = R.offsets[a][0], to = R.offsets[b][0] + R.offsets[b][1] - 1;
    const id = L.id;
    const p = (async () => {
      for (let attempt = 0; ; attempt++) {
        try {
          const res = await fetch(`${API}/videos/${id}/annot/files/${R.pack}`, {
            headers: { authorization: "Bearer " + token, range: `bytes=${from}-${to}` },
          });
          if (!res.ok) throw new Error("HTTP " + res.status);
          const buf = await res.arrayBuffer();
          if (!L || L.id !== id) return;
          for (let i = a; i <= b; i++) {
            const [o, n] = R.offsets[i];
            L.blobs.set(r + ":" + i, new Blob([buf.slice(o - from, o - from + n)], { type: "image/jpeg" }));
          }
          while (L.blobs.size > KEEP_BLOBS) L.blobs.delete(L.blobs.keys().next().value);
          return;
        } catch (e) {
          if (attempt >= 4) throw e;
          await new Promise((ok) => setTimeout(ok, 800 * (attempt + 1)));
        }
      }
    })().finally(() => { if (L && L.id === id) L.loading.delete(key); });
    L.loading.set(key, p);
    return p;
  }

  async function showFrame() {
    const r = L.r, i = L.i, key = r + ":" + i;
    if (L.bitmapKey === key && L.bitmap) { draw(); return; }
    draw(); // 先画框架和文字，图片到了再画一次
    try {
      if (!L.blobs.has(key)) await chunk(r, Math.floor(i / CHUNK));
      const blob = L && L.blobs.get(key);
      if (!blob || L.r !== r || L.i !== i) return;
      const bmp = await createImageBitmap(blob);
      if (!L || L.r !== r || L.i !== i) { bmp.close?.(); return; }
      L.bitmap?.close?.();
      L.bitmap = bmp;
      L.bitmapKey = key;
      draw();
    } catch (e) {
      if (L) { L.msg = "图片读取失败：" + e.message; draw(); }
    }
    // 预取后面几块
    const c = Math.floor(i / CHUNK);
    for (let k = 1; k <= AHEAD; k++) chunk(r, c + k)?.catch(() => {});
  }

  function go(r, i) {
    const R = L.state.rallies[r];
    L.r = r;
    L.i = Math.max(0, Math.min(R.offsets.length - 1, i));
    L.pending = null;
    L.armed = false;
    L.msg = "";
    showFrame();
    renderRallies();
  }

  // ---------- 保存 ----------

  function setLabel(frame, lab) {
    if (lab) L.labels.set(frame, lab);
    else L.labels.delete(frame);
    L.queue.set(frame, lab ? { frame, visibility: lab[0], x: lab[1], y: lab[2], status: lab[3] } : { frame, delete: true });
    flush();
  }

  async function flush() {
    if (!L || L.saving || !L.queue.size) { renderSave(); return; }
    const id = L.id, items = [...L.queue.values()];
    L.saving = true;
    L.queue.clear();
    renderSave();
    try {
      await api(`/videos/${id}/labels`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ items }) });
      if (L && L.id === id) L.saveError = "";
    } catch (e) {
      if (L && L.id === id) {
        for (const it of items) if (!L.queue.has(it.frame)) L.queue.set(it.frame, it); // 新的改动优先
        L.saveError = e.message;
        setTimeout(flush, 3000);
      }
    } finally {
      if (L && L.id === id) { L.saving = false; renderSave(); if (L.queue.size && !L.saveError) flush(); }
    }
  }

  function renderSave() {
    if (!L) return;
    const n = L.queue.size + (L.saving ? 1 : 0);
    const el = $l("lSave");
    el.textContent = L.saveError ? `${L.queue.size} 帧还没存上（${L.saveError}），会自动重试` : n ? "保存中…" : "已保存";
    el.className = L.saveError ? "bad" : "quiet";
  }

  // ---------- 操作 ----------

  function advance() {
    const R = rally();
    if (L.i + 1 < R.offsets.length) return go(L.r, L.i + 1);
    const next = firstTodo(L.r);
    if (next) { go(next.r, next.i); L.msg = `回合 ${L.r + 1}：接着标没标完的`; draw(); }
    else { L.msg = "全部标完了！"; draw(); }
  }

  function confirm() {
    const f = frameOf(L.r, L.i), R = rally(), p = R.pred[L.i];
    if (L.pending) setLabel(f, [1, round(L.pending.x), round(L.pending.y), "human"]);
    else if (done(L.labels.get(f))) { /* 已经标过：保持不变 */ }
    else if (p[0]) setLabel(f, [1, p[1], p[2], "human"]);
    else setLabel(f, [0, 0, 0, "human"]);
    advance();
  }

  function noShuttle() { setLabel(frameOf(L.r, L.i), [0, 0, 0, "human"]); advance(); }
  function cantTell() { setLabel(frameOf(L.r, L.i), [0, 0, 0, "skip"]); advance(); }
  function clearFrame() { setLabel(frameOf(L.r, L.i), null); L.pending = null; draw(); renderRallies(); }

  // 这个回合里模型有把握、还没人看过的帧，一次全部接受（标成 bulk，和电脑版一样）
  function confidentTodo() {
    const R = rally(), out = [];
    for (let i = 0; i < R.offsets.length; i++) {
      if (!done(L.labels.get(R.start + i)) && R.unc[i] < L.state.confident) out.push(i);
    }
    return out;
  }
  function acceptConfident() {
    const todo = confidentTodo();
    if (!todo.length) return;
    if (!L.armed) { L.armed = true; draw(); return; }
    const R = rally();
    for (const i of todo) {
      const p = R.pred[i];
      const lab = p[0] ? [1, p[1], p[2], "bulk"] : [0, 0, 0, "bulk"];
      L.labels.set(R.start + i, lab);
      L.queue.set(R.start + i, { frame: R.start + i, visibility: lab[0], x: lab[1], y: lab[2], status: "bulk" });
    }
    flush();
    L.armed = false;
    L.msg = `接受了 ${todo.length} 帧`;
    const next = firstTodo(L.r);
    if (next && next.r === L.r) go(next.r, next.i);
    else draw();
    renderRallies();
  }

  const round = (v) => Math.round(v * 10) / 10;

  // ---------- 画面 ----------

  function layout() {
    const cv = $l("lCanvas"), [vw, vh] = L.state.video.size;
    const landscape = window.innerWidth > window.innerHeight;
    let w = cv.parentElement.clientWidth;
    if (landscape) w = Math.min(w, (window.innerHeight - 24) * vw / vh);
    const h = w * vh / vw;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.width = w + "px";
      cv.style.height = h + "px";
    }
    return { w, h, dpr, s: w / vw };
  }

  function draw() {
    if (!L || !L.state) return;
    const { w, h, dpr, s } = layout();
    const cv = $l("lCanvas"), g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#000";
    g.fillRect(0, 0, w, h);
    const key = L.r + ":" + L.i;
    if (L.bitmap && L.bitmapKey === key) g.drawImage(L.bitmap, 0, 0, w, h);
    else {
      if (L.bitmap) { g.globalAlpha = 0.35; g.drawImage(L.bitmap, 0, 0, w, h); g.globalAlpha = 1; }
      g.fillStyle = "#fff"; g.font = "14px -apple-system, sans-serif"; g.fillText("图片读取中…", 12, 24);
    }
    const R = rally(), f = frameOf(L.r, L.i);
    // 前几帧的轨迹（标注优先，没有就用预测）
    g.lineWidth = 1.5;
    let prev = null;
    for (let k = TRAIL; k >= 1; k--) {
      const i = L.i - k;
      if (i < 0) continue;
      const lab = L.labels.get(R.start + i), p = done(lab) ? lab : R.pred[i];
      if (!p[0]) { prev = null; continue; }
      const pt = [p[1] * s, p[2] * s];
      g.fillStyle = g.strokeStyle = `rgba(255,214,0,${0.25 + 0.6 * (TRAIL - k + 1) / TRAIL})`;
      g.beginPath(); g.arc(pt[0], pt[1], 2.5, 0, 7); g.fill();
      if (prev) { g.beginPath(); g.moveTo(prev[0], prev[1]); g.lineTo(pt[0], pt[1]); g.stroke(); }
      prev = pt;
    }
    // 其它候选点（模型看到的、会动的亮点）
    if (R.cand) {
      g.strokeStyle = "rgba(255,255,255,.7)"; g.lineWidth = 1;
      for (const [x, y] of R.cand[L.i] || []) { g.beginPath(); g.arc(x * s, y * s, 7, 0, 7); g.stroke(); }
    }
    // 模型的预测
    const p = R.pred[L.i];
    if (p[0]) ring(g, p[1] * s, p[2] * s, 12, "#ff8c00", 2.5);
    // 已有的标注 / 正在改的点
    const lab = L.labels.get(f);
    if (L.pending) cross(g, L.pending.x * s, L.pending.y * s, "#2f81f7");
    else if (done(lab) && lab[0]) cross(g, lab[1] * s, lab[2] * s, lab[3] === "bulk" ? "#7ee787" : "#3fb950");
    if (L.pending) loupe(L.pending, s);
    else $l("lLoupe").hidden = true;
    renderInfo();
  }

  function ring(g, x, y, r, color, width) {
    g.strokeStyle = color; g.lineWidth = width;
    g.beginPath(); g.arc(x, y, r, 0, 7); g.stroke();
  }
  function cross(g, x, y, color) {
    g.strokeStyle = "#000"; g.lineWidth = 4;
    line(g, x, y);
    g.strokeStyle = color; g.lineWidth = 2;
    line(g, x, y);
  }
  function line(g, x, y) {
    g.beginPath();
    g.moveTo(x - 14, y); g.lineTo(x - 4, y); g.moveTo(x + 4, y); g.lineTo(x + 14, y);
    g.moveTo(x, y - 14); g.lineTo(x, y - 4); g.moveTo(x, y + 4); g.lineTo(x, y + 14);
    g.stroke();
  }

  // 放大镜：点附近的一小块放大显示，放在离点远的那一角
  function loupe(pt, s) {
    const lc = $l("lLoupe");
    if (!L.bitmap) { lc.hidden = true; return; }
    const size = 150, dpr = window.devicePixelRatio || 1;
    lc.hidden = false;
    lc.width = lc.height = size * dpr;
    lc.style.width = lc.style.height = size + "px";
    const cvW = $l("lCanvas").clientWidth;
    lc.style.left = pt.x * s > cvW / 2 ? "6px" : "";
    lc.style.right = pt.x * s > cvW / 2 ? "" : "6px";
    const g = lc.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.imageSmoothingEnabled = false;
    const k = L.bitmap.width / L.state.video.size[0];   // 视频像素 → 图片像素
    const span = size / (s * LOUPE_ZOOM) * k;           // 放大镜里看到的图片像素
    g.fillStyle = "#000"; g.fillRect(0, 0, size, size);
    g.drawImage(L.bitmap, pt.x * k - span / 2, pt.y * k - span / 2, span, span, 0, 0, size, size);
    const R = rally(), p = R.pred[L.i];
    const z = size / span * k;                          // 视频像素 → 放大镜像素
    if (p[0]) ring(g, size / 2 + (p[1] - pt.x) * z, size / 2 + (p[2] - pt.y) * z, 12, "#ff8c00", 2);
    cross(g, size / 2, size / 2, "#2f81f7");
  }

  function renderInfo() {
    const R = rally(), f = frameOf(L.r, L.i), lab = L.labels.get(f), p = R.pred[L.i];
    const n = R.offsets.length, fps = L.state.video.fps || 30;
    const doneN = countDone(L.r);
    $l("lWhere").textContent = `回合 ${L.r + 1}/${L.state.rallies.length} · 第 ${L.i + 1}/${n} 帧 · ${((f) / fps).toFixed(2)} 秒 · 这回合已标 ${doneN}/${n}`;
    const unsure = R.unc[L.i] >= L.state.doubtful ? "，模型很没把握" : R.unc[L.i] >= L.state.confident ? "，模型不太有把握" : "";
    const what = L.pending ? "你点的位置（蓝色）：点「对」保存，拖动可以微调"
      : done(lab) ? (lab[3] === "skip" ? "已标：看不清" : lab[0] ? `已标：有球（${lab[3] === "bulk" ? "一键接受的" : "绿色十字"}）` : "已标：没有球")
      : p[0] ? `模型猜的位置是橙色圈${unsure}` : `模型没找到球${unsure}`;
    $l("lWhat").textContent = L.msg || what;
    $l("lOk").textContent = L.pending ? "对（用蓝色的点）" : done(lab) ? "下一帧" : p[0] ? "对（橙圈）" : "对（没有球）";
    const todo = confidentTodo().length;
    const bulk = $l("lBulk");
    bulk.hidden = !todo;
    bulk.textContent = L.armed ? `再点一次：接受这回合 ${todo} 帧有把握的预测` : `这回合剩下有把握的 ${todo} 帧一键接受`;
    $l("lPrev").disabled = L.i === 0;
    $l("lClear").hidden = !done(lab) && !L.pending;
  }

  function countDone(r) {
    const R = L.state.rallies[r];
    let n = 0;
    for (let i = 0; i < R.offsets.length; i++) if (done(L.labels.get(R.start + i))) n++;
    return n;
  }

  function renderRallies() {
    if (!L || !L.state) return;
    const box = $l("lRallies");
    box.textContent = "";
    L.state.rallies.forEach((R, r) => {
      const b = document.createElement("button");
      const n = R.offsets.length, d = countDone(r);
      b.textContent = `回合 ${r + 1} · ${(n / (L.state.video.fps || 30)).toFixed(0)} 秒 · ${d === n ? "✓" : Math.floor(d * 100 / n) + "%"}`;
      b.className = r === L.r ? "on" : "";
      b.onclick = () => { const t = firstTodo(r); go(r, t && t.r === r ? t.i : 0); };
      box.append(b);
    });
    renderAccuracy();
  }

  // 模型在人标过的帧上准不准：和 TrackNetV3 的评测一样，距离在 4 个模型像素（288 高）以内算找对
  function renderAccuracy() {
    const [, vh] = L.state.video.size, tol = 4 * vh / 288;
    let vis = 0, hit = 0, miss = 0, far = 0, none = 0, fp = 0;
    const errs = [];
    L.state.rallies.forEach((R) => {
      for (let i = 0; i < R.offsets.length; i++) {
        const lab = L.labels.get(R.start + i);
        if (!lab || lab[3] !== "human") continue; // 一键接受的和看不清的不算：它们本来就和预测一样或者没法判断
        const p = R.pred[i];
        if (lab[0]) {
          vis++;
          if (!p[0]) { miss++; continue; }
          const d = Math.hypot(p[1] - lab[1], p[2] - lab[2]);
          errs.push(d);
          if (d <= tol) hit++; else far++;
        } else { none++; if (p[0]) fp++; }
      }
    });
    const el = $l("lAcc");
    if (!vis && !none) { el.textContent = "标几帧之后，这里会显示模型在你标过的帧上准不准。"; return; }
    errs.sort((a, b) => a - b);
    const pct = (a, b) => b ? Math.round(a * 100 / b) + "%" : "—";
    const mid = errs.length >> 1, med = errs.length ? (errs.length % 2 ? errs[mid] : (errs[mid - 1] + errs[mid]) / 2) : null;
    const m = L.state.model || {};
    el.textContent = "";
    const rows = [
      ["整体准确率", pct(hit + (none - fp), vis + none), `${vis + none} 帧，你亲手标的`],
      ["有球时找对", pct(hit, vis), `${vis} 帧有球：找对 ${hit}，位置偏了 ${far}，漏掉 ${miss}`],
      ["没球时误报", pct(fp, none), `${none} 帧没球，模型说有 ${fp}`],
      ["位置误差中位数", med === null ? "—" : med.toFixed(0) + " 像素", `找对 = 差距在 ${tol.toFixed(0)} 像素内（视频 ${L.state.video.size[0]}×${vh}）`],
    ];
    const t = document.createElement("table");
    for (const [a, b, c] of rows) {
      const tr = t.insertRow();
      tr.insertCell().textContent = a;
      tr.insertCell().textContent = b;
      const td = tr.insertCell(); td.textContent = c; td.className = "quiet";
    }
    el.append(t);
    const note = document.createElement("p");
    note.className = "quiet";
    note.textContent = `预测来自 ${m.predicted_at ? m.predicted_at.slice(0, 16).replace("T", " ") : "—"} 的模型${m.stamp ? "（" + m.stamp + "）" : ""}。微调后在录像页点「用新模型重新预测」，同样这些帧再看一次这个数。`;
    el.append(note);
  }

  // ---------- 手势：点 = 放点（靠近候选点会吸过去），拖 = 微调 ----------

  let touch = null;
  function onDown(e) {
    if (!L || !L.state) return;
    const cv = $l("lCanvas");
    cv.setPointerCapture(e.pointerId);
    const R = rally(), lab = L.labels.get(frameOf(L.r, L.i)), p = R.pred[L.i];
    const base = L.pending || (done(lab) && lab[0] ? { x: lab[1], y: lab[2] } : p[0] ? { x: p[1], y: p[2] } : null);
    touch = { x: e.clientX, y: e.clientY, base: base && { ...base }, dragging: false };
    e.preventDefault();
  }
  function onMove(e) {
    if (!touch) return;
    const dx = e.clientX - touch.x, dy = e.clientY - touch.y;
    if (!touch.dragging && Math.hypot(dx, dy) < DRAG_PX) return;
    touch.dragging = true;
    if (!touch.base) return;
    const s = layout().s, [vw, vh] = L.state.video.size;
    L.pending = { x: clamp(touch.base.x + dx * FINE / s, 0, vw), y: clamp(touch.base.y + dy * FINE / s, 0, vh) };
    L.msg = "";
    draw();
    e.preventDefault();
  }
  function onUp(e) {
    if (!touch) return;
    const t = touch;
    touch = null;
    if (t.dragging) return;
    const rect = $l("lCanvas").getBoundingClientRect(), s = layout().s;
    let x = (e.clientX - rect.left) / s, y = (e.clientY - rect.top) / s;
    // 吸到最近的候选点
    const R = rally();
    let best = SNAP_PX / s;
    for (const [cx, cy] of (R.cand && R.cand[L.i]) || []) {
      const d = Math.hypot(cx - x, cy - y);
      if (d < best) { best = d; x = cx; y = cy; }
    }
    L.pending = { x, y };
    L.msg = "";
    draw();
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function bind() {
    const cv = $l("lCanvas");
    cv.addEventListener("pointerdown", onDown);
    cv.addEventListener("pointermove", onMove);
    cv.addEventListener("pointerup", onUp);
    cv.addEventListener("pointercancel", () => { touch = null; });
    $l("lOk").onclick = () => confirm();
    $l("lNone").onclick = () => noShuttle();
    $l("lSkip").onclick = () => cantTell();
    $l("lPrev").onclick = () => go(L.r, L.i - 1);
    $l("lNext").onclick = () => { if (L.i + 1 < rally().offsets.length) go(L.r, L.i + 1); };
    $l("lClear").onclick = () => clearFrame();
    $l("lBulk").onclick = () => acceptConfident();
    window.addEventListener("resize", () => { if (L && L.state && !$l("vLabel").hidden) draw(); });
    // 电脑上也能用键盘：空格/回车 = 对，n = 没有球，s = 看不清，左右方向键 = 上一帧/下一帧，Backspace = 清除
    window.addEventListener("keydown", (e) => {
      if (!L || !L.state || $l("vLabel").hidden || e.target.tagName === "INPUT") return;
      const k = e.key;
      if (k === " " || k === "Enter") confirm();
      else if (k === "n" || k === "N") noShuttle();
      else if (k === "s" || k === "S") cantTell();
      else if (k === "ArrowLeft") go(L.r, L.i - 1);
      else if (k === "ArrowRight") { if (L.i + 1 < rally().offsets.length) go(L.r, L.i + 1); }
      else if (k === "Backspace") clearFrame();
      else if (k === "Escape") { L.pending = null; draw(); }
      else return;
      e.preventDefault();
    });
  }
  bind();

  return { open, close, get current() { return L && L.id; } };
})();
