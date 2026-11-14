// 手机上标注每一拍的技术类型（#/strokes/<录像编号>）。
// Mac 节点把每一拍前后 1.4 秒剪成一段，所有段首尾相接成一个 strokes.mp4，第 k 段从 k × 段长 秒开始。
// 这里循环播放当前这一段（默认慢放），选主类型（单选）、附加标记（多选）、击球人，可以写备注；
// 预先选好模型的判断，模型没把握的排在前面。每拍存到网站，Mac 节点再写回 .strokes.labels.csv。
"use strict";

const Strokes = (() => {
  let S = null;
  const $s = (id) => document.getElementById(id);
  const SPEEDS = [1, 0.5, 0.25];
  const NOT_A_HIT = "不是击球";   // 模型以为这里有一拍，其实没有

  async function open(id) {
    if (S && S.id === id) { show(); return; }
    close();
    S = { id, state: null, labels: new Map(), order: [], k: 0, sel: null, queue: new Map(), saving: false, saveError: "",
          speed: 0.5, reviewFirst: true, raf: 0 };
    $s("sMsg").textContent = "读取中";
    $s("sStage").hidden = true;
    try {
      const [state, labels] = await Promise.all([api(`/videos/${id}/strokes/files/strokes.state.json`), api(`/videos/${id}/stroke-labels`)]);
      if (!S || S.id !== id) return;
      S.state = state;
      if (!state.types.includes(NOT_A_HIT)) state.types.push(NOT_A_HIT); // 旧的小视频包里还没有
      for (const [f, lab] of Object.entries(labels.labels || {})) S.labels.set(Number(f), lab);
      const v = $s("sVideo");
      v.src = `${API}/videos/${id}/strokes/files/strokes.mp4`;
      v.muted = true;
      v.playsInline = true;
      $s("sMsg").textContent = "";
      $s("sStage").hidden = false;
      buildChoices();
      reorder();
      const first = S.order.findIndex((i) => !S.labels.has(state.strokes[i].frame));
      go(first >= 0 ? first : 0);
      loop();
    } catch (e) {
      $s("sMsg").textContent = e.status === 404 ? "击球小视频还没准备好：回到录像的「标注」里点「准备击球类型标注」。" : "读取失败：" + e.message;
    }
  }

  function close() {
    if (S) cancelAnimationFrame(S.raf);
    const v = $s("sVideo");
    if (v) { v.pause(); v.removeAttribute("src"); v.load(); }
    S = null;
  }

  // 没把握的先，其余按时间；或者全部按时间
  function reorder() {
    const n = S.state.strokes.length, all = [...Array(n).keys()];
    S.order = S.reviewFirst ? [...all.filter((i) => S.state.strokes[i].needs_review), ...all.filter((i) => !S.state.strokes[i].needs_review)] : all;
  }

  const stroke = () => S.state.strokes[S.order[S.k]];
  const clipLen = () => S.state.clip.frames / S.state.clip.fps;
  const clipStart = () => S.order[S.k] * clipLen();

  function go(k) {
    S.k = Math.max(0, Math.min(S.order.length - 1, k));
    const st = stroke(), lab = S.labels.get(st.frame);
    S.sel = lab ? { player_id: lab.player_id, type: lab.type, tags: [...lab.tags], note: lab.note || "" }
      : { player_id: st.player_id, type: st.type || "未知", tags: st.direction ? [st.direction] : [], note: "" };
    const v = $s("sVideo");
    v.currentTime = clipStart() + 0.001;
    v.playbackRate = S.speed;
    v.play().catch(() => {});
    show();
  }

  // 只在这一段里循环：到段尾就跳回段首
  function loop() {
    const v = $s("sVideo");
    const tick = () => {
      if (!S || !S.state) return;
      const a = clipStart(), b = a + clipLen() - 0.5 / S.state.clip.fps;
      if (!v.seeking && (v.currentTime >= b || v.currentTime < a - 0.05)) v.currentTime = a + 0.001;
      S.raf = requestAnimationFrame(tick);
    };
    S.raf = requestAnimationFrame(tick);
  }

  function playerName(pid) {
    if (pid === null || pid === undefined) return "不确定";
    const p = S.state.players.find((x) => String(x.id) === String(pid));
    const side = p && p.side === "near" ? "近端" : p && p.side === "far" ? "远端" : "";
    return (p && p.name) || `${side}${pid} 号`;
  }

  function buildChoices() {
    const mk = (box, items, cls) => {
      box.textContent = "";
      for (const [value, text] of items) {
        const b = document.createElement("button");
        b.textContent = text;
        b.dataset.v = value;
        b.className = cls;
        box.append(b);
      }
    };
    mk($s("sTypes"), S.state.types.map((t) => [t, t]), "chip");
    mk($s("sTags"), S.state.tags.map((t) => [t, t]), "chip");
    mk($s("sPlayers"), [...S.state.players.map((p) => [String(p.id), playerName(p.id)]), ["", "不确定"]], "chip");
    mk($s("sSpeeds"), SPEEDS.map((x) => [String(x), x === 1 ? "原速" : x + "×"]), "chip");
    $s("sTypes").onclick = (e) => { const v = e.target.dataset.v; if (v) { S.sel.type = v; show(); } };
    $s("sTags").onclick = (e) => {
      const v = e.target.dataset.v;
      if (!v) return;
      const t = S.sel.tags;
      S.sel.tags = t.includes(v) ? t.filter((x) => x !== v) : [...t.filter((x) => !(v === "直线" && x === "对角") && !(v === "对角" && x === "直线")), v];
      show();
    };
    $s("sPlayers").onclick = (e) => {
      const v = e.target.dataset.v;
      if (v === undefined) return;
      S.sel.player_id = v === "" ? null : /^\d+$/.test(v) ? Number(v) : v;
      show();
    };
    $s("sSpeeds").onclick = (e) => {
      const v = Number(e.target.dataset.v);
      if (!v) return;
      S.speed = v;
      $s("sVideo").playbackRate = v;
      show();
    };
  }

  function show() {
    if (!S || !S.state) return;
    const st = stroke(), lab = S.labels.get(st.frame), n = S.order.length;
    const done = S.state.strokes.filter((x) => S.labels.has(x.frame)).length;
    const review = S.state.strokes.filter((x) => x.needs_review).length;
    $s("sWhere").textContent = `第 ${S.k + 1}/${n} 拍 · ${fmtClock(st.time_s ?? st.frame / S.state.video.fps)} · 已标 ${done}/${n}` +
      (S.reviewFirst ? `（模型没把握的 ${review} 拍排在前面）` : "");
    const alts = (st.alternatives || []).map((a) => `${a.type} ${Math.round(a.prob * 100)}%`).join("、");
    $s("sModel").textContent = `模型：${playerName(st.player_id)}，${st.type} ${Math.round((st.prob || 0) * 100)}%` +
      (st.direction ? `，${st.direction}` : "") + (alts ? `（也可能：${alts}）` : "") + (st.needs_review ? " · 没把握" : "");
    $s("sModel").className = st.needs_review ? "warn" : "quiet";
    $s("sState").textContent = lab ? "已标：" + [lab.type, ...lab.tags].join(" · ") + (lab.note ? "（" + lab.note + "）" : "") : "还没标，下面已经按模型的判断选好";
    for (const b of $s("sTypes").children) b.classList.toggle("on", b.dataset.v === S.sel.type);
    for (const b of $s("sTags").children) b.classList.toggle("on", S.sel.tags.includes(b.dataset.v));
    const pid = S.sel.player_id === null || S.sel.player_id === undefined ? "" : String(S.sel.player_id);
    for (const b of $s("sPlayers").children) b.classList.toggle("on", b.dataset.v === pid);
    for (const b of $s("sSpeeds").children) b.classList.toggle("on", Number(b.dataset.v) === S.speed);
    if (document.activeElement !== $s("sNote")) $s("sNote").value = S.sel.note;
    $s("sPrev").disabled = S.k === 0;
    $s("sNext").disabled = S.k >= n - 1;
    $s("sClear").hidden = !lab;
    $s("sOrder").textContent = S.reviewFirst ? "改成按时间顺序" : "改成没把握的先";
    $s("sCsv").href = `${API}/videos/${S.id}/stroke-labels?format=csv`;
  }

  function save(notHit = false) {
    const st = stroke();
    if (notHit) S.sel.type = NOT_A_HIT;
    const no = S.sel.type === NOT_A_HIT;  // 不是击球：没有击球人，也没有附加标记
    const lab = { player_id: no ? null : S.sel.player_id ?? null, type: S.sel.type, tags: no ? [] : S.state.tags.filter((t) => S.sel.tags.includes(t)),
                  note: $s("sNote").value.trim() };
    S.labels.set(st.frame, lab);
    S.queue.set(st.frame, { frame: st.frame, ...lab });
    flush();
    if (S.k < S.order.length - 1) go(S.k + 1);
    else show();
  }

  function clear() {
    const st = stroke();
    S.labels.delete(st.frame);
    S.queue.set(st.frame, { frame: st.frame, delete: true });
    flush();
    go(S.k);
  }

  async function flush() {
    if (!S || S.saving || !S.queue.size) { renderSave(); return; }
    const id = S.id, items = [...S.queue.values()];
    S.saving = true;
    S.queue.clear();
    renderSave();
    try {
      await api(`/videos/${id}/stroke-labels`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ items }) });
      if (S && S.id === id) S.saveError = "";
    } catch (e) {
      if (S && S.id === id) {
        for (const it of items) if (!S.queue.has(it.frame)) S.queue.set(it.frame, it);
        S.saveError = e.message;
        setTimeout(flush, 3000);
      }
    } finally {
      if (S && S.id === id) { S.saving = false; renderSave(); if (S.queue.size && !S.saveError) flush(); }
    }
  }

  function renderSave() {
    if (!S) return;
    const el = $s("sSave");
    el.textContent = S.saveError ? `${S.queue.size} 拍还没存上（${S.saveError}），会自动重试` : S.queue.size || S.saving ? "保存中…" : "已保存";
    el.className = S.saveError ? "bad" : "quiet";
  }

  $s("sSave2").onclick = () => S && S.state && save();
  $s("sNotHit").onclick = () => S && S.state && save(true);
  $s("sPrev").onclick = () => S && go(S.k - 1);
  $s("sNext").onclick = () => S && go(S.k + 1);
  $s("sClear").onclick = () => S && clear();
  $s("sReplay").onclick = () => { if (S) { const v = $s("sVideo"); v.currentTime = clipStart() + 0.001; v.play().catch(() => {}); } };
  $s("sNote").oninput = () => { if (S && S.sel) S.sel.note = $s("sNote").value; };
  $s("sOrder").onclick = () => {
    if (!S) return;
    const i = S.order[S.k];
    S.reviewFirst = !S.reviewFirst;
    reorder();
    go(S.order.indexOf(i));
  };
  $s("sVideo").addEventListener("click", () => { const v = $s("sVideo"); if (v.paused) v.play().catch(() => {}); else v.pause(); });

  return { open, close };
})();
