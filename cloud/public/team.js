// 「标注」这一边：任务（#/tasks）、素材（#/materials，管理员）、审核（#/review，管理员）、标注页顶上的任务条；
// 还有账号菜单里的成员页（#/team，管理员）。
// 任务 = 一个回合的球位置，或约 20 拍的击球类型。领了之后只有自己能标这一段，标完提交。
"use strict";

const Team = (() => {
  const $t = (id) => document.getElementById(id);
  let tasks = [], info = null, timer = 0, page = null, barTask = null, lastList = "tasks", materials = null;
  const errBox = () => $t(page === "review" ? "rvError" : page === "materials" ? "mtError" : "tError");
  const post = (path, body) => api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });

  // 右上角的头像和账号菜单
  function setUser() {
    const name = (me && (me.name || me.email)) || "管理员";
    const av = $t("navAvatar");
    av.textContent = me && me.picture ? "" : name.slice(0, 1).toUpperCase();
    av.style.backgroundImage = me && me.picture ? `url("${me.picture}")` : "";
    $t("navName").textContent = name;
    $t("navEmail").textContent = (me && me.email) || "";
    $t("navRole").textContent = isOwner() ? "超级管理员" : isAdmin() ? "管理员" : "标注员";
  }
  // 点菜单外面就收起来
  document.addEventListener("click", (e) => { if (!$t("navMenu").contains(e.target)) $t("navMenu").open = false; });

  // ---------- 任务 ----------

  const linkOf = (t) => `#/${t.kind === "shuttle" ? "label" : "strokes"}/${t.videoId}/t${t.id}`;
  function describe(t) {
    const fps = t.fps || 30;
    if (t.kind === "shuttle") return `球位置 · 回合 ${t.idx + 1} · ${t.items} 帧（约 ${Math.max(1, Math.round(t.items / fps))} 秒）`;
    return `击球类型 · ${t.items} 拍 · ${fmtClock(t.start / fps)}–${fmtClock(t.end / fps)}`;
  }
  function leaseText(t) {
    if (!t.leaseUntil || !info) return "";
    const left = Math.max(0, Math.round((new Date(t.leaseUntil) - new Date(info.now)) / 60000));
    return left >= 60 ? `还剩 ${Math.floor(left / 60)} 小时 ${left % 60} 分` : `还剩 ${left} 分`;
  }
  const who = (t) => (t.assignee ? t.assignee.name || t.assignee.email : "");

  async function load() {
    info = await api("/tasks");
    tasks = info.tasks;
  }

  async function refresh() {
    clearTimeout(timer);
    const shown = page;
    try {
      await load();
      if (shown === "materials") materials = await api("/labeling/videos");
      errBox().hidden = true;
    } catch (e) {
      errBox().hidden = false;
      errBox().textContent = "读取失败：" + e.message;
    }
    if (page !== shown || !page) return;
    render();
    const preparing = shown === "materials" && materials && materials.videos.some((v) => [v.annot.status, v.strokes.status].some((x) => x === "queued" || x === "preparing"));
    timer = setTimeout(refresh, preparing ? 4000 : 20000);
  }

  async function act(t, action, body) {
    try {
      info = await post(`/tasks/${t.id}/${action}`, body);
      tasks = info.tasks;
      errBox().hidden = true;
      render();
      return true;
    } catch (e) {
      errBox().hidden = false;
      errBox().textContent = e.message;
      refresh();
      return false;
    }
  }

  function button(text, cls, fn) {
    const b = el("button", cls, text);
    b.onclick = fn;
    return b;
  }
  function link(text, href, cls) {
    const a = el("a", "pick " + (cls || ""), text);
    a.href = href;
    if (cls === "plain") a.style.cssText = "background:none;color:var(--accent);border:1px solid var(--line);font-weight:400";
    return a;
  }

  // 没标完就提交：再点一次才真的提交
  function submitButton(t, after) {
    const b = button("提交", "plain", async () => {
      const left = t.items - t.done;
      if (left > 0 && b.dataset.armed !== "1") {
        b.dataset.armed = "1";
        b.textContent = `还有 ${left} ${t.kind === "shuttle" ? "帧" : "拍"}没标，仍然提交`;
        return;
      }
      if ((await act(t, "submit")) && after) after();
    });
    return b;
  }

  function row(t, sub, btns) {
    const card = el("div", "card task");
    const body = el("div", "info");
    body.append(el("div", "name", t.video), el("div", "quiet", describe(t)), el("div", "quiet", sub));
    const bar = el("div", "mini"), fill = el("div");
    fill.style.width = (t.items ? (t.done * 100) / t.items : 0).toFixed(1) + "%";
    bar.append(fill);
    body.append(bar);
    const box = el("div", "btns");
    box.append(...btns);
    card.append(body, box);
    return card;
  }

  const unit = (t) => (t.kind === "shuttle" ? "帧" : "拍");
  const prog = (t) => `已标 ${t.done}/${t.items} ${unit(t)}`;
  function group(box, title, list, make) {
    if (!list.length) return;
    box.append(el("div", "tgroup", `${title}（${list.length}）`));
    for (const t of list) box.append(make(t));
  }

  function render() {
    const waiting = tasks.filter((t) => t.status === "submitted").length;
    for (const n of document.querySelectorAll("[data-review]")) n.textContent = isAdmin() && waiting ? String(waiting) : "";
    if (page === "tasks") renderTasks();
    else if (page === "review") renderReview();
    else if (page === "materials") renderMaterials();
  }

  // 任务：我手上的、可以领的
  function renderTasks() {
    const box = $t("tList");
    box.textContent = "";
    group(box, "我手上的", tasks.filter((t) => t.status === "claimed" && t.mine), (t) =>
      row(t, `${prog(t)} · ${leaseText(t)}自动放回`, [
        link("继续标注", linkOf(t)),
        submitButton(t),
        button("放弃", "plain danger", () => act(t, "release")),
      ]));
    group(box, "可以领取", tasks.filter((t) => t.status === "open"), (t) =>
      row(t, t.done ? `${prog(t)}（别人标了一部分）` : "还没有人标", [
        button("领取并开始", "", async () => { if (await act(t, "claim")) location.hash = linkOf(t); }),
      ]));
    if (!isAdmin()) {
      group(box, "我提交的", tasks.filter((t) => t.mine && t.status !== "claimed"), (t) =>
        row(t, `${prog(t)} · ${t.status === "approved" ? "管理员已通过" : "已提交"}`, []));
    }
    if (!box.children.length) box.append(el("div", "empty", isAdmin() ? "现在没有可以领的任务。到「素材」里给录像准备标注材料，准备好之后任务会出现在这里。" : "现在没有可以领的任务，过一会儿再来看看。"));
  }

  // 审核（管理员）：已提交等着看的、别人正在标的、已完成的
  function renderReview() {
    const box = $t("rvList");
    box.textContent = "";
    group(box, "已提交，等你看", tasks.filter((t) => t.status === "submitted"), (t) =>
      row(t, `${who(t)} · ${prog(t)} · ${fmtDate(new Date(t.submittedAt))}提交`, [
        link("查看", linkOf(t), "plain"),
        button("通过", "", () => act(t, "approve")),
        button(t.assignee ? "退回重标" : "放回任务里", "plain", () => act(t, "return")),
      ]));
    group(box, "正在标", tasks.filter((t) => t.status === "claimed"), (t) =>
      row(t, `${t.mine ? "我" : who(t)} · ${prog(t)} · ${leaseText(t)}`, [link("查看", linkOf(t), "plain"), button("收回", "plain danger", () => act(t, "release"))]));
    group(box, "已完成", tasks.filter((t) => t.status === "approved"), (t) =>
      row(t, `${who(t) || "以前标的"} · ${prog(t)}`, [link("查看", linkOf(t), "plain"), button(t.assignee ? "退回重标" : "放回任务里", "plain", () => act(t, "return"))]));
    if (!box.children.length) box.append(el("div", "empty", "还没有人领任务。"));
  }

  // ---------- 素材（管理员）：每段录像的标注材料 ----------

  async function prepare(v, path, method, body) {
    try {
      await api(`/videos/${v.id}${path}`, { method, headers: { "content-type": "application/json" }, body: body ? JSON.stringify(body) : undefined });
      $t("mtError").hidden = true;
    } catch (e) {
      $t("mtError").hidden = false;
      $t("mtError").textContent = `${v.title || v.filename}：${e.message}`;
    }
    refresh();
  }
  function armed(text, again, fn) {
    const b = button(text, "plain danger", () => {
      if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = again; return; }
      fn();
    });
    return b;
  }
  const taskLine = (c) => {
    const parts = [["open", "待领"], ["claimed", "正在标"], ["submitted", "待审核"], ["approved", "完成"]].filter(([k]) => c[k]).map(([k, name]) => `${c[k]} 个${name}`);
    return parts.length ? "任务：" + parts.join(" · ") : "";
  };
  function stateText(x, online, what) {
    return {
      none: "还没有准备",
      queued: online ? "排队中，Mac 马上开始准备" : "排队中。家里的 Mac 现在离线，上线后开始准备",
      preparing: "Mac 正在准备：" + (x.stage || "开始"),
      ready: `${what}准备好了` + (x.stage ? "：" + x.stage : ""),
      failed: "准备失败：" + (x.error || ""),
    }[x.status] || "";
  }

  function renderMaterials() {
    const box = $t("mtList");
    box.textContent = "";
    if (!materials) { box.append(el("div", "empty", "读取中")); return; }
    const node = materials.node, online = node && new Date(materials.now) - new Date(node.seenAt) < 2 * 60 * 1000;
    $t("mtNode").textContent = node ? (online ? `家里的 Mac 在线（${node.name}）` : `家里的 Mac 离线（${node.name}），上线后才会开始准备`) : "家里的 Mac 还没有连接过";
    const owner = isOwner();
    for (const v of materials.videos) {
      const card = el("section", "card mat");
      card.append(el("h2", "", v.title || v.filename));
      card.append(el("div", "quiet", `${fmtDate(new Date(v.createdAt))}上传` + (v.analyzed ? "" : " · 还没分析完，分析完才能准备标注材料")));

      const a = v.annot, c = a.labels || {}, n = (c.human || 0) + (c.bulk || 0) + (c.skip || 0);
      card.append(el("h3", "", "球的位置"));
      card.append(el("div", a.status === "failed" ? "bad" : "quiet", stateText(a, online, "图片")));
      const line = [n ? `已标 ${n} 帧（亲手确认 ${c.human || 0}，一键接受 ${c.bulk || 0}，看不清 ${c.skip || 0}）` : "", taskLine(a.tasks)].filter(Boolean).join(" · ");
      if (line) card.append(el("div", "quiet", line));
      const acts = el("div", "acts");
      if (owner && v.analyzed && (a.status === "none" || a.status === "failed")) acts.append(button(a.status === "failed" ? "重新准备" : "准备标注", "", () => prepare(v, "/annot", "POST", {})));
      if (owner && v.analyzed && a.status === "ready") acts.append(button("用新模型重新预测", "plain", () => prepare(v, "/annot", "POST", { fresh: true })));
      if (a.status === "ready") acts.append(link("整段打开", "#/label/" + v.id, "plain"));
      if (n) { const d = link("下载 CSV", `${API}/videos/${v.id}/labels?format=csv`, "plain"); acts.append(d); }
      if (owner && (a.status === "ready" || a.status === "failed")) acts.append(armed("删除图片", "再点一次：删除图片（标注保留）", () => prepare(v, "/annot", "DELETE")));
      card.append(acts);

      const k = v.strokes;
      card.append(el("h3", "", "击球类型"));
      card.append(el("div", k.status === "failed" ? "bad" : "quiet", stateText(k, online, "小视频")));
      const kline = [k.labels ? `已标 ${k.labels} 拍` : "", taskLine(k.tasks)].filter(Boolean).join(" · ");
      if (kline) card.append(el("div", "quiet", kline));
      const kacts = el("div", "acts");
      // 准备好了也能重新准备（比如 Mac 上的程序更新了）：已标的拍按帧号对应，不会丢
      if (owner && v.analyzed && k.status !== "queued" && k.status !== "preparing")
        kacts.append(button(k.status === "none" ? "准备击球类型标注" : k.status === "ready" ? "重新准备（已标的保留）" : "重新准备", k.status === "ready" ? "plain" : "", () => prepare(v, "/strokes", "POST", {})));
      if (k.status === "ready") kacts.append(link("整段打开", "#/strokes/" + v.id, "plain"));
      if (k.labels) kacts.append(link("下载 CSV", `${API}/videos/${v.id}/stroke-labels?format=csv`, "plain"));
      if (owner && (k.status === "ready" || k.status === "failed")) kacts.append(armed("删除小视频", "再点一次：删除小视频（标注保留）", () => prepare(v, "/strokes", "DELETE")));
      card.append(kacts);
      box.append(card);
    }
    if (!materials.videos.length) box.append(el("div", "empty", "还没有传完的录像。"));
  }

  // 切到「标注」的某一页（tasks | review | materials），或者离开（null）
  function show(name) {
    clearTimeout(timer);
    page = name;
    if (!name) return;
    if (name !== "materials") lastList = name;
    setUser();
    render();   // 先用手上的数据画，再去取新的
    refresh();
  }
  const cameFrom = () => lastList;

  // 标注页用：按任务号取任务（从服务器拿最新的）
  async function task(id) {
    await load();
    const t = tasks.find((x) => x.id === id);
    if (!t || (!isAdmin() && !t.mine)) throw new Error("这个任务不在你手上（没领，或者太久没保存已经放回大厅）。回任务大厅重新领取。");
    return t;
  }

  // 标注页顶上的任务条：写着是哪个任务，自己手上的可以直接提交
  function showBar(t) {
    barTask = t;
    for (const id of ["lTaskBar", "sTaskBar"]) $t(id).hidden = true;
    if (!t) return;
    const bar = $t(t.kind === "shuttle" ? "lTaskBar" : "sTaskBar");
    const mine = t.mine && t.status === "claimed";
    bar.hidden = false;
    bar.firstElementChild.textContent = mine ? "任务：" + describe(t) : `只是查看：${who(t) || "没有人"}的任务（${{ open: "还没人领", claimed: "正在标", submitted: "已提交", approved: "已完成" }[t.status]}）`;
    const b = bar.lastElementChild;
    b.hidden = !mine;
    b.textContent = "提交任务";
    delete b.dataset.armed;
    b.onclick = async () => {
      try {
        const now = await task(t.id); // 最新的进度
        const left = now.items - now.done;
        if (left > 0 && b.dataset.armed !== "1") {
          b.dataset.armed = "1";
          b.textContent = `还有 ${left} ${t.kind === "shuttle" ? "帧" : "拍"}没标，仍然提交`;
          return;
        }
        await post(`/tasks/${t.id}/submit`);
        location.hash = "#/tasks";
      } catch (e) {
        bar.firstElementChild.textContent = e.message;
      }
    };
  }

  // ---------- 成员 ----------

  async function saveUser(path, body) {
    try {
      renderTeam((await post(path, body)).users);
      $t("mError").hidden = true;
    } catch (e) {
      $t("mError").hidden = false;
      $t("mError").textContent = e.message;
    }
  }

  function renderTeam(users) {
    const box = $t("mList");
    box.textContent = "";
    const group = (title, list, make) => {
      if (!list.length) return;
      box.append(el("div", "tgroup", `${title}（${list.length}）`));
      for (const u of list) box.append(make(u));
    };
    const card = (u, sub, btns) => {
      const c = el("div", "card task"), body = el("div", "info"), b = el("div", "btns");
      body.append(el("div", "name", u.name ? `${u.name}（${u.email}）` : u.email), el("div", "quiet", sub));
      b.append(...btns);
      c.append(body, b);
      return c;
    };
    const stats = (u) => `标了 ${u.frames} 帧球位置、${u.strokes} 拍击球 · 完成 ${u.finished} 个任务，手上 ${u.held} 个` + (u.lastLoginAt ? ` · 上次登录 ${fmtDate(new Date(u.lastLoginAt))}` : " · 还没登录过");
    const remove = (u, text) => {
      const b = button(text, "plain danger", () => {
        if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = "再点一次确认"; return; }
        saveUser("/admin/users/remove", { email: u.email });
      });
      return b;
    };
    group("申请加入", users.filter((u) => u.status === "pending"), (u) =>
      card(u, "用 Google 登录过，还没有被邀请", [button("允许", "", () => saveUser("/admin/users", { email: u.email, status: "active" })), remove(u, "拒绝")]));
    group("成员", users.filter((u) => u.status === "active"), (u) => {
      const self = me && u.email === me.email;
      return card(u, (u.owner ? "超级管理员 · " : u.role === "admin" ? "管理员 · " : "") + stats(u), self || u.owner ? [] : [
        button("停用", "plain", () => saveUser("/admin/users", { email: u.email, status: "disabled" })),
        u.role === "admin" ? button("取消管理员", "plain", () => saveUser("/admin/users", { email: u.email, role: "annotator" }))
          : button("设为管理员", "plain", () => saveUser("/admin/users", { email: u.email, role: "admin" })),
      ]);
    });
    group("已停用", users.filter((u) => u.status === "disabled"), (u) =>
      card(u, stats(u), [button("恢复", "plain", () => saveUser("/admin/users", { email: u.email, status: "active" })), remove(u, "删除")]));
    if (!users.length) box.append(el("div", "empty", "还没有成员。你自己先用 Google 登录一次（登录后在「我是管理员」里输入口令），再在上面邀请别人。"));
  }

  async function openTeam() {
    try {
      renderTeam((await api("/admin/users")).users);
    } catch (e) {
      $t("mList").textContent = "读取失败：" + e.message;
    }
  }

  $t("mAdd").onclick = async () => {
    const email = $t("mEmail").value.trim();
    if (!email) return;
    await saveUser("/admin/users", { email, status: "active" });
    if ($t("mError").hidden) $t("mEmail").value = "";
  };

  return { setUser, show, cameFrom, openTeam, task, showBar };
})();
