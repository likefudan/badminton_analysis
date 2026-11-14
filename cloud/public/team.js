// 多人标注：任务大厅（#/tasks）、成员页（#/team，管理员）、标注页顶上的任务条。
// 任务 = 一个回合的球位置，或约 20 拍的击球类型。领了之后只有自己能标这一段，标完提交。
"use strict";

const Team = (() => {
  const $t = (id) => document.getElementById(id);
  let tasks = [], info = null, timer = 0, open = false, barTask = null;
  const post = (path, body) => api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });

  function setUser() {
    $t("tWho").textContent = me && me.email ? me.name || me.email : "";
  }

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
    try {
      await load();
      $t("tError").hidden = true;
    } catch (e) {
      $t("tError").hidden = false;
      $t("tError").textContent = "读取失败：" + e.message;
    }
    if (!open) return;
    render();
    timer = setTimeout(refresh, 20000);
  }

  async function act(t, action, body) {
    try {
      info = await post(`/tasks/${t.id}/${action}`, body);
      tasks = info.tasks;
      $t("tError").hidden = true;
      render();
      return true;
    } catch (e) {
      $t("tError").hidden = false;
      $t("tError").textContent = e.message;
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

  function render() {
    const box = $t("tList");
    box.textContent = "";
    const admin = isAdmin(), unit = (t) => (t.kind === "shuttle" ? "帧" : "拍");
    const prog = (t) => `已标 ${t.done}/${t.items} ${unit(t)}`;
    const group = (title, list, make) => {
      if (!list.length) return;
      box.append(el("div", "tgroup", `${title}（${list.length}）`));
      for (const t of list) box.append(make(t));
    };
    group("我手上的", tasks.filter((t) => t.status === "claimed" && t.mine), (t) =>
      row(t, `${prog(t)} · ${leaseText(t)}自动放回大厅`, [
        link("继续标注", linkOf(t)),
        submitButton(t),
        button("放弃", "plain danger", () => act(t, "release")),
      ]));
    group("可以领取", tasks.filter((t) => t.status === "open"), (t) =>
      row(t, t.done ? `${prog(t)}（别人标了一部分）` : "还没有人标", [
        button("领取并开始", "", async () => { if (await act(t, "claim")) location.hash = linkOf(t); }),
      ]));
    if (admin) {
      group("别人正在标", tasks.filter((t) => t.status === "claimed" && !t.mine), (t) =>
        row(t, `${who(t)} · ${prog(t)} · ${leaseText(t)}`, [link("查看", linkOf(t), "plain"), button("收回", "plain danger", () => act(t, "release"))]));
      group("已提交，等你看", tasks.filter((t) => t.status === "submitted"), (t) =>
        row(t, `${who(t)} · ${prog(t)} · ${fmtDate(new Date(t.submittedAt))}提交`, [
          link("查看", linkOf(t), "plain"),
          button("通过", "", () => act(t, "approve")),
          button(t.assignee ? "退回重标" : "放回大厅", "plain", () => act(t, "return")),
        ]));
      group("已完成", tasks.filter((t) => t.status === "approved"), (t) =>
        row(t, `${who(t) || "以前标的"} · ${prog(t)}`, [link("查看", linkOf(t), "plain"), button(t.assignee ? "退回重标" : "放回大厅", "plain", () => act(t, "return"))]));
    } else {
      group("我提交的", tasks.filter((t) => t.mine && t.status !== "claimed"), (t) =>
        row(t, `${prog(t)} · ${t.status === "approved" ? "管理员已通过" : "已提交"}`, []));
    }
    if (!box.children.length) box.append(el("div", "empty", admin ? "还没有标注任务。在录像的「标注」里点「准备标注」，Mac 准备好之后任务会出现在这里。" : "现在没有可以领的任务，过一会儿再来看看。"));
  }

  function openTasks() {
    open = true;
    setUser();
    refresh();
  }
  function closeTasks() {
    open = false;
    clearTimeout(timer);
  }

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

  return { setUser, openTasks, closeTasks, openTeam, task, showBar };
})();
