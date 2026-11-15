"use strict";
// 「Mac 状态」页（#/mac）：家里的 Mac 现在在做什么。内容来自 GET /status：
// Mac 节点每半分钟上报的状态（当前作业、后台任务和它们的日志、机器负载），加上云端知道的排队和失败的作业。
// 用到 app.js 里的 api、el、$、fmtDur。

const Mac = (() => {
  let timer = null;
  let data = null;
  let skew = 0;                 // 服务器时间 - 手机时间
  let error = "";
  const open = new Set();       // 展开了日志的任务：刷新后保持展开
  const ONLINE_MS = 90 * 1000;
  const KIND = { process: "分析录像", annotate: "准备球位置标注", strokes: "准备击球类型标注", labels: "同步球位置标注", "stroke-labels": "同步击球类型标注" };
  const now = () => Date.now() + skew;
  const ago = (iso) => (iso ? Math.max(0, (now() - new Date(iso)) / 1000) : NaN);
  const clock = (iso) => new Date(iso).toLocaleString("zh-CN", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

  function row(dl, label, value, cls) {
    dl.append(el("dt", "", label), el("dd", cls || "", value));
  }

  function card(title) {
    const c = el("section", "card");
    c.append(el("h2", "", title));
    return c;
  }

  function logBox(key, lines, label) {
    const d = el("details", "maclog");
    d.open = open.has(key);
    d.ontoggle = () => { if (d.open) open.add(key); else open.delete(key); };
    d.append(el("summary", "", label || `最近的输出（${lines.length} 行）`));
    const pre = el("pre", "", lines.join("\n"));
    d.append(pre);
    if (d.open) requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
    return d;
  }

  function bar(percent) {
    const b = el("div", "bar");
    const fill = el("div");
    fill.style.width = Math.max(0, Math.min(100, percent)) + "%";
    b.append(fill);
    return b;
  }

  function jobItem(j, extra) {
    const li = el("li", "macjob");
    const a = el("a", "name", j.name);
    a.href = "#/v/" + j.videoId;
    const head = el("div");
    head.append(a, el("span", "quiet", " · " + (KIND[j.kind] || j.kind)));
    li.append(head);
    if (extra) li.append(el("div", "quiet", extra));
    return li;
  }

  function renderNode(box, n) {
    const s = n.status;
    const online = now() - new Date(n.seenAt) < ONLINE_MS;
    const c = card("处理程序");
    const line = el("p", "macline");
    line.append(el("span", "dot " + (online ? "on" : "off")),
      online ? `在线（${n.name}）` : `离线（${n.name}），最后联系是 ${fmtDur(ago(n.seenAt))}前`);
    c.append(line);
    if (!s) {
      c.append(el("p", "quiet", "这台 Mac 上的处理程序是旧版本，还不会上报详细状态。重启处理程序后这里会显示它在做什么。"));
      box.append(c);
      return;
    }
    const stale = ago(n.statusAt) > 120;
    if (stale) c.append(el("p", "warn", `下面是 ${fmtDur(ago(n.statusAt))}前最后一次上报的内容，可能已经过时。`));
    const dl = el("dl");
    const job = s.job;
    row(dl, "正在做", job ? `${KIND[job.task] || job.task}：${job.name}` : "空闲，等新任务");
    if (job) row(dl, "已经做了", fmtDur(ago(job.since)));
    row(dl, "启动于", `${clock(s.startedAt)}（${fmtDur(ago(s.startedAt))}前）`);
    if (s.version) row(dl, "代码版本", s.version);
    const sys = s.system || {};
    if (sys.load) row(dl, "CPU 负载", `${sys.load[0]}（${sys.cpus} 核）`, sys.load[0] > sys.cpus ? "warn" : "");
    if (sys.memTotalGb) row(dl, "内存", `用了 ${sys.memUsedGb} / ${sys.memTotalGb} GB`);
    if (sys.diskTotalGb) row(dl, "磁盘剩余", `${sys.diskFreeGb} GB`, sys.diskFreeGb < 20 ? "bad" : "");
    c.append(dl);
    if (s.log && s.log.length) c.append(logBox("node:" + n.name, s.log, `处理程序的日志（最后 ${s.log.length} 行）`));
    box.append(c);

    const tasks = s.tasks || [];
    const running = tasks.filter((t) => t.running);
    const ended = tasks.filter((t) => !t.running);
    const t1 = card(`后台任务（${running.length} 个在跑）`);
    if (s.tasksError) t1.append(el("p", "bad", "读取后台任务失败：" + s.tasksError));
    if (!running.length) t1.append(el("p", "quiet", "现在没有训练或其它后台任务在跑。"));
    for (const t of running) t1.append(taskBlock(t, stale));
    t1.append(el("p", "quiet", `自动发现：在 ${(s.watch || []).join("、")} 下运行的 Python 程序。`));
    box.append(t1);
    if (ended.length) {
      const t2 = card("一天内结束的后台任务");
      for (const t of ended) t2.append(taskBlock(t, false));
      box.append(t2);
    }
  }

  function taskBlock(t, stale) {
    const d = el("div", "mactask");
    d.append(el("div", "cmd", t.command));
    const p = t.progress;
    let state;
    if (t.running) {
      state = `已经跑了 ${fmtDur(ago(t.startedAt))}`;
      if (p && p.label) state = p.label + " · " + state;
      if (p && p.etaSeconds != null) state += ` · 这一段还要约 ${fmtDur(p.etaSeconds)}`;
    } else {
      state = `${t.failed ? "结束了，日志里有报错" : "已结束"} · ${clock(t.endedAt)} · 跑了 ${fmtDur((new Date(t.endedAt) - new Date(t.startedAt)) / 1000)}`;
      if (p && p.label) state += ` · 最后进度 ${p.label}`;
    }
    d.append(el("div", t.running ? "" : t.failed ? "bad" : "quiet", state));
    if (t.running && p && p.percent != null) d.append(bar(p.percent));
    if (t.running && !stale) d.append(el("div", "quiet", `CPU ${t.cpu}% · 内存 ${t.rssMb >= 1024 ? (t.rssMb / 1024).toFixed(1) + " GB" : t.rssMb + " MB"}${t.processes > 1 ? ` · ${t.processes} 个进程` : ""} · 进程号 ${t.pid}`));
    if (t.log && t.log.length) d.append(logBox("task:" + t.id, t.log));
    else d.append(el("div", "quiet", "它的输出没有写到文件里，看不到日志。"));
    if (t.logFile) d.append(el("div", "quiet path", "日志文件 " + t.logFile));
    return d;
  }

  function render() {
    const box = $("macBody");
    if (!box) return;
    const y = window.scrollY;
    box.textContent = "";
    if (error) box.append(el("p", "bad", "读取失败：" + error));
    if (!data) {
      if (!error) box.append(el("div", "empty", "读取中"));
      return;
    }
    if (!data.nodes.length) box.append(el("div", "empty", "家里的 Mac 还没有连接过"));
    for (const n of data.nodes.slice(0, 1)) renderNode(box, n);
    const others = data.nodes.slice(1).filter((n) => now() - new Date(n.seenAt) < 7 * 86400 * 1000);
    if (others.length) {
      const c = card("其它连接过的机器");
      for (const n of others) c.append(el("p", "quiet", `${n.name}，最后联系是 ${fmtDur(ago(n.seenAt))}前`));
      box.append(c);
    }

    const q = card(`网站交给 Mac 的作业`);
    const ul = el("ul");
    for (const j of data.active) {
      ul.append(jobItem(j, j.stale ? `处理程序领了之后失联了（${fmtDur(ago(j.updatedAt))}没有消息），上线后会重新做`
        : `正在做：${j.stage || "已领取"}${j.startedAt ? " · 已用 " + fmtDur(ago(j.startedAt)) : ""}`));
    }
    data.queued.forEach((j, i) => ul.append(jobItem(j, `排队中，第 ${i + 1} 个`)));
    if (!ul.children.length) q.append(el("p", "quiet", "没有正在做或排队的作业。"));
    else q.append(ul);
    box.append(q);

    if (data.failed.length) {
      const f = card("失败的作业");
      const list = el("ul");
      for (const j of data.failed) {
        const li = jobItem(j, `${j.updatedAt ? clock(j.updatedAt) + " · " : ""}停在：${j.stage || "—"}`);
        if (j.error) li.append(logBox(`fail:${j.videoId}:${j.kind}`, j.error.split("\n"), "失败原因"));
        list.append(li);
      }
      f.append(list);
      box.append(f);
    }
    $("macAt").textContent = "每 10 秒自动刷新 · 上次刷新 " + new Date().toLocaleTimeString("zh-CN", { hour12: false });
    window.scrollTo(0, y);
  }

  async function refresh() {
    try {
      const d = await api("/status");
      skew = new Date(d.now) - Date.now();
      data = d;
      error = "";
    } catch (e) {
      error = e.message;
    }
    render();
  }

  return {
    open() {
      Mac.close();
      render();
      refresh();
      timer = setInterval(() => { if (!document.hidden) refresh(); }, 10000);
      document.addEventListener("visibilitychange", Mac.wake);
    },
    wake() { if (!document.hidden && timer) refresh(); },
    close() {
      clearInterval(timer);
      timer = null;
      document.removeEventListener("visibilitychange", Mac.wake);
    },
  };
})();
