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
  const AGENT_MS = 45 * 1000;   // 面板服务（badminton dashboard）每 10 秒来一次：这么久没来就算不在
  const ACTION = { start: "启动", stop: "立即停", "stop-after": "做完这个再停", "cancel-stop-after": "取消「做完再停」", restart: "重启", update: "更新代码并重启" };
  const COLORS = { cpu: "#2f6fd6", gpu: "#c2570c", anePowerW: "#7c4ddb", memUsedGb: "#0f8a7e" };
  let sending = "";             // 正在发出的指令
  let cmdError = "", cmdErrorUntil = 0;
  let fastUntil = 0;            // 发出指令后的半分钟里刷新得勤一些
  const KIND = { process: "分析录像", annotate: "准备球位置标注", strokes: "准备击球类型标注", labels: "同步球位置标注", "stroke-labels": "同步击球类型标注", "rally-labels": "同步回合胜负" };
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

  async function command(n, action, ask) {
    if (ask && !window.confirm(ask)) return;
    sending = action;
    cmdError = "";
    render();
    try {
      await api("/node/commands", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ node: n.name, action }) });
    } catch (e) {
      cmdError = `「${ACTION[action]}」没发出去：${e.message}`;
      cmdErrorUntil = Date.now() + 20000;
    }
    sending = "";
    fastUntil = Date.now() + 40000;
    refresh();
  }

  function spark(values, max, color) {
    const NS = "http://www.w3.org/2000/svg", W = 200, H = 40;
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const pts = [];
    const n = Math.max(values.length, 60);
    values.forEach((v, i) => {
      if (v != null) pts.push(((n - values.length + i) * W / (n - 1)).toFixed(1) + "," + (H - 2 - Math.min(1, v / max) * (H - 5)).toFixed(1));
    });
    if (pts.length > 1) {
      const area = document.createElementNS(NS, "polygon");
      area.setAttribute("points", `${pts[0].split(",")[0]},${H} ${pts.join(" ")} ${pts[pts.length - 1].split(",")[0]},${H}`);
      area.setAttribute("fill", color);
      area.setAttribute("opacity", "0.14");
      const line = document.createElementNS(NS, "polyline");
      line.setAttribute("points", pts.join(" "));
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", "1.5");
      line.setAttribute("vector-effect", "non-scaling-stroke");
      svg.append(area, line);
    }
    return svg;
  }

  function gauge(name, value, unit, sub, values, max, color) {
    const g = el("div", "macgauge");
    const v = el("div", "v", value == null ? "—" : String(value));
    if (value != null) v.append(el("small", "", unit));
    g.append(el("div", "quiet", name), v, el("div", "sub", sub || ""), spark(values || [], max, color));
    return g;
  }

  // 面板服务在线时：它每 10 秒上报一次，处理程序停着也报，还能替网页执行启停
  function renderAgent(box, n) {
    const a = n.agent, node = a.node || {}, c = card("处理程序");
    const mine = (data.commands || []).filter((x) => x.node === n.name);
    const waiting = mine.find((x) => x.status === "pending");
    const sent = sending || (waiting && waiting.action);   // 已经点了，Mac 还没来取
    const busy = sent || a.busy;
    const state = el("p", "macstate");
    let text, dot;
    if (sent) { dot = ""; text = `「${ACTION[sent]}」已发出，等 Mac 来取…`; }
    else if (busy) { dot = ""; text = `Mac 正在${busy}…`; }
    else if (node.running && node.stopAfter) { dot = "on"; text = "运行中，做完手上这个就停"; }
    else if (node.running) { dot = "on"; text = node.job ? "运行中" : "运行中，空闲"; }
    else if (node.loaded && node.lastExit) { dot = "off"; text = "出错退出了，系统会自动重启它"; }
    else { dot = "off"; text = "已停止"; }
    state.append(el("span", "dot " + dot), text);
    c.append(state);
    const meta = [n.name];
    if (node.running && node.startedAt) meta.push("已运行 " + fmtDur(ago(node.startedAt)));
    if (node.running && node.version) meta.push("代码 " + node.version);
    c.append(el("p", "quiet", meta.join(" · ")));
    if (a.error) c.append(el("p", "bad", a.error));
    const last = mine[0];
    if (last && ago(last.updatedAt) < 120 && last.status !== "pending")
      c.append(el("p", last.status === "failed" ? "bad" : "quiet", `「${ACTION[last.action] || last.action}」${last.status === "failed" ? "没成功：" + last.message : last.status === "taken" ? "已被 Mac 取走" : "：" + last.message}`));

    const btns = el("div", "macbtns");
    const add = (action, cls, ask) => {
      const b = el("button", cls, ACTION[action]);
      b.disabled = Boolean(busy);
      b.onclick = () => command(n, action, ask);
      btns.append(b);
    };
    const redo = node.job ? "正在做的这个会退回队列，之后从头重做。" : "";
    if (!node.running) add("start", "");
    else {
      add(node.stopAfter ? "cancel-stop-after" : "stop-after", "plain");
      add("stop", "plain stop", "立即停止 Mac 上的处理程序？" + redo);
      add("restart", "plain", redo ? "重启处理程序？" + redo : "");
    }
    add("update", "plain", "让 Mac 拉取最新代码（git pull），然后重启处理程序。" + redo + "继续？");
    c.append(btns);
    c.append(el("p", "quiet", "指令由 Mac 每 10 秒来取一次，点了之后要等几秒才生效。"));
    box.append(c);

    if (node.running) {
      const j = card("正在做");
      const job = node.job, p = node.progress;
      if (node.starting) j.append(el("p", "quiet", "刚启动，正在连接网站…"));
      else if (!job) j.append(el("p", "quiet", "空闲，等新任务。"));
      else {
        const head = el("div");
        head.append(el("strong", "", job.name || job.videoId), el("span", "quiet", " · " + (KIND[job.task] || job.task)));
        j.append(head);
        const line = [];
        if (node.stage) line.push(node.stage);
        line.push("已用 " + fmtDur(p ? p.elapsedSeconds + Math.max(0, ago(node.at)) : ago(job.since)));
        if (p && p.etaSeconds != null) line.push("预计还要 " + fmtDur(Math.max(0, p.etaSeconds - ago(node.at))));
        j.append(el("div", "quiet", line.join(" · ")));
        const steps = (p && p.steps) || [];
        if (steps.length) {
          const counted = steps.filter((x) => x.status !== "skipped");
          const done = counted.reduce((t, x) => t + (x.status === "done" ? 1 : x.status === "running" ? (x.percent || 0) / 100 : 0), 0);
          j.append(bar(counted.length ? done * 100 / counted.length : 0));
          const ul = el("ul", "steps");
          for (const x of steps) {
            const li = el("li", x.status);
            let time = "";
            if (x.status === "running") time = (x.percent != null ? x.percent + "% · " : "") + fmtDur(x.seconds) + (x.etaSeconds != null ? "，还要约 " + fmtDur(x.etaSeconds) : "");
            else if (x.status === "done") time = fmtDur(x.seconds);
            else if (x.status === "pending" && x.estimateSeconds != null) time = "约 " + fmtDur(x.estimateSeconds);
            else if (x.status === "skipped") time = "跳过";
            li.append(el("span", "ico", { done: "✓", running: "●", failed: "✕", skipped: "–", pending: "○" }[x.status] || ""), el("span", "lab", x.label), el("span", "time", time));
            if (x.detail && x.status !== "pending") li.append(el("span", "note", x.detail));
            ul.append(li);
          }
          j.append(ul);
        }
      }
      box.append(j);
    }

    const m = a.metrics || {}, now = m.now || {}, h = m.history || {}, sys = a.system || {};
    const r = card("资源");
    const peak = (list, floor) => Math.max(floor, ...(list || []).filter((v) => v != null));
    const g = el("div", "macgauges");
    g.append(gauge("CPU", now.cpu == null ? null : Math.round(now.cpu), "%", now.pcpu != null ? `性能核 ${Math.round(now.pcpu)}% · 能效核 ${Math.round(now.ecpu)}%` : "", h.cpu, 100, COLORS.cpu));
    g.append(gauge("GPU", now.gpu == null ? null : Math.round(now.gpu), "%", now.gpuPowerW != null ? `${now.gpuPowerW.toFixed(1)} W` : "", h.gpu, 100, COLORS.gpu));
    g.append(gauge("NPU 功耗", now.anePowerW == null ? null : now.anePowerW.toFixed(2), "W", now.anePowerW == null ? "需要 macmon" : "系统不公开利用率", h.anePowerW, peak(h.anePowerW, 2), COLORS.anePowerW));
    const memTotal = now.memTotalGb || sys.memTotalGb, memUsed = now.memUsedGb != null ? now.memUsedGb : sys.memUsedGb;
    g.append(gauge("内存", memUsed == null ? null : memUsed.toFixed(1), "GB", memTotal ? `共 ${memTotal} GB` : "", h.memUsedGb, memTotal || peak(h.memUsedGb, 8), COLORS.memUsedGb));
    r.append(g);
    const notes = ["曲线是最近 10 分钟"];
    if (now.cpuTemp) notes.push(`CPU ${now.cpuTemp}°C`);
    if (now.allPowerW != null) notes.push(`整机 ${now.allPowerW.toFixed(1)} W`);
    if (sys.diskTotalGb) notes.push(`磁盘剩余 ${sys.diskFreeGb} GB`);
    r.append(el("p", sys.diskFreeGb != null && sys.diskFreeGb < 20 ? "bad" : "quiet", notes.join(" · ")));
    if (m.note) r.append(el("p", "warn", m.note));
    box.append(r);
  }

  // 后台任务和日志来自处理程序自己每半分钟的上报；面板服务在线时上面已经有状态和资源，这里只补这两样
  function renderExtras(box, n) {
    const s = n.status;
    if (!s) return;
    const stale = ago(n.statusAt) > 120;
    if (s.log && s.log.length) {
      const c = card("处理程序的日志");
      if (stale) c.append(el("p", "quiet", `${fmtDur(ago(n.statusAt))}前最后一次上报的内容。`));
      c.append(logBox("node:" + n.name, s.log, `最后 ${s.log.length} 行`));
      box.append(c);
    }
    renderTasks(box, s, stale);
  }

  function renderNode(box, n) {
    if (n.agent && ago(n.agentAt) * 1000 < AGENT_MS) {
      renderAgent(box, n);
      renderExtras(box, n);
      return;
    }
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
    const u = sys.usage || {};
    if (u.cpu != null) row(dl, "CPU 利用率", `${u.cpu}%` + (u.pcpu != null ? `（性能核 ${u.pcpu}%，能效核 ${u.ecpu}%）` : ""));
    if (u.gpu != null) row(dl, "GPU 利用率", `${u.gpu}%` + (u.gpuPowerW != null ? `，${u.gpuPowerW.toFixed(1)} W` : ""));
    if (u.anePowerW != null) row(dl, "NPU 功耗", `${u.anePowerW.toFixed(2)} W（系统不公开利用率）`);
    if (sys.memTotalGb) row(dl, "内存", `用了 ${sys.memUsedGb} / ${sys.memTotalGb} GB`);
    if (sys.diskTotalGb) row(dl, "磁盘剩余", `${sys.diskFreeGb} GB`, sys.diskFreeGb < 20 ? "bad" : "");
    c.append(dl);
    if (s.log && s.log.length) c.append(logBox("node:" + n.name, s.log, `处理程序的日志（最后 ${s.log.length} 行）`));
    if (n.agentAt) c.append(el("p", "quiet", `Mac 上的面板服务 ${fmtDur(ago(n.agentAt))}前就没有消息了，现在不能远程启停。`));
    else c.append(el("p", "quiet", "要在这里启动和停止处理程序，在 Mac 上运行一次 badminton dashboard install。"));
    box.append(c);
    renderTasks(box, s, stale);
  }

  function renderTasks(box, s, stale) {
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
    if (cmdError && Date.now() < cmdErrorUntil) box.append(el("p", "bad", cmdError));
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
    $("macAt").textContent = "每 5 秒自动刷新 · 上次刷新 " + new Date().toLocaleTimeString("zh-CN", { hour12: false });
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
      let tick = 0;   // 平时 5 秒一次；刚发过指令时 1.5 秒一次，好尽快看到结果
      timer = setInterval(() => { if (!document.hidden && (Date.now() < fastUntil || ++tick % 3 === 0)) refresh(); }, 1500);
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
