// 分析区域：固定机位时，在一帧画面上画多边形，「只看这里」或「抹掉这里」。
// 存在这个视频的 options.region（坐标是画面宽高的比例 0–1），保存后按新区域重新分析。
// Mac 上：球的候选点、人的脚（检测框底边中点）不在区域里就不算。

const region = { draft: null, current: null, drag: null, vid: null };

function regionOf(v) {
  const r = (v && v.options && v.options.region) || {};
  return { keep: (r.keep || []).map((p) => p.map((q) => q.slice())), drop: (r.drop || []).map((p) => p.map((q) => q.slice())) };
}

function regionSummary(r) {
  const parts = [];
  if (r.keep.length) parts.push(`只看 ${r.keep.length} 块`);
  if (r.drop.length) parts.push(`抹掉 ${r.drop.length} 块`);
  return parts.join("，");
}

function regionImage(v) {
  const have = new Set(v.results.map((x) => x.name));
  for (const name of ["frame.jpg", "cover.jpg"]) if (have.has(name)) return resultUrl(v, name);
  return null;
}

// 详情页「处理」里那一块：现状 + 「画分析区域」按钮
function renderRegion(v) {
  const box = $("dRegion");
  const r = regionOf(v);
  const busyNow = v.job && v.job.status === "processing";
  const img = regionImage(v);
  $("rgSummary").textContent = r.keep.length || r.drop.length
    ? `现在：${regionSummary(r)}。区域外的球和人不参与分析。`
    : "现在分析整个画面。固定机位拍的时候，可以框出只看的范围，或者抹掉隔壁场、观众这些干扰。";
  $("rgEdit").hidden = region.vid === v.id;
  $("rgEdit").disabled = busyNow || !img || v.status !== "uploaded";
  $("rgEdit").textContent = r.keep.length || r.drop.length ? "修改分析区域" : "画分析区域";
  $("rgNote").textContent = busyNow ? "Mac 正在处理，处理完才能改。" : !img ? "处理完、有了画面截图之后才能画。" : "";
  box.hidden = false;
  if (region.vid && region.vid !== v.id) closeRegionEditor();
}

function openRegionEditor() {
  const v = videos.find((x) => x.id === openId);
  if (!v) return;
  const src = regionImage(v);
  if (!src) return;
  region.vid = v.id;
  region.draft = regionOf(v);
  region.current = null;
  region.kind = "keep";
  const ed = $("rgEditor");
  ed.hidden = false;
  $("rgEdit").hidden = true;
  const img = $("rgImg");
  img.onload = () => { $("rgSvg").setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`); drawRegion(); };
  img.src = src;
  setRegionKind("keep");
  drawRegion();
}

function closeRegionEditor() {
  region.vid = null;
  region.draft = null;
  region.current = null;
  $("rgEditor").hidden = true;
  $("rgEdit").hidden = false;
}

function setRegionKind(kind) {
  region.kind = kind;
  for (const b of $("rgKinds").children) b.classList.toggle("on", b.dataset.kind === kind);
  $("rgHint").textContent = kind === "keep"
    ? "在画面上一个个点出多边形的角，点完按「完成这一块」。注意把场地上方的空中也框进去，高球会飞出场地线很高。"
    : "点出要抹掉的范围（隔壁场、观众、反光的地方），点完按「完成这一块」。";
  if (region.current) region.current.kind = kind;
  drawRegion();
}

function svgEl(tag, attrs) {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, val] of Object.entries(attrs)) e.setAttribute(k, val);
  return e;
}

function drawRegion() {
  const svg = $("rgSvg");
  const img = $("rgImg");
  const W = img.naturalWidth || 1, H = img.naturalHeight || 1;
  svg.textContent = "";
  if (!region.draft) return;
  const pts = (p) => p.map(([x, y]) => `${x * W},${y * H}`).join(" ");
  const dot = Math.max(W, H) / 90;
  const line = Math.max(W, H) / 400;
  const d = region.draft;
  // 只看的范围以外压暗；抹掉的范围压暗再描红边
  if (d.keep.length) {
    const defs = svgEl("defs", {});
    const mask = svgEl("mask", { id: "rgMask" });
    mask.append(svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: "white" }));
    for (const p of d.keep) mask.append(svgEl("polygon", { points: pts(p), fill: "black" }));
    defs.append(mask);
    svg.append(defs);
    svg.append(svgEl("rect", { x: 0, y: 0, width: W, height: H, fill: "#000", "fill-opacity": .55, mask: "url(#rgMask)" }));
  }
  for (const p of d.keep) svg.append(svgEl("polygon", { points: pts(p), fill: "none", stroke: "#3fb950", "stroke-width": line }));
  for (const p of d.drop) svg.append(svgEl("polygon", { points: pts(p), fill: "#000", "fill-opacity": .6, stroke: "#f85149", "stroke-width": line }));
  const cur = region.current;
  if (cur && cur.points.length) {
    const color = cur.kind === "keep" ? "#3fb950" : "#f85149";
    svg.append(svgEl(cur.points.length >= 3 ? "polygon" : "polyline",
      { points: pts(cur.points), fill: "none", stroke: color, "stroke-width": line, "stroke-dasharray": `${line * 4} ${line * 3}` }));
  }
  // 所有的角都能拖
  const all = [...d.keep, ...d.drop, ...(cur ? [cur.points] : [])];
  for (const p of all) for (const [x, y] of p) {
    svg.append(svgEl("circle", { cx: x * W, cy: y * H, r: dot, fill: "rgba(255,255,255,.9)", stroke: "#000", "stroke-width": line / 2 }));
  }
  const list = $("rgList");
  list.textContent = "";
  d.keep.forEach((p, i) => list.append(regionItem("只看", i + 1, () => { d.keep.splice(i, 1); drawRegion(); })));
  d.drop.forEach((p, i) => list.append(regionItem("抹掉", i + 1, () => { d.drop.splice(i, 1); drawRegion(); })));
  $("rgDone").disabled = !cur || cur.points.length < 3;
  $("rgUndo").disabled = !cur || !cur.points.length;
}

function regionItem(kind, n, onDelete) {
  const li = el("li", "", "");
  li.append(el("span", "", `第 ${n} 块 · ${kind}`));
  const a = el("a", "", "删掉");
  a.onclick = onDelete;
  li.append(a);
  return li;
}

function regionPoint(ev) {
  const r = $("rgSvg").getBoundingClientRect();
  return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
}

function nearestCorner(p, ev) {
  const r = $("rgSvg").getBoundingClientRect();
  const lists = [...region.draft.keep, ...region.draft.drop, ...(region.current ? [region.current.points] : [])];
  let best = null, bestD = 22; // 手指点到角 22 像素以内算拖这个角
  for (const poly of lists) poly.forEach((q, i) => {
    const dd = Math.hypot((q[0] - p[0]) * r.width, (q[1] - p[1]) * r.height);
    if (dd < bestD) { bestD = dd; best = { poly, i }; }
  });
  return best;
}

function regionDown(ev) {
  if (!region.draft) return;
  ev.preventDefault();
  const p = regionPoint(ev);
  const hit = nearestCorner(p, ev);
  if (hit) {
    region.drag = hit;
    $("rgSvg").setPointerCapture(ev.pointerId);
    return;
  }
  if (!region.current) region.current = { kind: region.kind, points: [] };
  region.current.points.push(p);
  drawRegion();
}

function regionMove(ev) {
  if (!region.drag) return;
  ev.preventDefault();
  region.drag.poly[region.drag.i] = regionPoint(ev);
  drawRegion();
}

function regionUp() { region.drag = null; }

function finishRegionPolygon() {
  const cur = region.current;
  if (!cur || cur.points.length < 3) return;
  region.draft[cur.kind].push(cur.points);
  region.current = null;
  drawRegion();
}

async function saveRegion() {
  const v = videos.find((x) => x.id === region.vid);
  if (!v) return;
  if (region.current && region.current.points.length >= 3) finishRegionPolygon();
  const r = region.draft;
  const options = { ...v.options };
  if (r.keep.length || r.drop.length) options.region = r;
  else delete options.region;
  $("rgSave").disabled = true;
  try {
    await api(`/videos/${v.id}/job/retry`, {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ options }),
    });
    log(r.keep.length || r.drop.length ? `分析区域已保存（${regionSummary(r)}），重新排队` : "已去掉分析区域，重新排队");
    closeRegionEditor();
  } catch (e) { log("保存分析区域失败：" + e.message); }
  $("rgSave").disabled = false;
  refreshList();
}

$("rgEdit").onclick = openRegionEditor;
$("rgCancel").onclick = () => { closeRegionEditor(); const v = videos.find((x) => x.id === openId); if (v) renderRegion(v); };
$("rgDone").onclick = finishRegionPolygon;
$("rgUndo").onclick = () => { if (region.current) { region.current.points.pop(); if (!region.current.points.length) region.current = null; drawRegion(); } };
$("rgClear").onclick = () => { region.draft = { keep: [], drop: [] }; region.current = null; drawRegion(); };
$("rgSave").onclick = saveRegion;
for (const b of $("rgKinds").children) b.onclick = () => setRegionKind(b.dataset.kind);
$("rgSvg").addEventListener("pointerdown", regionDown);
$("rgSvg").addEventListener("pointermove", regionMove);
$("rgSvg").addEventListener("pointerup", regionUp);
$("rgSvg").addEventListener("pointercancel", regionUp);
