// 下拉刷新：添加到主屏幕后（standalone）Safari 没有自带的下拉刷新，这里自己做。
// 在页面顶部往下拉过一段距离再松手 → 重新加载整个页面（顺便拿到新部署的网页）。
// 正在上传时不重新加载（会把上传打断），只刷新数据；标注页不启用（会丢掉没保存的标注）。
(function () {
  const standalone = window.navigator.standalone === true ||
    (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches);
  if (!standalone) return; // 普通 Safari / 电脑浏览器自己有刷新

  const TRIGGER = 70;   // 指示器拉到这么多像素松手就刷新
  const MAX = 110;      // 最多拉这么远
  const box = document.createElement("div");
  box.id = "ptr";
  box.innerHTML = '<div class="ptr-dot"><svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">' +
    '<path d="M12 4v11m0 0l-5-5m5 5l5-5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg></div>' +
    '<div class="ptr-text">下拉刷新</div>';
  document.body.prepend(box);
  const text = box.querySelector(".ptr-text");

  let startX = 0, startY = 0, pull = 0, tracking = false, active = false, busy = false;

  const uploading = () => {
    try { return Boolean(queueBusy || (run && !run.finished)); } catch { return false; }
  };
  // 手指按在这些东西上时不算下拉：输入框、视频、标注画布，或者本身还能往上滚的小区域
  function blocked(target) {
    if (document.body.classList.contains("labeling")) return true;
    for (let n = target; n && n !== document.body && n.nodeType === 1; n = n.parentElement) {
      if (/^(INPUT|TEXTAREA|SELECT|VIDEO|CANVAS)$/.test(n.tagName)) return true;
      if (n.scrollTop > 0) return true;
      if (n.getAttribute("contenteditable") === "true") return true;
    }
    return false;
  }
  function show(y, label) {
    box.style.transform = `translateY(${y - 56}px)`;
    box.style.opacity = String(Math.min(1, y / TRIGGER));
    box.classList.toggle("ready", y >= TRIGGER);
    if (label) text.textContent = label;
  }
  function hide() {
    box.classList.add("back");
    box.classList.remove("ready", "spin");
    show(0);
    setTimeout(() => box.classList.remove("back"), 250);
  }

  document.addEventListener("touchstart", (e) => {
    if (busy || e.touches.length !== 1 || window.scrollY > 0 || blocked(e.target)) { tracking = false; return; }
    tracking = true;
    active = false;
    pull = 0;
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener("touchmove", (e) => {
    if (!tracking) return;
    const dx = e.touches[0].clientX - startX;
    const dy = e.touches[0].clientY - startY;
    if (!active) {
      // 先动的方向决定是不是下拉：横着划、往上推、页面已经滚下去了，都交还给页面
      if (Math.abs(dy) < 8 && Math.abs(dx) < 8) return;
      if (dy <= 0 || Math.abs(dx) > dy || window.scrollY > 0) { tracking = false; return; }
      active = true;
      box.classList.remove("back");
    }
    if (e.cancelable) e.preventDefault(); // 不让 iOS 自己的回弹把页面也拉下来
    pull = Math.min(MAX, Math.max(0, dy) * 0.5);
    show(pull, pull >= TRIGGER ? "松手刷新" : "下拉刷新");
  }, { passive: false });

  async function end() {
    if (!tracking) return;
    tracking = false;
    if (!active) return;
    active = false;
    if (pull < TRIGGER) { hide(); return; }
    busy = true;
    box.classList.add("spin");
    show(TRIGGER, "正在刷新…");
    if (!uploading()) {
      location.reload();
      return;
    }
    // 正在上传：只重新读录像列表，不打断上传
    try { if (typeof refreshList === "function") await refreshList(); } catch {}
    show(TRIGGER, "正在上传，只刷新了数据");
    setTimeout(() => { busy = false; hide(); }, 900);
  }
  document.addEventListener("touchend", end);
  document.addEventListener("touchcancel", () => { if (tracking && active) { tracking = active = false; hide(); } else tracking = false; });
})();
