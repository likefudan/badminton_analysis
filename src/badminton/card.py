"""Movement card: one PNG per player (court heatmap + trajectory + key numbers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .court import LENGTH, NET_Y, WIDTH, court_lines

_CJK_FONTS = ["PingFang SC", "Hiragino Sans GB", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS",
              "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "WenQuanYi Zen Hei", "Microsoft YaHei",
              "SimHei"]


def _setup_font() -> bool:
    """Use a font that has Chinese glyphs; False if none is installed (labels go English)."""
    from matplotlib import font_manager, rcParams

    names = {f.name for f in font_manager.fontManager.ttflist}
    for n in _CJK_FONTS:
        if n in names:
            rcParams["font.sans-serif"] = [n] + list(rcParams["font.sans-serif"])
            rcParams["font.family"] = "sans-serif"
            rcParams["axes.unicode_minus"] = False
            return True
    return False


_TEXT = {
    True: dict(title="跑动卡片", dist="跑动距离", speed_max="最高速度", speed_mean="平均速度", sprints="冲刺次数",
               in_play="回合内时间", zones="站位", front="前场", mid="中场", back="后场", left="左侧", right="右侧",
               unit_m="米", unit_mps="米/秒", unit_s="秒", times="次", near="近端", far="远端",
               note="只统计回合内时间；左右按球员面向球网的方向"),
    False: dict(title="Movement card", dist="Distance", speed_max="Top speed", speed_mean="Mean speed",
                sprints="Sprints", in_play="In-play time", zones="Zones", front="front", mid="mid", back="back",
                left="left", right="right", unit_m="m", unit_mps="m/s", unit_s="s", times="", near="near",
                far="far", note="In-play time only; left/right as the player faces the net"),
}


def draw_court(ax, color="#ffffff", lw=1.0) -> None:
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((-1, -1), WIDTH + 2, LENGTH + 2, color="#2e7d5b", zorder=0))
    for (x0, y0), (x1, y1) in court_lines():
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1)
    ax.plot([0, WIDTH], [NET_Y, NET_Y], color="#dddddd", lw=2.5, zorder=1)
    ax.set_xlim(-1, WIDTH + 1)
    ax.set_ylim(-1, LENGTH + 1)
    ax.set_aspect("equal")
    ax.axis("off")


def render_card(player: dict, xy: np.ndarray, in_play: np.ndarray, out_path: str | Path, title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zh = _setup_font()
    T = _TEXT[zh]
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_axes([0.02, 0.03, 0.45, 0.9])
    draw_court(ax)
    hm = player["heatmap"]
    counts = np.asarray(hm["counts"], float)
    if counts.size and counts.max() > 0:
        import cv2

        dens = cv2.GaussianBlur(counts, (0, 0), 2.0)  # 2 cells = 0.5 m
        dens /= dens.max()
        masked = np.ma.masked_where(dens < 0.06, dens)
        ax.pcolormesh(hm["x_edges"], hm["y_edges"], masked, cmap="YlOrRd", alpha=0.8, zorder=2,
                      shading="flat", vmin=0, vmax=1)
    pts = np.where(in_play[:, None], xy, np.nan)
    ax.plot(pts[:, 0], pts[:, 1], color="#1f3b73", lw=0.5, alpha=0.6, zorder=3)
    side = T["near"] if player["side"] == "near" else T["far"]
    fig.text(0.52, 0.9, f"{player['name']}", fontsize=20, weight="semibold")
    fig.text(0.52, 0.85, f"{T['title']} · {side}" + (f" · {title}" if title else ""), fontsize=10, color="#555")
    rows = [
        (T["dist"], f"{player['distance_m']:.0f} {T['unit_m']}"),
        (T["speed_max"], f"{player['max_speed_mps']:.1f} {T['unit_mps']}"),
        (T["speed_mean"], f"{player['mean_speed_mps']:.1f} {T['unit_mps']}"),
        (T["sprints"], f"{player['sprints']} {T['times']}".strip()),
        (T["in_play"], f"{player['in_play_s']:.0f} {T['unit_s']}"),
    ]
    y = 0.75
    for k, v in rows:
        fig.text(0.52, y, k, fontsize=11, color="#555")
        fig.text(0.95, y, v, fontsize=15, weight="semibold", ha="right")
        y -= 0.075
    z = player["zones_pct"]
    fig.text(0.52, y - 0.01, T["zones"], fontsize=11, color="#555")
    y -= 0.07
    for keys in (("front", "mid", "back"), ("left", "right")):
        fig.text(0.52, y, "   ".join(f"{T[k]} {z[k]:.0f}%" for k in keys), fontsize=12)
        y -= 0.055
    fig.text(0.52, 0.04, T["note"], fontsize=8, color="#888")
    out_path = Path(out_path)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)
    return out_path
