"""Generate presentation figures for the final WIP talk.

Outputs PNGs into rmd/figures/. Figures that illustrate algorithms (DTW path,
DBA barycenter, soft-DTW expected alignment) are computed with the REAL
implementations in experiments/, not mockups.

Usage:  python make_figures.py            # all figures
        python make_figures.py arch dba   # subset
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "experiments"
sys.path.insert(0, str(EXP))

# --- style ------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"],
    "axes.unicode_minus": False,
    "font.size": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.9,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})

NAVY = "#1F3864"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREEN = "#70AD47"
GRAY = "#A6A6A6"
RED = "#C00000"
PURPLE = "#7030A0"
LIGHT = "#F2F2F2"

CYCLE_COLORS = {1: GREEN, 2: BLUE, 3: PURPLE, 4: ORANGE}
CYCLE_TINTS = {1: "#E2EFDA", 2: "#DEEBF7", 3: "#EDE4F3", 4: "#FBE5D6"}


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fontsize=11, fontcolor=NAVY,
         lw=1.6, weight="bold", zorder=3, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                fc=fc, ec=ec, lw=lw, zorder=zorder, linestyle=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, weight=weight, zorder=zorder + 1)


def _arrow(ax, x0, y0, x1, y1, color=NAVY, lw=2.2, style="-|>", ls="-", zorder=2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=15, color=color, lw=lw,
                                 linestyle=ls, zorder=zorder))


def _callout(ax, x, y, w, h, title, body, color, tint, ls="-",
             title_fs=11, body_fs=9.5, dy=0.0):
    """Rounded box with a bold title and the description packed directly
    underneath (the title/body pair is vertically centered as one block;
    dy nudges the whole text block up/down)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008",
                                fc=tint, ec=color, lw=1.8, linestyle=ls, zorder=3))
    lh = 0.055                                  # approx. line height (axes coords)
    n_body = body.count("\n") + 1
    split = y + h / 2 + 0.5 * lh * (n_body - 1) + dy
    ax.text(x + w / 2, split + 0.008, title, ha="center", va="bottom",
            fontsize=title_fs, color=color, weight="bold", zorder=4)
    ax.text(x + w / 2, split - 0.006, body, ha="center", va="top",
            fontsize=body_fs, color="#333333", zorder=4, linespacing=1.5)


# ============================================================================
# 1. Architecture: where each improvement lives
# ============================================================================
def fig_arch():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1); ax.set_ylim(0.06, 1.0); ax.axis("off")

    ax.text(0.5, 0.985, "",
            ha="center", va="top", fontsize=16, color=NAVY, weight="bold")

    # ---- inference pipeline (middle row) --------------------------------------
    Y, H = 0.55, 0.17
    boxes = [
        (0.015, 0.085, "Webcam\n映像"),
        (0.122, 0.108, "MediaPipe\nHands 検出"),
        (0.244, 0.136, "ランドマーク\nバッファ\n(30, 42, 3)"),
        (0.402, 0.122, "埋め込みネット\nCNN + TCN"),
        (0.546, 0.118, "クエリ埋め込み\n(30, 256)"),
        (0.686, 0.100, "Partial DTW\n照合"),
        (0.808, 0.100, "判定\n(閾値・発火)"),
        (0.930, 0.057, "字幕"),
    ]
    for i, (x, w, t) in enumerate(boxes):
        _box(ax, x, Y, w, H, t, fontsize=10.5)
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            _arrow(ax, x + w + 0.002, Y + H / 2, nx - 0.002, Y + H / 2)

    # ---- ① fallback: callout above the detection→buffer hand-off --------------
    _callout(ax, 0.06, 0.815, 0.335, 0.135,
             "① 手の検出失敗フレームの穴埋め",
             "減衰 (5f で手なし扱い)・バッファ破棄・可視率の記録",
             CYCLE_COLORS[1], CYCLE_TINTS[1])
    _arrow(ax, 0.243, 0.812, 0.243, Y + H + 0.008, color=CYCLE_COLORS[1], lw=2)

    # ---- ③ decoder calibration: callout above the decision box ----------------
    _callout(ax, 0.665, 0.815, 0.32, 0.135,
             "③ 連続入力の復号較正",
             "no-match 閾値・発火条件をチューニング",
             CYCLE_COLORS[3], CYCLE_TINTS[3])
    _arrow(ax, 0.858, 0.812, 0.858, Y + H + 0.008, color=CYCLE_COLORS[3], lw=2)

    # ---- ④ soft-DTW: training-only box, feeds the embedding net ---------------
    _callout(ax, 0.215, 0.18, 0.335, 0.19,
             "④ Soft-DTW 損失 (学習時のみ)",
             "min → softmin で全整合経路に勾配を流す\n学習が終われば上の流れはそのまま\n(推論側は一切変わらない)",
             CYCLE_COLORS[4], CYCLE_TINTS[4], ls=(0, (4, 3)), title_fs=11.5,
             dy=-0.02)
    _arrow(ax, 0.42, 0.375, 0.458, Y - 0.006, color=ORANGE, ls=(0, (4, 3)))

    # ---- ② DBA: prototype DB box, feeds the matcher ----------------------------
    _callout(ax, 0.60, 0.18, 0.335, 0.19,
             "② 原型データベースを DBA で集約",
             "N クラス × K 録画の「お手本」を\nクラスごとに整合平均して 1 本へ\n→ 照合回数 1/K = 4.6× 高速",
             CYCLE_COLORS[2], CYCLE_TINTS[2], title_fs=11.5, dy=-0.02)
    _arrow(ax, 0.755, 0.375, 0.733, Y - 0.006, color=BLUE)

    fig.savefig(HERE / "fig_arch_where.png")
    plt.close(fig)


# ============================================================================
# 2. Fallback (hand-detection failure) — the 3 fixes
# ============================================================================
def fig_fallback():
    fig = plt.figure(figsize=(13.5, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], wspace=0.06)

    # ---- top: frame timelines -------------------------------------------------
    axT = fig.add_subplot(gs[0, :])
    n = 22
    lose_at = 7                                  # hand lost from frame 7 on
    axT.set_xlim(-5.0, 36.5); axT.set_ylim(-0.65, 4.35); axT.axis("off")
    axT.set_title("手検出失敗フォールバック: フレーム 7 以降カメラから手が消えた場合",
                  fontsize=14.5, color=NAVY, weight="bold", pad=10)

    # legend row
    for x, fc, alpha, t in [(-5.0, GREEN, 1.0, "手を検出できたフレーム"),
                            (7.5, ORANGE, 1.0, "直前ポーズで穴埋め"),
                            (17.0, "#FFFFFF", 1.0, "ゼロ埋め (手なし扱い)")]:
        axT.add_patch(Rectangle((x, 3.78), 0.85, 0.42, fc=fc, alpha=alpha,
                                ec="#888888", lw=0.7))
        axT.text(x + 1.15, 3.99, t, va="center", fontsize=10)

    lanes = [
        (2.5, "旧 Handy", "ghost",
         "直前の手の形を無期限に貼り続ける\n→ 静止した「幽霊ポーズ」が埋め込みを汚染",
         RED),
        (1.25, "Fix#1 減衰\n(5フレーム)", "decay",
         "5フレーム (~250ms) だけ保持し,\nその後ゼロへ → 正しく「手なし」になる",
         NAVY),
        (0.0, "Fix#2 バッファ破棄\n(両手 8 フレーム超)", "discard",
         "両手とも 8 フレーム超消えたら\n溜めかけの 30f バッファごと破棄して再開",
         NAVY),
    ]
    for y, label, mode, note, notecolor in lanes:
        axT.text(-0.45, y + 0.4, label, ha="right", va="center", fontsize=10.5,
                 weight="bold", color=NAVY, linespacing=1.4)
        for f in range(n):
            if f < lose_at:
                fc, alpha = GREEN, 1.0
            else:
                run = f - lose_at + 1
                if mode == "ghost":
                    fc, alpha = ORANGE, 1.0
                elif run <= 5:
                    fc, alpha = ORANGE, max(0.25, 1.0 - 0.17 * (run - 1))
                else:
                    fc, alpha = "#FFFFFF", 1.0
            axT.add_patch(Rectangle((f * 1.0, y), 0.9, 0.8, fc=fc, alpha=alpha,
                                    ec="#888888", lw=0.7))
        if mode == "discard":
            fx = lose_at + 8                      # 9th both-missing frame
            axT.text(fx + 0.45, y + 0.4, "×", ha="center", va="center",
                     fontsize=16, color=RED, weight="bold")
            axT.annotate("ここで破棄", xy=(fx + 0.45, y - 0.04),
                         xytext=(fx + 0.45, y - 0.52), ha="center", fontsize=9,
                         color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
        axT.text(23.0, y + 0.4, note, ha="left", va="center", fontsize=10,
                 color=notecolor, linespacing=1.5,
                 weight="bold" if notecolor == RED else "normal")

    # ---- bottom-left: Fix#3 presence penalty ---------------------------------
    axP = fig.add_subplot(gs[1, 0])
    q = [0.95, 0.10]; p = [0.95, 0.85]
    xs = np.arange(2)
    bq = axP.bar(xs - 0.18, q, 0.32, color=ORANGE, label="クエリ (入力) の可視率")
    bp = axP.bar(xs + 0.18, p, 0.32, color=BLUE, label="原型 (お手本) の可視率")
    for bars in (bq, bp):
        for b in bars:
            axP.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
                     f"{b.get_height():.2f}", ha="center", fontsize=9.5)
    axP.set_xticks(xs); axP.set_xticklabels(["左手", "右手"], fontsize=11)
    axP.set_ylim(0, 1.42); axP.set_ylabel("手が映っていたフレームの割合", fontsize=10)
    axP.legend(fontsize=9.5, loc="upper right", framealpha=0.9)
    axP.annotate("", xy=(1.18, 0.85), xytext=(1.18, 0.10),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
    axP.text(1.24, 0.46, "差 0.75", color=RED, fontsize=10.5, weight="bold")
    axP.set_title("Fix#3 可視率ペナルティ (照合時)", fontsize=12.5, color=NAVY,
                  weight="bold")
    axP.text(0.03, 0.955, "DTW距離 += 0.3 × |可視率の差|\n→ 片手のクエリが両手のお手本に化けない",
             transform=axP.transAxes, fontsize=9.5, va="top", color="#333333",
             bbox=dict(boxstyle="round,pad=0.4", fc="#FFF7F0", ec=ORANGE, lw=1))
    for sp in ["top", "right"]:
        axP.spines[sp].set_visible(False)

    # ---- bottom-right: why it matters + effect --------------------------------
    axW = fig.add_subplot(gs[1, 1]); axW.axis("off")
    axW.text(0.0, 1.00, "なぜこれが効くのか", fontsize=12.5, weight="bold", color=NAVY)
    msg = ("・WLASL novel クラスでは 38.2% のフレームで手が非検出\n"
           "    (録画の 93% が少なくとも 1 フレームの欠損を含む)\n"
           "・旧仕様では欠損区間が「静止した偽の手」になり,\n"
           "    どの原型とも低コストで整合 → 誤マッチの温床\n"
           "・3 つともモデル変更なし (前処理と照合の修正のみ)")
    axW.text(0.0, 0.86, msg, fontsize=10.5, va="top", linespacing=1.85)
    axW.add_patch(FancyBboxPatch((0.0, 0.00), 0.97, 0.17, boxstyle="round,pad=0.02",
                                 fc="#E2EFDA", ec=GREEN, lw=1.6))
    axW.text(0.485, 0.085,
             "効果:  5-way 1-shot  44.08 → 52.71 %  (+8.6pt)      "
             "10-way 5-shot  47.33 → 54.65 %",
             ha="center", va="center", fontsize=10.5, weight="bold", color="#375623")
    fig.savefig(HERE / "fig_fallback.png")
    plt.close(fig)


# ============================================================================
# 3. DTW alignment (the "stretchy ruler")
# ============================================================================
def fig_dtw():
    from dba import full_dtw_with_path
    # same hand movement, recording B is simply faster (22f vs 34f)
    t1 = np.linspace(0, 1, 34)
    a = np.exp(-((t1 - 0.32) / 0.10) ** 2) + 0.55 * np.exp(-((t1 - 0.72) / 0.08) ** 2)
    t2 = np.linspace(0, 1, 22)
    b = np.exp(-((t2 - 0.32) / 0.10) ** 2) + 0.55 * np.exp(-((t2 - 0.72) / 0.08) ** 2)
    _, path = full_dtw_with_path(a[:, None].astype(np.float64),
                                 b[:, None].astype(np.float64))
    off = 1.7
    p1a = int(np.argmax(a[:17])); p2a = 17 + int(np.argmax(a[17:]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True,
                             layout="constrained")
    for ax in axes:
        ax.plot(np.arange(len(a)), a + off, color=BLUE, lw=3.2, zorder=3)
        ax.plot(np.arange(len(b)), b, color=ORANGE, lw=3.2, zorder=3)
        ax.set_xlim(-1.5, len(a) + 0.5); ax.set_ylim(-0.55, off + 1.85)
        ax.set_xlabel("フレーム (時間 →)", fontsize=11)
        ax.set_yticks([]); ax.set_xticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    # curve labels (left panel only)
    axes[0].text(0.0, off + 1.18, "録画 A: ある手話表現を\nゆっくり行う (34 フレーム)",
                 fontsize=10, color=BLUE, weight="bold", linespacing=1.45)
    axes[0].text(0.0, -0.50, "録画 B: 同じ表現を 速く行う (22 フレーム)",
                 fontsize=10, color=ORANGE, weight="bold")

    # ---- (a) naive frame-by-frame comparison ----------------------------------
    axA = axes[0]
    for i in range(0, len(b), 2):
        axA.plot([i, i], [a[i] + off, b[i]], color="#E08A8A", lw=1.1, zorder=1)
    axA.plot([10, 10], [a[10] + off, b[10]], color=RED, lw=2.6, zorder=2)
    axA.annotate("A はまだ山の途中なのに\nB はもう山を越えている\n→ 別の場面どうしを比較",
                 xy=(10.4, 2.12), xytext=(18.5, 2.88),
                 fontsize=9.5, color=RED, ha="left", va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axA.plot([22, 33], [1.42, 1.42], color="#999999", lw=1.0,
             linestyle=(0, (3, 3)))
    axA.text(27.5, 1.52, "A の残り 12 フレームは比べる相手なし",
             fontsize=9, color="#888888", ha="center")
    axA.set_title("(a) フレーム番号どうしで単純に比較すると…\n速度のズレ → 距離が大きい → 「別の表現」と誤判定",
                  fontsize=12, color=RED, weight="bold", linespacing=1.5)

    # ---- (b) DTW alignment -----------------------------------------------------
    axB = axes[1]
    for (i, j) in path[::2]:
        axB.plot([i, j], [a[i] + off, b[j]], color="#BBBBBB", lw=0.9, zorder=1)
    hi = {}
    for (i, j) in path:
        if i in (p1a, p2a) and i not in hi:
            hi[i] = j
    for i, j in hi.items():
        axB.plot([i, j], [a[i] + off, b[j]], color=GREEN, lw=2.8, zorder=2)
    axB.annotate("時間を伸び縮みさせて\n山と山・谷と谷を対応づける\n→ 距離が小さい → 「同じ表現」",
                 xy=(15.5, 1.35), xytext=(23.5, 2.88),
                 fontsize=9.5, color="#375623", weight="bold", ha="left",
                 va="center", linespacing=1.5,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    axB.set_title("(b) DTW = 時間の「伸び縮み定規」で比較すると…\n速度差を吸収して対応づけ → 正しく「同じ表現」と判定",
                  fontsize=12, color="#375623", weight="bold", linespacing=1.5)

    fig.suptitle("同じ手話表現でも, 行う速さは人や場面で変わる — その速度差を DTW が吸収する",
                 fontsize=13.5, color=NAVY, weight="bold")
    fig.savefig(HERE / "fig_dtw.png")
    plt.close(fig)


# ============================================================================
# 4. DBA vs naive mean vs medoid  (uses real experiments/dba.py)
# ============================================================================
def fig_dba():
    from dba import dba, medoid
    rng = np.random.default_rng(3)
    T = 60
    base_t = np.linspace(0, 1, T)

    def make_variant():
        k = rng.uniform(0.55, 1.8)
        shift = rng.uniform(-0.06, 0.06)
        w = np.clip(base_t ** k + shift, 0, 1)
        sig = (np.exp(-((w - 0.35) / 0.09) ** 2)
               + 0.6 * np.exp(-((w - 0.75) / 0.07) ** 2))
        return (sig * rng.uniform(0.9, 1.1)
                + rng.normal(0, 0.015, T))[:, None].astype(np.float64)

    seqs = [make_variant() for _ in range(5)]
    naive = np.mean(np.stack(seqs), axis=0)[:, 0]
    bary = dba(seqs, max_iters=20)[:, 0]
    med = seqs[medoid(seqs)][:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True,
                             layout="constrained")
    for ax in axes:
        for s in seqs:
            ax.plot(s[:, 0], color=GRAY, lw=1.1, alpha=0.65)
        ax.set_xlabel("フレーム", fontsize=11)
        ax.set_yticks([])
        ax.set_ylim(-0.12, 1.38)
        for sp in ["top", "right", "left"]:
            ax.spines[sp].set_visible(False)
    axes[0].plot(naive, color=RED, lw=3, ls="--")
    axes[0].set_title("(a) フレームごとの単純平均", fontsize=13, color=RED,
                      weight="bold")
    axes[0].annotate("タイミングがずれたまま平均\n→ 山が潰れて「誰の動きでもない」形に",
                     xy=(27, naive[27]), xytext=(30, 1.16), fontsize=10.5,
                     color=RED, ha="center",
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    axes[1].plot(med, color=BLUE, lw=2.0, ls=":")
    axes[1].plot(bary, color=ORANGE, lw=3.4)
    axes[1].set_title("(b) DBA: DTW で対応づけてから平均", fontsize=13,
                      color="#9C4B00", weight="bold")
    bi = int(np.argmax(bary))
    axes[1].annotate("山の形・高さが保たれた代表 1 本", xy=(bi, bary[bi]),
                     xytext=(38, 1.22), fontsize=10.5, color="#9C4B00",
                     ha="center", weight="bold",
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
    axes[1].plot([], [], color=GRAY, lw=1.1, alpha=0.8, label="入力 5 録画 (K=5)")
    axes[1].plot([], [], color=ORANGE, lw=3.4, label="DBA barycenter")
    axes[1].plot([], [], color=BLUE, lw=2.0, ls=":", label="medoid (対照: 平均せず 1 本選抜)")
    axes[1].legend(fontsize=9.5, loc="upper left", framealpha=0.9)
    fig.suptitle("K=5 録画 (灰) から「お手本」1 本を作る — experiments/dba.py の実出力",
                 fontsize=14, color=NAVY, weight="bold")
    fig.savefig(HERE / "fig_dba.png")
    plt.close(fig)


# ============================================================================
# 5. Soft-DTW: one hard path vs weighted all-path gradient (real soft_dtw.py)
# ============================================================================
def fig_softdtw():
    import torch
    from dba import full_dtw_with_path
    from soft_dtw import soft_dtw_from_cost

    rng = np.random.default_rng(0)
    t1 = np.linspace(0, 1, 26)
    a = np.stack([np.sin(2 * np.pi * t1), np.cos(2 * np.pi * t1 * 0.5)], 1)
    t2 = np.linspace(0, 1, 20) ** 1.3
    b = np.stack([np.sin(2 * np.pi * t2), np.cos(2 * np.pi * t2 * 0.5)], 1)
    a += rng.normal(0, 0.04, a.shape); b += rng.normal(0, 0.04, b.shape)

    _, path = full_dtw_with_path(a.astype(np.float64), b.astype(np.float64))
    hard = np.zeros((len(a), len(b)))
    for i, j in path:
        hard[i, j] = 1.0
    pi = np.array([p[0] for p in path]); pj = np.array([p[1] for p in path])

    D = torch.cdist(torch.tensor(a), torch.tensor(b)).double().requires_grad_(True)
    soft_dtw_from_cost(D, gamma=1.0).backward()
    E = D.grad.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout="constrained")
    im0 = axes[0].imshow(hard.T, origin="lower", aspect="auto", cmap="Blues",
                         vmin=0, vmax=1)
    axes[0].set_title("hard DTW (現行の学習)\n最良の 1 経路だけ → 勾配もそこにしか流れない",
                      fontsize=12, color=NAVY, weight="bold", linespacing=1.5)
    im1 = axes[1].imshow(E.T, origin="lower", aspect="auto", cmap="Oranges")
    axes[1].plot(pi, pj, color="white", ls="--", lw=1.8)
    axes[1].annotate("白破線 = hard の 1 経路\n(soft の重みの「稜線」に一致)",
                     xy=(pi[13], pj[13]), xytext=(15.5, 4.0), fontsize=10,
                     color="#9C4B00", weight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color="#9C4B00", lw=1.4))
    axes[1].set_title("Soft-DTW (γ=1, 提案する学習)\n全経路の重み付き平均 → 全整合可能性が学習信号に",
                      fontsize=12, color="#9C4B00", weight="bold", linespacing=1.5)
    for ax, im in [(axes[0], im0), (axes[1], im1)]:
        ax.set_xlabel("クエリ系列のフレーム", fontsize=10.5)
        ax.set_ylabel("原型系列のフレーム", fontsize=10.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("整合の重み (= 勾配の流れる量)", fontsize=9.5)
    fig.suptitle("min → softmin: 「どの対応もあり得る」を重みとして学習に使う — "
                 "experiments/soft_dtw.py の実勾配",
                 fontsize=13.5, color=NAVY, weight="bold")
    fig.savefig(HERE / "fig_softdtw.png")
    plt.close(fig)


# ============================================================================
# 6. Latency vs vocabulary size (from results_latency.json)
# ============================================================================
def fig_latency():
    data = json.loads((EXP / "results_latency.json").read_text())
    n = [d["n_classes"] for d in data]
    pr = [d["ms_per_query_per_recording"] for d in data]
    db = [d["ms_per_query_dba"] for d in data]

    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    ax.plot(n, pr, "o-", color=RED, lw=2.6, ms=8,
            label="per-recording: 全 K=5 録画と照合  →  O(N·K)")
    ax.plot(n, db, "s-", color=GREEN, lw=2.6, ms=8,
            label="DBA 集約: クラスごとに 1 本と照合  →  O(N)")
    for x, y in zip(n, pr):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9.5, color=RED)
    db_offsets = [(10, 2), (10, 2), (0, 9), (0, 9), (10, 0)]
    for x, y, off in zip(n, db, db_offsets):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=off,
                    ha="left" if off[0] else "center", fontsize=9.5, color=GREEN)
    ax.annotate("", xy=(80, db[-1]), xytext=(80, pr[-1]),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.8))
    ax.text(71, np.sqrt(pr[-1] * db[-1]), "4.6×", fontsize=16, weight="bold",
            color=NAVY, ha="center")

    ax.set_xscale("log", base=2)
    ax.set_xticks(n); ax.set_xticklabels([str(v) for v in n], fontsize=11)
    ax.minorticks_off()
    ax.set_xlabel("登録クラス数 N  (各クラス K=5 録画, 対数軸)", fontsize=11)
    ax.set_ylabel("1 クエリの照合時間 [ms]", fontsize=11)
    ax.set_ylim(0, 125)
    ax.set_title("原型を増やすほど照合が遅くなる問題を DBA 集約が 1/K に抑える (実測)",
                 fontsize=13.5, color=NAVY, weight="bold")
    ax.legend(fontsize=10.5, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.savefig(HERE / "fig_latency.png")
    plt.close(fig)


# ============================================================================
# 7. Continuous-stream WER decomposition (from results_continuous.json)
# ============================================================================
def fig_continuous():
    data = json.loads((EXP / "results_continuous.json").read_text())
    rows = [
        ("配備そのまま\n(per-rec, 閾値0.9)", data[0],
         "挿入の氾濫 = 喋っていないのに字幕が出続ける", RED),
        ("復号器を較正\n(トリム+発火3+閾値0.35)", data[1],
         "挿入は止まったが, 置換 (誤認識) が露出", "#333333"),
        ("較正 + DBA 集約\n(16ms/バッファ)", data[2],
         "今度は削除支配へ反転 → 閾値の最適値は原型戦略に依存", PURPLE),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.6), layout="constrained")
    ys = np.arange(len(rows))[::-1]
    for y, (label, d, note, notecolor) in zip(ys, rows):
        s, dl, ins = 100 * d["sub_rate"], 100 * d["del_rate"], 100 * d["ins_rate"]
        ax.barh(y, s, 0.55, color=BLUE, label="置換 S (別の単語と誤認)" if y == ys[0] else "")
        ax.barh(y, dl, 0.55, left=s, color=GRAY, label="削除 D (見逃し)" if y == ys[0] else "")
        ax.barh(y, ins, 0.55, left=s + dl, color=RED, label="挿入 I (幻の単語)" if y == ys[0] else "")
        for v, l, c in [(s, 0, BLUE), (dl, s, GRAY), (ins, s + dl, RED)]:
            if v > 9:
                ax.text(l + v / 2, y, f"{v:.0f}", ha="center", va="center",
                        fontsize=10, color="white", weight="bold")
        ax.text(s + dl + ins + 5, y + 0.07, f"WER {100 * d['WER']:.0f}%",
                va="bottom", fontsize=13, weight="bold", color=NAVY)
        ax.text(s + dl + ins + 5, y - 0.10, note, va="top", fontsize=9.5,
                color=notecolor)
    ax.axvline(100, color="#444444", ls="--", lw=1.2)
    ax.text(100, -0.62, "WER 100%", fontsize=9.5, color="#444444", ha="center")
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("単語誤り率の内訳 [%]   (Slerp 合成 100 ストリーム × 8 手話表現, novel 40 クラス 5-shot)",
                  fontsize=10.5)
    ax.set_xlim(0, 305); ax.set_ylim(-0.75, 2.75)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_title("孤立クリップで 52–55% でも, 連続入力では崩壊する — 較正で WER 280→94%, 誤りの種類は戦略で反転",
                 fontsize=13, color=NAVY, weight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.savefig(HERE / "fig_continuous.png")
    plt.close(fig)


# ============================================================================
# 8. Related-work positioning map (2 axes, quadrant explanations)
# ============================================================================
def fig_positioning():
    fig, ax = plt.subplots(figsize=(13, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#AAAAAA")

    # quadrant fills
    ax.add_patch(Rectangle((0, 0), 0.5, 0.5, fc="#EDF2F8", ec="none", zorder=0))
    ax.add_patch(Rectangle((0, 0.5), 0.5, 0.5, fc="#F0F0F0", ec="none", zorder=0))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.5, fc="#EBF3E7", ec="none", zorder=0))
    ax.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, fc="#FDF0E6", ec="none", zorder=0))
    ax.axhline(0.5, color="#AAAAAA", lw=1.2); ax.axvline(0.5, color="#AAAAAA", lw=1.2)

    # quadrant headers (title + one condensed ◎/× line)
    def quad(x, y, title, line, color):
        ax.text(x, y, title, ha="center", fontsize=12.5, weight="bold", color=color)
        ax.text(x, y - 0.045, line, ha="center", fontsize=9, color="#666666")

    quad(0.25, 0.455, "教師あり 単語SLR",
         "◎ 大規模データで高精度   × 新語は再学習・クリップ前提", BLUE)
    quad(0.25, 0.955, "連続手話認識 (CSLR)",
         "◎ 文単位の連続入力   × 語彙固定・大規模教師データ前提", "#666666")
    quad(0.75, 0.455, "Zero / Few-shot SLR",
         "◎ 新語を例示ゼロ〜少数で追加   × 孤立クリップ前提", GREEN)
    quad(0.75, 0.955, "目標領域 (空白地帯)",
         "語彙追加 × 連続入力 × 端末動作 を同時に扱う研究は未だない", "#9C4B00")

    # papers: (name, one-line what-it-does/accuracy, x, y, color)
    # x = how easily vocabulary extends (zero-shot < few-shot < instant 1-recording)
    # y = how continuous the handled input is
    pts = [
        ("I3D 基線 (Li+ 2020)", "WLASL-2000: 32.5%・RGB 3D-CNN",
         0.075, 0.305, BLUE),
        ("SignBERT+ (Hu+ 2023)", "WLASL-2000: 55.6%・自己教師あり事前学習",
         0.125, 0.125, BLUE),
        ("Tunga+ (2021)", "骨格 GCN+BERT・WLASL-100 対象",
         0.30, 0.305, BLUE),
        ("NLA-SLR (Zuo+ 2023)", "WLASL-2000: 61.1%・語義テキスト併用",
         0.385, 0.125, BLUE),
        ("Cui+ (2017)", "CNN+RNN 段階最適化の連続認識 (PHOENIX14)",
         0.135, 0.645, "#666666"),
        ("Camgöz+ (2018)", "連続手話 → ドイツ語文への翻訳 (PHOENIX14T)",
         0.32, 0.78, "#666666"),
        ("Bilge+ (2022)", "意味記述で例示ゼロの新クラス認識",
         0.60, 0.305, GREEN),
        ("Saad (2025)", "ST-GCN + ProtoNet の few-shot ASL 認識",
         0.665, 0.115, GREEN),
        ("Bilge+ (2024)", "言語横断 few-shot 認識 (Pattern Recognition)",
         0.865, 0.225, GREEN),
    ]
    for name, desc, x, y, c in pts:
        ax.scatter(x, y, s=140, color=c, zorder=3, ec="white", lw=1.2)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9.5, color=c, weight="bold")
        ax.annotate(desc, (x, y), textcoords="offset points", xytext=(0, -19),
                    ha="center", fontsize=8.2, color="#777777")

    # ours
    sx, sy = 0.78, 0.71
    ax.scatter(sx, sy, s=750, marker="*", color=ORANGE, ec=NAVY, lw=1.5, zorder=4)
    ax.annotate("Handy + 本研究", (sx, sy), textcoords="offset points",
                xytext=(0, 19), ha="center", fontsize=13.5, color="#9C4B00",
                weight="bold")
    ax.annotate("録画 1 本でその場登録・連続ストリームを評価・端末アプリ実装 (DTW照合 23ms)",
                (sx, sy), textcoords="offset points", xytext=(0, -24),
                ha="center", fontsize=8.8, color="#9C4B00")

    # 3rd axis (lightweight / on-device) note
    ax.text(0.5, 1.022,
            "※ 第3の軸「端末動作」は本図では省略 — ● の先行研究はいずれも GPU 推論前提で,"
            "軽量・低遅延の端末動作まで扱うのは ★ のみ",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#777777")

    # axis arrows + labels
    ax.annotate("", xy=(0.99, -0.05), xytext=(0.01, -0.05),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8),
                annotation_clip=False)
    ax.text(0.5, -0.095,
            "語彙の増やしやすさ:   固定 (新語ごとに再学習)   →   意味記述で追加 (zero-shot)   →   少数例でその場追加 (few-shot)",
            ha="center", fontsize=10.5, color=NAVY, weight="bold", clip_on=False)
    ax.annotate("", xy=(-0.035, 0.99), xytext=(-0.035, 0.01),
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.8),
                annotation_clip=False)
    ax.text(-0.063, 0.5, "入力の形:   切り出した単語クリップ   →   区切りのない連続手話",
            ha="center", va="center", fontsize=10.5, color=NAVY, weight="bold",
            rotation=90, clip_on=False)

    ax.set_title("関連研究の 2 軸ポジショニング — 位置 = 語彙の追加しやすさ × 入力の連続性",
                 fontsize=14, color=NAVY, weight="bold", pad=30)
    fig.savefig(HERE / "fig_positioning.png")
    plt.close(fig)


# ============================================================================
# 9. Cycle-1 results (detection fixes)
# ============================================================================
def fig_cycle1():
    settings = ["5-way 1-shot\n(chance 20%)", "10-way 1-shot\n(chance 10%)",
                "10-way 5-shot\n(chance 10%)"]
    base = [44.08, 32.61, 47.33]
    v2 = [52.71, 39.56, 52.37]
    pres = [52.22, 39.30, 54.65]
    chance = [20, 10, 10]
    x = np.arange(3); w = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    b1 = ax.bar(x - w, base, w, color=GRAY, label="共同開発時 (ベースライン)")
    b2 = ax.bar(x, v2, w, color=BLUE, label="+ 減衰・バッファ破棄 (Fix#1,2)")
    b3 = ax.bar(x + w, pres, w, color=GREEN, label="+ 可視率ペナルティ (Fix#3)")
    for bars, c in [(b1, "#777777"), (b2, BLUE), (b3, GREEN)]:
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                    f"{b.get_height():.1f}", ha="center", fontsize=9.5, color=c,
                    weight="bold")
    for xi, c in zip(x, chance):
        ax.plot([xi - 1.55 * w, xi + 1.55 * w], [c, c], color=RED, ls="--", lw=1.5)
    ax.text(x[0] - 1.5 * w, chance[0] + 1.2, "chance", color=RED, fontsize=9)

    ax.annotate("+8.6pt", xy=(x[0] + 0.02, v2[0] + 3), xytext=(x[0] + 0.52, 60),
                fontsize=12.5, weight="bold", color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))
    ax.annotate("+7.3pt\n(可視率は5-shotで有効)", xy=(x[2] + w, pres[2] + 4),
                xytext=(x[2] + w - 0.06, 63.5), fontsize=10.5, weight="bold",
                color="#375623", ha="center", linespacing=1.4,
                arrowprops=dict(arrowstyle="->", color="#375623", lw=1.4))

    ax.set_xticks(x); ax.set_xticklabels(settings, fontsize=11, linespacing=1.5)
    ax.set_ylabel("Top-1 精度 [%]  (1000 エピソード平均, novel 40 クラス)", fontsize=10.5)
    ax.set_ylim(0, 72)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("サイクル①: 手検出フォールバック修正の効果 (WLASL OOD, class-disjoint)",
                 fontsize=13.5, color=NAVY, weight="bold")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.savefig(HERE / "fig_cycle1_results.png")
    plt.close(fig)


FIGS = {
    "arch": fig_arch,
    "fallback": fig_fallback,
    "dtw": fig_dtw,
    "dba": fig_dba,
    "softdtw": fig_softdtw,
    "latency": fig_latency,
    "continuous": fig_continuous,
    "positioning": fig_positioning,
    "cycle1": fig_cycle1,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(FIGS)
    for t in targets:
        print(f"-> {t}")
        FIGS[t]()
    print("done:", ", ".join(f"fig_{t}" for t in targets))
