"""Generate poster figures (A1) for the final WIP presentation, paper style.

Serif (Times / Yu Mincho) + Computer Modern math, muted colors, booktabs
table. Data comes from the canonical results (rmd/RESEARCH.md); the
calibration figure uses the REAL leave-one-out scores computed through the
continuous_bench code path (cached in calibration_scores.json).

Usage:  python make_poster_figures.py            # all
        python make_poster_figures.py transfer eq # subset
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "experiments"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP.parent))

plt.rcParams.update({
    "font.family": ["Times New Roman", "Yu Mincho", "Hiragino Mincho ProN",
                    "MS Mincho"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 13,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

# muted, CVD-safe (Tol): baseline gray / proposed blue; S/D/I = blue/sand/rose
GRAY = "#b8b8b8"
BLUE = "#4477AA"
SAND = "#DDCC77"
ROSE = "#CC6677"
EDGE = "black"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# 1. transfer: baseline vs soft-DTW+L2 (clean11)
# ---------------------------------------------------------------------------
def fig_transfer():
    settings = ["5-way 1-shot", "10-way 1-shot", "10-way 5-shot"]
    base = [65.07, 52.95, 67.13]
    ours = [69.30, 58.11, 76.05]
    x = np.arange(3)
    w = 0.30

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    b1 = ax.bar(x - w / 2 - 0.015, base, w, color=GRAY, ec=EDGE, lw=0.8,
                zorder=3, label="学習前 (baseline)")
    b2 = ax.bar(x + w / 2 + 0.015, ours, w, color=BLUE, ec=EDGE, lw=0.8,
                zorder=3, label="Soft-DTW + 正規化学習 (提案)")
    for r in list(b1) + list(b2):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.9,
                f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=11.5)
    ax.set_xticks(x, settings, fontsize=12.5)
    ax.set_ylabel("Top-1 精度 [%]", fontsize=12.5)
    ax.set_ylim(0, 92)
    ax.legend(loc="upper left", fontsize=11.5)
    _despine(ax)
    fig.savefig(HERE / "fig_poster_transfer.png")
    plt.close(fig)
    print("fig_poster_transfer.png")


# ---------------------------------------------------------------------------
# 2. WER progression with S/D/I breakdown
# ---------------------------------------------------------------------------
def fig_wer():
    configs = ["配備時\n(per-rec, 0.9)", "手動較正\n(per-rec, 0.35)",
               "手動較正\n(DBA, 0.35)", "自動較正 (提案)\n(DBA, 0.448)"]
    S = [61.1, 70.5, 20.8, 44.0]
    D = [0.0, 10.1, 66.9, 38.8]
    I = [218.6, 13.3, 0.0, 0.4]
    tot = [279.8, 93.9, 87.6, 83.1]
    x = np.arange(4)
    w = 0.5

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x, S, w, color=BLUE, ec=EDGE, lw=0.8, zorder=3, label="置換 S")
    ax.bar(x, D, w, bottom=S, color=SAND, ec=EDGE, lw=0.8, zorder=3, label="削除 D")
    ax.bar(x, I, w, bottom=np.array(S) + np.array(D), color=ROSE, ec=EDGE,
           lw=0.8, zorder=3, label="挿入 I")
    for xi, t in zip(x, tot):
        lbl = f"$\\bf{{{t:.1f}}}$" if xi == 3 else f"{t:.1f}"
        ax.text(xi, t + 6, lbl, ha="center", va="bottom", fontsize=12.5)
    ax.axhline(100, color="#555555", lw=0.8, ls=(0, (2, 3)), zorder=2)
    ax.text(2.45, 104, "WER 100%", fontsize=9.5, color="#555555", ha="center")
    ax.set_xticks(x, configs, fontsize=10.5)
    ax.set_ylabel("WER [%]", fontsize=12.5)
    ax.set_ylim(0, 305)
    ax.legend(loc="upper right", fontsize=11.5)
    _despine(ax)
    fig.savefig(HERE / "fig_poster_wer.png")
    plt.close(fig)
    print("fig_poster_wer.png")


# ---------------------------------------------------------------------------
# 3. conformal calibration: real LOO genuine-score distributions
# ---------------------------------------------------------------------------
def _loo_scores():
    cache = HERE / "calibration_scores.json"
    if cache.exists():
        return json.load(open(cache))
    import random
    import torch
    from continuous_bench import trim_active_span, auto_calibrate_threshold
    from eval_harness import load_wlasl_landmarks, make_class_disjoint_split, load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    landmark_dict, presence_dict = load_wlasl_landmarks(EXP / "wlasl_landmarks_v2")
    _, novel = make_class_disjoint_split(list(landmark_dict.keys()))
    model = load_model(EXP.parent / "server" / "model" / "weights.h5", device=device)

    rng = random.Random(0)  # mirrors continuous_bench seed/k_shot/trim_blank
    support_embs, support_pres = {}, {}
    for cls in novel:
        recs = list(zip(landmark_dict[cls], presence_dict[cls]))
        rng.shuffle(recs)
        embs, press = [], []
        for lm, pres in recs[:5]:
            lm, pres = trim_active_span(lm, pres)
            if len(lm) < 2:
                continue
            with torch.no_grad():
                x = torch.tensor(lm, dtype=torch.float32, device=device).unsqueeze(0)
                embs.append(model(x).squeeze(0).cpu().numpy())
            press.append(pres)
        support_embs[cls], support_pres[cls] = embs, press

    out = {}
    for strat in ("per_recording", "dba"):
        thr, scores = auto_calibrate_threshold(support_embs, support_pres,
                                               strat, use_presence=True, q=0.5)
        out[strat] = {"threshold": thr, "scores": scores}
        print(f"  {strat}: median {thr:.4f} ({len(scores)} scores)")
    json.dump(out, open(cache, "w"))
    return out


def fig_calibration():
    data = _loo_scores()
    per, dba = data["per_recording"], data["dba"]
    bins = np.linspace(0, 1.0, 41)

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    ax.hist(per["scores"], bins=bins, density=True, histtype="stepfilled",
            color=BLUE, alpha=0.25, ec=BLUE, lw=1.4, zorder=3,
            label="per-recording 本人スコア")
    ax.hist(dba["scores"], bins=bins, density=True, histtype="stepfilled",
            color=ROSE, alpha=0.25, ec=ROSE, lw=1.4, zorder=3,
            label="DBA 本人スコア")
    # headroom for a dedicated label band so annotations never sit on the bars
    ymax = ax.get_ylim()[1] * 1.42
    ax.set_ylim(0, ymax)
    ax.axvline(per["threshold"], color=BLUE, lw=1.6, ls="--", zorder=4)
    ax.axvline(dba["threshold"], color=ROSE, lw=1.6, ls="--", zorder=4)
    ax.axvline(0.35, color="black", lw=1.2, ls=(0, (1, 2)), zorder=4)
    # 0.343 (auto) and 0.35 (manual) nearly coincide -> stagger vertically,
    # each with a leader line back to its own axvline
    bb = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9)
    lead = dict(arrowstyle="-", lw=0.7, shrinkA=0, shrinkB=1)
    ax.annotate(f"自動閾値 {per['threshold']:.3f} (per-rec.)",
                (per["threshold"], ymax * 0.995),
                xytext=(per["threshold"] - 0.035, ymax * 0.995),
                ha="right", va="top", fontsize=10.5, color=BLUE,
                bbox=bb, zorder=6,
                arrowprops=dict(color=BLUE, **lead))
    ax.annotate("手動探索値 0.35", (0.35, ymax * 0.855),
                xytext=(0.35 - 0.035, ymax * 0.855),
                ha="right", va="top", fontsize=10.5, color="black",
                bbox=bb, zorder=6,
                arrowprops=dict(color="black", **lead))
    ax.annotate(f"自動閾値 {dba['threshold']:.3f} (DBA)",
                (dba["threshold"], ymax * 0.995),
                xytext=(dba["threshold"] + 0.035, ymax * 0.995),
                ha="left", va="top", fontsize=10.5, color=ROSE,
                bbox=bb, zorder=6,
                arrowprops=dict(color=ROSE, **lead))
    ax.set_xlabel("leave-one-out 本人スコア (正規化 DTW コスト)", fontsize=12.5)
    ax.set_ylabel("密度", fontsize=12.5)
    ax.set_xlim(0, 1.0)
    # right side beyond ~0.65 is near-empty; anchor below the label band
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.70), fontsize=10.5)
    _despine(ax)
    fig.savefig(HERE / "fig_poster_calibration.png")
    plt.close(fig)
    print("fig_poster_calibration.png")


# ---------------------------------------------------------------------------
# 4. results summary table (booktabs style)
# ---------------------------------------------------------------------------
def fig_table():
    rows = [
        ("① 手検出フォールバック", "Top-1 精度 (5-way 1-shot)", "44.1 %", "52.7 %", "+8.6"),
        ("② DBA 原型集約", "照合時間 (80クラス)", "108 ms", "23 ms", "×4.6"),
        ("③ conformal 自動較正", "連続入力 WER (DBA)", "87.6 %", "83.1 %", "−4.5"),
        ("④ Soft-DTW + 正規化学習", "Top-1 精度 (10-way 5-shot)", "67.1 %", "76.1 %", "+8.9"),
    ]
    header = ["改良", "評価指標", "改良前", "改良後", "効果"]
    widths = [0.30, 0.30, 0.125, 0.125, 0.15]
    aligns = ["left", "left", "center", "center", "center"]

    fig, ax = plt.subplots(figsize=(9.4, 2.6))
    ax.axis("off")
    n = len(rows) + 1
    rh = 1.0 / n
    xs = np.concatenate([[0], np.cumsum(widths)])
    # booktabs rules: toprule, midrule (under header), bottomrule
    ax.plot([0, 1], [1, 1], color="black", lw=1.6, clip_on=False)
    ax.plot([0, 1], [1 - rh, 1 - rh], color="black", lw=0.8, clip_on=False)
    ax.plot([0, 1], [0, 0], color="black", lw=1.6, clip_on=False)
    for j, h in enumerate(header):
        xa = xs[j] + (0.008 if aligns[j] == "left" else widths[j] / 2)
        ax.text(xa, 1 - rh / 2, h, ha=aligns[j] if aligns[j] != "left" else "left",
                va="center", fontsize=13, weight="bold")
    for i, row in enumerate(rows):
        y = 1 - (i + 2) * rh
        for j, cell in enumerate(row):
            xa = xs[j] + (0.008 if aligns[j] == "left" else widths[j] / 2)
            kw = dict(ha="left" if aligns[j] == "left" else "center",
                      va="center", fontsize=12.5)
            if j == 4:
                kw.update(weight="bold")
            ax.text(xa, y + rh / 2, cell, **kw)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(HERE / "fig_poster_results_table.png")
    plt.close(fig)
    print("fig_poster_results_table.png")


# ---------------------------------------------------------------------------
# 5. equations (Computer Modern)
# ---------------------------------------------------------------------------
def _equation(fname, tex, caption, fs=24):
    fig = plt.figure(figsize=(7.0, 1.5))
    fig.text(0.5, 0.68, tex, ha="center", va="center", fontsize=fs)
    fig.text(0.5, 0.12, caption, ha="center", va="center", fontsize=11.5,
             color="#444444")
    fig.savefig(HERE / fname)
    plt.close(fig)
    print(fname)


def fig_equations():
    _equation(
        "fig_eq_dtw.png",
        r"$D_{i,j} \;=\; d(x_i, y_j) \;+\; \min\left\{ D_{i-1,j-1},\; "
        r"D_{i,j-1},\; D_{i-1,j} \right\}$",
        "Partial DTW: 時間の伸び縮みを許して系列を照合する距離の漸化式")
    _equation(
        "fig_eq_softdtw.png",
        r"$\mathrm{sdtw}_{\gamma}(X,Y) \;=\; -\gamma \,\log "
        r"\sum_{\pi \in \mathcal{A}} \exp\!\left(-\frac{\langle \pi,\, "
        r"\Delta(X,Y) \rangle}{\gamma}\right)$",
        "Soft-DTW: 最良の 1 経路ではなく全整合経路から学ぶ (γ→0 で hard DTW に一致)")
    _equation(
        "fig_eq_conformal.png",
        r"$\tau \;=\; Q_{q}\left(\{ s_i \}\right), \qquad "
        r"s_i = \min_{t}\; c\!\left(r_i,\; \mathrm{proto}(R_{c_i} "
        r"\setminus r_i)\right)$",
        "Conformal 較正: 登録録画の leave-one-out 本人スコアの分位点を棄却閾値 τ に")


# ---------------------------------------------------------------------------
FIGS = {"transfer": fig_transfer, "wer": fig_wer, "calibration": fig_calibration,
        "table": fig_table, "eq": fig_equations}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(FIGS)
    for t in targets:
        FIGS[t]()
