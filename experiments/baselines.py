"""Published WLASL baseline numbers from prior sign language recognition work.

These are all fully-supervised: the model is trained on the same WLASL-100
classes it is tested on, and scored on the standard test split. Our evaluation
is class-disjoint few-shot, so the two are not directly comparable and any
table mixing them needs to say so.
"""

from typing import Dict, List, NamedTuple


class BaselineEntry(NamedTuple):
    method: str
    dataset: str
    protocol: str       # "fully_supervised" or "few_shot_5w1s" etc.
    top1: float         # accuracy (0-1)
    citation_key: str
    notes: str


# ---------------------------------------------------------------------------
# Published baselines (fully-supervised WLASL — strict asterisk)
# ---------------------------------------------------------------------------

PUBLISHED_BASELINES: List[BaselineEntry] = [
    # WLASL-100 results from Li et al. WACV 2020 (the dataset paper itself).
    # These are reported in Table 5 of the paper.
    BaselineEntry(
        method="I3D (RGB)",
        dataset="WLASL-100",
        protocol="fully_supervised",
        top1=0.6588,
        citation_key="li2020wlasl",
        notes="Trained on WLASL-100 train split, evaluated on WLASL-100 test split. Same classes.",
    ),
    BaselineEntry(
        method="Pose-TGCN (skeleton)",
        dataset="WLASL-100",
        protocol="fully_supervised",
        top1=0.5543,
        citation_key="li2020wlasl",
        notes="Skeleton-only, OpenPose keypoints, TGCN backbone. Same train/test classes.",
    ),

    # SL-GCN — Jiang et al. CVPRW 2021. Strongest published skeleton-only baseline.
    BaselineEntry(
        method="SL-GCN",
        dataset="WLASL-100",
        protocol="fully_supervised",
        top1=0.8156,
        citation_key="jiang2021slgcn",
        notes=(
            "Multi-stream skeleton GCN. Numbers are skeleton-only stream "
            "(joint stream). Full multi-modal version higher. Same classes."
        ),
    ),

    # MS-G3D: the original paper evaluated on NTU RGB+D, so WLASL numbers come
    # from community reimplementations and vary. Verify before citing.
    BaselineEntry(
        method="MS-G3D (transferred)",
        dataset="WLASL-100",
        protocol="fully_supervised",
        top1=None,   # not reported on WLASL in the original paper
        citation_key="liu2020msg3d",
        notes=(
            "Original paper evaluates on NTU RGB+D, not WLASL. Number not "
            "directly available. Use community reimpl or skip."
        ),
    ),

    # Bilge et al. — zero-shot SLR. Different dataset (MS-ASL), but the
    # evaluation protocol (class-disjoint splits) is what you'll adopt.
    # Cite the methodology, not the number.
    BaselineEntry(
        method="Bilge ZSL (class-disjoint)",
        dataset="MS-ASL-1000",
        protocol="zero_shot",
        top1=None,
        citation_key="bilge2022zsl",
        notes=(
            "Different dataset (MS-ASL). Cited for evaluation protocol "
            "(class-disjoint splits with auxiliary class embeddings), not "
            "the accuracy number."
        ),
    ),
]


# Our own numbers are not duplicated here — read them from
# results_baseline.json, which eval_harness.py appends to.


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

CITATIONS: Dict[str, str] = {
    "li2020wlasl": (
        "Li, Dongxu, et al. 'Word-level deep sign language recognition from "
        "video: A new large-scale dataset and methods comparison.' "
        "WACV 2020."
    ),
    "jiang2021slgcn": (
        "Jiang, Songyao, et al. 'Skeleton aware multi-modal sign language "
        "recognition.' CVPRW 2021."
    ),
    "liu2020msg3d": (
        "Liu, Ziyu, et al. 'Disentangling and unifying graph convolutions for "
        "skeleton-based action recognition.' CVPR 2020."
    ),
    "bilge2022zsl": (
        "Bilge, Yunus Can, Nazli Ikizler-Cinbis, and Ramazan Gokberk Cinbis. "
        "'Towards zero-shot sign language recognition.' TPAMI 2022."
    ),
    "snell2017protonet": (
        "Snell, Jake, Kevin Swersky, and Richard Zemel. 'Prototypical networks "
        "for few-shot learning.' NeurIPS 2017."
    ),
    "cuturi2017softdtw": (
        "Cuturi, Marco, and Mathieu Blondel. 'Soft-DTW: a differentiable loss "
        "function for time-series.' ICML 2017."
    ),
    "petitjean2011dba": (
        "Petitjean, François, Alain Ketterlin, and Pierre Gançarski. "
        "'A global averaging method for dynamic time warping, with applications "
        "to clustering.' Pattern Recognition 44.3 (2011): 678-693."
    ),
    "bai2018tcn": (
        "Bai, Shaojie, J. Zico Kolter, and Vladlen Koltun. 'An empirical "
        "evaluation of generic convolutional and recurrent networks for "
        "sequence modeling.' arXiv:1803.01271 (2018)."
    ),
    "muller2007ir": (
        "Müller, Meinard. 'Information retrieval for music and motion.' "
        "Springer 2007. (Chapter 4 — subsequence DTW.)"
    ),
}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_baseline_row(entry: BaselineEntry, max_method_width: int = 28) -> str:
    """One-line formatter for a baseline entry. Useful for printing tables."""
    top1_str = f"{entry.top1 * 100:6.2f}%" if entry.top1 is not None else "  —  "
    method = entry.method[:max_method_width].ljust(max_method_width)
    return f"  {method}  {entry.dataset:12s}  {entry.protocol:20s}  {top1_str}  [{entry.citation_key}]"


def print_baselines():
    """Pretty-print the published baselines."""
    print("Published baselines (fully-supervised; not comparable to few-shot):")
    print(f"  {'method'.ljust(28)}  {'dataset':12s}  {'protocol':20s}  {'top1':>7s}  citation")
    print(f"  {'-' * 28}  {'-' * 12}  {'-' * 20}  {'-' * 7}  {'-' * 20}")
    for entry in PUBLISHED_BASELINES:
        print(format_baseline_row(entry))


if __name__ == "__main__":
    print_baselines()
