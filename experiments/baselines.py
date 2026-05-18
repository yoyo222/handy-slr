"""
baselines.py — Published baseline numbers from prior SLR papers.

This is a LOOKUP TABLE of numbers from papers, with citations. Use it to
populate the "Comparison against published baselines" row in your results
tables and slides.

⚠️ CRITICAL CAVEAT (re-read this every time you cite these numbers):
   ALL of the numbers below are from FULLY-SUPERVISED evaluation:
   - the model was trained on the same WLASL-100 classes it's tested on
   - no class-disjoint hold-out
   - the reported accuracy is on the standard WLASL test split, not a
     few-shot episodic eval

   Your work uses a CLASS-DISJOINT FEW-SHOT eval (40 novel classes held out
   from any training). Your numbers and these numbers are NOT directly
   comparable. Always add an asterisk and explanation in any table that
   mixes them.

   For the final talk's headline table, show your numbers (your few-shot
   protocol) prominently and the published numbers in a separate "for
   reference, fully-supervised baselines achieve" footnote-style row.

Format of each entry:
    (method_name, dataset, protocol, top1_accuracy, citation_key, notes)

Citations are in CITATIONS dict below. Pull the BibTeX from each paper's
official page or Google Scholar when you write the final paper.

USAGE:
    from experiments.baselines import PUBLISHED_BASELINES, format_baseline_row
    for entry in PUBLISHED_BASELINES:
        print(format_baseline_row(entry))
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

    # MS-G3D — strong general skeleton action recognition baseline; numbers for
    # WLASL come from community reimplementations rather than the original
    # paper (which evaluated on NTU RGB+D, not WLASL).
    # When you cite this, verify the WLASL number — there's reimplementation
    # variability.
    BaselineEntry(
        method="MS-G3D (transferred)",
        dataset="WLASL-100",
        protocol="fully_supervised",
        top1=None,   # TODO: fill in from community impl / your own; not in original paper
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


# ---------------------------------------------------------------------------
# Your-system baselines (filled in as Phase 0/1 results come in)
# ---------------------------------------------------------------------------

YOUR_BASELINES: List[BaselineEntry] = [
    # These will be populated by eval_harness.py runs.
    # Suggested entries to fill in:
    #   ("weights.h5 per-recording, 5w1s", "WLASL-100", "few_shot_5w1s", XX.XX, "ours_phase0", "...")
    #   ("weights.h5 per-recording, 10w1s", ...)
    #   ("weights.h5 per-recording, 10w5s", ...)
    #   ("weights.h5 DBA-aggregated, 5w1s", ...)
    #   ...
    # Don't manually edit; have eval_harness.py append to results_baseline.json
    # and then a small script populate this list. Or just read the JSON in your
    # slide-building scripts directly.
]


# ---------------------------------------------------------------------------
# Citations (TODO: replace with BibTeX entries when writing paper)
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
    """Pretty-print all published baselines, then yours (when populated)."""
    print("PUBLISHED BASELINES (fully-supervised; NOT directly comparable to few-shot):")
    print(f"  {'method'.ljust(28)}  {'dataset':12s}  {'protocol':20s}  {'top1':>7s}  citation")
    print(f"  {'-' * 28}  {'-' * 12}  {'-' * 20}  {'-' * 7}  {'-' * 20}")
    for entry in PUBLISHED_BASELINES:
        print(format_baseline_row(entry))

    if YOUR_BASELINES:
        print("\nYOUR RESULTS (few-shot, class-disjoint):")
        print(f"  {'method'.ljust(28)}  {'dataset':12s}  {'protocol':20s}  {'top1':>7s}  citation")
        print(f"  {'-' * 28}  {'-' * 12}  {'-' * 20}  {'-' * 7}  {'-' * 20}")
        for entry in YOUR_BASELINES:
            print(format_baseline_row(entry))
    else:
        print("\nYOUR RESULTS: not yet populated. Run eval_harness.py and add entries.")


if __name__ == "__main__":
    print_baselines()
