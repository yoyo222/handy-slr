"""
eval_sweep.py — few-shot transfer evaluation of fine-tuned checkpoints.

Answers "does (soft-)DTW fine-tuning transfer to novel classes?" on two class
subsets (see experiments/novel_classes.json):

  novel40 : the standard eval split. CAVEAT: 29/40 of these classes were in
            the wlasl100 training set of the ft checkpoints (leakage), so
            novel40 numbers for ft ckpts are optimistic.
  clean11 : novel40 minus WLASL top-100 — classes the ft runs never saw.
            The honest transfer number.

Loads landmarks once, reuses per-checkpoint embedding caches, appends every
result row to results_ft_transfer.json. Presence penalty off (matches the
52.71/39.56/52.37 v2 baseline rows in results_baseline.json).

Usage:
    python experiments/eval_sweep.py
    # evaluate additional checkpoints (skips the default grid):
    python experiments/eval_sweep.py --add ftv2_soft_g1_ep03=experiments/checkpoints/wlasl_ftv2_soft_g1_ep03.h5
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_harness import (
    load_wlasl_landmarks, make_class_disjoint_split, load_model,
    embed_all_classes, crop_presence, sample_episode, evaluate_episode,
)
from server.model.classify import partial_DTW  # noqa: E402  (path set by eval_harness import)

HERE = Path(__file__).resolve().parent
LANDMARKS = HERE / "wlasl_landmarks_v2"
OUTPUT = HERE / "results_ft_transfer.json"

CHECKPOINTS = [
    ("soft_g1_ep01", HERE / "checkpoints" / "wlasl_ft_soft_g1_ep01.h5"),
    ("soft_g1_ep02", HERE / "checkpoints" / "wlasl_ft_soft_g1_ep02.h5"),
    ("hard_ft_ep01", HERE / "checkpoints" / "wlasl_ft_ep01.h5"),
    ("baseline", HERE.parent / "server" / "model" / "weights.h5"),
]
CACHES = {
    "soft_g1_ep01": HERE / "soft_g1_ep01_v2_embeddings.pkl",
    "soft_g1_ep02": HERE / "soft_g1_ep02_v2_embeddings.pkl",
    "hard_ft_ep01": HERE / "hard_ft_ep01_v2_embeddings.pkl",
    "baseline": HERE / "wlasl_v2_embeddings.pkl",
}
SETTINGS = [(5, 1), (10, 1), (10, 5)]  # (n_way, k_shot)
N_EPISODES = 1000
N_QUERY = 5
SEED = 0

# baseline novel40 rows already exist in results_baseline.json — skip them.
SKIP = {("baseline", "novel40")}


def main():
    global CHECKPOINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], metavar="NAME=PATH",
                    help="checkpoint(s) to evaluate instead of the default grid")
    ap.add_argument("--normalize", choices=["none", "l2"], default="none",
                    help="l2 = unit-normalize embedding frames before DTW; "
                         "must match the checkpoint's training setting")
    args = ap.parse_args()
    if args.add:
        CHECKPOINTS = []
        for spec in args.add:
            name, path = spec.split("=", 1)
            CHECKPOINTS.append((name, Path(path).resolve()))
            CACHES[name] = HERE / f"{name}_v2_embeddings.pkl"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading landmarks from {LANDMARKS}...")
    landmark_dict, presence_dict = load_wlasl_landmarks(LANDMARKS)
    _, novel40 = make_class_disjoint_split(list(landmark_dict.keys()))
    subsets = {"novel40": novel40,
               "clean11": json.load(open(HERE / "novel_classes.json"))["clean11"]}
    presence_dict = crop_presence(presence_dict)

    # Embed with every checkpoint first (GPU phase), then episodes (CPU phase).
    embeddings = {}
    for name, ckpt in CHECKPOINTS:
        print(f"\n--- embedding with {name} ({ckpt.name}) ---")
        model = load_model(ckpt, device=device)
        embeddings[name] = embed_all_classes(model, landmark_dict,
                                             cache_path=CACHES[name], device=device)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.normalize == "l2":
        # caches hold raw model output; normalize after load so the same
        # cache serves both settings
        for name in embeddings:
            embeddings[name] = {
                cls: [r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)
                      for r in recs]
                for cls, recs in embeddings[name].items()}

    any_class = next(iter(next(iter(embeddings.values())).values()))
    _ = partial_DTW(any_class[0], any_class[1])  # numba warm-up

    results = []
    if OUTPUT.exists():
        try:
            results = json.load(open(OUTPUT))
        except json.JSONDecodeError:
            pass
    done_keys = {(r["model"], r["subset"], r["n_way"], r["k_shot"]) for r in results}

    for name, ckpt in CHECKPOINTS:
        for subset_name, classes in subsets.items():
            if (name, subset_name) in SKIP:
                continue
            for n_way, k_shot in SETTINGS:
                key = (name, subset_name, n_way, k_shot)
                if key in done_keys:
                    print(f"skip (done): {key}")
                    continue
                rng = random.Random(SEED)
                accs = []
                t0 = time.time()
                for ep in range(N_EPISODES):
                    support, query = sample_episode(
                        embeddings[name], presence_dict, classes,
                        n_way, k_shot, N_QUERY, rng)
                    accs.append(evaluate_episode(support, query,
                                                 "per_recording", use_presence=False))
                row = {
                    "model": name, "checkpoint": str(ckpt),
                    "subset": subset_name, "n_subset_classes": len(classes),
                    "n_way": n_way, "k_shot": k_shot, "n_query": N_QUERY,
                    "n_episodes": N_EPISODES, "use_presence": False,
                    "normalize": args.normalize,
                    "mean_accuracy": float(np.mean(accs)),
                    "std_accuracy": float(np.std(accs)),
                    "seed": SEED, "seconds": round(time.time() - t0, 1),
                }
                results.append(row)
                json.dump(results, open(OUTPUT, "w"), indent=2)
                print(f"[{name} | {subset_name} | {n_way}w{k_shot}s] "
                      f"{row['mean_accuracy']*100:.2f}% ± {row['std_accuracy']*100:.2f}% "
                      f"({row['seconds']}s)", flush=True)

    print(f"\nAll done -> {OUTPUT}")


if __name__ == "__main__":
    main()
