"""
latency_bench.py: inference-latency comparison of per-recording vs DBA prototypes.

Measures wall-clock
classify time per query as a function of database size for both prototype
strategies, plus the one-off DBA aggregation cost per class.

The timed operation mirrors classify_query_simple from eval_harness.py:
partial_DTW against every prototype + length-normalized min. This is the
per-30-frame-buffer cost the deployed system pays every 1.5 s.

Usage:
    python latency_bench.py                       # default settings
    python latency_bench.py --k_shot 5 --n_queries 200
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.model.classify import partial_DTW

from dba import dba


def classify_once(query_emb, database):
    """Length-normalized min partial-DTW classification (timing target)."""
    best_class, best_cost = None, float("inf")
    for cls, emb in database:
        costs = partial_DTW(query_emb, emb)
        c = float(np.min(costs)) / max(len(emb), 1)
        if c < best_cost:
            best_cost, best_class = c, cls
    return best_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_path", type=Path,
                    default=Path(__file__).resolve().parent / "wlasl_embeddings.pkl")
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--class_counts", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "results_latency.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading embedding cache: {args.cache_path}")
    with open(args.cache_path, "rb") as f:
        embedding_dict = pickle.load(f)

    # Classes with enough recordings for k_shot prototypes + 1 query
    usable = sorted(c for c, v in embedding_dict.items() if len(v) >= args.k_shot + 1)
    print(f"{len(usable)} usable classes (need >= {args.k_shot + 1} recordings).")
    if max(args.class_counts) > len(usable):
        args.class_counts = [n for n in args.class_counts if n <= len(usable)]
        print(f"Capped class counts to {args.class_counts}")

    # Warm up numba JIT (both kernels)
    any_embs = embedding_dict[usable[0]]
    _ = partial_DTW(any_embs[0], any_embs[1])
    _ = dba(any_embs[: args.k_shot])

    results = []
    for n_classes in args.class_counts:
        classes = rng.sample(usable, n_classes)

        support = {}   # cls -> list of k_shot embeddings
        queries = []   # embeddings not used as support
        for cls in classes:
            recs = embedding_dict[cls][:]
            rng.shuffle(recs)
            support[cls] = recs[: args.k_shot]
            queries.extend(recs[args.k_shot:])
        rng.shuffle(queries)
        queries = queries[: args.n_queries]

        db_per_rec = [(cls, emb) for cls, embs in support.items() for emb in embs]

        t0 = time.perf_counter()
        db_dba = [(cls, dba(embs)) for cls, embs in support.items()]
        dba_build_s = time.perf_counter() - t0

        timings = {}
        for name, db in [("per_recording", db_per_rec), ("dba", db_dba)]:
            t0 = time.perf_counter()
            for q in queries:
                classify_once(q, db)
            total = time.perf_counter() - t0
            timings[name] = total / len(queries) * 1000.0  # ms / query

        speedup = timings["per_recording"] / timings["dba"]
        row = {
            "n_classes": n_classes,
            "k_shot": args.k_shot,
            "n_prototypes_per_recording": len(db_per_rec),
            "n_prototypes_dba": len(db_dba),
            "ms_per_query_per_recording": timings["per_recording"],
            "ms_per_query_dba": timings["dba"],
            "speedup": speedup,
            "dba_build_total_s": dba_build_s,
            "dba_build_per_class_ms": dba_build_s / n_classes * 1000.0,
            "n_queries": len(queries),
        }
        results.append(row)
        print(f"N={n_classes:3d} K={args.k_shot}: "
              f"per-rec {timings['per_recording']:7.2f} ms/query ({len(db_per_rec)} protos)  "
              f"DBA {timings['dba']:6.2f} ms/query ({len(db_dba)} protos)  "
              f"speedup {speedup:.2f}x  "
              f"(DBA build {dba_build_s / n_classes * 1000.0:.0f} ms/class, one-off)")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
