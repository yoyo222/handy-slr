"""
Few-shot evaluation of weights.h5 on WLASL landmarks.

Constructs n-way k-shot episodes from MediaPipe-extracted landmark .npy files,
embeds with the deployed TCNSignEmbedding + weights.h5, classifies each query
against the support set using partial_DTW (length-normalized minimum-cost
score, min over k support clips per class), reports top-1 accuracy.

This mirrors the deployed system's matching logic from server/model/classify.py
adapted for episodic few-shot evaluation.

Usage:
    python experiments/few_shot_eval.py                            # 5-way 1-shot, 200 episodes
    python experiments/few_shot_eval.py --n_way 10 --k_shot 5
    python experiments/few_shot_eval.py --n_way 20 --k_shot 5 --n_episodes 500
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add server/ to sys.path so we can import the deployed model + classifier.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from model.model import get_model
from model.classify import partial_DTW


def load_class_index(landmarks_dir, min_per_class):
    """Return {gloss: [np.array(T,42,3), ...]} keeping only classes with enough clips."""
    index = {}
    for gloss_dir in Path(landmarks_dir).iterdir():
        if not gloss_dir.is_dir():
            continue
        clips = []
        for p in gloss_dir.glob("*.npy"):
            arr = np.load(p)
            if arr.shape[0] >= 5:  # skip suspiciously short clips
                clips.append(arr)
        if len(clips) >= min_per_class:
            index[gloss_dir.name] = clips
    return index


def center_30(clip):
    """30-frame center crop of the clip (pad by edge-repeat if shorter)."""
    T = clip.shape[0]
    if T == 30:
        return clip
    if T > 30:
        s = (T - 30) // 2
        return clip[s:s + 30]
    pad_before = (30 - T) // 2
    pad_after = 30 - T - pad_before
    return np.concatenate([
        np.repeat(clip[:1], pad_before, axis=0),
        clip,
        np.repeat(clip[-1:], pad_after, axis=0),
    ], axis=0)


def embed(model, clip, device):
    """(T,42,3) landmark array -> (T,256) embedding."""
    t = torch.tensor(clip, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        z = model(t).squeeze(0).cpu().numpy()
    return z


def precompute_embeddings(model, index, device):
    """For each clip: embed full (for support) and 30-frame center crop (for query).
    Returns {gloss: [(emb_full, emb_q30), ...]}.
    """
    out = {}
    t0 = time.time()
    n_clips = sum(len(v) for v in index.values())
    done = 0
    for gloss, clips in index.items():
        out[gloss] = []
        for clip in clips:
            emb_full = embed(model, clip, device)
            emb_q30 = embed(model, center_30(clip), device)
            out[gloss].append((emb_full, emb_q30))
            done += 1
            if done % 100 == 0:
                rate = done / (time.time() - t0)
                eta = (n_clips - done) / rate
                print(f"  embedded {done}/{n_clips}  ({rate:.1f}/s, eta {eta:.0f}s)",
                      flush=True)
    print(f"Embedded {n_clips} clips in {time.time() - t0:.1f}s")
    return out


def score_query(query_emb, support_emb):
    """Length-normalized minimum partial-DTW cost: how well does support_emb
    align ending somewhere within query_emb? Lower = better match."""
    costs = partial_DTW(query_emb, support_emb)
    return float(np.min(costs)) / len(support_emb)


def run_episode(emb_index, n_way, k_shot, q_query, rng):
    classes = rng.sample(list(emb_index.keys()), n_way)
    support = {}  # gloss -> [emb_full, ...]
    queries = []  # [(true_gloss, emb_q30), ...]
    for c in classes:
        clips = emb_index[c][:]
        rng.shuffle(clips)
        for emb_full, _ in clips[:k_shot]:
            support.setdefault(c, []).append(emb_full)
        for _, emb_q30 in clips[k_shot:k_shot + q_query]:
            queries.append((c, emb_q30))

    correct = 0
    for true_c, q_emb in queries:
        best_c, best_score = None, float("inf")
        for c, s_embs in support.items():
            for s_emb in s_embs:
                s = score_query(q_emb, s_emb)
                if s < best_score:
                    best_score, best_c = s, c
        if best_c == true_c:
            correct += 1
    return correct, len(queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="experiments/wlasl_landmarks")
    ap.add_argument("--weights", default="server/model/weights.h5")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=1)
    ap.add_argument("--q_query", type=int, default=5)
    ap.add_argument("--n_episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {args.weights}...")
    model = get_model(path=args.weights, device=device)
    model.eval()

    min_needed = args.k_shot + args.q_query
    print(f"Scanning {args.landmarks} (classes need ≥ {min_needed} clips)...")
    index = load_class_index(args.landmarks, min_per_class=min_needed)
    print(f"Usable classes: {len(index)}")

    if len(index) < args.n_way:
        print(f"ERROR: only {len(index)} usable classes; need ≥ {args.n_way} "
              f"for {args.n_way}-way eval. Lower --n_way or --k_shot.")
        sys.exit(1)

    print(f"Precomputing embeddings...")
    emb_index = precompute_embeddings(model, index, device)

    rng = random.Random(args.seed)
    print(f"\nRunning {args.n_episodes} episodes: "
          f"{args.n_way}-way {args.k_shot}-shot, {args.q_query} queries/class")
    print(f"(first episode is slow: numba JIT-compiling partial_DTW)")

    correct_total, q_total = 0, 0
    t0 = time.time()
    for ep in range(args.n_episodes):
        c, t = run_episode(emb_index, args.n_way, args.k_shot, args.q_query, rng)
        correct_total += c
        q_total += t
        if (ep + 1) % 20 == 0:
            acc = correct_total / q_total
            rate = (ep + 1) / (time.time() - t0)
            print(f"  [{ep+1}/{args.n_episodes}] running acc = {acc:.4f}  "
                  f"({rate:.1f} ep/s)", flush=True)

    acc = correct_total / q_total
    print(f"\n=== Result ===")
    print(f"{args.n_way}-way {args.k_shot}-shot top-1: "
          f"{acc:.4f}  ({correct_total}/{q_total})")


if __name__ == "__main__":
    main()
