"""
eval_harness.py — n-way k-shot evaluation of TCNSignEmbedding checkpoints on WLASL.

USAGE:
    # First time (no embedding cache):
    python eval_harness.py --checkpoint ../server/model/weights.h5 \\
                           --landmarks_dir wlasl_landmarks_v2 \\
                           --n_way 5 --k_shot 1 --n_episodes 1000

    # Subsequent runs reuse the cache automatically.

    # Multiple n-way k-shot configurations:
    for n in 5 10; do for k in 1 5; do
        python eval_harness.py --checkpoint ../server/model/weights.h5 \\
                               --n_way $n --k_shot $k \\
                               --output results_baseline.json
    done; done

DESIGN NOTES:

1. CLASS-DISJOINT SPLIT.
   N_NOVEL_CLASSES of the available classes are held out as novel. The split
   is deterministic via SPLIT_SEED. Episodes are sampled ONLY from novel
   classes — this is what makes the eval few-shot.

2. EMBEDDING CACHE.
   Computing model embeddings for every recording takes a few minutes. After
   the first run, embeddings are cached to disk (--cache_dir) and reused.

3. FIXED 30-FRAME WINDOW.
   WLASL videos vary in length. We center-crop to 30 frames (the deployed
   model's input size). Short clips are padded with last-frame repetition.
   Document this in your final talk — it's a real source of eval noise.

4. THRESHOLD AT INFERENCE.
   For closed-set eval (every query has a correct class in the support set),
   we disable the no-match class by using a huge threshold.

5. NUMBA WARM-UP.
   partial_DTW is @njit; first call compiles. We do one warm-up call before
   timing/eval to amortize.
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# Add parent dir so we can import the deployed model + classifier
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.model.model import TCNSignEmbedding
from server.model.classify import partial_DTW


# Fixed seed for class-disjoint split (DO NOT change once experiments start)
SPLIT_SEED = 42
N_NOVEL_CLASSES = 40

# Fixed input window (matches deployed system)
WINDOW_FRAMES = 30

# Minimum recordings per class to be usable (need k_shot + n_query)
# For up to (5w-1s + 5q) we need >=6; for (10w-5s + 5q) we need >=10.
# Filter to at least 10 to support the headline configurations.
MIN_RECORDINGS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wlasl_landmarks(
    landmarks_dir: Path,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]]]:
    """Load all WLASL preprocessed landmark files + their presence sidecars.

    Returns: (landmarks_by_class, presence_by_class). Both dicts are keyed by
    class name; values are lists aligned by index. Only classes with
    >= MIN_RECORDINGS examples are kept. Missing presence sidecars get an
    all-ones fallback so old caches still work.
    """
    out = {}
    presence_out = {}
    for class_dir in sorted(landmarks_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        recordings = []
        presences = []
        for npy_path in sorted(class_dir.glob("*.npy")):
            if npy_path.name.endswith(".presence.npy"):
                continue
            try:
                arr = np.load(npy_path)
                if arr.ndim == 3 and arr.shape[1:] == (42, 3) and arr.shape[0] > 0:
                    recordings.append(arr)
                    pres_path = npy_path.with_name(npy_path.stem + ".presence.npy")
                    if pres_path.exists():
                        pres = np.load(pres_path).astype(np.uint8)
                    else:
                        pres = np.ones((arr.shape[0], 2), dtype=np.uint8)
                    presences.append(pres)
            except Exception as e:
                print(f"WARNING: skipping {npy_path}: {e}", file=sys.stderr)
        if len(recordings) >= MIN_RECORDINGS:
            out[class_dir.name] = recordings
            presence_out[class_dir.name] = presences
    return out, presence_out


def crop_to_window(landmarks: np.ndarray, window: int = WINDOW_FRAMES) -> np.ndarray:
    """Crop or pad a (T, ...) array to exactly `window` frames.

    Center-crop if T > window; pad with last-frame repetition if T < window.
    Works for landmarks (T, 42, 3) and for presence masks (T, 2).
    """
    T = landmarks.shape[0]
    if T == window:
        return landmarks
    if T > window:
        start = (T - window) // 2
        return landmarks[start:start + window]
    pad_count = window - T
    pad_shape = (pad_count,) + landmarks.shape[1:]
    pad = np.broadcast_to(landmarks[-1:], pad_shape)
    return np.concatenate([landmarks, pad], axis=0)


def make_class_disjoint_split(
    class_names: List[str], n_novel: int = N_NOVEL_CLASSES, seed: int = SPLIT_SEED
) -> Tuple[List[str], List[str]]:
    """Deterministic split into (base, novel). Novel is what eval uses."""
    sorted_names = sorted(class_names)
    rng = random.Random(seed)
    shuffled = sorted_names[:]
    rng.shuffle(shuffled)
    n_novel = min(n_novel, len(shuffled))
    novel = sorted(shuffled[:n_novel])
    base = sorted(shuffled[n_novel:])
    return base, novel


# ---------------------------------------------------------------------------
# Model loading + embedding
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path, device: str = "cpu") -> TCNSignEmbedding:
    """Load TCNSignEmbedding from a state_dict file (.h5 or .pt)."""
    model = TCNSignEmbedding().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints may lack the threshold parameter
    if "threshold" not in state_dict:
        state_dict["threshold"] = torch.tensor(4.0)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def embed_sequence(model: TCNSignEmbedding, landmarks: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Embed a single (T, 42, 3) landmark sequence.

    Crops to WINDOW_FRAMES, runs through the model, returns (WINDOW_FRAMES, 256).
    """
    cropped = crop_to_window(landmarks)
    x = torch.from_numpy(cropped).float().unsqueeze(0).to(device)  # (1, 30, 42, 3)
    with torch.no_grad():
        z = model(x)  # (1, 30, 256)
    return z.squeeze(0).cpu().numpy()


def crop_presence(presence_dict: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
    """Crop every presence mask to WINDOW_FRAMES, mirroring embed_sequence."""
    return {
        cls: [crop_to_window(p) for p in plist]
        for cls, plist in presence_dict.items()
    }


def embed_all_classes(
    model: TCNSignEmbedding,
    landmark_dict: Dict[str, List[np.ndarray]],
    cache_path: Path = None,
    device: str = "cpu",
) -> Dict[str, List[np.ndarray]]:
    """Embed every recording. Cache to disk if cache_path given.

    Returns: {class_name: [embedding (30, 256), ...]}
    """
    if cache_path is not None and cache_path.exists():
        print(f"Loading embeddings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Embedding all recordings...")
    t0 = time.time()
    out = {}
    total = sum(len(v) for v in landmark_dict.values())
    done = 0
    for class_name, recordings in landmark_dict.items():
        out[class_name] = [embed_sequence(model, r, device) for r in recordings]
        done += len(recordings)
        if done % 200 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] {rate:.1f}/s")

    print(f"Embedded {done} recordings in {time.time() - t0:.1f}s.")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
        print(f"Cached to {cache_path}")

    return out


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------

def sample_episode(
    embedding_dict: Dict[str, List[np.ndarray]],
    presence_dict: Dict[str, List[np.ndarray]],
    novel_classes: List[str],
    n_way: int,
    k_shot: int,
    n_query: int,
    rng: random.Random,
) -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]],
           List[Tuple[str, np.ndarray, np.ndarray]]]:
    """Sample one n-way k-shot episode from the novel classes.

    Returns (support, query) lists of (class_name, embedding, presence) triples.
    """
    eligible = [c for c in novel_classes if len(embedding_dict.get(c, [])) >= k_shot + n_query]
    if len(eligible) < n_way:
        raise RuntimeError(
            f"Not enough eligible novel classes (have {len(eligible)}, need {n_way})."
        )
    chosen_classes = rng.sample(eligible, n_way)

    support = []
    query = []
    for cls in chosen_classes:
        recordings = embedding_dict[cls]
        presences = presence_dict[cls]
        picks = rng.sample(range(len(recordings)), k_shot + n_query)
        for idx in picks[:k_shot]:
            support.append((cls, recordings[idx], presences[idx]))
        for idx in picks[k_shot:k_shot + n_query]:
            query.append((cls, recordings[idx], presences[idx]))
    return support, query


# ---------------------------------------------------------------------------
# Prototype strategies
# ---------------------------------------------------------------------------

def build_database(
    support: List[Tuple[str, np.ndarray, np.ndarray]],
    strategy: str = "per_recording",
) -> List[Tuple[str, np.ndarray, None, None, np.ndarray]]:
    """Build the database list expected by classify-like routines.

    Strategies:
      - "per_recording": each support example is its own prototype. Baseline.
      - "medoid": one prototype per class — the support recording with the
              smallest sum of DTW distances to its classmates. Control for
              DBA: isolates "1 prototype instead of k" from "averaging".
      - "dba": one DBA-aggregated barycenter per class. Phase 1 contribution.

    5th tuple field is the presence mask (shape (T, 2)) matching the prototype.
    """
    if strategy == "per_recording":
        return [(cls, emb, None, None, pres) for cls, emb, pres in support]

    if strategy == "medoid":
        from collections import defaultdict
        from dba import medoid
        per_class = defaultdict(list)
        for cls, emb, pres in support:
            per_class[cls].append((emb, pres))
        out = []
        for cls, items in per_class.items():
            idx = medoid([emb for emb, _ in items]) if len(items) > 1 else 0
            out.append((cls, items[idx][0], None, None, items[idx][1]))
        return out

    if strategy == "dba":
        from collections import defaultdict
        per_class_emb = defaultdict(list)
        per_class_pres = defaultdict(list)
        for cls, emb, pres in support:
            per_class_emb[cls].append(emb)
            per_class_pres[cls].append(pres)

        try:
            from dba import dba
        except ImportError as e:
            raise RuntimeError(
                "DBA strategy requires experiments/dba.py to be implemented. "
                f"Import error: {e}"
            )
        # For DBA: average presence masks across recordings (continuous-valued
        # average reflects per-frame visibility frequency).
        out = []
        for cls in per_class_emb:
            bary = dba(per_class_emb[cls])
            pres_avg = np.mean(
                np.stack([p.astype(np.float32) for p in per_class_pres[cls]], axis=0),
                axis=0,
            )
            out.append((cls, bary, None, None, pres_avg))
        return out

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

PRESENCE_LAMBDA = 0.3  # mirror server/model/classify.py


def classify_query_simple(
    query_embedding: np.ndarray,
    database: List[Tuple[str, np.ndarray, None, None, np.ndarray]],
    query_presence: np.ndarray = None,
    use_presence: bool = False,
) -> str:
    """Classify a single query by min length-normalized partial-DTW distance.

    If use_presence is True and both query/prototype have a presence mask, add
    PRESENCE_LAMBDA * L1(q_frac, p_frac) to each prototype's normalized cost.
    """
    q_frac = None
    if use_presence and query_presence is not None and len(query_presence) > 0:
        q_frac = np.asarray(query_presence, dtype=np.float32).mean(axis=0)

    best_class = None
    best_cost = float("inf")
    for entry in database:
        cls = entry[0]
        target_emb = entry[1]
        proto_presence = entry[4] if len(entry) >= 5 else None

        costs = partial_DTW(query_embedding, target_emb)
        min_cost = float(np.min(costs)) / max(len(target_emb), 1)

        if q_frac is not None and proto_presence is not None and len(proto_presence) > 0:
            p_frac = np.asarray(proto_presence, dtype=np.float32).mean(axis=0)
            min_cost += PRESENCE_LAMBDA * float(np.sum(np.abs(q_frac - p_frac)))

        if min_cost < best_cost:
            best_cost = min_cost
            best_class = cls
    return best_class


def evaluate_episode(
    support: List[Tuple[str, np.ndarray, np.ndarray]],
    query: List[Tuple[str, np.ndarray, np.ndarray]],
    prototype_strategy: str = "per_recording",
    use_presence: bool = False,
) -> float:
    """Run one episode. Return accuracy = fraction of queries classified correctly."""
    database = build_database(support, prototype_strategy)
    correct = 0
    for true_cls, q_emb, q_pres in query:
        pred = classify_query_simple(q_emb, database, q_pres, use_presence=use_presence)
        if pred == true_cls:
            correct += 1
    return correct / len(query)


# ---------------------------------------------------------------------------
# Outer eval loop
# ---------------------------------------------------------------------------

def eval_checkpoint(
    checkpoint_path: Path,
    landmarks_dir: Path,
    n_way: int,
    k_shot: int,
    n_query: int = 5,
    n_episodes: int = 1000,
    prototype_strategy: str = "per_recording",
    cache_path: Path = None,
    device: str = "cpu",
    seed: int = 0,
    use_presence: bool = False,
) -> Tuple[float, float]:
    """Main evaluation. Returns (mean_accuracy, std_accuracy)."""
    print(f"Loading landmarks from {landmarks_dir}...")
    landmark_dict, presence_dict = load_wlasl_landmarks(landmarks_dir)
    print(f"Loaded {len(landmark_dict)} classes with >= {MIN_RECORDINGS} recordings.")

    base, novel = make_class_disjoint_split(list(landmark_dict.keys()))
    print(f"Split: {len(base)} base, {len(novel)} novel.")
    print(f"Novel classes (first 10): {novel[:10]}{'...' if len(novel) > 10 else ''}")

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device=device)

    embedding_dict = embed_all_classes(model, landmark_dict, cache_path=cache_path, device=device)
    presence_dict = crop_presence(presence_dict)

    # Warm up numba JIT
    print("Warming up numba JIT...")
    any_class = next(iter(embedding_dict.values()))
    _ = partial_DTW(any_class[0], any_class[1])

    presence_tag = "presence-on" if use_presence else "presence-off"
    print(f"Running {n_episodes} episodes of {n_way}-way {k_shot}-shot "
          f"({prototype_strategy}, {presence_tag})...")
    rng = random.Random(seed)
    accs = []
    t0 = time.time()
    for ep in range(n_episodes):
        support, query = sample_episode(
            embedding_dict, presence_dict, novel, n_way, k_shot, n_query, rng
        )
        acc = evaluate_episode(support, query, prototype_strategy, use_presence=use_presence)
        accs.append(acc)
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t0
            mean_so_far = float(np.mean(accs))
            print(f"  [{ep + 1}/{n_episodes}] mean acc so far: {mean_so_far * 100:.2f}% "
                  f"({elapsed:.1f}s elapsed)")

    return float(np.mean(accs)), float(np.std(accs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parent.parent / "server" / "model" / "weights.h5")
    ap.add_argument("--landmarks_dir", type=Path,
                    default=Path(__file__).resolve().parent / "wlasl_landmarks_v2")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=1)
    ap.add_argument("--n_query", type=int, default=5)
    ap.add_argument("--n_episodes", type=int, default=1000)
    ap.add_argument("--prototype_strategy", choices=["per_recording", "medoid", "dba"],
                    default="per_recording")
    ap.add_argument("--cache_path", type=Path,
                    default=Path(__file__).resolve().parent / "wlasl_v2_embeddings.pkl")
    ap.add_argument("--device", default=None,
                    help="cpu or cuda. Default: cuda if available, else cpu.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_presence", action="store_true",
                    help="Apply the per-prototype presence-mismatch DTW penalty "
                         "(Fix #3). Requires .presence.npy sidecars in landmarks_dir.")
    ap.add_argument("--output", type=Path, default=None,
                    help="If set, append results to this JSON file.")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    mean_acc, std_acc = eval_checkpoint(
        checkpoint_path=args.checkpoint,
        landmarks_dir=args.landmarks_dir,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        n_episodes=args.n_episodes,
        prototype_strategy=args.prototype_strategy,
        cache_path=args.cache_path,
        device=args.device,
        seed=args.seed,
        use_presence=args.use_presence,
    )

    print()
    presence_tag = "presence-on" if args.use_presence else "presence-off"
    print(f"=== {args.n_way}-way {args.k_shot}-shot ({args.prototype_strategy}, {presence_tag}) ===")
    print(f"Mean accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}% "
          f"over {args.n_episodes} episodes")

    if args.output is not None:
        result = {
            "checkpoint": str(args.checkpoint),
            "landmarks_dir": str(args.landmarks_dir),
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "n_query": args.n_query,
            "n_episodes": args.n_episodes,
            "prototype_strategy": args.prototype_strategy,
            "use_presence": args.use_presence,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "seed": args.seed,
        }
        existing = []
        if args.output.exists():
            try:
                with open(args.output) as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(result)
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Results appended to {args.output}")


if __name__ == "__main__":
    main()
