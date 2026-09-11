"""
continuous_bench.py: continuous-stream evaluation of the deployed pipeline.

Fills the gap WLASL cannot measure: recognition on UNSEGMENTED signing.
Synthesizes continuous streams by Slerp-stitching isolated novel-class clips
(same transition mechanism the training pipeline uses), then replays the
deployed inference loop faithfully:

    30-frame non-overlapping buffers -> TCNSignEmbedding -> classify()
    (partial DTW + 10-frame chunk argmin + threshold no-match)
    -> duplicate-collapsing subtitle decoder (mirrors server recieve()).

METRICS
  - WER: Levenshtein distance between decoded word sequence and the true
    stitched sign sequence, normalized by reference length. Reported with
    substitution/deletion/insertion breakdown.
  - Sequence accuracy: fraction of streams decoded exactly.
  - Boundary precision/recall/F1: predicted boundaries = chunk positions
    where the per-chunk argmin label changes; true boundaries = stitch
    midpoints; greedy one-to-one matching within --tolerance frames.

PROTOTYPE STRATEGIES
  per_recording | medoid | dba  (same semantics as eval_harness.py).
  Support = k_shot recordings per novel class (all 40 novel classes are
  registered, mirroring a user vocabulary); streams draw from the remainder.

USAGE
    python continuous_bench.py --strategy per_recording
    python continuous_bench.py --strategy dba --n_streams 200
    python continuous_bench.py --strategy per_recording --no_presence
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server.model.classify import classify, partial_DTW
from eval_harness import (
    load_wlasl_landmarks, make_class_disjoint_split, load_model,
)
from dba import dba, medoid

WINDOW = 30          # deployed buffer size (frames)
CHUNK = 10           # classify() chunk size -> 3 windows per buffer


def trim_active_span(landmarks, presence):
    """Cut leading/trailing no-hands frames (WLASL clips start/end blank;
    in-app recordings are user-bounded, so trimming mirrors deployment)."""
    active = np.where(presence.sum(axis=1) > 0)[0]
    if len(active) == 0:
        return landmarks, presence
    lo, hi = active[0], active[-1] + 1
    return landmarks[lo:hi], presence[lo:hi]


# ---------------------------------------------------------------------------
# Stream synthesis (Slerp transitions, mirrors train_wlasl.HandGestureDataset)
# ---------------------------------------------------------------------------

def slerp_transition(end_frame, start_frame, length):
    """(42,3) -> (42,3) rotation-vector Slerp, `length` intermediate frames."""
    out = []
    for j in range(42):
        rots = Rotation.from_rotvec(np.vstack([end_frame[j], start_frame[j]]))
        s = Slerp([0, 1], rots)
        out.append(s(np.linspace(0, 1, length)).as_rotvec())
    return np.transpose(np.array(out), (1, 0, 2))


def build_stream(pool, classes, rng, signs_per_stream, min_trans=5, max_trans=30):
    """Sample a sign sequence and stitch clips into one continuous stream.

    pool: {cls: [(landmarks, presence), ...]} non-support recordings.
    Returns (stream_landmarks (T,42,3), stream_presence (T,2),
             ref_words [cls,...], frame_labels (T,) int index into ref or -1).
    """
    seq = []
    prev = None
    for _ in range(signs_per_stream):
        c = rng.choice(classes)
        while c == prev:  # avoid adjacent repeats (undetectable by decoder)
            c = rng.choice(classes)
        seq.append(c)
        prev = c

    parts, pres_parts, labels = [], [], []
    for i, c in enumerate(seq):
        lm, pres = pool[c][rng.randrange(len(pool[c]))]
        if i > 0:
            tlen = rng.randint(min_trans, max_trans)
            trans = slerp_transition(parts[-1][-1], lm[0], tlen)
            parts.append(trans)
            pres_parts.append(np.ones((tlen, 2), dtype=np.uint8))
            labels.extend([-1] * tlen)
        parts.append(lm)
        pres_parts.append(pres)
        labels.extend([i] * len(lm))

    return (np.vstack(parts), np.vstack(pres_parts), seq,
            np.array(labels, dtype=np.int64))


# ---------------------------------------------------------------------------
# Database construction (mirrors eval_harness.build_database, on embeddings)
# ---------------------------------------------------------------------------

def build_db(support_embs, support_pres, strategy):
    """support_embs/pres: {cls: [emb (30,256) / pres (30,2), ...]}.
    Returns the 5-tuple list classify() expects."""
    db = []
    for cls in sorted(support_embs):
        embs, press = support_embs[cls], support_pres[cls]
        if strategy == "per_recording":
            for e, p in zip(embs, press):
                db.append((cls, e, None, None, p))
        elif strategy == "medoid":
            i = medoid(embs) if len(embs) > 1 else 0
            db.append((cls, embs[i], None, None, press[i]))
        elif strategy == "dba":
            bary = dba(embs)
            # classify()'s penalty uses only the time-mean visibility fraction,
            # so average the per-recording fractions (masks differ in length).
            pres_avg = np.mean(
                np.stack([p.astype(np.float32).mean(axis=0) for p in press]),
                axis=0, keepdims=True)
            db.append((cls, bary, None, None, pres_avg))
        else:
            raise ValueError(strategy)
    return db


# ---------------------------------------------------------------------------
# Auto-calibration of the no-match threshold (leave-one-out conformal)
# ---------------------------------------------------------------------------

def auto_calibrate_threshold(support_embs, support_pres, strategy,
                             use_presence, q=0.95):
    """Set the no-match threshold from the registered recordings themselves.

    For each recording, hold it out, build its class's prototype(s) from the
    remaining recordings UNDER THE ACTIVE STRATEGY, and score the held-out
    recording with the same normalized partial-DTW cost classify() thresholds
    against (+ presence penalty when enabled). These are genuine
    (within-class) nonconformity scores; their q-quantile is a
    distribution-free threshold that accepts a true registered sign with
    probability ~q (split-conformal guarantee under exchangeability).
    Strategy dependence is absorbed automatically: DBA scores calibrate DBA's
    threshold, per-recording scores calibrate per-recording's.

    Returns (threshold, genuine_scores).
    """
    PRESENCE_LAMBDA = 0.3  # matches classify()
    scores = []
    for cls in sorted(support_embs):
        embs, press = support_embs[cls], support_pres[cls]
        if len(embs) < 2:
            continue
        for i in range(len(embs)):
            rest_e = embs[:i] + embs[i + 1:]
            rest_p = press[:i] + press[i + 1:]
            db_i = build_db({cls: rest_e}, {cls: rest_p}, strategy)
            q_frac = (np.asarray(press[i], dtype=np.float32).mean(axis=0)
                      if use_presence else None)
            best = np.inf
            for entry in db_i:
                proto, proto_pres = entry[1], entry[4]
                cost = float((partial_DTW(embs[i], proto) / len(proto)).min())
                if q_frac is not None and proto_pres is not None and len(proto_pres) > 0:
                    p_frac = np.asarray(proto_pres, dtype=np.float32).mean(axis=0)
                    cost += PRESENCE_LAMBDA * float(np.sum(np.abs(q_frac - p_frac)))
                best = min(best, cost)
            scores.append(best)
    return float(np.quantile(scores, q)), scores


# ---------------------------------------------------------------------------
# Deployed decode loop (mirrors server/model/main.py:recieve translate branch)
# ---------------------------------------------------------------------------

def decode_stream(model, device, stream, presence, db, use_presence,
                  threshold, debounce=1):
    """Replay the deployed loop. Returns (decoded_words, chunk_preds) where
    chunk_preds is one predicted class-or-None per 10-frame chunk.

    debounce=1 reproduces the deployed decoder exactly (classify()'s internal
    change-detection + the server's cross-buffer dedup). debounce>1 instead
    decodes from the chunk-winner sequence, emitting a word only after it
    wins `debounce` consecutive chunks.
    """
    decoded = []
    chunk_preds = []
    last_word = None
    n_buffers = len(stream) // WINDOW
    for b in range(n_buffers):
        window = stream[b * WINDOW:(b + 1) * WINDOW]
        pres = presence[b * WINDOW:(b + 1) * WINDOW]
        with torch.no_grad():
            x = torch.tensor(window, dtype=torch.float32, device=device).unsqueeze(0)
            emb = model(x).squeeze(0).cpu().numpy()
        qp = pres.astype(np.uint8) if use_presence else None
        sequence, costs = classify(emb, threshold, db, chunk=CHUNK, query_presence=qp)

        # per-chunk winner (for boundary metrics + debounced decoding):
        # argmin over rows; the last row is the threshold/no-match pseudo-class.
        names = [entry[0] for entry in db] + [None]
        for col in np.argmin(costs, axis=0):
            chunk_preds.append(names[col])

        if debounce == 1:
            # subtitle dedup, mirroring server recieve()
            if len(sequence) == 0:
                last_word = None
                continue
            if last_word != sequence[0]:
                decoded.append(sequence[0])
            decoded.extend(sequence[1:])
            last_word = sequence[-1]

    if debounce > 1:
        decoded = []
        prev_emitted = None
        run_label, run_len = None, 0
        for lab in chunk_preds:
            if lab == run_label:
                run_len += 1
            else:
                run_label, run_len = lab, 1
            if run_label is not None and run_len == debounce:
                if run_label != prev_emitted:
                    decoded.append(run_label)
                    prev_emitted = run_label
            if run_label is None and run_len >= debounce:
                prev_emitted = None
    return decoded, chunk_preds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def wer_counts(ref, hyp):
    """Levenshtein alignment. Returns (S, D, I) counts."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int64)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1])
            d[i, j] = min(sub, d[i - 1, j] + 1, d[i, j - 1] + 1)
    # backtrace for S/D/I breakdown
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += int(ref[i - 1] != hyp[j - 1]); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I


def boundaries_from_chunks(chunk_preds):
    """Chunk indices where the predicted label changes (frame positions)."""
    bounds = []
    for k in range(1, len(chunk_preds)):
        if chunk_preds[k] != chunk_preds[k - 1]:
            bounds.append(k * CHUNK)
    return bounds


def true_boundaries(frame_labels):
    """Midpoints of transition (-1) runs + hard label changes."""
    bounds = []
    T = len(frame_labels)
    k = 0
    while k < T - 1:
        if frame_labels[k] != frame_labels[k + 1]:
            if frame_labels[k + 1] == -1:  # entering a transition: midpoint
                j = k + 1
                while j < T and frame_labels[j] == -1:
                    j += 1
                bounds.append((k + 1 + j) // 2)
                k = j
                continue
            bounds.append(k + 1)
        k += 1
    return bounds


def boundary_prf(true_b, pred_b, tol):
    """Greedy one-to-one matching within tol frames -> precision/recall/F1."""
    matched_pred = set()
    tp = 0
    for tb in true_b:
        best, best_d = None, tol + 1
        for pi, pb in enumerate(pred_b):
            if pi in matched_pred:
                continue
            dd = abs(pb - tb)
            if dd <= tol and dd < best_d:
                best, best_d = pi, dd
        if best is not None:
            matched_pred.add(best)
            tp += 1
    prec = tp / len(pred_b) if pred_b else 0.0
    rec = tp / len(true_b) if true_b else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", type=Path,
                    default=Path(__file__).resolve().parent / "wlasl_landmarks_v2")
    ap.add_argument("--checkpoint", type=Path,
                    default=Path(__file__).resolve().parent.parent / "server" / "model" / "weights.h5")
    ap.add_argument("--strategy", choices=["per_recording", "medoid", "dba"],
                    default="per_recording")
    ap.add_argument("--k_shot", type=int, default=5)
    ap.add_argument("--n_streams", type=int, default=100)
    ap.add_argument("--signs_per_stream", type=int, default=8)
    ap.add_argument("--tolerance", type=int, default=15,
                    help="boundary match tolerance in frames")
    ap.add_argument("--threshold", type=float, default=0.9,
                    help="no-match threshold (deployed default: 0.9)")
    ap.add_argument("--auto_threshold", type=float, default=None, metavar="Q",
                    help="override --threshold with the Q-quantile of "
                         "leave-one-out genuine scores from the support set "
                         "(conformal calibration; e.g. 0.95)")
    ap.add_argument("--debounce", type=int, default=1,
                    help="chunks a class must win consecutively to emit "
                         "(1 = deployed decoder)")
    ap.add_argument("--trim_blank", action="store_true",
                    help="trim support clips to their active (hands-visible) "
                         "span before embedding, mirroring in-app recording")
    ap.add_argument("--no_presence", action="store_true",
                    help="disable the presence-mismatch penalty (deployed default: on)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "results_continuous.json")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)
    use_presence = not args.no_presence

    print(f"Loading landmarks from {args.landmarks_dir}...")
    landmark_dict, presence_dict = load_wlasl_landmarks(args.landmarks_dir)
    _, novel = make_class_disjoint_split(list(landmark_dict.keys()))
    print(f"{len(novel)} novel classes registered as the vocabulary.")

    model = load_model(args.checkpoint, device=device)

    # Split each class: k_shot support, rest stream pool.
    support_embs, support_pres = {}, {}
    pool = {}
    for cls in novel:
        recs = list(zip(landmark_dict[cls], presence_dict[cls]))
        rng.shuffle(recs)
        sup, rest = recs[:args.k_shot], recs[args.k_shot:]
        embs, press = [], []
        for lm, pres in sup:
            # support is embedded like an in-app recording: the full clip,
            # optionally trimmed to its active span (--trim_blank)
            if args.trim_blank:
                lm, pres = trim_active_span(lm, pres)
            if len(lm) < 2:
                continue
            with torch.no_grad():
                x = torch.tensor(lm, dtype=torch.float32, device=device).unsqueeze(0)
                embs.append(model(x).squeeze(0).cpu().numpy())
            press.append(pres)
        support_embs[cls], support_pres[cls] = embs, press
        pool[cls] = rest

    print(f"Building {args.strategy} database...")
    t0 = time.time()
    db = build_db(support_embs, support_pres, args.strategy)
    print(f"{len(db)} prototypes ({time.time() - t0:.1f}s).")

    if args.auto_threshold is not None:
        thr, genuine = auto_calibrate_threshold(
            support_embs, support_pres, args.strategy, use_presence,
            q=args.auto_threshold)
        print(f"Auto-calibrated threshold: {thr:.4f} "
              f"(q={args.auto_threshold} of {len(genuine)} LOO genuine scores; "
              f"median {np.median(genuine):.4f})")
        args.threshold = thr

    agg = {"S": 0, "D": 0, "I": 0, "ref_len": 0, "exact": 0,
           "b_prec": [], "b_rec": [], "b_f1": [], "ms_per_buffer": []}

    for si in range(args.n_streams):
        stream, presence, ref, frame_labels = build_stream(
            pool, novel, rng, args.signs_per_stream)
        t0 = time.time()
        decoded, chunk_preds = decode_stream(model, device, stream, presence,
                                             db, use_presence,
                                             args.threshold, args.debounce)
        n_buf = max(1, len(stream) // WINDOW)
        agg["ms_per_buffer"].append((time.time() - t0) / n_buf * 1000)

        S, D, I = wer_counts(ref, decoded)
        agg["S"] += S; agg["D"] += D; agg["I"] += I; agg["ref_len"] += len(ref)
        agg["exact"] += int(decoded == ref)

        tb = true_boundaries(frame_labels)
        pb = boundaries_from_chunks(chunk_preds)
        p, r, f = boundary_prf(tb, pb, args.tolerance)
        agg["b_prec"].append(p); agg["b_rec"].append(r); agg["b_f1"].append(f)

        if (si + 1) % 20 == 0:
            wer = (agg["S"] + agg["D"] + agg["I"]) / agg["ref_len"]
            print(f"  [{si + 1}/{args.n_streams}] running WER {wer * 100:.1f}%")

    wer = (agg["S"] + agg["D"] + agg["I"]) / agg["ref_len"]
    result = {
        "strategy": args.strategy,
        "k_shot": args.k_shot,
        "use_presence": use_presence,
        "n_streams": args.n_streams,
        "signs_per_stream": args.signs_per_stream,
        "threshold": args.threshold,
        "auto_threshold_q": args.auto_threshold,
        "debounce": args.debounce,
        "trim_blank": args.trim_blank,
        "tolerance_frames": args.tolerance,
        "WER": wer,
        "sub_rate": agg["S"] / agg["ref_len"],
        "del_rate": agg["D"] / agg["ref_len"],
        "ins_rate": agg["I"] / agg["ref_len"],
        "sequence_accuracy": agg["exact"] / args.n_streams,
        "boundary_precision": float(np.mean(agg["b_prec"])),
        "boundary_recall": float(np.mean(agg["b_rec"])),
        "boundary_f1": float(np.mean(agg["b_f1"])),
        "ms_per_30f_buffer": float(np.mean(agg["ms_per_buffer"])),
        "n_prototypes": len(db),
        "seed": args.seed,
    }

    print(f"\n=== continuous stream: {args.strategy}, k={args.k_shot}, "
          f"presence={'on' if use_presence else 'off'}, thr={args.threshold}, "
          f"debounce={args.debounce}, trim={args.trim_blank} ===")
    print(f"WER {wer * 100:.2f}%  (S {result['sub_rate'] * 100:.1f} / "
          f"D {result['del_rate'] * 100:.1f} / I {result['ins_rate'] * 100:.1f})")
    print(f"sequence acc {result['sequence_accuracy'] * 100:.1f}%   "
          f"boundary P/R/F1 {result['boundary_precision']:.2f}/"
          f"{result['boundary_recall']:.2f}/{result['boundary_f1']:.2f}")
    print(f"{result['ms_per_30f_buffer']:.1f} ms per 30-frame buffer "
          f"({len(db)} prototypes)")

    existing = []
    if args.output.exists():
        try:
            existing = json.load(open(args.output))
        except json.JSONDecodeError:
            pass
    existing.append(result)
    json.dump(existing, open(args.output, "w"), indent=2)
    print(f"Appended to {args.output}")


if __name__ == "__main__":
    main()
