# Conformal auto-calibration of the no-match threshold from the user's own
# registered recordings. Mirrors experiments/continuous_bench.py, which
# validated it: the LOO median reproduces the hand-tuned threshold, and beats
# it for DBA prototypes.

import numpy as np

from model.classify import partial_DTW

PRESENCE_LAMBDA = 0.3        # must match classify()
FALLBACK_THRESHOLD = 0.35    # manually calibrated value (WLASL bench)

# Operating point on the LOO genuine-score distribution. q=0.5 rejects about
# half of all real signs by construction, which hits movement signs hardest
# (their take-to-take timing varies more than a held handshape's). Measured on
# a 7-class self-recorded set: q=0.5 -> 3/8 recognized, q=0.8 -> 5/8,
# q=0.9 -> 6/8. Raise toward 1.0 for recall, lower for precision.
QUANTILE = 0.8


def trim_active_span(landmarks, presence):
    """Cut leading/trailing frames where no hand is visible, so blank spans
    never become part of a prototype (they align with anything)."""
    landmarks = np.asarray(landmarks)
    presence = np.asarray(presence)
    if len(presence) == 0:
        return landmarks, presence
    active = np.where(presence.sum(axis=1) > 0)[0]
    if len(active) == 0:
        return landmarks, presence
    lo, hi = active[0], active[-1] + 1
    return landmarks[lo:hi], presence[lo:hi]


def auto_threshold(database, q=QUANTILE, fallback=FALLBACK_THRESHOLD):
    """Leave-one-out genuine scores over classes with >=2 recordings; the
    q-quantile is the no-match threshold. Falls back to the hand-calibrated
    value when there is not enough data to calibrate."""
    by_class = {}
    for entry in database:
        by_class.setdefault(entry[0], []).append(entry)

    scores = []
    for entries in by_class.values():
        if len(entries) < 2:
            continue
        for i, held_out in enumerate(entries):
            emb_i = held_out[1]
            pres_i = held_out[4] if len(held_out) >= 5 else None
            q_frac = (np.asarray(pres_i, dtype=np.float32).mean(axis=0)
                      if pres_i is not None and len(pres_i) > 0 else None)
            best = np.inf
            for j, other in enumerate(entries):
                if j == i:
                    continue
                proto = other[1]
                cost = float((partial_DTW(emb_i, proto) / len(proto)).min())
                pres_j = other[4] if len(other) >= 5 else None
                if q_frac is not None and pres_j is not None and len(pres_j) > 0:
                    p_frac = np.asarray(pres_j, dtype=np.float32).mean(axis=0)
                    cost += PRESENCE_LAMBDA * float(np.sum(np.abs(q_frac - p_frac)))
                best = min(best, cost)
            if np.isfinite(best):
                scores.append(best)

    if len(scores) < 4:
        return fallback
    return float(np.quantile(scores, q))
