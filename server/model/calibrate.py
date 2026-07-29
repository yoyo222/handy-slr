# server/model/calibrate.py
#
# Conformal auto-calibration of the no-match threshold from the user's own
# registered recordings (mirrors experiments/continuous_bench.py, validated
# there: reproduces the hand-tuned threshold as the LOO median and beats it
# for DBA prototypes).

import numpy as np

from model.classify import partial_DTW

PRESENCE_LAMBDA = 0.3        # must match classify()
FALLBACK_THRESHOLD = 0.35    # manually calibrated value (WLASL bench)
QUANTILE = 0.5               # LOO-median operating point


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
