"""
WLASL preprocessing: raw videos -> per-instance MediaPipe landmark .npy files.

Reads WLASL_v0.3.json, locates each instance's source video in raw_videos/,
trims to its [frame_start, frame_end] range, runs MediaPipe Hands per frame
using the SAME configuration as server/model/main.py:convert_mediapipe(), and
writes landmarks of shape (T, 42, 3) to:

    out_root/{gloss}/{video_id}.npy

This replaces WLASL/start_kit/preprocess.py + a separate MediaPipe pass.
No ffmpeg, no bash, Windows-friendly. Resumable (skips already-processed
instances).

Usage:
    python preprocess_wlasl.py
    python preprocess_wlasl.py --top_k_classes 100      # WLASL-100 only
    python preprocess_wlasl.py --top_k_classes 300      # WLASL-300 only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


MISSING_RESET = 5  # mirror server/model/main.py:convert_mediapipe


def extract_landmarks(video_path, frame_start, frame_end, hands):
    """Return ((T, 42, 3) landmark array, (T, 2) uint8 presence mask) for
    frames [frame_start, frame_end] inclusive. frame_end < 0 means 'use all
    frames from frame_start to end'. Returns (None, None) if the video can't
    be opened or has zero usable frames.

    Mirrors the post-fix server/model/main.py:convert_mediapipe:
      - prev-hand cache decays to zeros after MISSING_RESET missing frames
      - per-frame (left_seen, right_seen) flags are emitted as a sidecar mask
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    if frame_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    prev_left = np.zeros((21, 3))
    prev_right = np.zeros((21, 3))
    left_missing = 0
    right_missing = 0
    out = []
    presence_out = []
    idx = frame_start

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_end >= 0 and idx > frame_end:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        frame_lm = np.zeros((42, 3))
        left_seen = False
        right_seen = False

        if results.multi_hand_landmarks:
            for hl, handedness in zip(results.multi_hand_landmarks,
                                      results.multi_handedness):
                label = handedness.classification[0].label
                if label == "Left":
                    left_seen = True
                    for i, lm in enumerate(hl.landmark):
                        frame_lm[i, :] = [lm.x, lm.y, lm.z]
                    prev_left = frame_lm[:21, :].copy()
                elif label == "Right":
                    right_seen = True
                    for i, lm in enumerate(hl.landmark):
                        frame_lm[i + 21, :] = [lm.x, lm.y, lm.z]
                    prev_right = frame_lm[21:, :].copy()

        if left_seen:
            left_missing = 0
        else:
            left_missing += 1
            if left_missing > MISSING_RESET:
                prev_left = np.zeros((21, 3))
        if right_seen:
            right_missing = 0
        else:
            right_missing += 1
            if right_missing > MISSING_RESET:
                prev_right = np.zeros((21, 3))

        if not left_seen:
            frame_lm[:21, :] = prev_left
        if not right_seen:
            frame_lm[21:, :] = prev_right

        out.append(frame_lm)
        presence_out.append([1 if left_seen else 0, 1 if right_seen else 0])
        idx += 1

    cap.release()

    if not out:
        return None, None
    return np.stack(out), np.array(presence_out, dtype=np.uint8)


def resolve_video_path(inst, raw_videos_dir):
    """Map a WLASL instance to its source video file in raw_videos/.
    YouTube instances are keyed by the YouTube ID (last 11 chars of URL).
    Non-YouTube instances are keyed by video_id.
    """
    url = inst["url"]
    if "youtube" in url or "youtu.be" in url:
        key = url[-11:]
    else:
        key = inst["video_id"]
    for ext in (".mp4", ".mkv", ".webm"):
        candidate = raw_videos_dir / f"{key}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json",
                    default=r"data/wlasl/WLASL_v0.3.json")
    ap.add_argument("--raw_videos",
                    default=r"data/wlasl/raw_videos")
    ap.add_argument("--out",
                    default=r"experiments\wlasl_landmarks")
    ap.add_argument("--top_k_classes", type=int, default=0,
                    help="If >0, only process the top-k most-frequent classes "
                         "(100 -> WLASL-100, 300 -> WLASL-300, etc.)")
    args = ap.parse_args()

    with open(args.json) as f:
        content = json.load(f)

    if args.top_k_classes > 0:
        content.sort(key=lambda x: -len(x["instances"]))
        content = content[:args.top_k_classes]

    raw_dir = Path(args.raw_videos)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )

    total = sum(len(e["instances"]) for e in content)
    done = saved = skipped = missing_src = empty = 0
    t0 = time.time()

    for entry in content:
        gloss = entry["gloss"]
        gloss_dir = out_root / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)

        for inst in entry["instances"]:
            done += 1
            video_id = inst["video_id"]
            out_path = gloss_dir / f"{video_id}.npy"
            presence_path = gloss_dir / f"{video_id}.presence.npy"

            if out_path.exists() and presence_path.exists():
                skipped += 1
            else:
                src = resolve_video_path(inst, raw_dir)
                if src is None:
                    missing_src += 1
                else:
                    fs = max(0, inst["frame_start"] - 1)
                    fe = inst["frame_end"] - 1 if inst["frame_end"] > 0 else -1
                    arr, presence = extract_landmarks(src, fs, fe, hands)
                    if arr is None or len(arr) == 0:
                        empty += 1
                    else:
                        np.save(out_path, arr)
                        np.save(presence_path, presence)
                        saved += 1

            if done % 100 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"[{done}/{total}] saved={saved} skipped={skipped} "
                      f"missing_src={missing_src} empty={empty} "
                      f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"\nDone. saved={saved} skipped={skipped} "
          f"missing_src={missing_src} empty={empty} of {total} total")


if __name__ == "__main__":
    main()
