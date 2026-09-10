"""
Process PNG/JPG images in this directory and write annotated copies with
MediaPipe Hands landmarks drawn on top.

For each `<name>.png` (excluding files already ending in `_annotated`),
writes `<name>_annotated.png` alongside it.

Usage (from this directory or anywhere):
    python overlay_landmarks.py
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp


HERE = Path(__file__).resolve().parent
INPUT_EXTS = {".png", ".jpg", ".jpeg"}

# ---------- visualisation knobs (tweak freely) ----------------------------
# Colors are BGR tuples (OpenCV convention), 0–255 each.
LANDMARK_COLOR    = (0, 255, 0)       # green dots
LANDMARK_RADIUS   = 4                  # dot size in pixels (default ~5)
LANDMARK_THICK    = -1                 # -1 = filled circle

CONNECTION_COLOR  = (255, 0, 0)        # blue bones
CONNECTION_THICK  = 2                  # bone line width in pixels (default 2)

# Background: drop the original photo, draw landmarks on a clean canvas.
#   None              -> transparent PNG (best for slide compositing)
#   (B, G, R) tuple   -> solid colour (e.g. (255, 255, 255) for white)
BACKGROUND        = None
# --------------------------------------------------------------------------


def main() -> None:
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    )
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS

    landmark_spec = drawing.DrawingSpec(
        color=LANDMARK_COLOR,
        thickness=LANDMARK_THICK,
        circle_radius=LANDMARK_RADIUS,
    )
    connection_spec = drawing.DrawingSpec(
        color=CONNECTION_COLOR,
        thickness=CONNECTION_THICK,
    )

    for img_path in sorted(HERE.iterdir()):
        if img_path.suffix.lower() not in INPUT_EXTS:
            continue
        if img_path.stem.endswith("_annotated"):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"SKIP (unreadable): {img_path.name}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Build a clean canvas the same size as the input (drop the photo).
        # MediaPipe's draw_landmarks insists on 3-channel BGR, so even when
        # we want transparent output, draw on a 3-channel BLACK canvas first
        # and synthesise the alpha channel afterwards from non-black pixels.
        h, w = img.shape[:2]
        if BACKGROUND is None:
            out = np.zeros((h, w, 3), dtype=np.uint8)      # black, becomes alpha=0
        else:
            out = np.full((h, w, 3), BACKGROUND, dtype=np.uint8)

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                drawing.draw_landmarks(
                    out,
                    hand_lm,
                    connections,
                    landmark_spec,
                    connection_spec,
                )
            n = len(result.multi_hand_landmarks)
        else:
            n = 0

        # Transparent mode: derive alpha from non-black pixels and write BGRA.
        if BACKGROUND is None:
            alpha = (out.any(axis=2).astype(np.uint8)) * 255
            out = np.dstack([out, alpha])
            # Force .png extension for transparency support
            out_path = img_path.with_name(f"{img_path.stem}_annotated.png")
        else:
            out_path = img_path.with_name(f"{img_path.stem}_annotated{img_path.suffix}")
        cv2.imwrite(str(out_path), out)
        print(f"{img_path.name} -> {out_path.name} ({n} hand{'s' if n != 1 else ''})")

    hands.close()


if __name__ == "__main__":
    main()
