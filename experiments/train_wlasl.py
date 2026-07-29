"""
Fine-tune TCNSignEmbedding on WLASL landmarks.

Adapts the friend's NB2 training pipeline (`train/finalfinal_*.ipynb`) to:
  - load weights.h5 as the starting checkpoint (NEVER overwritten),
  - train on `experiments/wlasl_landmarks_v2/` (clean preprocessing, presence
    sidecars) with the eval-novel classes EXCLUDED (experiments/
    novel_classes.json) — the June runs on wlasl100_landmarks had 29/40
    novel classes leaked into training AND v1 ghost-pose artifacts,
  - hold out --n_val_classes classes (class-disjoint) for validation, so
    val_acc measures transfer to unseen classes instead of memorization,
  - save fine-tuned checkpoints to `experiments/checkpoints/`.

Loss = `dtw_partition_loss` (NB2's recipe). Two gradient-flow variants:
  --loss hard  (default): alignment chosen by numba argmin (non-differentiable);
               PyTorch gradients flow only through per-frame distances along
               the chosen path.
  --loss soft : Cuturi-Blondel soft-DTW (experiments/soft_dtw.py), batched
               across all (episode, prototype) pairs per batch; softmin_gamma
               makes the whole alignment landscape differentiable.

Augmentation changes vs NB2 (which always-mirrored without swapping hand
slots — anatomically impossible data + a systematic train/eval orientation
mismatch): mirroring is now a 50% augmentation that flips x AND swaps the
left/right hand blocks, and augmentations preserve exact zeros for absent
hands (the v2 "no hand" encoding).

Usage:
    python experiments/train_wlasl.py --loss soft --gamma 1.0
    python experiments/train_wlasl.py --loss hard --epochs 15 --lr 1e-4
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from model.model import get_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from soft_dtw import soft_dtw_pairs


# ---------------------------------------------------------------------------
# Presence-safe augmentation helpers
# ---------------------------------------------------------------------------
def _present_mask(g):
    """(T, 42, 3) -> (T, 2) bool: hand block not all-zero. v2 preprocessing
    encodes an absent hand as exact zeros; augmentations must keep it so."""
    return np.stack([g[:, :21].any(axis=(1, 2)),
                     g[:, 21:].any(axis=(1, 2))], axis=1)


def mirror_hands(g):
    """Horizontal mirror: flip x -> 1-x AND swap the left/right hand slots
    (a mirrored left hand IS a right hand — flipping without swapping, as NB2
    did, produces configurations MediaPipe can never emit). Absent (all-zero)
    hand blocks pass through unchanged."""
    g = g.copy()
    pres = _present_mask(g)
    left, right = g[:, :21], g[:, 21:]
    left[pres[:, 0], :, 0] = 1.0 - left[pres[:, 0], :, 0]
    right[pres[:, 1], :, 0] = 1.0 - right[pres[:, 1], :, 0]
    return np.concatenate([right, left], axis=1)


# ---------------------------------------------------------------------------
# Dataset (NB2 recipe + class filtering / class-disjoint mode)
# ---------------------------------------------------------------------------
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, num_classes_per_batch, num_distractions_per_batch,
                 max_sequence_length, batch_length=100, mirror=True,
                 speed_variation=0.5, random_slice=True, sample_all=False, train=True,
                 class_filter=None, use_all_files=False):
        self.root_dir = root_dir
        self.num_classes_per_batch = num_classes_per_batch
        self.num_distractions_per_batch = num_distractions_per_batch
        self.max_sequence_length = max_sequence_length
        self.speed_variation = speed_variation
        self.random_slice = random_slice
        self.batch_length = batch_length
        self.sample_all = sample_all
        self.train = train
        self.mirror = mirror

        names = sorted(os.listdir(root_dir))
        if class_filter is not None:
            keep = set(class_filter)
            names = [c for c in names if c in keep]
        self.class_names = names
        self.class_to_idx = {}
        self.data_files = {}
        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            files = sorted(f for f in os.listdir(class_dir)
                           if f.endswith(".npy") and not f.endswith(".presence.npy"))
            self.data_files[class_name] = files
            self.class_to_idx[class_name] = i

        self.train_files = {}
        self.val_files = {}
        for class_name, files in self.data_files.items():
            if use_all_files:
                # class-disjoint train/val: this dataset owns its classes
                # outright, so every file is usable
                self.train_files[class_name] = files
                self.val_files[class_name] = files
            else:
                split_idx = int(0.8 * len(files))
                self.train_files[class_name] = files[:split_idx]
                self.val_files[class_name] = files[split_idx:]

    def __len__(self):
        return self.batch_length

    def vary_speed(self, gesture_data):
        speed_factor = np.random.uniform(1 - self.speed_variation, 1 + self.speed_variation)
        original_frames = gesture_data.shape[0]
        new_frames = max(2, int(original_frames / speed_factor))
        old_times = np.arange(original_frames)
        new_times = np.linspace(0, original_frames - 1, new_frames)
        return interp1d(old_times, gesture_data, axis=0, kind="linear")(new_times)

    def vary_position(self, gesture_data):
        shifted = gesture_data + np.random.uniform(-0.05, 0.05, (3,))
        # don't shift absent (all-zero) hand blocks off their zero encoding
        mask = np.repeat(_present_mask(gesture_data), 21, axis=1)[..., None]
        return np.where(mask, shifted, gesture_data)

    def concatenate_hand_gestures(self, gesture_data_list, class_indices):
        concatenated_data = []
        concatenated_labels = []
        transition_lengths = np.random.randint(5, 30, size=len(gesture_data_list) - 1)
        for i, (current_data, class_idx) in enumerate(zip(gesture_data_list, class_indices)):
            concatenated_data.append(current_data)
            concatenated_labels.extend([class_idx] * len(current_data))
            if i < len(gesture_data_list) - 1:
                next_data = gesture_data_list[i + 1]
                end_frame = current_data[-1]
                start_frame = next_data[0]
                transition = []
                for j in range(42):
                    rots = Rotation.from_rotvec(np.vstack([end_frame[j], start_frame[j]]))
                    slerp = Slerp([0, 1], rots)
                    transition.append(slerp(np.linspace(0, 1, transition_lengths[i])).as_rotvec())
                transition = np.transpose(np.array(transition), (1, 0, 2))
                concatenated_data.append(transition)
                concatenated_labels.extend([-1] * transition_lengths[i])
        return np.vstack(concatenated_data), np.array(concatenated_labels)

    def _load(self, class_name, file_set, mirrored=False):
        data_file = random.choice(file_set[class_name])
        gesture_data = np.load(os.path.join(self.root_dir, class_name, data_file))
        gesture_data = self.vary_speed(self.vary_position(gesture_data))
        if mirrored:
            gesture_data = mirror_hands(gesture_data)
        return gesture_data

    def __getitem__(self, index):
        file_set = self.train_files if self.train else self.val_files
        # Filter classes that have any files in this split
        usable = [c for c in self.class_names if file_set[c]]

        # one mirror decision per episode: queries and their prototypes stay
        # in the same orientation (as they are at deployment), while the
        # model still sees both orientations across episodes
        mirrored = self.mirror and random.random() < 0.5

        input_classes = random.sample(usable, min(self.num_classes_per_batch, len(usable)))
        gesture_data_list, class_indices = [], []
        for c in input_classes:
            gesture_data_list.append(self._load(c, file_set, mirrored))
            class_indices.append(self.class_to_idx[c])

        input_sequence, input_label = self.concatenate_hand_gestures(gesture_data_list, class_indices)

        if self.random_slice:
            start = random.randint(0, min(15, len(input_sequence) // 2))
            end = len(input_sequence) - random.randint(0, min(15, len(input_sequence) // 2))
            input_sequence = input_sequence[start:end]
            curr_idx = start
            prev = input_label[start]
            while curr_idx < len(input_label) and input_label[curr_idx] == prev:
                input_label[curr_idx] = -2
                curr_idx += 1
            curr_idx = end
            prev = input_label[end - 1]
            while curr_idx > 0 and input_label[curr_idx - 1] == prev:
                input_label[curr_idx - 1] = -2
                curr_idx -= 1
            input_label = input_label[start:end]

        input_sequence = torch.from_numpy(input_sequence).float()
        input_label = torch.from_numpy(input_label).long()

        additional = random.choices(usable, k=self.num_distractions_per_batch)
        sample_classes = list(set(input_classes + additional))
        if self.sample_all:
            sample_classes = usable

        sample_seqs, sample_labels = [], []
        for c in sample_classes:
            sample_seqs.append(torch.from_numpy(self._load(c, file_set, mirrored)).float())
            sample_labels.append(self.class_to_idx[c])

        max_len = max(s.shape[0] for s in sample_seqs)
        padded, masks = [], []
        for s in sample_seqs:
            pad = max_len - s.shape[0]
            padded.append(F.pad(s.permute(1, 2, 0), (0, pad), value=0).permute(2, 0, 1))
            mask = torch.ones(max_len, dtype=torch.bool)
            mask[s.shape[0]:] = 0
            masks.append(mask)

        return (input_sequence, input_label,
                torch.stack(padded), torch.tensor(sample_labels), torch.stack(masks))


# ---------------------------------------------------------------------------
# Loss (NB2's dtw_partition_loss)
# ---------------------------------------------------------------------------
@njit
def dtw_path(x, y):
    N, _ = x.shape
    M, _ = y.shape
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = np.linalg.norm(x[i - 1] - y[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    path = []
    i, j = N, M
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            choices = np.array([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
            k = np.argmin(choices)
            if k == 0:
                i -= 1
            elif k == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
    path.reverse()
    return path


def partition_sequence(sequence, labels):
    parts = []
    cur_label = labels[0]
    cur_part = []
    for i, label in enumerate(labels):
        if label == cur_label:
            cur_part.append(sequence[i])
        else:
            parts.append((torch.stack(cur_part), cur_label))
            cur_label = label
            cur_part = [sequence[i]]
    parts.append((torch.stack(cur_part), cur_label))
    return parts


def dtw_partition_loss(z_in, z_tgt, labels, threshold, alpha=0.05,
                       loss_mode="hard", gamma=1.0):
    """NB2's partition loss. loss_mode selects the per-target distance:

    - "hard": alignment path from numba argmin (non-differentiable choice);
              gradients flow through per-frame distances along that path.
    - "soft": soft-DTW value (fully differentiable through the alignment
              landscape). Same Euclidean local cost, same scale. Computed
              for ALL (episode, prototype) pairs in one batched DP
              (soft_dtw_pairs) instead of a Python loop per pair.
    """
    episodes = [(seq, lab) for seq, lab in partition_sequence(z_in, labels)
                if lab.item() != -2]
    if not episodes:
        return z_in.sum() * 0.0, 0, 0

    if loss_mode == "soft":
        xs = [seq for seq, _ in episodes]
        lab_t = torch.stack([lab for _, lab in episodes]).long()
        dists = soft_dtw_pairs(xs, list(z_tgt), gamma=gamma)   # (E, B)
        thr_col = threshold.reshape(1, 1).expand(dists.shape[0], 1)
        logits = -torch.cat([dists, thr_col], dim=1)           # (E, B+1)
        loss = F.cross_entropy(logits, lab_t, reduction="sum")
        real = lab_t < dists.shape[1]                          # pull term only
        if real.any():                                         # for real targets
            loss = loss + alpha * dists[real, lab_t[real]].sum()
        correct = int((logits.argmax(dim=1) == lab_t).sum())
        return loss, correct, len(episodes)

    loss = 0
    correct = total = 0
    for episode, label in episodes:
        distances = []
        correct_dist = None
        for tgt_label, tgt_emb in enumerate(z_tgt):
            path = dtw_path(episode.detach().cpu().numpy(),
                            tgt_emb.detach().cpu().numpy())
            d = torch.tensor(0.0, device=z_in.device)
            for (i, j) in path:
                d = d + torch.norm(episode[i] - tgt_emb[j])
            distances.append(d)
            if tgt_label == label.item():
                correct_dist = d
        distances.append(threshold)
        logits = -torch.stack(distances)
        loss = loss + F.cross_entropy(logits, label)
        if correct_dist is not None:
            loss = loss + alpha * correct_dist
        if torch.argmax(logits) == label:
            correct += 1
        total += 1
    return loss, correct, total


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None, loss_mode="hard", gamma=1.0):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0
    correct = total = 0
    for batch in loader:
        input_seq, input_lbl, tgt_seqs, tgt_lbls, tgt_masks = [b.to(device)[0] for b in batch]
        label_to_idx = {l.item(): i for i, l in enumerate(tgt_lbls)}
        label_to_idx[-1] = len(tgt_lbls)
        label_to_idx[-2] = -2
        labels = torch.empty_like(input_lbl, device=device)
        for i, v in enumerate(input_lbl):
            labels[i] = label_to_idx.get(v.item(), len(tgt_lbls))

        with torch.set_grad_enabled(is_train):
            z_in = model(input_seq.unsqueeze(0)).squeeze(0)
            z_tgt = [model(t[m].unsqueeze(0)).squeeze(0)
                     for t, m in zip(tgt_seqs, tgt_masks)]
            loss, c, t = dtw_partition_loss(z_in, z_tgt, labels, model.threshold,
                                            loss_mode=loss_mode, gamma=gamma)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        correct += c
        total += t

    return total_loss / max(1, len(loader)), correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", default="experiments/wlasl_landmarks_v2")
    ap.add_argument("--init_weights", default="server/model/weights.h5")
    ap.add_argument("--out_dir", default="experiments/checkpoints")
    ap.add_argument("--exclude_classes", default="experiments/novel_classes.json",
                    help="JSON with a 'novel40' list of eval-only classes to "
                         "exclude from training/validation entirely. '' disables.")
    ap.add_argument("--n_val_classes", type=int, default=20,
                    help="classes held out (class-disjoint) for validation")
    ap.add_argument("--val_class_seed", type=int, default=777)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after this many epochs without val_acc improvement")
    ap.add_argument("--n_classes_per_batch", type=int, default=10)
    ap.add_argument("--n_distractors", type=int, default=10)
    ap.add_argument("--train_episodes", type=int, default=50)
    ap.add_argument("--val_episodes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--loss", choices=["hard", "soft"], default="hard",
                    help="hard = NB2 path-sum gradients; soft = batched soft-DTW")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="soft-DTW temperature (only used with --loss soft)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; aborting (training on CPU is not viable).")
        sys.exit(1)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoints to {out_dir} (init from {args.init_weights}, NOT overwritten)")

    model = get_model(path=args.init_weights, device=device)

    # weight decay only on actual weight matrices — biases, BatchNorm params
    # and the learned no-match threshold must not be pulled toward zero
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "threshold" or name.endswith(".bias") or ".bn" in name or p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    # class-disjoint train/val: exclude eval-novel classes, then hold out
    # n_val_classes for validation so val_acc measures unseen-class transfer
    all_classes = sorted(
        c for c in os.listdir(args.landmarks)
        if os.path.isdir(os.path.join(args.landmarks, c)))
    excluded = []
    if args.exclude_classes:
        excluded = json.load(open(args.exclude_classes))["novel40"]
        all_classes = [c for c in all_classes if c not in set(excluded)]
    rng = random.Random(args.val_class_seed)
    val_classes = sorted(rng.sample(all_classes, min(args.n_val_classes, len(all_classes))))
    train_classes = [c for c in all_classes if c not in set(val_classes)]

    train_ds = HandGestureDataset(args.landmarks, args.n_classes_per_batch,
                                  args.n_distractors, 150,
                                  batch_length=args.train_episodes, train=True,
                                  class_filter=train_classes, use_all_files=True)
    val_ds = HandGestureDataset(args.landmarks, args.n_classes_per_batch,
                                args.n_distractors, 150,
                                batch_length=args.val_episodes, train=False,
                                class_filter=val_classes, use_all_files=True)
    print(f"Classes: {len(train_ds.class_names)} train / {len(val_ds.class_names)} val "
          f"(class-disjoint), {len(excluded)} eval-novel excluded  "
          f"train episodes/epoch: {args.train_episodes}  "
          f"val episodes/epoch: {args.val_episodes}")

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=0, batch_size=1)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=0, batch_size=1)

    history = []
    best_val_acc = 0.0
    best_epoch = -1
    t_start = time.time()

    ckpt_stem = "wlasl_ftv2" if args.loss == "hard" else f"wlasl_ftv2_soft_g{args.gamma:g}"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer=optimizer,
                                          loss_mode=args.loss, gamma=args.gamma)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None,
                                      loss_mode=args.loss, gamma=args.gamma)
        ckpt_path = out_dir / f"{ckpt_stem}_ep{epoch:02d}.h5"
        torch.save(model.state_dict(), ckpt_path)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "ckpt": str(ckpt_path)})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        elapsed = time.time() - t0
        print(f"[ep {epoch:02d}/{args.epochs}] "
              f"train_loss={train_loss:7.3f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:7.3f} val_acc={val_acc:.3f}  "
              f"({elapsed:.0f}s)  -> {ckpt_path.name}", flush=True)

        if epoch - best_epoch >= args.patience:
            print(f"Early stop: no val_acc improvement in {args.patience} epochs "
                  f"(best {best_val_acc:.3f} at ep {best_epoch}).", flush=True)
            break

    # Per-run history file (a shared history.json previously got overwritten
    # by later runs); timestamp keeps reruns of the same config distinct.
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"history_{ckpt_stem}_{run_tag}.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_val_acc": best_val_acc,
                   "val_classes": val_classes,
                   "n_train_classes": len(train_ds.class_names),
                   "excluded_novel": len(excluded),
                   "args": vars(args)}, f, indent=2, default=str)

    print(f"\n=== Training complete ({(time.time()-t_start)/60:.1f} min) ===")
    print(f"Best val_acc {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"Recommended checkpoint for eval: "
          f"{out_dir / f'{ckpt_stem}_ep{best_epoch:02d}.h5'}")


if __name__ == "__main__":
    main()
