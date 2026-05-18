"""
Fine-tune TCNSignEmbedding on WLASL landmarks.

Adapts the friend's NB2 training pipeline (`train/finalfinal_*.ipynb`) to:
  - load weights.h5 as the starting checkpoint (NEVER overwritten),
  - train on `experiments/wlasl100_landmarks/`,
  - save fine-tuned checkpoints to `experiments/checkpoints/wlasl_ft_ep{N}.h5`.

Loss = `dtw_partition_loss` only (NB2's recipe). Hard DTW with PyTorch
gradients flowing through per-frame distances along the chosen path.

Usage:
    python experiments/train_wlasl.py
    python experiments/train_wlasl.py --epochs 30 --lr 1e-3
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


# ---------------------------------------------------------------------------
# Dataset (verbatim from NB2)
# ---------------------------------------------------------------------------
class HandGestureDataset(Dataset):
    def __init__(self, root_dir, num_classes_per_batch, num_distractions_per_batch,
                 max_sequence_length, batch_length=100, mirror=True,
                 speed_variation=0.5, random_slice=True, sample_all=False, train=True):
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

        self.class_names = sorted(os.listdir(root_dir))
        self.class_to_idx = {}
        self.data_files = {}
        for i, class_name in enumerate(self.class_names):
            class_dir = os.path.join(root_dir, class_name)
            self.data_files[class_name] = sorted(os.listdir(class_dir))
            self.class_to_idx[class_name] = i

        self.train_files = {}
        self.val_files = {}
        for class_name, files in self.data_files.items():
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
        return gesture_data + np.random.uniform(-0.05, 0.05, (3,))

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

    def _load(self, class_name, file_set):
        data_file = random.choice(file_set[class_name])
        gesture_data = np.load(os.path.join(self.root_dir, class_name, data_file))
        gesture_data = self.vary_speed(self.vary_position(gesture_data))
        if self.mirror:
            gesture_data[:, :, 0] *= -1
            gesture_data[:, :, 0] += 1
        return gesture_data

    def __getitem__(self, index):
        file_set = self.train_files if self.train else self.val_files
        # Filter classes that have any files in this split
        usable = [c for c in self.class_names if file_set[c]]

        input_classes = random.sample(usable, min(self.num_classes_per_batch, len(usable)))
        gesture_data_list, class_indices = [], []
        for c in input_classes:
            gesture_data_list.append(self._load(c, file_set))
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
            sample_seqs.append(torch.from_numpy(self._load(c, file_set)).float())
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


def dtw_partition_loss(z_in, z_tgt, labels, threshold, alpha=0.05):
    episodes = partition_sequence(z_in, labels)
    loss = 0
    correct = total = 0
    for episode, label in episodes:
        if label.item() == -2:
            continue
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
def run_epoch(model, loader, device, optimizer=None):
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
            loss, c, t = dtw_partition_loss(z_in, z_tgt, labels, model.threshold)

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
    ap.add_argument("--landmarks", default="experiments/wlasl100_landmarks")
    ap.add_argument("--init_weights", default="server/model/weights.h5")
    ap.add_argument("--out_dir", default="experiments/checkpoints")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_classes_per_batch", type=int, default=10)
    ap.add_argument("--n_distractors", type=int, default=10)
    ap.add_argument("--train_episodes", type=int, default=100)
    ap.add_argument("--val_episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = HandGestureDataset(args.landmarks, args.n_classes_per_batch,
                                  args.n_distractors, 150,
                                  batch_length=args.train_episodes, train=True)
    val_ds = HandGestureDataset(args.landmarks, args.n_classes_per_batch,
                                args.n_distractors, 150,
                                batch_length=args.val_episodes, train=False)
    print(f"Train classes: {len(train_ds.class_names)}  "
          f"train episodes/epoch: {args.train_episodes}  "
          f"val episodes/epoch: {args.val_episodes}")

    train_loader = DataLoader(train_ds, shuffle=True, num_workers=0, batch_size=1)
    val_loader = DataLoader(val_ds, shuffle=False, num_workers=0, batch_size=1)

    history = []
    best_val_acc = 0.0
    best_epoch = -1
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, optimizer=None)
        ckpt_path = out_dir / f"wlasl_ft_ep{epoch:02d}.h5"
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

    with open(out_dir / "history.json", "w") as f:
        json.dump({"history": history, "best_epoch": best_epoch,
                   "best_val_acc": best_val_acc}, f, indent=2)

    print(f"\n=== Training complete ({(time.time()-t_start)/60:.1f} min) ===")
    print(f"Best val_acc {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"Recommended checkpoint for eval: "
          f"{out_dir / f'wlasl_ft_ep{best_epoch:02d}.h5'}")


if __name__ == "__main__":
    main()
