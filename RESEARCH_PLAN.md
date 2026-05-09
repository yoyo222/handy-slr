# Research Plan — Few-Shot Sign Language Recognition (Grad Lab Topic Scoping)

> **Deliverable**: a ~20-slide lab presentation in 2 weeks whose purpose is to **decide a grad-paper research topic** built on top of this system. Not a publication target yet.
>
> **Authorship**: friend who built the original system has explicitly allowed you to treat it as your own; he is named as a **contributor** in any future writeup (acknowledgments section). No co-authorship conversation needed; no need to reproduce his training as a controlled baseline.
>
> **Comparison strategy**: evaluate this system (`weights.h5` as-is, or fine-tuned) against **published external baselines** on WLASL — I3D, Pose-TGCN, SL-GCN, MS-G3D, ProtoNet over skeleton features. Not against a reconstructed version of the friend's training pipeline.
>
> **Long-term venue**: undecided; depends on which research direction you commit to. Most likely a workshop short paper later (CVPR-W SLRTP / WACV / ASSETS) or a grad-thesis chapter.

---

## Table of contents

- **Part 1** — The system in plain English (algorithm narrative)
- **Part 2** — How retraining / fine-tuning works here
- **Part 3** — What you'd be making, which component, how you'd test
- **Part 4** — Tear-down (mathematical detail; for talk slides 7–10)
- **Part 4.5** — How the model was actually trained (from `train/` notebooks)
- **Part 5** — Related work / reference papers (for talk slides 4–6)
- **Part 6** — Limitations of the inherited system (these are your research opportunities)
- **Part 7** — Candidate research directions, ranked
- **Part 8** — External baselines you'll compare against, and the experimental harness
- **Part 9** — Dataset acquisition (WLASL)
- **Part 10** — 2-week plan to the lab talk; longer-term grad timeline
- **Part 11** — Risks & contingencies
- **Part 12** — Slide outline (20 slides)
- **Part 13** — Action items this week

---

## Part 1: The system in plain English

### The whole pipeline in one paragraph

A webcam frame goes in. **MediaPipe** turns it into 42 hand-landmark points (21 per hand × XYZ). The system buffers 30 such frames (~1.5 s of video). It feeds those 30 frames through a small neural network (CNN+TCN) which converts them into 30 vectors of 256 numbers each — call this an "embedding sequence." A database of previously-recorded sign embeddings sits in memory. For each stored sign, the system runs **partial-DTW** (a sequence-comparison algorithm) between the live embedding sequence and the stored one; this returns a number — "how well does the live clip match this sign?" The lowest number wins, unless it's above a threshold, in which case the system says "no match." That label is shown as a subtitle.

### The interesting trick: why this can recognize signs it was never trained on

Most sign-recognition models have a fixed vocabulary baked in: 100 classes in the training set → exactly 100 outputs. Adding a new sign requires retraining.

This system has *no* output classes. The neural network's only job is to take a 30-frame clip and produce an embedding sequence such that **two clips of the same sign produce similar sequences, two clips of different signs produce different sequences.** That's it. The classifier is just a lookup: "find the stored sequence most similar to the live one." Adding a new sign = adding one more entry to the lookup table. No retraining.

This is the **few-shot learning / metric learning** paradigm. The neural network learns a *general metric*, not a specific classifier.

### Where DTW comes in

Hand motions take different amounts of time. The same sign might take 20 frames one time and 28 the next. Naive vector comparison ("compute distance between sequence A and sequence B") doesn't handle this. **Dynamic Time Warping** (DTW) is a 1970s algorithm from speech recognition that finds the *best alignment* between two variable-length sequences and returns the cost of that alignment. Slow signers and fast signers of the same sign get matched.

The friend's twist — **partial-DTW** (also called subsequence DTW in the literature; Müller 2007 ch. 4) — extends this so the system can find a stored sign *anywhere within the 30-frame window*, not just at the start. Useful because the user's sign doesn't start cleanly at frame 0; it begins somewhere in the middle of the buffer.

### The two modules to remember

| Module | What it does | Trained? |
|---|---|---|
| `TCNSignEmbedding` (`server/model/model.py`) | 30 frames of landmarks → 30 vectors of 256 numbers | **Yes** — this is what `weights.h5` holds |
| `partial_DTW` + `classify` (`server/model/classify.py`) | Compare embedding sequences, return the best-matching label | **No** — pure algorithm, no parameters |

Everything in this project sits on top of those two pieces.

---

## Part 2: How retraining / fine-tuning works here

### The honest situation

Training code now lives in `train/` (two Colab notebooks — see Part 4.5 for what they do and how they differ). The deployed `weights.h5` was produced by an earlier version of this pipeline. So "retraining" here means: **adapt one of those notebooks to a new dataset (e.g., WLASL) and/or a new loss formulation, then run it.**

For your grad-research scoping, you have **three possible relationships with the trained model**, in order of effort:

1. **Use `weights.h5` as a frozen feature extractor.** No retraining. You evaluate the existing model on WLASL and compare to published baselines. *This is the right starting point and what I recommend for the 2-week talk.*
2. **Fine-tune `weights.h5` on WLASL.** Continue training from the existing weights using the friend's `train/` pipeline pointed at WLASL data — so the model adapts to a public dataset's distribution.
3. **Train from scratch with a modified objective (e.g., soft-DTW replacing the hard-DTW + cross-entropy used in `train/`).** Most ambitious; this is the headline-paper version.

Pick option 1 for the lab talk. Decide between 2 and 3 *as your grad research direction* after the talk.

### What "retraining from scratch" looks like (option 3)

The friend's notebooks already do this. The actual loop (simplified from `train/`):

```python
model = TCNSignEmbedding()                       # or `Model` in NB1, with two thresholds
optim = Adam(model.parameters(), lr=1e-3)
for batch in HandGestureDataset(...):            # synthetic continuous concatenated streams
    input_seq, input_labels, target_seqs, target_labels, target_masks = batch
    z_in  = model(input_seq.unsqueeze(0)).squeeze(0)             # (T_in, 256)
    z_tgt = [model(t[m].unsqueeze(0)).squeeze(0)
             for t, m in zip(target_seqs, target_masks)]         # list of (T_c, 256)

    # Both losses use HARD DTW (numpy/numba alignment) + PyTorch sum along path
    p_loss, *_ = dtw_partition_loss(z_in, z_tgt, labels, model.partition_threshold, alpha=0.05)
    m_loss, _  = dtw_match_loss   (z_in, z_tgt, labels, model.match_threshold)   # NB1 only
    loss = p_loss + 0.5 * m_loss
    loss.backward(); optim.step()
```

The choices the friend made (now visible in `train/`):

1. **Loss function** — DTW-based cross-entropy, NOT triplet. See Part 4.5.3 for the full breakdown of `dtw_partition_loss` and `dtw_match_loss`.
2. **Sampling** — episodic, but augmented with **continuous-stream synthesis**: each batch is one synthetic continuous signing sequence built by concatenating N class recordings with Slerp transitions. See Part 4.5.2.
3. **Augmentation** — temporal stretch (0.5–1.5×), random global position offset, left-right hand mirror.
4. **Optimizer** — Adam, lr=1e-3, no schedule visible. NB1 ran 50 epochs scheduled, NB2 ran 400 scheduled (only ~8 visible in cell output).
5. **Validation** — 80/20 file split per class; episodic eval with same loss formulation. *Note: this is NOT class-disjoint few-shot eval — for a paper-grade meta-learning evaluation you'll need to add class-disjoint splits.*

### What "fine-tuning" means here (option 2)

Two distinct things people call fine-tuning, both available:

| Type | What it means here | Use when |
|---|---|---|
| **Domain fine-tuning** | Take `weights.h5`, continue training on WLASL with the same (newly-written) loss. | You want the model to handle a different signing population/style. |
| **Per-user adaptation** | Take `weights.h5`, do a few gradient steps using one user's recordings. | You want the model to specialize to a single signer at runtime. |

### How you'd validate that retraining "worked"

Three checkpoints, in order:
1. **Loss decreases** during training (basic sanity).
2. **Few-shot accuracy** on held-out novel classes is non-trivial (well above 1/N where N is the n-way).
3. **Your trained checkpoint** is competitive with or better than `weights.h5` itself when both are evaluated on the same WLASL test split.

---

## Part 3: What you'd be making, which component, how you'd test it

This is the section you said you were confused about. Here it is literally.

### 3.1 The 2-week deliverable (no training, just evaluation)

**Code (3 Python files):**

| File | New? | Purpose |
|---|---|---|
| `experiments/preprocess_wlasl.py` | NEW | Take WLASL videos → run MediaPipe Hands → save `(T, 42, 3)` numpy arrays per video |
| `experiments/few_shot_eval.py` | NEW | Load `weights.h5`, embed query + support clips, classify with `partial_DTW`, compute n-way k-shot accuracy |
| `experiments/baselines/` | NEW | Stubs for published-baseline comparison numbers (I3D, Pose-TGCN, SL-GCN — pulled from their papers, not re-run yet) |

**Code you do NOT change:**
- `server/model/model.py`, `server/model/classify.py` — keep them as-is so your eval mirrors the deployed system

**Artifacts (what you walk into the talk with):**
- 1 results table on slide 16: your system's few-shot accuracy on WLASL-100 vs. published baseline numbers
- The talk slides themselves (Part 12)

That's it for the 2-week scope. No training, no loss-function design, no checkpoint-comparing. Just MediaPipe + load weights + evaluate + table.

### 3.2 The grad-research deliverable (after the talk, picks one direction)

If you commit to a direction (e.g., soft-DTW) for your grad work, the additional code becomes:

| File | Purpose |
|---|---|
| `train/dataset.py` | PyTorch `Dataset` + episodic batch sampler over the .npy files |
| `train/train.py` | Train `TCNSignEmbedding` with your chosen new loss → produce `your_checkpoint.pt` |
| Updated `experiments/few_shot_eval.py` | Loop over multiple checkpoints, produce comparison table |

### 3.3 Which component you'd be changing (per direction)

For the lab talk you don't change anything. For the grad work, depending on which direction you pick:

```
                              ↓ Soft-DTW direction touches HERE ↓
  WLASL videos ─→ preprocess ─→ episodic batches ─→ [LOSS FUNCTION] ─→ trained model
                                                                              │
                                                                              ▼
  Live webcam ─→ MediaPipe ─→ TCNSignEmbedding ─→ partial_DTW ─→ predicted label
                              ↑ GCN direction         ↑ DBA / pruning
                              touches HERE            touches HERE
```

Each candidate research direction (Part 7) modifies a different module and leaves the others fixed. That isolation is exactly what makes a controlled experiment possible.

### 3.4 How you'd test (concrete recipe, used for any direction)

This is the actual experimental procedure. Read it as a recipe.

**Step 1 — preprocess WLASL.** Download WLASL videos. Run `preprocess_wlasl.py`. Output: a folder of `.npy` files, one per video, shape `(T, 42, 3)`, with class label encoded in directory name.

**Step 2 — make a few-shot split.** WLASL-100 has 100 classes. For *meta-learning* evaluation, hold out 40 classes as **novel** (the model never sees these during any training). Evaluate exclusively on the novel 40.

**Step 3 — pick what you're testing.** For 2-week talk: just `weights.h5`. For grad-research direction: `weights.h5` (baseline) AND `your_new_checkpoint.pt` (your contribution).

**Step 4 — n-way k-shot evaluation, per checkpoint:**
- Pick 5 random novel classes ("5-way")
- Pick 1 random recording per class as the support set ("1-shot")
- Pick 5 other recordings per class as the query set
- Embed all 5 + 25 = 30 clips
- For each query, classify it via `partial_DTW` against the 5 support clips
- Record accuracy
- Repeat 1000 random episodes; report mean and std

Report (5-way, 1-shot), (10-way, 1-shot), (10-way, 5-shot), (20-way, 5-shot) — that's your results table.

**Step 5 — compare against published baselines.** I3D, Pose-TGCN (both from the WLASL paper, Li et al. WACV 2020), SL-GCN (Jiang CVPRW 2021), MS-G3D (Liu CVPR 2020). Pull their reported numbers from their papers; for fair comparison you must use the same WLASL-100 split they used (the official one).

**Step 6 — pass/fail check (for the grad-research version).** If your trained checkpoint beats `weights.h5` by ≥ 2 absolute points across most cells of the table — and is competitive with or better than the strongest published baseline — that direction is worth committing to. If it's flat or negative, fall back to a different candidate from Part 7.

---

## Part 4: Tear-down (mathematical detail)

Read this until you can present it. The talk needs you fluent on slides 7–10.

### 4.1 The embedding network in math

Input `x ∈ ℝ^{B × 30 × 42 × 3}`. Output `z ∈ ℝ^{B × 30 × 256}`. One embedding vector per input frame — temporal structure preserved. (This matters: it's what lets DTW align frame-by-frame.)

**Spatial extractor.** Each frame is reshaped to `(B·30, 3, 42, 1)` and pushed through 5× `Conv2d(k=3×3, p=1)` with channels `3→16→16→16→16→32`. The width=1 means the kernel slides only along the landmark-index axis. The CNN learns co-activation patterns between adjacent landmark indices in MediaPipe's numbering.

> **Limitation to flag in talk:** "adjacent landmark indices" ≠ "anatomically adjacent joints." MediaPipe's index 5 (thumb base) is not next to index 6 (index-finger MCP) in real space. The CNN's inductive bias is wrong; it works in spite of this, not because of it. → GCN-replacement opportunity (Part 7, direction B).

**Dim reduction.** `Conv1d(32·42 = 1344 → 256, k=1)` over the time axis. Mathematically a learned linear projection per timestep; compresses spatial features to a 256-dim per-frame vector.

**TCN.** Four `TemporalBlock`s (each: 2× Conv1d + BN + ReLU + residual) along the time axis. Dilations are `[1, 1, 1, 2]`. Effective receptive field ≈ 7–9 frames (~350–450 ms at 20 fps). Residual is identity when in/out channels match, else `Conv1d(k=1)` projection.

> **Limitation to flag:** receptive field is shorter than the 30-frame window. The last TCN layer cannot integrate the start and end of a sign jointly. Easy fix in a deeper-dilation version (`[1, 2, 4, 8]`).

**Learnable margin (now known from `train/`):** `self.threshold = nn.Parameter(torch.tensor(4.0))` is the learnable "no-match" class distance, used inside cross-entropy losses over per-class DTW distances. NB1's later iteration splits this into **two** parameters: `partition_threshold` (used in the partition loss, full DTW, scale ~5.0) and `match_threshold` (used in the subsequence-DTW match loss, length-normalized, scale ~0.5). At inference the deployed `classify()` is hardcoded to `0.9` — close to but not identical with the learned `match_threshold`. There's a small calibration gap here (could be tightened by saving and loading the learned `match_threshold`) but it's much smaller than what was hypothesized in earlier drafts. See Part 4.5 for the full training breakdown.

### 4.2 Partial-DTW in math

Given query `x ∈ ℝ^{N × 256}` (N=30) and prototype `y ∈ ℝ^{M × 256}` (variable M):

**Standard DTW recurrence:** `D[i,j] = ‖x_i − y_j‖₂ + min(D[i−1,j−1], D[i−1,j], D[i,j−1])` with `D[0,0] = ‖x_0 − y_0‖₂`.

**Friend's modification — the key trick.** Initialize the *first column* differently:

```
D[i, 0] = ‖x_i − y_0‖₂      for all i ∈ [0, N)
origins[i, 0] = i
```

This says "the prototype can start at any frame of the query." Then a backward pass produces `min_costs[i]`: the cheapest alignment cost of *the prototype ending at query frame i*. This is **subsequence DTW** (Müller 2007 ch. 4) — it lets you detect a sign that occurs anywhere within the 30-frame window without segmentation.

**Chunking + winner-takes-all.** `chunk_min_arr(costs, K=10)` divides the N=30 query into 3 windows of 10 frames; min within each window. Then `argmin` over (classes ∪ {no-match-threshold}) per chunk. Predicted sequence = deduplicated argmins. The threshold class wins when no real class is below 0.9 — this is the implicit open-set rejection.

### 4.3 What's novel and what isn't

| Claim | Novel? | Why |
|---|---|---|
| CNN+TCN over MediaPipe landmarks | No | Bai/Kolter/Koltun TCN (2018), ST-GCN (Yan 2018), SL-GCN (Jiang 2021) all use similar recipes |
| Subsequence DTW on learned embeddings | **Mildly** | Subseq-DTW is classical (Müller); applying it to a learned 256-d embedding for streaming SLR is uncommon |
| Runtime sign registration / few-shot pipeline | Partly | Prototypical Networks (Snell 2017) is the obvious comparison; the inherited system is closer to k-NN-with-DTW-distance |
| End-to-end training the embedding for DTW alignment | **Partially — and the remaining gap is your opening** | Friend's training (now visible in `train/`) already uses DTW-based cross-entropy losses, NOT triplet. But it uses **hard DTW** — alignment chosen non-differentiably, gradients flow only through per-frame distances along the chosen path. Soft-DTW (Cuturi-Blondel) would propagate gradient through the alignment too. → soft-DTW direction (Part 7, direction A); see Part 4.5.4 for the gradient-flow detail. |

---

## Part 4.5: How the model was actually trained (from `train/`)

The repo now includes two Colab training notebooks under `train/`. **Read these before the talk** — the actual training is much more sophisticated than the "inferred triplet loss" hypothesized from `model.py` alone, and it changes the framing of any improvement you propose.

### 4.5.1 What we learned that earlier drafts had wrong

`ARCHITECTURE.md` and the first version of this plan inferred — from the presence of `self.threshold = nn.Parameter(...)` — that training was a **triplet/contrastive loss in raw embedding space**. That was wrong. The actual training already uses **DTW-aware cross-entropy losses**: the embedding is trained against the same DTW alignment that classification uses at inference. The remaining gap (which is still your soft-DTW opening) is that the alignment selection is itself non-differentiable; see 4.5.4.

### 4.5.2 The dataset construction (`HandGestureDataset`)

This is the most creative part of the friend's pipeline and worth understanding well.

- Per `__getitem__`, sample N classes (default 10) plus M distractor classes (10–30).
- For each input class, load one recording, apply augmentations: speed variation (linear interp, factor 0.5–1.5×), random global position offset (±0.05), optional left-right hand mirror.
- **Concatenate the N input recordings into one continuous sequence** with **Slerp-interpolated transitions** between them (5–30 transition frames each, computed by treating each landmark coordinate as a rotvec and spherical-linearly interpolating between the last frame of one sign and the first frame of the next). Transition frames get label `-1` (no class).
- Optionally crop the start/end of the sequence; cropped frames get label `-2` (ignore in loss).
- Separately, build a **target gallery**: one recording per (input ∪ distractor) class, each with its own variable length, padded to a common max length, with masks for valid frames.
- Output: `(input_sequence, input_label, target_sequences, target_labels, target_masks)`.

**Why this matters**: most isolated-SLR pipelines train on one-sign-per-clip and then deploy on continuous streams — a domain gap. The friend's pipeline closes that gap by training directly on synthetic continuous streams. This is itself a defensible methodological contribution.

### 4.5.3 The two losses

**Partition loss — `dtw_partition_loss`** (used in BOTH notebooks):

1. Partition `z_in` into segments where consecutive frames share a label.
2. For each segment with class `c` (skip transitions and ignored frames), compute **standard full-DTW** between the segment and every gallery embedding. Per-frame Euclidean distances are summed along the chosen alignment path.
3. Append the learnable `partition_threshold` as the "no-match" class.
4. Apply softmax over `-distances`; cross-entropy against true class `c`.
5. Add `alpha * dist_to_correct_class` (alpha=0.05) as a regularizer pulling the correct class's distance down explicitly.

**Match loss — `dtw_match_loss`** (used ONLY in NB1):

1. For each gallery class, compute **subsequence DTW** (the partial-DTW from `classify.py`, but with PyTorch path reconstruction so gradients flow), length-normalized.
2. Stack as logits per query frame; append `match_threshold` as no-match class.
3. Cross-entropy per frame against the frame-level labels.
4. Decode predicted sequence (deduplicated argmax, optionally gated by confidence > 0.6) and compute Levenshtein-distance-based sequence accuracy as a metric.

### 4.5.4 How gradients flow through DTW (the soft-DTW opening)

Both losses use **hard DTW** — alignment computed in numpy/numba via argmin selections, non-differentiable. PyTorch gradients then flow by **summing per-frame distances along the chosen path**:

```python
path = dtw_path(...)              # numba; non-differentiable
dist = torch.tensor(0.)
for (i, j) in path:
    dist += torch.norm(z_in[i] - z_tgt[j])   # gradient flows through this sum
```

So the model is updated to make per-frame distances *along the selected alignment* better — but the **choice of alignment receives no gradient**. If a slightly different alignment would have produced a smaller cost on positives, the model isn't told.

**This is exactly what soft-DTW (Cuturi & Blondel 2017) fixes**: replacing argmin with softmin propagates gradient to all alignment paths weighted by their probability under the soft-min temperature γ. Your soft-DTW direction (Part 7, A) is reframed: it's no longer "first DTW-aware training," it's "**fully differentiable alignment vs. hard-aligned gradient flow**." Subtler claim, still genuinely novel for landmark-based few-shot SLR, still publishable.

### 4.5.5 Differences between the two notebooks

| Aspect | NB2 (`finalfinal_actualwtfwhywontitwork…`) | NB1 (`DTWMatching_with_prototypical_learning`) |
|---|---|---|
| Model class | `TCNSignEmbedding` (matches deployed) | `Model` (renamed) |
| Threshold parameters | one — `threshold = 4.0` | two — `partition_threshold = 5.0`, `match_threshold = 0.5` |
| Losses used | partition only | partition + 0.5 × match (joint) |
| Resumes from | `model-5.h5` | `model-10.h5` |
| Epochs scheduled | 400 (only ~8 visible in cell output) | 50 |
| Distractors per batch | 30 train / 20 val | 10 / 10 |
| Match-decode confidence gate | n/a | argmax must exceed 0.6 to emit a sign (NB1's later code) |
| Inference visualization | yes (cells 9–17: per-frame DTW costs over a real recording) | no |

**Architecturally identical.** The `spatialExtractor`, `TemporalBlock`, and the CNN+TCN backbone are the same in both. Only the threshold *parameters* and the loss combination differ.

**Which notebook produced `weights.h5`?** Almost certainly an earlier checkpoint of NB2's pipeline (or a precursor of it):

- Deployed `model.py` declares `self.threshold` as a single learnable parameter — matching NB2's `TCNSignEmbedding`, not NB1's `Model` (which has two thresholds).
- Deployed `classify.py` uses a hardcoded threshold `0.9` (length-normalized scale) — close to NB1's `match_threshold` initialization (0.5) and consistent with using the subsequence-DTW match formulation at inference time.
- NB2's training loop visible in cell output ran ~8 epochs with `train_acc` climbing from 0.69 → 0.86; NB1 was the next attempt to extend training with the additional joint loss.

**Suggested reading order**: NB2 first (simpler, single-loss, matches the deployed model file), then NB1 (adds match loss for direct optimization of the streaming decode metric).

### 4.5.6 Implications for the talk and your research framing

1. **Soft-DTW pitch survives but the wording changes.** Direction A in Part 7 is now "fully differentiable alignment via soft-DTW vs. hard-aligned gradient flow." Don't claim "first DTW-aware training" — that's NB1 and NB2 already.
2. **Partition vs. partition+match is itself an unpublished ablation.** NB2 (partition only) vs NB1 (partition + 0.5·match) is a clean A/B that the friend never measured rigorously. You could add this as a small ablation in any direction you commit to.
3. **The Slerp-transition continuous-stream training is genuinely creative.** Cite it as a methodological strength of the inherited system (slide 8 or 9). If you want, this is itself a candidate research claim — most SLR work does NOT train on synthetic continuous streams.
4. **Per-class learnable thresholds** become a clean direction-D variant: scale the `match_threshold` per class or query-dependent (currently scalar; could be a small MLP head).
5. **Validation in `train/` is NOT few-shot.** It's an 80/20 file split per class; same classes in train and val. For a paper-grade meta-learning claim you'll need to add **class-disjoint** splits (hold out 40 of WLASL-100's classes from any training).

---

## Part 5: Related work / reference papers (for slides 4–6)

Group into 4 buckets. ★ = cite; ★★ = cite + read end-to-end.

### Few-shot / metric learning
- ★ **Snell, Swersky, Zemel.** *Prototypical Networks for Few-shot Learning.* NeurIPS 2017.
- Vinyals et al. *Matching Networks for One Shot Learning.* NeurIPS 2016.
- Finn, Abbeel, Levine. *Model-Agnostic Meta-Learning.* ICML 2017.
- ★ Sung et al. *Learning to Compare: Relation Network for Few-Shot Learning.* CVPR 2018.
- Schroff, Kalenichenko, Philbin. *FaceNet: A Unified Embedding for Face Recognition and Clustering.* CVPR 2015. — *The triplet-with-learnable-margin pattern; almost certainly the inherited system's training shape.*
- Sohn. *Improved Deep Metric Learning with Multi-class N-pair Loss.* NeurIPS 2016.

### DTW & differentiable DTW (central to direction A)
- ★★ **Cuturi, Blondel.** *Soft-DTW: a Differentiable Loss Function for Time-Series.* ICML 2017. — *Read until you can write the recurrence on a whiteboard.*
- Sakoe, Chiba. *Dynamic programming algorithm optimization for spoken word recognition.* IEEE TASSP 1978.
- ★ **Müller.** *Information Retrieval for Music and Motion.* Springer 2007. — *Best treatment of subsequence DTW (Ch. 4); cite for the partial-DTW algorithm.*
- Petitjean, Ketterlin, Gançarski. *A global averaging method for dynamic time warping (DBA).* Pattern Recognition 2011.
- Keogh, Ratanamahatana. *Exact indexing of dynamic time warping.* KAIS 2005.
- Mensch, Blondel. *Differentiable Dynamic Programming for Structured Prediction and Attention.* ICML 2018.

### Temporal models / skeleton networks
- ★ **Bai, Kolter, Koltun.** *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv 2018. — *The TCN paper; the inherited `TemporalBlock` is a direct adaptation.*
- ★ **Yan, Xiong, Lin.** *Spatial Temporal Graph Convolutional Networks (ST-GCN).* AAAI 2018. — *Cite for direction B (GCN replacement of spatial CNN).*
- Liu et al. *Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition (MS-G3D).* CVPR 2020.

### Sign language recognition (the baselines you'll compare against)
- ★ **Li et al.** *Word-level Deep Sign Language Recognition (WLASL).* WACV 2020. — *Dataset paper. Reports I3D and Pose-TGCN on WLASL-100/2000. These are your primary published baselines.*
- Joze, Koller. *MS-ASL: A Large-Scale Data Set and Benchmark.* BMVC 2019.
- Sincan, Keles. *AUTSL: A Large Scale Multi-modal Turkish Sign Language Dataset.* IEEE Access 2020. — *Backup dataset.*
- ★ **Jiang et al.** *Skeleton Aware Multi-modal Sign Language Recognition (SL-GCN).* CVPRW 2021. — *Strongest skeleton-only SLR baseline.*
- ★ **Bilge, Ikizler-Cinbis, Cinbis.** *Towards Zero-shot Sign Language Recognition.* TPAMI 2022. — *Closest peer to your few-shot framing. Read end-to-end. Adopt their evaluation protocol (class-disjoint splits).*
- Camgoz et al. *Neural Sign Language Translation.* CVPR 2018.
- ★ **Zhang et al.** *MediaPipe Hands: On-device Real-time Hand Tracking.* CVPRW 2020. — *Cite for the landmark front-end.*

---

## Part 6: Limitations of the inherited system (your research opportunities)

These are the gaps you can claim to address. Each maps to a candidate research direction in Part 7.

| # | Limitation | Candidate direction |
|---|---|---|
| 1 | Hard-DTW alignment during training: gradient flows through summed per-frame distances along the chosen path, but the alignment selection itself is non-differentiable (numpy/numba argmin). See Part 4.5.4. | **A** Soft-DTW training |
| 2 | Spatial CNN over arbitrary landmark order, ignores hand bone structure | **B** GCN spatial encoder |
| 3 | Linear scaling with database size (every recording = its own DTW computation) | **C** DBA-aggregated prototypes |
| 4 | Hardcoded `threshold=0.9` for "no match"; no calibration; fragile rejection | **D** Calibrated open-set rejection |
| 5 | Non-overlapping 30-frame windows; signs straddling window boundaries lost | (engineering, not research) |
| 6 | Missing-hand previous-frame fallback creates artificial consistency | (data preprocessing, not research) |
| 7 | TCN receptive field (~9 frames) shorter than window (30 frames) | (architecture tuning, weak claim alone) |

---

## Part 7: Candidate research directions, ranked

For the lab talk, present **all four** in slides 11–14 with my recommendation as the closing slide. Don't commit to one publicly until your lab gives feedback.

### Direction A — Soft-DTW–trained embedding ★ recommended

**The pitch (revised after reading `train/`):** the inherited model is already trained with DTW-based cross-entropy losses (`dtw_partition_loss` + optionally `dtw_match_loss`), but the alignment is selected with **hard argmin** in numpy/numba — gradients flow only through per-frame distances along the chosen path. Replace the hard alignment with **Cuturi-Blondel soft-DTW** (ICML 2017), so the gradient also propagates through alignment uncertainty (all paths weighted by softmin probability under temperature γ). End-to-end differentiable alignment.

- **Effort**: 6–8 weeks
- **Difficulty**: moderate (PyTorch soft-DTW implementations exist; integration into NB1's joint partition+match loss is the work)
- **Strength of paper**: high — narrower but cleaner claim than originally framed; the precise contribution is "fully differentiable alignment vs. hard-aligned gradient flow for landmark-based few-shot SLR"
- **Risk**: γ is finicky (start γ=1, anneal down). Soft-DTW is known to be costlier than hard DTW (no early termination, smoother loss landscape may need lower lr).
- **Required baselines**: WLASL paper baselines (I3D, Pose-TGCN); SL-GCN; `weights.h5` as inherited; **NB2-trained-on-WLASL** (replicating partition-loss only on WLASL); **NB1-trained-on-WLASL** (joint loss). The latter two isolate "soft vs hard DTW" without conflating with the dataset shift.
- **Expected gain**: +1 to +4 points top-1 over the hard-DTW baseline on WLASL-100 5-way 1-shot. Smaller than the previously-claimed +2 to +5 because the hard-DTW baseline is already DTW-aware.

### Direction B — GCN spatial encoder over MediaPipe hand graph

**The pitch:** replace the `spatialExtractor` CNN (which treats 42 landmarks as an arbitrary 1D sequence) with a Graph Convolutional Network respecting the MediaPipe Hands skeleton (21 nodes/hand, 20 bones/hand, optionally an inter-hand virtual edge). Anatomically meaningful inductive bias.

- **Effort**: 6–10 weeks
- **Difficulty**: moderate-to-hard (need to design graph adjacency, integrate GCN layer into TCN pipeline, train from scratch)
- **Strength of paper**: medium — strong inductive-bias story, but ST-GCN/SL-GCN already do GCN on skeleton SLR. The novelty would be the *combination* (GCN spatial + TCN temporal + partial-DTW classification + few-shot eval).
- **Risk**: lots of prior GCN-SLR work means reviewers will want exhaustive comparison
- **Expected gain**: +3 to +6 points; strongest candidate for hand-specific SLR

### Direction C — DBA prototypes for sublinear inference

**The pitch:** the system stores every recording as its own prototype, so inference cost grows linearly with database size. Use **DTW Barycenter Averaging** (Petitjean 2011) to aggregate K recordings per class into one barycenter. Inference goes from O(D×N×M̄×E) to O(C×N×M̄×E) where C ≪ D.

- **Effort**: 3–5 weeks
- **Difficulty**: easy-to-medium (DBA implementations exist)
- **Strength of paper**: medium — clean systems contribution, easy to evaluate (latency + accuracy table), but narrower scope than A or B
- **Risk**: DBA may lose intra-class variance information, hurting accuracy
- **Expected gain**: 5–10× inference speedup, ≤ 1 point accuracy loss

### Direction D — Calibrated open-set rejection

**The pitch:** the hardcoded `threshold=0.9` for "no match" is brittle. Replace with a principled rejection mechanism: per-class threshold learned on validation, OR extreme-value-theory rejection (Weibull on negative-class distance distribution; Bendale & Boult CVPR 2016).

- **Effort**: 4 weeks
- **Difficulty**: medium
- **Strength of paper**: medium-narrow — important for the runtime-registration UX, but may feel "engineering" to ML reviewers
- **Risk**: hard to evaluate without an explicit "negative-class" dataset
- **Expected gain**: better precision/recall trade-off on rejection; sharper open-set numbers

### Direction E — Per-user adaptation (fine-tuning at registration time)

**The pitch:** when a user registers a new sign, also do a few gradient steps on the embedding to specialize for that user's signing style. Few-shot personalization.

- **Effort**: 5–8 weeks
- **Difficulty**: medium-hard (eval protocol is tricky — need per-user splits, which WLASL doesn't naturally provide)
- **Strength of paper**: medium — interesting story, lines up with the runtime-registration UX
- **Risk**: catastrophic forgetting; need regularization
- **Expected gain**: hard to predict without preliminary numbers

### My ranking for grad work
**A > B > C ≈ E > D.** A has the cleanest "isolatable variable" experimental design and the most surprising finding if it works (someone *should* have tried this and apparently nobody has for landmark SLR specifically). B is the safest if you want to commit to a richer architectural project for a thesis chapter. C is the right choice if you want to ship a workshop paper fast. D and E are interesting but harder to evaluate cleanly.

---

## Part 8: External baselines & experimental harness

You compare against published numbers, not against re-trained internal baselines. This means most of your evaluation effort is **building a clean evaluation harness**, then plugging in different checkpoints.

### 8.1 The published baselines (with their numbers)

These numbers are public in the cited papers and are what you compete against on slide 16. You do NOT need to reproduce these — you cite their papers and use their reported numbers.

| Method | Source | WLASL-100 top-1 | Notes |
|---|---|---|---|
| I3D | Li et al. WACV 2020 | ~65% | RGB; trained on full WLASL-100, not few-shot |
| Pose-TGCN | Li et al. WACV 2020 | ~55% | Skeleton; trained on full WLASL-100 |
| SL-GCN | Jiang CVPRW 2021 | ~80%+ | Strongest skeleton baseline; multimodal version higher |
| MS-G3D (transferred) | Liu CVPR 2020 + community impl. | varies | Skeleton, full-supervision SOTA-class |
| Bilge et al. ZSL | TPAMI 2022 | reports zero-shot numbers on MS-ASL — adopt their *protocol* even if numbers don't directly compare |

> **Critical caveat for slide 16:** the published numbers are for **fully-supervised** WLASL classification (train on the same classes you test on), not **few-shot** evaluation (train on base classes, test on novel classes). For an honest comparison, you must either:
> (a) Restrict your eval to the same fully-supervised protocol — register all WLASL-100 classes from a small set of recordings, evaluate on held-out recordings of those same classes, OR
> (b) Re-run a couple of the baselines yourself in the few-shot protocol (this is real work — ~2 weeks of additional engineering)
>
> For the 2-week talk, do (a). For the eventual paper, do (b).

### 8.2 The evaluation harness (what you build in 2 weeks)

```python
# experiments/few_shot_eval.py — pseudo-code shape

def eval_checkpoint(checkpoint_path, wlasl_npy_dir, n_way, k_shot, n_episodes=1000):
    model = load_model(checkpoint_path)
    classes = list_classes(wlasl_npy_dir)

    accs = []
    for _ in range(n_episodes):
        episode_classes = random.sample(classes, n_way)
        support, query = sample_support_query(episode_classes, k_shot, n_query=5)

        # support: list of (class_label, np_array shape (T,42,3))
        # query: list of (class_label, np_array shape (T,42,3))

        # Embed with model
        support_z = [model(s.input.unsqueeze(0))[0] for _, s in support]
        query_z = [(label, model(q.input.unsqueeze(0))[0]) for label, q in query]

        # Build episodic database
        database = [(label, z, None, None) for (label, _), z in zip(support, support_z)]

        # Classify each query
        for true_label, qz in query_z:
            sentence, _ = classify(qz.cpu().numpy(), 0.9, database)
            pred = sentence[0] if sentence else None
            accs.append(pred == true_label)

    return mean(accs), std(accs)
```

That + a preprocessing script + a thin CLI is your entire 2-week deliverable.

### 8.3 The headline table mockup (slide 16)

```
                       5w-1s      10w-1s      10w-5s      20w-5s
Pose-TGCN (lit.)        --         --          --          --     <- supervised numbers, asterisk
SL-GCN (lit.)           --         --          --          --     <- supervised numbers, asterisk
weights.h5 (ours)       XX.X       XX.X        XX.X        XX.X
```

You walk into the talk with these XX.X numbers measured. That's the talk's grounding.

---

## Part 9: Dataset acquisition (WLASL)

### 9.1 Pipeline

1. Clone https://github.com/dxli94/WLASL
2. Run their `video_downloader.py`. **Expect ~70–80% video survival** (link rot). Plan a week. Run it in background today.
3. For each surviving video, extract MediaPipe Hands landmarks → save as `.npy` of shape `(T, 42, 3)`.
4. Variable T per video → for 30-frame eval, take a center crop of 30 frames; for full-eval, sliding window across the video.
5. Splits: WLASL ships official train/val/test. For *few-shot*, follow Bilge et al.'s class-disjoint protocol: hold out k classes for evaluation only.

### 9.2 Critical: match the deployed system's MediaPipe configuration

```python
mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
)
```

Same handedness assignment, same previous-frame fallback as `server/model/main.py:convert_mediapipe()`. Train and eval must agree on this — otherwise your eval numbers are measuring distribution shift, not model quality.

### 9.3 Backup: AUTSL

If WLASL link rot is catastrophic (>40% loss), pivot to AUTSL — 226 classes, RGB + depth, well-curated, downloadable in one shot. Smaller scale → narrower paper, but reproducible.

---

## Part 10: Plan to the lab talk; longer-term grad timeline

### 10.1 The 2 weeks until the talk

| Day | Action |
|---|---|
| 1 | Read this doc end-to-end. Verify Part 4 math against `classify.py` line by line. Start WLASL download in background. |
| 2 | Read **Cuturi & Blondel 2017** + **Snell ProtoNet 2017**. |
| 3 | Read **Li WLASL 2020** + **Jiang SL-GCN 2021** + **Bilge 2022**. |
| 4 | Skim Müller Ch. 4 (subseq DTW). Skim Bai TCN 2018. |
| 5 | Write `preprocess_wlasl.py`. Run on whatever videos have downloaded so far. |
| 6 | Write `few_shot_eval.py` against the inherited `weights.h5`. Get a first number on WLASL-100 5-way 1-shot. |
| 7 | (Buffer + WLASL download likely still running). Run eval on more episodes; compute (5w-1s, 10w-1s, 10w-5s, 20w-5s) numbers. |
| 8 | Build slide deck draft (slides 1–10, the "what is" half). |
| 9 | Build slide deck draft (slides 11–20, the "what to do" half). |
| 10 | Rehearse once. Identify gaps in your understanding. Re-read the gap papers. |
| 11 | Polish slides. Make the headline-table figure on slide 16 from your real numbers. |
| 12 | Send draft to advisor for pre-review. |
| 13 | Buffer for advisor edits. |
| 14 | Present. |

### 10.2 Longer-term: if you commit to a direction after the talk

Direction A (soft-DTW) example:

| Month | Milestone |
|---|---|
| 1 | Implement soft-DTW loss, write `train.py`, train first checkpoint on WLASL-100 |
| 2 | γ ablation, baseline comparisons |
| 3 | WLASL-2000 final training run on uni GPU |
| 4 | Latency study, additional baselines |
| 5–6 | Writing |

---

## Part 11: Risks & contingencies

| Risk | Probability | Mitigation |
|---|---|---|
| WLASL link rot >40% | Med | Pivot to AUTSL. Smaller scale, but clean. |
| MediaPipe extraction is slow on your machine | Low-Med | Run in background overnight; batch; cache aggressively. |
| `weights.h5` baseline number is bad on WLASL (e.g., <30% on 5w-1s) | Med | Two interpretations: (a) the model wasn't trained on enough variety to generalize → Direction A or B is *more* attractive, not less; (b) bug in your eval harness → debug. Either way, having the bad number is informative for the talk. |
| Your eval doesn't finish in time for the talk | Low | Have a backup slide showing only `weights.h5` on the held-out user recordings in `database/` instead. Less compelling but always available. |
| Lab pushes back on direction A in the talk | Expected | Have B, C, E ranked and ready as alternatives in your back pocket. The talk is exploratory; this is the right time for that pushback. |
| You can't reach 5+ point improvement on direction A | Med | Direction A is your top pick *for a paper*; for a grad project, even a rigorous ablation that *establishes* the gap (positive or negative) is publishable as a finding. |

---

## Part 12: Slide outline (20 slides)

The talk is in **3 acts**: what is the system (1–9), what's interesting about it (10–14), what should I do for grad work (15–20).

| # | Slide | Source |
|---|---|---|
| 1 | Title + your name + advisor + lab + date. Brief credit line: "*Original system contributed by [friend]*." | — |
| 2 | The problem: continuous, few-shot, landmark-based SLR. Why it matters (accessibility, vocabulary scaling). | Part 1 |
| 3 | One-sentence what-this-is: "A real-time sign-language recognizer where new signs can be registered without retraining the network." | Part 1 |
| 4 | Related work: few-shot / metric learning (ProtoNet, FaceNet, MAML) | Part 5 |
| 5 | Related work: DTW & soft-DTW (Sakoe-Chiba, Müller subseq, Cuturi-Blondel) | Part 5 |
| 6 | Related work: SLR (WLASL, ST-GCN, SL-GCN, Bilge zero-shot) | Part 5 |
| 7 | System overview block diagram (Part 1 narrative version) | Part 1 |
| 8 | Embedding network: architecture + math (CNN+TCN, shapes) | Part 4.1 |
| 9 | Partial-DTW: the subsequence-DTW trick + chunking + winner-takes-all | Part 4.2 |
| 10 | **Training pipeline** (new — from `train/`): synthetic continuous streams via Slerp transitions; `dtw_partition_loss` + optional `dtw_match_loss`; hard-DTW + path-summed gradients | Part 4.5 |
| 11 | What's open: hard-DTW alignment is non-differentiable → soft-DTW opening; plus 2 secondary gaps | Parts 4.5.4 + 6 |
| 12 | Candidate direction A: Soft-DTW–trained embedding (fully differentiable alignment) | Part 7 A |
| 13 | Candidate direction B: GCN over hand skeleton | Part 7 B |
| 14 | Candidate directions C/D/E: DBA prototypes / open-set rejection / per-user adaptation | Part 7 C, D, E |
| 15 | What I built in the past 2 weeks: preprocessing + eval harness on WLASL | Part 3 |
| 16 | **Headline table**: weights.h5 on WLASL-100 few-shot vs. published baselines | Part 8.3 |
| 17 | Where I'd go: my recommended direction (A) and why | Part 7 + ranking |
| 18 | Experimental plan if I commit to A: timeline, expected results, ablations | Part 10.2 |
| 19 | Risks & alternative directions if A doesn't pan out | Part 11 |
| 20 | Asks: advisor sign-off on direction, GPU time on uni cluster, feedback on framing | — |

---

## Part 13: Action items this week

1. **Today**: clone the WLASL repo and start the video downloader running. Will take days; don't block on it.
2. **Today / tomorrow**: read Cuturi & Blondel 2017. Take notes on the recurrence and gradient — slide 12 needs you fluent.
3. **This week**: write `preprocess_wlasl.py` + `few_shot_eval.py`. Run `weights.h5` on whatever WLASL data is available. Get your first real number.
4. **End of week**: have slides 1–10 drafted (the "what is" half). Save 11–20 for week 2.
5. **Read `train/` end-to-end** (both notebooks) — this is the highest-leverage task this week and supersedes the earlier "ask friend for training script" item. The notebooks ARE the training code, and Part 4.5 of this doc is your map for reading them.

---

## Notes for future-you

- The talk is **scoping**, not committing. If your advisor pushes you toward a different direction than A, that's a successful talk — you've identified the right question to commit to.
- Whichever direction you pick, the **eval harness you build in week 1** is reusable across all of them. That's the highest-leverage code you'll write this month.
- Cite the inherited system honestly. Acknowledgments line is enough for the talk; for any future paper, include a clear "Built on a system originally developed by [name] (contributor)" sentence in §1.
- Every slide should fit on one whiteboard. If it doesn't, split it.
