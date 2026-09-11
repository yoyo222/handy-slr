# Handy

Real-time sign language recognition that learns a new sign from two recordings. No retraining, no fine-tuning, no GPU.

![Handy recognising a newly registered sign](docs/demo.gif)

Adding a word to most sign recognition systems means collecting data and retraining the model. I wanted one where you could show it a sign twice and have it work straight away, so Handy compares hand motion in a learned embedding space rather than classifying into a fixed vocabulary. Registering a new sign takes about ten seconds.

## Quick start

You need Node.js with Yarn, Python 3.10 to 3.12, and a webcam.

```bash
yarn
pip install -r server/requirements.txt
```

Run the server and the web app in two terminals, both from the repository root:

```bash
python server/server.py   # websocket server on :8765
yarn start                # web app on :3000
```

Open <http://localhost:3000> and create an account. Accounts are stored locally in `server/login/users.json`, which is created on first signup and is not tracked in git. The server binds to `127.0.0.1` and is a single-user local application.

The sign database starts empty, so nothing is recognised until you record something. Go to *Record*, click to start, and sign after the three-second countdown, then open your mouth to stop. Record each sign twice: the rejection threshold is fitted from your own recordings using leave-one-out, and it needs at least two per class to do that. Once two classes are registered, the dashboard starts producing subtitles.

## How it works

The browser captures a webcam frame every 50 ms and sends it over a websocket as base64 JPEG. On the server, MediaPipe pulls out 42 hand keypoints per frame, and every 30 frames are encoded into a 256-dimensional embedding by a CNN and TCN network. Recognition is a partial DTW alignment against every sign you have registered, which absorbs differences in signing speed and copes with a query that only partly overlaps a sign.

Two things turned out to matter more than the model itself:

- **The rejection threshold is calibrated rather than hardcoded.** A fixed threshold either fires constantly or never fires at all, and the original hardcoded 0.9 never fired once in benchmarking. Handy takes the leave-one-out distances of your own recordings and uses their median, recomputed whenever you add or delete a sign.
- **Detection failures need handling.** MediaPipe fails to find a hand in roughly 38% of WLASL frames. Zero-filling those puts a landmark at the origin that matches almost anything; decaying the last known pose and discarding unrecoverable windows instead was worth about 8 points of accuracy.

The architecture and websocket protocol are written up in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Research

Handy is also the subject of an ongoing research project on few-shot sign recognition, evaluated on WLASL under a class-disjoint protocol where the evaluation classes are never seen during training.

- **Paper** (Japanese): [`research/WIP_paper_2026s.pdf`](research/WIP_paper_2026s.pdf)
- **Poster** (Japanese): [`research/WIP_poster_2026s.pdf`](research/WIP_poster_2026s.pdf)
- **Code**: [`experiments/`](experiments/), with reproduction steps and protocol details in [experiments/README.md](experiments/README.md)

Results so far, all measured on classes disjoint from the training vocabulary:

| Change | Effect |
|---|---|
| Detection fallback for missing hands | 5-way 1-shot 44.1% to 52.7% |
| DBA prototype aggregation | 4.6× faster inference |
| Soft-DTW loss on L2-normalised embeddings | 10-way 5-shot transfer 67.1% to 76.1% |
| Conformal threshold calibration | Continuous-stream WER 280% to 83.1%, no manual tuning |

## Repository layout

```
src/            React frontend (TypeScript)
server/         Python websocket server: MediaPipe, embedding model, DTW matching
experiments/    Research code: evaluation harness, Soft-DTW, DBA, benchmarks
train/          Original Colab training notebooks
research/       Paper, poster, and figure generation
```

## Credits

The Handy application, meaning the React frontend, the websocket server and the original embedding model in `server/model/weights.h5`, was co-developed, as were the training notebooks in `train/`. Full attribution is in [NOTICE](NOTICE).

The research in `experiments/` and `research/` is my own work, carried out at the Nakazawa-Ohkoshi Laboratory, Faculty of Policy Management, Keio University. It covers the evaluation harness, the detection fallback, DBA prototype aggregation, the batched Soft-DTW implementation, the continuous-stream benchmark and conformal threshold calibration.

If you use this work, please cite it. See [CITATION.cff](CITATION.cff).

## License

Code is under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution requirements. The paper, poster and figures under `research/` are under [CC BY 4.0](LICENSE-docs). The notebooks in `train/` are excluded from both grants. See [train/README.md](train/README.md).

Third-party material is not redistributed here: the WLASL dataset has to be obtained from its own maintainers, and the favicon is used under the [Flaticon Basic License](https://www.flaticon.com/free-icons/sign-language).
