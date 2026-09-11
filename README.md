# Handy

Real-time sign language recognition from a webcam. Record a sign twice and it is recognized from then on, with no retraining.

![Handy recognizing a newly registered sign](docs/demo.gif)

Most sign recognition models have a fixed vocabulary, so adding a word means collecting data and training again. Handy stores each sign as an embedding and matches input against it with dynamic time warping.

## Setup

Node.js with Yarn, Python 3.11, and a webcam.

```bash
yarn
pip install -r server/requirements.txt
```

Two terminals, both from the repository root:

```bash
python server/server.py   # websocket server, :8765
yarn start                # web app, :3000
```

Open <http://localhost:3000> and create an account. Accounts live in `server/login/users.json`, created on first signup and untracked. The server binds to `127.0.0.1`.

### First run

The database starts empty. Open *Record*, click to start, sign after the 3 second countdown, then open your mouth to stop.

Record **two classes with two recordings each** before expecting output. The rejection threshold is fitted from leave-one-out distances over your own recordings and needs at least four scores. Below that it falls back to a fixed 0.35 and recognition is unreliable. The server prints `[calibration] threshold = ...` whenever it recalculates.

If the UI hangs on the loading screen, read the server terminal. That means an exception partway through the video stream, not a frontend fault.

## How it works

Webcam frame every 50 ms over a websocket. MediaPipe extracts 42 hand keypoints per frame; 30 frames are encoded to a 256-dim embedding by a CNN+TCN net. Recognition is partial DTW against every registered sign, which tolerates differences in signing speed.

Two non-obvious parts:

- The rejection threshold is fitted per user from leave-one-out distances over their own recordings. The original hardcoded 0.9 never fired once in benchmarking.
- MediaPipe misses a hand in ~38% of WLASL frames. Zero-filling those puts a landmark at the origin that matches almost anything. Decaying the last known pose instead is worth ~8 points of accuracy.

Protocol and component detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Research

Few-shot sign recognition evaluated on WLASL, class-disjoint, so evaluation classes are never seen in training.

| Change | Effect |
|---|---|
| Detection fallback for missing hands | 5-way 1-shot 44.1% to 52.7% |
| DBA prototype aggregation | 4.6x faster inference |
| Soft-DTW loss on L2-normalized embeddings | 10-way 5-shot transfer 67.1% to 76.1% |
| Conformal threshold calibration | Continuous-stream WER 280% to 83.1%, no manual tuning |

Paper and poster are in Japanese: [`research/WIP_paper_2026s.pdf`](research/WIP_paper_2026s.pdf), [`research/WIP_poster_2026s.pdf`](research/WIP_poster_2026s.pdf). Reproduction steps and evaluation protocol: [experiments/README.md](experiments/README.md).

## Repository layout

```
src/            React frontend (TypeScript)
server/         Websocket server: MediaPipe, embedding model, DTW matching
experiments/    Evaluation harness, Soft-DTW, DBA, benchmarks
train/          Original Colab training notebooks
research/       Paper, poster, figure generation
```

## Credits

The application was co-developed: React frontend, websocket server, the original embedding model in `server/model/weights.h5`, and the notebooks in `train/`. Full attribution in [NOTICE](NOTICE).

The research in `experiments/` and `research/` is mine, done at the Nakazawa-Ohkoshi Laboratory, Faculty of Policy Management, Keio University. It covers the evaluation harness, detection fallback, DBA prototype aggregation, batched Soft-DTW, the continuous-stream benchmark, and conformal threshold calibration.

To cite, see [CITATION.cff](CITATION.cff).

## License

Code is Apache 2.0 ([LICENSE](LICENSE), attribution terms in [NOTICE](NOTICE)). Paper, poster and figures under `research/` are CC BY 4.0 ([LICENSE-docs](LICENSE-docs)). The notebooks in `train/` are excluded from both, see [train/README.md](train/README.md).

WLASL is not redistributed here and must be obtained from its maintainers. The favicon is used under the [Flaticon Basic License](https://www.flaticon.com/free-icons/sign-language).
