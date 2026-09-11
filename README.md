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

The browser sends a webcam frame every 50 ms over a websocket. MediaPipe extracts 42 hand keypoints per frame. Every 30 frames are encoded into a 256-dimensional embedding by a CNN+TCN network and matched against each registered sign with partial DTW, which tolerates differences in signing speed.

The rejection threshold is fitted per user from leave-one-out distances over their own recordings, and recalculated whenever a sign is added or deleted.

Frames where MediaPipe finds no hand are not zero-filled. The last known pose is decayed forward, windows with no recoverable hand are dropped, and a per-frame presence mask is stored alongside the landmarks.

Recognized signs pass a 3-buffer debounce before being shown, which suppresses single-window misfires.

Protocol and component detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Research

The matching pipeline is evaluated as a few-shot problem on WLASL, using a class-disjoint split so evaluation classes are never seen during training. The harness, benchmarks and results live in [`experiments/`](experiments/), documented in [experiments/README.md](experiments/README.md).

Paper and poster are in Japanese: [`research/WIP_paper_2026s.pdf`](research/WIP_paper_2026s.pdf), [`research/WIP_poster_2026s.pdf`](research/WIP_poster_2026s.pdf).

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
