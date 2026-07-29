# Handy AI

A real-time sign language translation app using few-shot learning — new signs can be recorded and recognized instantly without retraining the model.

## How it works

Hand landmarks are extracted via MediaPipe, embedded into a 256-dimensional space by a CNN+TCN neural network, then matched against stored sign recordings using Partial DTW. See [CLAUDE.md](./CLAUDE.md) for the architecture overview and [rmd/STATUS.md](./rmd/STATUS.md) for research status.

## Requirements

- Node.js + Yarn
- Python 3.8+
- A webcam

## Installation

```bash
yarn
pip install -r server/requirements.txt
```

## Usage

Start the Python WebSocket server (terminal 1):
```bash
python server/server.py
```

Start the web app (terminal 2):
```bash
yarn start
```

Open [localhost:3000](http://localhost:3000).

Default login — username: `TEST`, password: `test123`

You can also create a new account. Recorded signs are saved per user.

## License

Favicon — [Flaticon Basic License](https://www.flaticon.com/free-icons/sign-language)

[MIT](https://choosealicense.com/licenses/mit/)
