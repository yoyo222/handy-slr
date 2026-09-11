# Architecture

A React frontend talks to a Python backend over a websocket. The backend owns
MediaPipe, the embedding model, and the sign database; the frontend owns the
camera and the UI. Nothing is sent to a third-party service, so everything runs
locally.

```
React (:3000)  ←── websocket ──→  Python asyncio server (:8765)
                                       │
                                       ├── MediaPipe Hands / FaceMesh
                                       ├── CNN + TCN embedding model (PyTorch)
                                       └── prototype database (.npy + .webm)
```

## Websocket protocol

Every message is JSON. Requests carry a function name and its arguments;
responses echo the function name back so the frontend can route them.

```jsonc
// request
{ "function": "send_descriptions", "args": [], "kwargs": {} }
// response
{ "result": { "hello": "a greeting" }, "function": "send_descriptions" }
```

The server dispatches on `function` against `Session.functions`. Names listed in
`Session.async_functions` (`send_files`, `stop_recording`) are async generators
and stream their result as a sequence of messages instead of one.

Errors come back as `{ "error": "..." }`: `Invalid JSON`, `Function not found`,
`Invalid message format`, or `Not authenticated`.

## Sessions and authentication

`server/server.py` gives every websocket connection its own `LoginSession`.
Until a `login` or `signup` succeeds, only `login`, `signup` and `onOpen` are
dispatched; anything else is refused with `Not authenticated`.

Accounts live in `server/login/users.json` as bcrypt hashes. The file is created
on first signup and is gitignored. A credential database is runtime state, not
source.

The expensive object is `Session` (`server/model/main.py`), which loads
MediaPipe, the embedding model, and the user's sign database. It is cached per
authenticated username in `sessionsList` and shared across that user's
connections, because rebuilding it takes several seconds.

> **A note on the session cache.** Sessions were originally keyed by client IP,
> so every connection from localhost shared the key `127.0.0.1`. React
> StrictMode opens two connections in development; whichever closed first
> popped the shared session, and the survivor then raised `KeyError` on its
> next message. Per-connection state now lives on the connection itself, and
> cache entries are deliberately never evicted on disconnect.

The server pushes `onOpen` unprompted as its first message; the frontend never
sends it. `result: true` means a remembered user was restored and the login
screen is skipped. Because the server binds to `127.0.0.1` and is a single-user
local application, "remember me" is process memory rather than a token.

## Sign database

```
server/database/
├── sampleSet/          # optional, shared with every user
│   └── hello/
└── {username}/
    └── hello/
        ├── 20260730_125224.npy           # (T, 42, 3) landmarks
        ├── 20260730_125224.presence.npy  # per-frame hand-detection mask
        ├── 20260730_125224.webm          # preview video
        └── description.txt
```

One folder per sign class, holding one triple per recording. `Session` loads
`sampleSet` and the user's own folder into an in-memory list of
`(class_name, embeddings, npy_path, video_path)`.

The directory is gitignored. Recordings are video of whoever used the app.

## Recognition pipeline

1. The frontend captures a webcam frame every 50 ms and sends it as base64 JPEG
   via `recieve`.
2. The backend extracts 42 hand keypoints (2 hands × 21) per frame and records
   whether each hand was actually detected.
3. Once 30 frames are buffered, they are encoded into a `(30, 256)` embedding.
4. `classify()` (`server/model/classify.py`) runs `partial_DTW`, a numba-JIT
   alignment that allows the query to match a subsequence of a prototype, against
   every stored prototype.
5. Matches above the calibrated threshold are rejected. Surviving matches pass a
   3-buffer debounce before being emitted, which suppresses single-window
   misfires.
6. The frontend renders the last 15 recognized words as subtitles.

### Detection failures

MediaPipe frequently fails to locate a hand. Zero-filling those frames places a
landmark at the origin, and that "ghost pose" aligns cheaply with anything,
which corrupts matching. Instead the last known pose is decayed forward, windows
with no recoverable hand are discarded, and a per-frame presence mask is stored
alongside the landmarks so downstream code can tell absence from a real pose.

### Threshold calibration

`server/model/calibrate.py` sets the no-match threshold from the user's own
data rather than a constant. It computes leave-one-out distances, scoring each
recording against the prototypes built from that class's *other* recordings,
and takes a quantile of that distribution (`QUANTILE = 0.8`).

The quantile is an operating point, not an accuracy figure. At 0.5 the
threshold sits at the median of genuine scores, so roughly half of real signs
are rejected by construction, which hits movement signs hardest because their
timing varies more between takes. Measured on a 7-class self-recorded set:
0.5 recognized 3 of 8, 0.8 recognized 5 of 8, 0.9 recognized 6 of 8. Raise it
toward 1.0 for recall, lower it for precision.

Calibration needs at least four leave-one-out scores, which means at least two
classes holding two or more recordings each. Below that it falls back to
`FALLBACK_THRESHOLD = 0.35`. Registering or deleting a sign triggers a
recalculation; the chosen value is printed as `[calibration] threshold = ...` at
startup and on every change.

A fixed threshold does not work here, because the right value depends on which
prototype strategy is in use. The app's original hardcoded 0.9 never fired once
in benchmarking.

## Recording a sign

1. The user starts recording in the *Record* view and signs after a 3-second
   countdown.
2. Frames stream to `record`. The backend watches for an open mouth via
   MediaPipe FaceMesh (`mouth_open`) and stops automatically.
3. `stop_recording` receives the base64 video blob, trims leading and trailing
   frames with no hand present, saves the `.npy`/`.presence.npy`/`.webm` triple,
   recalculates the threshold, and streams the video back to the client.

## Video streaming

Videos are too large for one websocket message, so `send_files` and
`stop_recording` split base64 payloads into 256 KB chunks. A `null` chunk marks
end-of-file. The client reassembles them in `VideoChunkProcessor` (`src/App/App.tsx`).

Startup order is `send_descriptions` → `send_files` → `setIsReady(true)`; the
loading screen stays up until every video has arrived.

## Frontend

| Path | Role |
|---|---|
| `src/App/App.tsx` | Root. Websocket lifecycle (reconnects every 5 s), auth state, video reassembly, view switching |
| `src/Login/Auth.tsx` | Login and signup form |
| `src/Record/Record.tsx` | Recording UI, countdown, hand-detection badge |
| `src/Saved/FileViewer.tsx` | Browse and delete recorded signs |
| `src/Settings/Settings.tsx` | Theme and language |
| `src/components/Sidebar.tsx` | Navigation between dashboard, record, saved, settings |
| `src/contexts/` | `ThemeContext`, `LanguageContext` (i18n via `src/translation.ts`) |

Views are selected by a `currentComponent` string in `App.tsx` rather than a
router.
