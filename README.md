# Wyoming Parakeet

A speech-to-text server for NVIDIA Parakeet offering **the best of both worlds**:
first-class [Home Assistant](https://www.home-assistant.io/integrations/wyoming)
support over the Wyoming protocol, and broad compatibility with everything else
over an OpenAI-compatible API. One process, one set of model weights, two
front doors.

| Protocol | Endpoint | Who it's for |
| --- | --- | --- |
| Wyoming | `tcp://0.0.0.0:10300` | Home Assistant, ESPHome, Wyoming satellites |
| OpenAI | `http://0.0.0.0:10301/v1/audio/transcriptions` | OpenAI SDKs, LibreChat, and any client that speaks `/v1/audio/transcriptions` |

## Why both?

These two protocols cover complementary audiences, which is why running both is
worth the small amount of extra code:

- **Wyoming is what Home Assistant speaks.** It is the protocol behind Home
  Assistant's Wyoming integration and works well with ESPHome and Wyoming
  satellites. But Home Assistant's integration is TCP-only and Wyoming-specific,
  so Wyoming alone locks you into that ecosystem.
- **The OpenAI transcription API is the de-facto universal STT interface.**
  `/v1/audio/transcriptions` is implemented by the official OpenAI SDKs and a
  long tail of applications, so serving it means this server drops into tools
  that know nothing about Wyoming.

Both listeners share a single loaded model, so exposing the OpenAI endpoint costs
no extra memory, no extra model load, and no extra inference queue.

## Credits

This project is a fork of
[jpwoodbu/wyoming-parakeet](https://github.com/jpwoodbu/wyoming-parakeet) by
**Jonathan Woodbury**. All credit for the original Wyoming protocol server, the
onnx-asr based model loading, and the Docker packaging goes to him, as well as
for the Ryzen 5500 latency numbers quoted above.

This fork builds on that foundation to add:

- An **OpenAI-compatible `/v1/audio/transcriptions`** endpoint (aiohttp), serving
  the same model over HTTP alongside Wyoming.
- **Compressed audio decoding** for uploads: `mp3`, `m4a`, `mp4`, `mpga`, `webm`
  and `aac` via ffmpeg, with libsndfile handling `wav`, `flac` and `ogg`.
- **Fixes**: an `AttributeError` crash whenever a client sent `transcribe`
  without a language, inference blocking the asyncio event loop, and unbounded
  per-connection audio buffering.
- **Correct language metadata**: the Wyoming `info describe` response now
  advertises all 25 languages the model supports instead of English only.
- **int8 quantization by default**, cutting the first-run download from 2.43GB to
  639MB. `-q none` restores fp32.
- **Fork packaging**: the distribution is now `parakeet-docker` so it no longer
  collides with upstream's `wyoming-parakeet` on PyPI.

## Performance

Inference runs on the CPU, not a GPU. As a reference point from the upstream
project, on Jonathan Woodbury's Ryzen 5500 most Home Assistant commands,
_e.g. "Turn off the light"_, transcribe in about 200ms. GPU support was
considered, but CPU latency proved adequate.

## Quick start with Docker

```sh
docker run --name parakeet-docker -u 1000:1000 \
  -p 10300:10300 -p 10301:10301 -d ghcr.io/gargomoma/parakeet-docker:latest
```

Or build and run your working tree:

```sh
docker compose up --build -d
```

Alternatively, using _Docker compose_:
```yaml
services:
  parakeet-docker:
    build: .
    image: ghcr.io/gargomoma/parakeet-docker:latest
    container_name: parakeet-docker
    restart: unless-stopped
    ports:
      - "10300:10300"
      - "10301:10301"
    user: "1000:1000"
    volumes:
      - models:/models

volumes:
  models:
```

## Using it

### Wyoming

Point Home Assistant's [Wyoming integration][ha-wyoming] at the host and port
`10300`. The server implements the `info describe`, `asr transcribe`,
`audio-start`, `audio-chunk`, and `audio-stop` events, and replies with a single
`asr transcript` per connection. Like most Wyoming ASR servers it relies on the
client for voice activity detection, so audio is transcribed when the client
sends `audio-stop`.

[ha-wyoming]: https://www.home-assistant.io/integrations/wyoming

### OpenAI-compatible API

`POST /v1/audio/transcriptions`, multipart form, following OpenAI's contract.

```sh
curl http://localhost:10301/v1/audio/transcriptions \
  -F file=@speech.mp3 \
  -F model=whisper-1 \
  -F response_format=verbose_json
```

`response_format` accepts `json` (the default), `text`, `srt`, `verbose_json`,
and `vtt`. `verbose_json` additionally honours:

- `timestamp_granularities[]=word` to include word-level timings.
- `language` to echo back the requested language code.
- `include[]=logprobs` to include token log probabilities.

Accepted upload formats are the same as OpenAI's: `flac`, `mp3`, `mp4`, `mpeg`,
`mpga`, `m4a`, `ogg`, `wav`, and `webm`. `wav`, `flac`, and `ogg` are decoded
in-process via libsndfile; the rest are piped through `ffmpeg`.

From the OpenAI SDKs, just point the base URL at this server:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:10301/v1", api_key="not-used")
print(client.audio.transcriptions.create(model="whisper-1", file=open("speech.mp3", "rb")).text)
```

`GET /health` reports readiness, the loaded model, whether inference is
currently busy, and whether `ffmpeg` was found.

### What is not implemented

Only `/v1/audio/transcriptions` exists. There is deliberately no
`/v1/audio/translations` route, since translating to English would require a
separate translation model; requesting a `target_language` returns a `400`
explaining why. `/v1/models` is also absent, because the model name is fixed, so
pass any value for `model`.

`language` is echoed rather than detected. Parakeet v3 is multilingual but does
not report which language it detected, so `language` is `null` unless you send
one. Word and segment timings come from Parakeet's own token timestamps. Because
no voice activity detection is applied, `verbose_json` returns a single segment
spanning the whole clip rather than Whisper-style segments.

## Configuration

| Flag | Default | Description |
| --- | --- | --- |
| `--uri` | `tcp://0.0.0.0:10300` | Wyoming listen URI (`tcp://`, `unix://`, or `stdio://`) |
| `--openai-uri` | `http://0.0.0.0:10301` | OpenAI API listen URI, or `none` to disable |
| `-q`, `--quantization` | `int8` | `int8` (~639MB), or `none` for fp32 (~2.43GB) |
| `--max-audio-seconds` | `30` | Audio past this length is trimmed, since the model ignores it |

## Model size and quantization

The model downloads into the `/models` volume on first start, so the first run
is slow whichever you pick. It persists there across restarts.

| `-q` | Download | Peak RAM | Notes |
| --- | --- | --- | --- |
| `int8` (default) | ~639 MB | ~1 GB | Smallest published variant of this model |
| `none` (fp32) | ~2.43 GB | ~3 GB | Full precision weights |

int8 is the default because a 3.8x smaller download and roughly a third of the
memory is what makes this deployable on a Raspberry Pi or other constrained
host.

Two honest caveats:

- **int8 is not verified lossless for this model.** Third-party benchmarks report
  int8 and fp32 scoring identically, but NVIDIA's own model card publishes no
  int8-versus-fp32 comparison for `parakeet-tdt-0.6b-v3`. Validate on your own
  audio before relying on it, and use `-q none` if accuracy matters more than
  footprint.
- **There is nothing smaller than int8 available.** onnx-asr downloads
  pre-published variants by filename (`encoder-model.int8.onnx` and friends) and
  never quantizes on your behalf, so `-q int4` fails with a file-not-found error
  rather than silently working. An `fp16` build does exist in a third-party repo
  at ~1216 MB, but that is _larger_ than int8 and fp16 weights are usually
  slower than int8 on CPU without AVX512-FP16 or AMX.

## Language support

Parakeet TDT v3 is **multilingual, not English-only**. It transcribes 25 European
languages and auto-detects which one was spoken:

> Bulgarian, Czech, Danish, German, Greek, English, Spanish, Estonian, Finnish,
> French, Croatian, Hungarian, Italian, Lithuanian, Latvian, Maltese, Dutch,
> Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Ukrainian

Because the model auto-detects, a language sent by a client is **not** used to
condition decoding. On the Wyoming side `asr transcribe` accepts `language` and
logs it at debug level, but transcribing Spanish audio produces Spanish whether
or not `language` was set. The Wyoming `info describe` response advertises the
full 25-language list so that clients such as Home Assistant populate their
language picker correctly.

## Setup without Docker

From the project root:
```sh
python3 -m venv .venv
. .venv/bin/activate
pip install .
```

`ffmpeg` must be on `PATH` for `mp3`, `m4a`, `mp4`, and `webm` uploads. Without
it the server still works but rejects those formats with a clear error. The
Docker image installs it for you.

### Running the server

```sh
. .venv/bin/activate
python -m wyoming_parakeet
```

To see flags, run:
```sh
python -m wyoming_parakeet --help
```

## Implementation notes

`transcribe.py` owns the model and is shared by both protocols, so the weights
are loaded once regardless of how many clients connect. It holds a single lock
because inference is CPU bound, and runs inference in a worker thread via
`asyncio.to_thread` so the event loop stays free to accept connections and serve
other clients' sockets while one utterance is being transcribed.

`audio.py` holds all decoding. `handler.py` normalizes incoming Wyoming chunks
to 16 kHz mono signed 16-bit PCM with `AudioChunkConverter`, which handles
whatever sample rate and channel count a client sends. `openai.py` owns the
HTTP surface and shape conversions only.