"""OpenAI-compatible speech-to-text HTTP endpoint.

Only ``POST /v1/audio/transcriptions`` is implemented, since Parakeet is
transcription-only. There is deliberately no ``/v1/audio/translations`` route:
translating would require a second translation model.
"""
from __future__ import annotations

import logging
import zlib
from typing import Any

from aiohttp import web

from wyoming_parakeet import audio as audio_utils
from wyoming_parakeet import transcribe


_LOGGER = logging.getLogger(__name__)


TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
HEALTH_PATH = "/health"

RESPONSE_FORMATS = ("json", "text", "srt", "verbose_json", "vtt")

DEFAULT_RESPONSE_FORMAT = "json"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
"""Matches the OpenAI API limit so oversized uploads fail predictably."""

TRANSLATION_FIELDS = ("target_language",)
"""Rejected explicitly so callers learn why there is no translations route."""

TRANSCRIBER_KEY: web.AppKey[transcribe.ParakeetTranscriber] = web.AppKey(
    "transcriber", transcribe.ParakeetTranscriber
)


def create_app(transcriber: transcribe.ParakeetTranscriber) -> web.Application:
    """Build the aiohttp application around a shared transcriber."""
    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app[TRANSCRIBER_KEY] = transcriber
    app.router.add_post(TRANSCRIPTIONS_PATH, handle_transcriptions)
    app.router.add_get(HEALTH_PATH, handle_health)
    return app


async def handle_health(request: web.Request) -> web.Response:
    """Report readiness and whether inference is currently busy."""
    transcriber = request.app[TRANSCRIBER_KEY]
    return web.json_response(
        {
            "status": "ok",
            "model": transcribe.MODEL_REF,
            "busy": transcriber.is_busy,
            "ffmpeg": audio_utils.ffmpeg_available(),
        }
    )


async def handle_transcriptions(request: web.Request) -> web.Response:
    """Transcribe an uploaded file, mirroring OpenAI's transcriptions contract."""
    transcriber = request.app[TRANSCRIBER_KEY]

    if not request.content_type.startswith("multipart/form-data"):
        return _error(
            415,
            "Expected a multipart/form-data request body.",
            type="invalid_request_error",
        )

    try:
        form = await request.post()
    except Exception as err:  # noqa: BLE001 - surfaced to the client as a 400
        _LOGGER.exception("Failed to parse the request body")
        return _error(400, f"Could not parse the request body: {err}")

    rejected = _reject_unsupported_fields(form)
    if rejected is not None:
        return rejected

    upload = form.get("file")
    if upload is None:
        return _error(
            400,
            "Missing required parameter: 'file'.",
            type="invalid_request_error",
            param="file",
        )

    if form.get("model") is None:
        return _error(
            400,
            "Missing required parameter: 'model'.",
            type="invalid_request_error",
            param="model",
        )

    response_format = (form.get("response_format") or DEFAULT_RESPONSE_FORMAT).lower()
    if response_format not in RESPONSE_FORMATS:
        return _error(
            400,
            f"Unsupported response_format '{response_format}'. "
            f"Supported values: {', '.join(RESPONSE_FORMATS)}.",
            type="invalid_request_error",
            param="response_format",
        )

    data = upload.file.read()
    if not data:
        return _error(
            400,
            "The uploaded audio file is empty.",
            type="invalid_request_error",
            param="file",
        )

    language = form.get("language") or None
    granularities = _multi(form, "timestamp_granularities")
    include = _multi(form, "include")
    want_words = "word" in granularities

    try:
        waveform, sample_rate = await audio_utils.decode(data)
    except audio_utils.AudioDecodeError as err:
        return _error(400, str(err), type="invalid_request_error", param="file")

    # The timestamped adapter costs one extra ONNX pass, so only pay for it when
    # the response actually needs logprobs or word timings.
    with_details = (
        response_format == "verbose_json" or want_words or "logprobs" in include
    )

    transcript = await transcriber.transcribe(
        waveform, sample_rate, words=with_details
    )

    _LOGGER.info(
        "Transcribed %s via OpenAI API (format=%s)",
        upload.filename or "upload",
        response_format,
    )

    if response_format == "text":
        return web.Response(text=transcript.text, content_type="text/plain")

    if response_format == "srt":
        return web.Response(text=_to_srt(transcript), content_type="text/plain")

    if response_format == "vtt":
        return web.Response(text=_to_vtt(transcript), content_type="text/plain")

    if response_format == "verbose_json":
        return web.json_response(
            _verbose_json(transcript, language, granularities, want_words)
        )

    payload: dict[str, Any] = {"text": transcript.text}
    if "logprobs" in include:
        payload["logprobs"] = _logprobs(transcript)

    return web.json_response(payload)


def _multi(form: Any, name: str) -> list[str]:
    """Read a repeated form field, with or without OpenAI's trailing [].

    The OpenAI SDKs send ``name[]`` while some clients send bare ``name``.
    """
    return form.getall(f"{name}[]", []) or form.getall(name, [])


def _reject_unsupported_fields(form: Any) -> web.Response | None:
    """Explain the absent translations route instead of silently ignoring it."""
    for field in TRANSLATION_FIELDS:
        if form.get(field) is not None:
            return _error(
                400,
                f"Unsupported parameter: '{field}'. This server implements "
                f"transcription only, see {TRANSCRIPTIONS_PATH}.",
                type="invalid_request_error",
                param=field,
            )
    return None


def _verbose_json(
    transcript: transcribe.Transcript,
    language: str | None,
    granularities: list[str],
    want_words: bool,
) -> dict[str, Any]:
    """Build the verbose_json body.

    Parakeet is non-streaming and VAD is not enabled, so there is a single
    segment spanning the whole clip rather than Whisper-style segments.
    """
    payload: dict[str, Any] = {
        "task": "transcribe",
        "language": language,
        "duration": transcript.duration,
        "text": transcript.text,
    }

    if "segment" in granularities or not want_words:
        payload["segments"] = [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": transcript.duration,
                "text": transcript.text,
                "tokens": transcript.tokens,
                "temperature": 0.0,
                "avg_logprob": transcript.avg_logprob,
                "compression_ratio": _compression_ratio(transcript.text),
                # Parakeet has no no-speech classifier to report here.
                "no_speech_prob": 0.0,
            }
        ]

    if want_words:
        payload["words"] = [
            {"word": word.word, "start": word.start, "end": word.end}
            for word in transcript.words
        ]

    return payload


def _logprobs(transcript: transcribe.Transcript) -> list[dict[str, Any]]:
    return [
        {
            "token": token,
            "bytes": list(token.encode("utf-8")),
            "logprob": logprob,
        }
        for token, logprob in zip(transcript.tokens, transcript.logprobs)
    ]


def _compression_ratio(text: str) -> float:
    """Whisper's ratio of raw to gzipped text length."""
    if not text:
        return 0.0
    return len(text.encode("utf-8")) / len(zlib.compress(text.encode("utf-8")))


def _to_srt(transcript: transcribe.Transcript) -> str:
    return (
        f"1\n{_timestamp(0.0, comma=True)} --> "
        f"{_timestamp(transcript.duration, comma=True)}\n{transcript.text}\n"
    )


def _to_vtt(transcript: transcribe.Transcript) -> str:
    return (
        "WEBVTT\n\n"
        f"{_timestamp(0.0)} --> {_timestamp(transcript.duration)}\n"
        f"{transcript.text}\n"
    )


def _timestamp(seconds: float, *, comma: bool = False) -> str:
    """Format seconds as HH:MM:SS,mmm or HH:MM:SS.mmm."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, ms = divmod(total_ms, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}"


def _error(
    status: int,
    message: str,
    *,
    type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> web.Response:
    """Return an error body shaped like OpenAI's."""
    return web.json_response(
        {"error": {"message": message, "type": type, "param": param, "code": code}},
        status=status,
    )