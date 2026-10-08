"""Audio decoding helpers shared by both protocol servers.

The Wyoming side already receives raw PCM, so it only needs :func:`pcm_to_waveform`.
The OpenAI side receives arbitrary uploads, so it needs :func:`decode`.
"""
from __future__ import annotations

import asyncio
import io
import logging
import shutil

import numpy as np
import soundfile as sf


_LOGGER = logging.getLogger(__name__)


TARGET_SAMPLE_RATE = 16_000
"""Sample rate the ASR model is fed at. Wyoming and OpenAI both land here."""

FFMPEG_BINARY = "ffmpeg"
"""External binary used for the container formats libsndfile cannot read."""

MAX_ERROR_DETAIL = 300
"""Characters of ffmpeg/decoder output to include in an error message."""


class AudioDecodeError(ValueError):
    """Raised when audio cannot be decoded to PCM."""


def ffmpeg_available() -> bool:
    """Return True if the ffmpeg fallback is usable."""
    return shutil.which(FFMPEG_BINARY) is not None


def pcm_to_waveform(pcm: bytes) -> np.ndarray:
    """Convert signed 16-bit little-endian mono PCM to float32 in [-1, 1)."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


async def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Decode encoded audio into a mono float32 waveform and its sample rate.

    Tries libsndfile first since it is in-process and covers wav/flac/ogg, then
    falls back to ffmpeg for mp3/m4a/mp4/aac/webm.
    """
    try:
        return _decode_libsndfile(data)
    except AudioDecodeError as err:
        _LOGGER.debug("libsndfile could not decode the upload: %s", err)
        libsndfile_error = str(err)

    if not ffmpeg_available():
        raise AudioDecodeError(
            f"Unsupported or corrupt audio: {libsndfile_error}. This build has no "
            f"ffmpeg, so only wav, flac and ogg uploads can be decoded."
        )

    return await _decode_ffmpeg(data)


def _decode_libsndfile(data: bytes) -> tuple[np.ndarray, int]:
    """Decode via soundfile, averaging any extra channels down to mono."""
    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(data), dtype="float32", always_2d=True
        )
    except Exception as err:
        raise AudioDecodeError(str(err)) from err

    return waveform.mean(axis=1), int(sample_rate)


async def _decode_ffmpeg(data: bytes) -> tuple[np.ndarray, int]:
    """Decode by piping the upload through ffmpeg into mono s16le at 16 kHz."""
    command = [
        FFMPEG_BINARY,
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate(data)
    except BaseException:
        process.kill()
        await process.wait()
        raise

    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()[:MAX_ERROR_DETAIL]
        raise AudioDecodeError(f"ffmpeg could not decode the audio: {detail}")

    return pcm_to_waveform(stdout), TARGET_SAMPLE_RATE