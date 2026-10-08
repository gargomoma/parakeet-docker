"""Transcription service shared by the Wyoming and OpenAI servers.

The model is loaded once at start up and every adapter shares the same
onnxruntime sessions, so both protocols run off one set of weights.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import onnx_asr

from wyoming_parakeet import audio


_LOGGER = logging.getLogger(__name__)


REPO_ID = "istupakov"
MODEL_ID = "parakeet-tdt-0.6b-v3-onnx"
MODEL_REF = f"{REPO_ID}/{MODEL_ID}"

MODEL_LANGUAGES = (
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr",
    "hr", "hu", "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "uk",
)
"""ISO 639-1 codes the model transcribes, per NVIDIA's parakeet-tdt-0.6b-v3 card.

Bulgarian, Czech, Danish, German, Greek, English, Spanish, Estonian, Finnish,
French, Croatian, Hungarian, Italian, Lithuanian, Latvian, Maltese, Dutch,
Polish, Portuguese, Romanian, Russian, Slovak, Slovenian, Swedish, Ukrainian.

The model auto-detects the spoken language, so these are advertised for
discovery only; a language sent by a client is not used to condition decoding.
"""

DEFAULT_MAX_AUDIO_SECONDS = 30.0
"""Most ASR models degrade past 20-30s of audio, so longer input is trimmed."""

SUPPORTED_SAMPLE_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000)
"""Rates the bundled onnx-asr resampler knows about. Others get resampled first."""


@dataclass
class Word:
    """A word with timings in seconds."""

    word: str
    start: float
    end: float


@dataclass
class Transcript:
    """Result of transcribing a single utterance."""

    text: str
    duration: float
    words: list[Word] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)

    @property
    def avg_logprob(self) -> float:
        """Mean token log probability, or 0.0 when unavailable."""
        if not self.logprobs:
            return 0.0
        return sum(self.logprobs) / len(self.logprobs)


class ParakeetTranscriber:
    """Serializes inference over a single shared Parakeet model."""

    def __init__(
        self,
        quantization: str | None = None,
        max_audio_seconds: float = DEFAULT_MAX_AUDIO_SECONDS,
        model_ref: str = MODEL_REF,
    ) -> None:
        _LOGGER.info(
            "Loading model %s (quantization=%s)...",
            model_ref,
            quantization or "fp32",
        )
        self._model = onnx_asr.load_model(model_ref, quantization=quantization)
        _LOGGER.info("Model loaded")

        # with_timestamps() wraps the same sessions, so this costs no extra memory.
        self._timestamped_model = self._model.with_timestamps()

        self._lock = asyncio.Lock()
        self.max_samples = int(max_audio_seconds * audio.TARGET_SAMPLE_RATE)
        self.max_audio_seconds = max_audio_seconds

    @property
    def is_busy(self) -> bool:
        """True while an inference is in flight."""
        return self._lock.locked()

    async def transcribe(
        self,
        waveform: np.ndarray,
        sample_rate: int = audio.TARGET_SAMPLE_RATE,
        *,
        words: bool = False,
    ) -> Transcript:
        """Transcribe a mono float32 waveform.

        Audio longer than ``max_audio_seconds`` is trimmed, since the model does
        not benefit from it and it would otherwise inflate latency.
        """
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)

        # The cap is a duration, so it has to scale with the incoming rate: a
        # 30s clip at 48kHz holds three times as many samples as one at 16kHz.
        max_samples = int(self.max_audio_seconds * sample_rate)
        if waveform.size > max_samples:
            dropped = (waveform.size - max_samples) / sample_rate
            _LOGGER.warning(
                "Trimming %.1fs of audio beyond the %.0fs limit",
                dropped,
                self.max_audio_seconds,
            )
            waveform = waveform[:max_samples]

        if waveform.size == 0:
            return Transcript(text="", duration=0.0)

        start_time = time.perf_counter()
        # onnx-asr is synchronous and CPU bound, so keep it off the event loop.
        async with self._lock:
            transcript = await asyncio.to_thread(
                self._recognize, waveform, sample_rate, words
            )

        _LOGGER.info(
            "Transcribed %.1fs in %.3fs: %s",
            transcript.duration,
            time.perf_counter() - start_time,
            transcript.text,
        )
        return transcript

    def _recognize(
        self, waveform: np.ndarray, sample_rate: int, words: bool
    ) -> Transcript:
        if sample_rate not in SUPPORTED_SAMPLE_RATES:
            _LOGGER.debug(
                "Resampling from unsupported %dHz to %dHz",
                sample_rate,
                audio.TARGET_SAMPLE_RATE,
            )
            waveform = _resample_linear(waveform, sample_rate, audio.TARGET_SAMPLE_RATE)
            sample_rate = audio.TARGET_SAMPLE_RATE

        duration = waveform.size / sample_rate

        if not words:
            return Transcript(
                text=self._model.recognize(waveform, sample_rate=sample_rate),
                duration=duration,
            )

        result = self._timestamped_model.recognize(waveform, sample_rate=sample_rate)
        return Transcript(
            text=result.text,
            duration=duration,
            words=_words_from_tokens(result.tokens, result.timestamps, duration),
            tokens=list(result.tokens or []),
            logprobs=list(result.logprobs or []),
        )


def _words_from_tokens(
    tokens: list[str] | None,
    timestamps: list[float] | None,
    duration: float,
) -> list[Word]:
    """Group SentencePiece subword tokens into words with timings.

    onnx-asr emits tokens with the SentencePiece word marker rendered as a
    leading space, so a token starting with whitespace opens a new word.
    """
    if not tokens:
        return []

    pieces: list[tuple[str, float]] = []
    for index, token in enumerate(tokens):
        stripped = token.strip()
        if not stripped:
            continue

        start = 0.0
        if timestamps is not None and index < len(timestamps):
            start = float(timestamps[index])

        if token[:1].isspace() or not pieces:
            pieces.append((stripped, start))
        else:
            previous, previous_start = pieces[-1]
            pieces[-1] = (previous + stripped, previous_start)

    words = []
    for index, (word, start) in enumerate(pieces):
        end = pieces[index + 1][1] if index + 1 < len(pieces) else duration
        words.append(Word(word=word, start=start, end=max(end, start)))

    return words


def _resample_linear(waveform: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Cheap linear resample for rates onnx-asr has no resampler for."""
    if waveform.size == 0 or in_rate == out_rate:
        return waveform

    out_size = int(round(waveform.size * (out_rate / in_rate)))
    if out_size == 0:
        return waveform

    positions = np.linspace(0, waveform.size - 1, out_size, dtype=np.float64)
    return np.interp(positions, np.arange(waveform.size), waveform).astype(np.float32)