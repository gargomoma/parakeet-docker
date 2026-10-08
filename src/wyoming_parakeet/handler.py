"""Wyoming protocol event handler for clients of the server."""
from __future__ import annotations

import logging

from wyoming import asr
from wyoming import audio
from wyoming import info
from wyoming import server
from wyoming.event import Event

from wyoming_parakeet import audio as audio_utils
from wyoming_parakeet import transcribe


_LOGGER = logging.getLogger(__name__)


SAMPLE_WIDTH = 2
"""Bytes per sample after conversion. Parakeet is fed signed 16-bit PCM."""

_INFO = info.Info(
    asr=[
        info.AsrProgram(
            name="Parakeet ASR",
            description="Parakeet transcription",
            attribution=info.Attribution(
                name="Jonathan Woodbury",
                url="https://github.com/jpwoodbu/wyoming-parakeet",
            ),
            installed=True,
            version="0.0.1",
            models=[
                info.AsrModel(
                    name=transcribe.MODEL_ID,
                    description="NVIDIA Parakeet multilingual automatic speech recognition (ASR) model",
                    attribution=info.Attribution(
                        name="Ilya Stupakov",
                        url=f"https://huggingface.co/{transcribe.REPO_ID}/{transcribe.MODEL_ID}",
                    ),
                    installed=True,
                    languages=list(transcribe.MODEL_LANGUAGES),
                    version="v3",
                ),
            ],
        )
    ]
)


class ParakeetEventHandler(server.AsyncEventHandler):
    """Handles one Wyoming ASR session per connection."""

    def __init__(self, transcriber: transcribe.ParakeetTranscriber, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.info_event = _INFO.event()
        self.transcriber = transcriber
        self._max_bytes = transcriber.max_samples * SAMPLE_WIDTH
        self._start()

    async def handle_event(self, event: Event) -> bool:
        if audio.AudioStart.is_type(event.type):
            self._start()
            return True

        if audio.AudioChunk.is_type(event.type):
            self._append(audio.AudioChunk.from_event(event))
            return True

        if audio.AudioStop.is_type(event.type):
            await self._stop()
            return False

        if asr.Transcribe.is_type(event.type):
            language = asr.Transcribe.from_event(event).language
            if language:
                _LOGGER.debug(
                    "Language %s requested, but this model auto-detects the "
                    "spoken language so the request is ignored",
                    language,
                )
            return True

        if info.Describe.is_type(event.type):
            await self.write_event(self.info_event)
            _LOGGER.debug("Sent info")
            return True

        return True

    def _start(self) -> None:
        """Reset the buffer for a new utterance on this connection."""
        self._pcm = bytearray()
        self._truncated = False
        # One converter per utterance: it carries resampler state between chunks.
        self._converter = audio.AudioChunkConverter(
            rate=audio_utils.TARGET_SAMPLE_RATE, width=SAMPLE_WIDTH, channels=1
        )

    def _append(self, chunk: audio.AudioChunk) -> None:
        """Buffer converted PCM, dropping anything past the duration limit."""
        converted = self._converter.convert(chunk)

        remaining = self._max_bytes - len(self._pcm)
        if remaining <= 0:
            self._truncated = True
            return

        if len(converted.audio) > remaining:
            self._pcm.extend(converted.audio[:remaining])
            self._truncated = True
        else:
            self._pcm.extend(converted.audio)

    async def _stop(self) -> None:
        """Transcribe the buffered audio and send the transcript."""
        _LOGGER.debug("Audio stopped. Transcribing...")

        if self._truncated:
            _LOGGER.warning(
                "Input exceeded %.0fs and was trimmed to the model limit",
                self.transcriber.max_audio_seconds,
            )

        if not self._pcm:
            _LOGGER.warning("No audio received, sending an empty transcript")
            await self.write_event(asr.Transcript(text="").event())
            return

        transcript = await self.transcriber.transcribe(
            audio_utils.pcm_to_waveform(bytes(self._pcm))
        )

        await self.write_event(asr.Transcript(text=transcript.text).event())
        _LOGGER.debug("Completed request")