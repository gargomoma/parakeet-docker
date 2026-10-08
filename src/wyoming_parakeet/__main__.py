import argparse
import asyncio
import contextlib
import functools
import logging
import signal
from urllib.parse import urlparse

from aiohttp import web
from wyoming import server as wyoming_server

from wyoming_parakeet import audio as audio_utils
from wyoming_parakeet import handler
from wyoming_parakeet import openai
from wyoming_parakeet import transcribe


_LOGGER = logging.getLogger(__package__)


DEFAULT_OPENAI_URI = "http://0.0.0.0:10301"

DISABLED_URIS = ("", "none", "off", "disabled")
"""Values that turn a listener off."""

FP32_ALIASES = ("", "none", "no", "off", "false", "fp32")
"""Values of --quantization that mean 'ship the unquantized fp32 weights'.

onnx-asr derives model filenames from this string, so anything unrecognized would
look for a file that does not exist.
"""

DEFAULT_QUANTIZATION = "int8"
"""int8 is the smallest published variant: ~639MB instead of ~2.43GB."""


def _normalize_quantization(value: str | None) -> str | None:
    """Map the fp32 aliases onto None, which onnx-asr reads as 'no quantization'."""
    if value is None or value.strip().lower() in FP32_ALIASES:
        return None
    return value.strip()


def _parse_http_uri(uri: str) -> tuple[str, int]:
    """Split an http://host:port URI into an aiohttp bind target."""
    parsed = urlparse(uri)
    if parsed.scheme != "http":
        raise ValueError(
            f"Only 'http://' is supported for the OpenAI API, got '{parsed.scheme}://'"
        )
    if parsed.port is None:
        raise ValueError(f"A port must be specified when using an '{uri}' URI")

    return parsed.hostname or "0.0.0.0", parsed.port


async def start(
    uri: str,
    openai_uri: str,
    quantization: str | None,
    max_audio_seconds: float,
) -> None:
    # The model is expensive to load and shared by every connection on both
    # protocols, so load it once up front.
    transcriber = transcribe.ParakeetTranscriber(
        quantization=quantization, max_audio_seconds=max_audio_seconds
    )

    if not audio_utils.ffmpeg_available():
        _LOGGER.warning(
            "ffmpeg was not found. The OpenAI endpoint will only accept wav, "
            "flac and ogg uploads; mp3/m4a/mp4/webm will be rejected."
        )

    server_tasks: list[asyncio.Task[None]] = []

    wyoming_server_instance = wyoming_server.AsyncServer.from_uri(uri)
    _LOGGER.info("Starting Wyoming server at %s", uri)
    wyoming_task = asyncio.create_task(
        wyoming_server_instance.run(
            functools.partial(handler.ParakeetEventHandler, transcriber)
        )
    )
    server_tasks.append(wyoming_task)

    runner: web.AppRunner | None = None
    if openai_uri.lower() not in DISABLED_URIS:
        host, port = _parse_http_uri(openai_uri)
        runner = web.AppRunner(openai.create_app(transcriber), shutdown_timeout=10.0)
        try:
            await runner.setup()
            await web.TCPSite(runner, host, port).start()
        except OSError:
            _cancel_all(server_tasks)
            await runner.cleanup()
            raise

        _LOGGER.info(
            "Starting OpenAI API at http://%s:%d%s", host, port, openai.TRANSCRIPTIONS_PATH
        )
    else:
        _LOGGER.info("OpenAI API is disabled")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Not implemented on Windows, where Ctrl+C already cancels the task.
        # partial is required: _cancel_all(tasks) would execute immediately and
        # cancel the servers instead of registering a handler.
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, functools.partial(_cancel_all, server_tasks))

    try:
        await asyncio.gather(*server_tasks)
    except asyncio.CancelledError:
        _LOGGER.info("Server stopped gracefully.")
    except Exception as e:
        _LOGGER.error(f"Server stopped due to an unexpected error: {e}")
        _cancel_all(server_tasks)
    finally:
        with contextlib.suppress(Exception):
            await wyoming_server_instance.stop()
        if runner is not None:
            with contextlib.suppress(Exception):
                await runner.cleanup()


def _cancel_all(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10300",
        help="URI the Wyoming server should listen on",
    )
    parser.add_argument(
        "--openai-uri",
        default=DEFAULT_OPENAI_URI,
        help="URI the OpenAI-compatible API should listen on, or 'none' to disable",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        default=DEFAULT_QUANTIZATION,
        help="Model quantization, e.g. int8 (~639MB download). "
        "Use 'none' for fp32 weights (~2.43GB download).",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=transcribe.DEFAULT_MAX_AUDIO_SECONDS,
        help="Audio beyond this length is trimmed, since most models ignore it",
    )
    args = parser.parse_args()

    if args.openai_uri.lower() not in DISABLED_URIS:
        try:
            _parse_http_uri(args.openai_uri)
        except ValueError as err:
            parser.error(str(err))

    try:
        asyncio.run(
            start(
                args.uri,
                args.openai_uri,
                _normalize_quantization(args.quantization),
                args.max_audio_seconds,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()