FROM python:3.13-slim

WORKDIR /app

# ffmpeg decodes the mp3/m4a/mp4/webm uploads the OpenAI endpoint accepts;
# libsndfile backs the soundfile dependency used for wav/flac/ogg.
RUN apt-get update && \
    apt-get install --no-install-recommends -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv .venv
ENV PATH="/app/.venv/bin:$PATH"

COPY . .

RUN pip install --upgrade pip && \
    pip install .

# Set up a cache directory for models that is writable by any user
ENV XDG_CACHE_HOME=/models
RUN mkdir -p /models && chmod 1777 /models

EXPOSE 10300 10301

ENTRYPOINT ["python", "-m", "wyoming_parakeet"]
