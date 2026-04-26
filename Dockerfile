FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ffmpeg \
    libmagic1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN curl -fsSL https://ollama.ai/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"
ENV OLLAMA_HOST=0.0.0.0:11434

COPY . .

RUN mkdir -p output/audio

EXPOSE 7860 11434

CMD ["python", "app.py"]