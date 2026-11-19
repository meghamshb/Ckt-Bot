# EXPERIMENTAL LINUX CPU PACKAGING; NOT A VALIDATED REPLACEMENT FOR THE MAC DEMO.
FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 SAM2_BUILD_CUDA=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    YOLO_OFFLINE=true YOLO_AUTOINSTALL=false \
    PATH=/home/kbot/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ngspice socat libgl1 libglib2.0-0 ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/homebrew/bin \
    && ln -s /usr/bin/ngspice /opt/homebrew/bin/ngspice \
    && ln -sf /usr/local/bin/python3 /usr/bin/python3

COPY docker/requirements.txt /tmp/kbot-requirements.txt
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.15.0 torchvision==0.29.2
RUN python -m pip install --no-cache-dir -r /tmp/kbot-requirements.txt
RUN useradd --create-home --uid 1000 kbot \
    && mkdir -p /state /home/kbot/.codex \
    && chown -R kbot:kbot /state /home/kbot

COPY . /opt/kbot-source
USER kbot
WORKDIR /state
VOLUME ["/state", "/home/kbot/.codex"]
EXPOSE 8766
ENTRYPOINT ["python", "/opt/kbot-source/docker/entrypoint.py"]
