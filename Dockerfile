# ─────────────────────────────────────────────────────────────────────────────
# CHARM-MARL — Docker image with Google Research Football built in.
#
# Build:
#   docker build -t charm-marl .
#
# Run (open a shell inside the container):
#   docker run -it --rm -v "$(pwd)/experiments:/app/experiments" charm-marl bash
#
# The volume mount keeps your experiment outputs on the host machine.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System libraries needed to build the Google Research Football C++ engine.
RUN apt-get update && apt-get install --no-install-recommends -yq \
      git cmake build-essential \
      libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev \
      libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev \
      mesa-utils xvfb x11vnc \
    && rm -rf /var/lib/apt/lists/*

# gym 0.21.0 does not install with newer setuptools/wheel, so pin them first.
RUN python -m pip install --upgrade "pip==23.3.2" \
    && python -m pip install "setuptools==65.5.0" "wheel==0.38.4" psutil

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

# Copy the project source.
COPY . /app

# A quick import check so a broken build fails early.
RUN python -c "import torch, numpy, scipy, faiss, gym; import gfootball.env as e; print('all imports OK')"

CMD ["bash"]
