# Environment for analysis/explain.py, which needs shap.
#
#   docker build -t upov-sombor .
#   docker run --rm -v "$PWD:/repo" upov-sombor python /repo/analysis/explain.py
#
# shap will not install on every host. This always works.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY requirements.txt requirements-explain.txt /tmp/
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r /tmp/requirements.txt -r /tmp/requirements-explain.txt

ENV MPLBACKEND=Agg OMP_NUM_THREADS=4
