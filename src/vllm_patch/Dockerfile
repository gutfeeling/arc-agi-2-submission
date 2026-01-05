FROM vllm/vllm-openai:v0.11.0
# Add patched file into the image
COPY harmony_utils.py /tmp/harmony_utils.py
# Copy it into the installed vllm site-packages directory
RUN python3 - <<'PY'
import shutil, pathlib, vllm, sys
pkg = pathlib.Path(vllm.__file__).parent
target = pkg / 'entrypoints' / 'harmony_utils.py'
shutil.copy('/tmp/harmony_utils.py', target)

# Nuke stale bytecode for this module (if present)
cache_dir = target.parent / '__pycache__'
for pyc in cache_dir.glob('harmony_utils.*.pyc'):
    pyc.unlink(missing_ok=True)
PY

# arm64 version was missing encodings
# See https://github.com/vllm-project/vllm/issues/22525
RUN mkdir -p /etc/encodings && \
    curl -fsSL -o /etc/encodings/o200k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    curl -fsSL -o /etc/encodings/cl100k_base.tiktoken https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken

ENV TIKTOKEN_ENCODINGS_BASE=/etc/encodings