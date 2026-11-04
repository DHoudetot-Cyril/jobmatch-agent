# syntax=docker/dockerfile:1
# jevos decision server ("System 1", CPU): `jev serve` from the pinned GitHub release jevos-v5
# (feder-cr/jev, MIT). Both release assets are checked against the release's SHA256SUMS.txt
# (values copied below). See bench/decisions/BACKENDS.md.
#
# Same base image as the api (python:3.12.13-slim-trixie): layers are shared, and Python's
# stdlib gives us download + sha256 + unzip + healthcheck without installing curl/unzip.

ARG BASE_IMAGE=python:3.12.13-slim-trixie

FROM ${BASE_IMAGE} AS fetch
ARG JEVOS_RELEASE=jevos-v5
ARG JEV_TGZ_SHA256=1ce77566405514ed2f20d64f8222cf9eba9b714d332f05f6296e0cad84dd7b04
ARG JEVOS_MODEL_SHA256=0a1159c31b9c2abc484af9c866cdcfe5279fdf592b368636625c5a795f970acc
WORKDIR /dl
RUN python - <<'EOF'
import hashlib, os, shutil, tarfile, urllib.request, zipfile

base = f"https://github.com/feder-cr/jev/releases/download/{os.environ['JEVOS_RELEASE']}/"
assets = {
    "jev-linux-x64.tar.gz": os.environ["JEV_TGZ_SHA256"],
    "jevos-v5-openvino-int8.zip": os.environ["JEVOS_MODEL_SHA256"],
}
for name, expected in assets.items():
    with urllib.request.urlopen(base + name, timeout=120) as r, open(name, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    h = hashlib.sha256()
    with open(name, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != expected:
        raise SystemExit(f"SHA256 mismatch for {name}: {h.hexdigest()} != {expected}")
    print("verified", name)

os.makedirs("/opt", exist_ok=True)
with tarfile.open("jev-linux-x64.tar.gz") as t:
    t.extractall("/opt", filter="tar")  # -> /opt/jev/{jev, lib*.so*}
with zipfile.ZipFile("jevos-v5-openvino-int8.zip") as z:
    z.extractall("/opt/jev")  # -> /opt/jev/model/
assert os.path.isfile("/opt/jev/jev") and os.path.isdir("/opt/jev/model"), os.listdir("/opt/jev")
os.chmod("/opt/jev/jev", 0o755)
EOF

FROM ${BASE_IMAGE} AS runtime
RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /opt/jev --no-create-home app
COPY --from=fetch --chown=app:app /opt/jev /opt/jev
USER app
WORKDIR /opt/jev
ENV JEVOS_PORT=8017
EXPOSE 8017
# /health answers {"status": "ready", ...} once the model is loaded and warmed up (~20-30 s).
HEALTHCHECK --interval=10s --timeout=3s --start-period=90s --retries=5 \
    CMD ["python", "-c", "import json,os,sys,urllib.request; r=urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"JEVOS_PORT\"]}/health', timeout=2); sys.exit(0 if json.load(r).get('status') == 'ready' else 1)"]
CMD ["sh", "-c", "exec /opt/jev/jev serve --host 0.0.0.0 --port \"$JEVOS_PORT\""]
