# Experimental Linux CPU container

This packaging adapter has not been built or run end to end. The Docker client is installed on the source Mac, but its Docker daemon was unavailable during preparation. Native Mac execution is the demonstrated route; see [run.md](../run.md).

The container runs the same Python pipeline and web UI source. It generates a separate private runtime copy with CPU device selection, Linux paths, the actual installed dependency versions and a new environment manifest. This is a derived environment, not proof of the original Mac timings or behaviour. Apple MPS is not exposed inside this Linux container.

## Required private inputs

- Three approved checkpoint files in one private directory: `yolo26n-final50.pt`, `sam2.1_hiera_large.pt`, and `circuitvision_finetuned.pth`. The shared configuration helper checks their existing frozen SHA-256 values. No checkpoint is included in the image or automatically downloaded.
- Full Git clones of the two approved vendor trees, at the commits recorded in `runtime/config/sam.json`. The clones must include their `.git` directories; host worktree pointer files are not portable mounts. The SAM2 package is installed editably from its private copied checkout with CUDA extension build disabled.
- An official, architecture-matching Linux Codex CLI release directory containing an executable named `codex`. The Mac app binary does not run on Linux. The CLI must support the frozen command flags and have access to the configured model. The helper checks managed login and CLI flag parsing before startup; it does not strip unsupported flags or change providers/models.
- A managed ChatGPT login created through that CLI in a private Docker auth volume. Do not add credential files to the image, build context or Git.
- Network access for the hosted model stages. Local detector/SAM model loading remains offline.

The configured model and app-bundled alpha CLI may not be available in a public Linux release. In that case this route is blocked until a compatible approved CLI/model is available. It does not fall back to another model. Even a successful precheck does not establish inference access or capacity.

## Build

Building downloads the base image, system packages and Python dependencies, but no model checkpoints. Direct package versions were checked against public PyPI release metadata; CPU PyTorch wheels were checked against the official PyTorch index. Transitive dependencies and system packages are not fully hash-locked, and no image digest or resolver/build success is claimed.

```sh
docker build -t kbot:frozen-cpu-experimental .
```

## Login

Set `KBOT_LINUX_CODEX` to the absolute path of the approved Linux release directory. Use a fresh private login through the official CLI; do not copy the Mac app's token files into the repository.

```sh
docker run --rm -it \
  --mount type=bind,source="$KBOT_LINUX_CODEX",target=/opt/codex,readonly \
  --mount type=volume,source=kbot-codex-auth,target=/home/kbot/.codex \
  --entrypoint /opt/codex/codex \
  kbot:frozen-cpu-experimental login --device-auth
```

## Run

Use independent Git clones for the two vendor directories. Linked worktrees with a `.git` pointer file are rejected because their host Git storage is not available inside the container.

Set the four host path variables below to approved private directories before running. Checkouts, private state and checkpoints stay outside the source repository. Publish the web port on host loopback only.

```sh
docker run --rm --name kbot-demo \
  -p 127.0.0.1:8766:8766 \
  --mount type=bind,source="$KBOT_MODELS",target=/models,readonly \
  --mount type=bind,source="$KBOT_VENDOR",target=/vendor/circuitvision,readonly \
  --mount type=bind,source="$KBOT_SAM2",target=/vendor/sam2,readonly \
  --mount type=bind,source="$KBOT_LINUX_CODEX",target=/opt/codex,readonly \
  --mount type=volume,source=kbot-codex-auth,target=/home/kbot/.codex \
  --mount type=volume,source=kbot-private-runtime,target=/state \
  kbot:frozen-cpu-experimental
```

Open `http://127.0.0.1:8766/`. Do not start this container while the original demo already owns host port 8766. CPU SAM inference can be substantially slower than the native MPS demo; no container timing is available.

The frozen server binds its own loopback port 8767. `socat` forwards container port 8766 to that port without changing the web application. The frozen spice module expects `/opt/homebrew/bin/ngspice`; the image provides that path as a symlink to its installed Linux ngspice. The Linux ngspice version can differ from the recorded Mac version.

Stop with `docker stop kbot-demo`. The private state and auth volumes are retained. After source or dependency changes, use a distinct private state volume and revalidate; do not reuse a historical pass. After a failed first-time configuration, inspect the private volume before reusing it.

## What has actually been checked

- The Python adapters parse and their help/argument handling can be checked without loading models.
- Published direct dependency versions and official CPU wheel availability were inspected.
- Docker daemon access failed, so image build, dependency resolution, authenticated Linux CLI execution, CPU SAM loading, container networking and complete image-to-answer behaviour remain unverified.
- The derived runtime records its source identity and does not change the frozen pipeline or web UI files.

Maintainers must run a full circuit with independently checked connections and answer before calling this a reproduced demo. See `runtime/notices/` for required third-party terms.
