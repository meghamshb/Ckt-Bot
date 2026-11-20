# Docker (experimental)

The Mac setup in [run.md](../run.md) is the tested route. This Linux CPU container has not been validated end to end.

You need Docker, the three downloaded models, the two pinned source clones from `run.md`, and a compatible Linux Codex executable. Mac CLI binaries cannot run inside this container.

## Build

```sh
docker build -t ckt-bot:frozen-cpu-experimental .
```

## Sign in

Set `KBOT_LINUX_CODEX` to the full path of the Linux CLI folder containing the `codex` executable. Use your own account:

```sh
docker run --rm -it \
  --mount type=bind,source="$KBOT_LINUX_CODEX",target=/opt/codex,readonly \
  --mount type=volume,source=kbot-codex-auth,target=/home/kbot/.codex \
  --entrypoint /opt/codex/codex \
  ckt-bot:frozen-cpu-experimental login --device-auth
```

## Run

Set these variables to full host paths:

- `KBOT_MODELS`: the folder containing all three models.
- `KBOT_VENDOR`: the pinned CircuitVision clone.
- `KBOT_SAM2`: the pinned SAM2 clone.
- `KBOT_LINUX_CODEX`: the Linux CLI folder.

Use full source clones with their own `.git` directories.

```sh
docker run --rm --name ckt-bot-demo \
  -p 127.0.0.1:8766:8766 \
  --mount type=bind,source="$KBOT_MODELS",target=/models,readonly \
  --mount type=bind,source="$KBOT_VENDOR",target=/vendor/circuitvision,readonly \
  --mount type=bind,source="$KBOT_SAM2",target=/vendor/sam2,readonly \
  --mount type=bind,source="$KBOT_LINUX_CODEX",target=/opt/codex,readonly \
  --mount type=volume,source=kbot-codex-auth,target=/home/kbot/.codex \
  --mount type=volume,source=kbot-private-runtime,target=/state \
  ckt-bot:frozen-cpu-experimental
```

Open **http://127.0.0.1:8766/**. Stop with `docker stop ckt-bot-demo`. Private outputs and login remain in the Docker volumes.

The container uses CPU rather than Apple's MPS. It needs network access and your own hosted-model access. No model weights or credentials are built into the image. Use a new state volume after changing source or dependencies, and verify a real circuit before relying on this route.
