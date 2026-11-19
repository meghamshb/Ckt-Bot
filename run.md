# Run KBOT

This guide is for authorized internal users. Keep the source repository, circuit images, credentials and run outputs private. The model archive is hosted separately with public visibility.

## What you need before you start

The private GitHub repository contains the application. The three frozen model files are in the public [K-Bot model archive](https://huggingface.co/Meghamsh111/KBOT-models). This guide downloads an exact, checksum-verified revision. You still need access to the private source repository, the required software and your own hosted-model login.

The model archive is public for inspection; the project notice and third-party terms still apply. Public visibility does not make the project open source.

This Mac setup targets **Apple Silicon and macOS 14 or later**, as required by the pinned PyTorch wheel. The recorded demo ran on an Apple M4 Pro Mac with 24 GiB of memory, using Apple's MPS accelerator for SAM. Another Mac is a new installation that needs checking; this guide does not claim that a clean machine has already been tested.

You need:

- Access to the private `meghamshb/KBOT` repository.
- Python 3.12 and the recorded packages. Install commands are below. The original interpreter was Python 3.12.14; a different build will be recorded as a derived local environment.
- The approved YOLO checkpoint and both SAM checkpoints.
- The two pinned source checkouts listed below.
- Apple's command-line developer tools, including `/usr/bin/python3`, which the launchers use to locate the configured Python environment.
- ngspice installed at `/opt/homebrew/bin/ngspice`. The frozen Mac adapter requires this path; the demo used ngspice 47.
- Node.js and npm to install the pinned official Codex CLI, `codex-cli 0.169.0-alpha.11.1`.
- Your own managed ChatGPT login with access to the configured hosted model, `gpt-6.1-terra`.
- Internet access and access to the configured hosted model. Detection and segmentation are local; the language/vision calls are not offline.

Do not copy someone else's login files or API keys. The current provider uses the official CLI login and does not need an API key pasted into this repository.

## Install the Mac tools first

If you already have Homebrew, open Terminal and run:

```sh
brew install python@3.12 ngspice git
```

If Homebrew is missing, follow the installation instructions at [the official Homebrew website](https://brew.sh/), then run the command above. Do not install a second copy if you already have it.

If Node.js or npm is missing, install them with:

```sh
brew install node
```

If macOS reports that command-line developer tools are missing, run this once and complete the macOS installation dialog:

```sh
xcode-select --install
```

Homebrew may install a newer Python 3.12 patch or ngspice release than the recorded demo. Check the versions; this is a new local environment, not an identical original-machine restore.

Check the required system paths before continuing:

```sh
/usr/bin/python3 --version
python3.12 --version
/opt/homebrew/bin/ngspice --version
node --version
npm --version
```

If `/usr/bin/python3` is missing, finish the developer-tools installation. If ngspice is installed elsewhere, stop and contact the maintainer: adding it to your command path does not satisfy the frozen adapter's absolute path. This guide does not cover Intel Macs.

## 1. Get the application

Open Terminal. If the private repository is already on your Mac, open its folder in Terminal and skip the clone.

```sh
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
git clone https://github.com/meghamshb/KBOT.git
cd KBOT
```

Use your own GitHub access when asked. Do not make the repository public to get around an access error.

## 2. Identify the frozen model files

The download command in step 4 obtains these exact files, approximately **1.82 GB** in total. No Hugging Face login is required. Their expected identities are:

| File | Size in bytes | SHA-256 |
| --- | ---: | --- |
| `yolo26n-final50.pt` | 21344349 | `3d05f8ede9c8aa25ab19261fad7df60a1a5fbd61dcd00f8312ee88122c254005` |
| `sam2.1_hiera_large.pt` | 898083611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| `circuitvision_finetuned.pth` | 900710020 | `ff219e3ea853bcd23fef7d22642c3c6341237e2a2a3f11c1f3898c8f48becd01` |

Do not substitute a different model with the same filename. The project notice does not replace upstream model terms; the adapted SAM checkpoint's separate redistribution rights remain unverified.

You can check a file in Terminal with:

```sh
shasum -a 256 /path/to/yolo26n-final50.pt
```

Replace `/path/to/` with the folder where you put the file. Its output must match the table.

The pinned third-party source checkouts are:

| Source | Required commit |
| --- | --- |
| `https://github.com/JKc66/CircuitVision.git` | `fc5b775043b577668906b6d0bbb238a690a82965` |
| `https://github.com/facebookresearch/sam2.git` | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |

These are source checkouts, not the CircuitVision website. KBOT does not need you to start CircuitVision's web app.

## 3. Install the pinned CLI and sign in

Install the exact official CLI package into your private working folder. This keeps it separate from any other Codex installation and does not require the original Mac app:

```sh
mkdir -p "$HOME/KBOT-PRIVATE/tools"
npm install --prefix "$HOME/KBOT-PRIVATE/tools" --save-exact @openai/codex@0.169.0-alpha.11.1
KBOT_CODEX="$HOME/KBOT-PRIVATE/tools/node_modules/.bin/codex"
"$KBOT_CODEX" --version
```

The version must report `codex-cli 0.169.0-alpha.11.1`. This exact package has been downloaded from the public npm registry and its version checked. That verifies CLI acquisition, not a successful circuit run. The setup helper also checks the frozen command-line options; do not substitute an untested CLI version.

Use a separate login folder for this installation. Keep the same Terminal window open for the remaining setup and startup commands:

```sh
export CODEX_HOME="$HOME/KBOT-PRIVATE/codex-home"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"
"$KBOT_CODEX" login --device-auth
"$KBOT_CODEX" login status
```

Open the URL printed by the official CLI, enter its code and sign in with **your own authorized ChatGPT account**. The status must say `Logged in using ChatGPT`. Each teammate must sign in separately; do not copy the original machine's authentication or another person's login files. An API-key login is not the current configuration.

The frozen model name is `gpt-6.1-terra`. Your account must have access to it; installing the CLI does not grant access. Model access and capacity can change, and a successful login check does not prove that a live model request will succeed. If access is unavailable, stop and contact the maintainer rather than changing the model silently.

## 4. Prepare your private working folder

Keep your model files and run outputs outside the Git checkout. In the example below, the private files live under `$HOME/KBOT-PRIVATE/`. You may use another location, but the command must point to the real files.

- The download command below puts all three frozen model files in `$HOME/KBOT-PRIVATE/models/`.
- Put the pinned CircuitVision checkout in `$HOME/KBOT-PRIVATE/CircuitVision/`.
- Put the pinned SAM2 checkout in `$HOME/KBOT-PRIVATE/sam2/`.
- Keep the `KBOT_CODEX` value and exported `CODEX_HOME` from step 3 available in the same Terminal window.

These commands obtain the two pinned source checkouts. Model weights and Python packages are downloaded separately below. Skip a clone if the correct checkout already exists:

```sh
mkdir -p "$HOME/KBOT-PRIVATE"
git clone https://github.com/JKc66/CircuitVision.git "$HOME/KBOT-PRIVATE/CircuitVision"
git -C "$HOME/KBOT-PRIVATE/CircuitVision" checkout --detach fc5b775043b577668906b6d0bbb238a690a82965
git clone https://github.com/facebookresearch/sam2.git "$HOME/KBOT-PRIVATE/sam2"
git -C "$HOME/KBOT-PRIVATE/sam2" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
```

Now create a separate Python environment and install the recorded package versions. These commands download packages; they do not train models or download the excluded model weights. Run from your KBOT checkout:

```sh
python3.12 -m venv "$HOME/KBOT-PRIVATE/.venv"
KBOT_PYTHON="$HOME/KBOT-PRIVATE/.venv/bin/python"
"$KBOT_PYTHON" -m pip install -r requirements-macos.txt
SAM2_BUILD_CUDA=0 "$KBOT_PYTHON" -m pip install --no-build-isolation --no-deps -e "$HOME/KBOT-PRIVATE/sam2"
"$KBOT_PYTHON" -m pip check
```

`requirements-macos-observed.txt` preserves the original package record. `requirements-macos.txt` has the same lines except the editable SAM2 URL, which is installed from the pinned local checkout by the next command.

This recipe passed in a fresh Python environment on the original Mac on 19 November 2026, including an uncached package download and `pip check`. A separate physical Mac has not been tested. If installation or `pip check` fails on your machine, stop and give that error to the maintainer; do not silently unpin packages.

### Download the three frozen checkpoints

Run this in the same Terminal window after installing the packages above. It downloads the models and their notices from the verified Hugging Face revision into your private working folder. It does not train anything or change the source checkout.

```sh
"$KBOT_PYTHON" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Meghamsh111/KBOT-models",
    revision="8e59f5f03b30cb02eeace43e5df904066a233f2b",
    local_dir=str(Path.home() / "KBOT-PRIVATE" / "models"),
    allow_patterns=["*.pt", "*.pth", "ARTIFACTS.json", "README.md", "LICENSE.md", "notices/*"],
    token=False,
)
print("Model download complete. The setup helper below verifies all three SHA-256 hashes.")
PY
```

If you already have all three matching files locally, skip the download. The next command checks their identities either way. [The model archive receipt](provenance/HUGGING_FACE_MODELS.json) records the verified revision, sizes and checksums; it contains no credentials.

Next prepare the private runtime copy:

```sh
"$KBOT_PYTHON" scripts/configure-local.py \
  --output "$HOME/KBOT-PRIVATE/runtime" \
  --yolo "$HOME/KBOT-PRIVATE/models/yolo26n-final50.pt" \
  --sam-artifacts "$HOME/KBOT-PRIVATE/models" \
  --vendor-root "$HOME/KBOT-PRIVATE/CircuitVision" \
  --sam2-root "$HOME/KBOT-PRIVATE/sam2" \
  --codex "$KBOT_CODEX" \
  --device mps
```

The output folder must be new and outside the Git checkout. The helper prepares a private runtime copy, creates the shared lock required by the first upload, and records its local paths and environment. It does not download weights, install packages, train or run model inference. It checks model identities, pinned sources and CLI requirements. If an approved file or required software is missing, stop and get it from the maintainer.

The local configuration is a derived installation record, not a claim that a different Python build or machine is identical to the original demo. The original freeze records and pipeline/UI source stay preserved.

## 5. Start the web interface

After the helper succeeds, run the private copy:

```sh
export CODEX_HOME="$HOME/KBOT-PRIVATE/codex-home"
cd "$HOME/KBOT-PRIVATE/runtime"
./check-ready
./run-web --port 8766
```

The first command checks files, versions and login. It does not run a circuit, train a model or download model files. If it says `NOT READY`, fix the reported missing file or mismatch before continuing.

Keep that Terminal window open. In your browser, open:

```text
http://127.0.0.1:8766/
```

Upload a circuit image that you are authorized to use, enter the question and press the run button. For the known AC walkthrough, choose `docs/images/ac-source.png` in your source checkout. Selected documentation images are included with their source record; full textbook datasets and the prepared demo question manifest are excluded.

The full workflow is: image → component boxes → SAM mask → pins and nets → values and pin roles → final circuit → the question's physical target → exact linear calculation → ngspice check → worked explanation. It does more than component detection.

For the previously tested Sadiku figure 10.52 shown in `docs/images/ac-source.png`, enter:

```text
Find v_o(t) in sinusoidal steady state, using the marked polarity across the 1 Ω resistor. Show the phasor conversion, nodal equation and final time-domain voltage.
```

This is a known demo example, not an unseen test. Another image needs its own correct question and measurement reference.

The pipeline shows detections, the mask, connections, readings and the answer as they become available. A missing or ambiguous connection can stop the calculation. A model capacity or network error can stop a language/vision stage. Do not present a saved result as a live run.

## Stopping and restarting

Press Control-C in the server's Terminal to stop it. To restart, repeat the startup commands above, including the `CODEX_HOME` export if you open a new Terminal window. If another server already uses port 8766, close that server or ask the maintainer before starting a second instance.

Outputs are saved under `UI-RUNS/` and are excluded from Git. They can contain the uploaded circuit and model outputs; keep them private.

## What has and has not been checked

On **19 November 2026**, the handoff was checked using a fresh GitHub clone, a new Python environment with uncached downloads, new pinned upstream checkouts, all three models downloaded anonymously from Hugging Face, the pinned CLI installed from npm, and a separate fresh ChatGPT login. Existing project models, Python environments and saved answers were not used.

One known AC circuit was uploaded through the actual browser interface. YOLO, SAM, graph assembly, visual reading, question interpretation, calculation, SPICE and the worked explanation completed in **55.76 seconds**. The result, $v_o(t)=3.83555\cos(4t-35.0249^\circ)\,\mathrm{V}$, matched an independent nodal calculation. The graph and rendered equations were visible, with no browser script errors. Setup now creates the missing shared lock file; pipeline and Web UI source were unchanged.

This was a fresh environment on the existing Apple M4 Pro Mac, using its installed Python 3.12.14, ngspice 47, Node.js and macOS developer tools. It was **not a test on a second physical machine**, an offline run, or an unseen-circuit accuracy benchmark. Each teammate still needs private source access and their own hosted-model login and access. See [the reproduction record](provenance/HANDOFF_REPRODUCTION_20261119.json) and `EXCLUSIONS.md`.

## Optional Docker route

See `docker/README.md` for the experimental Linux/CPU container route. It requires private model mounts, a compatible Linux Codex CLI and your own managed login. Docker on a Mac does not provide the original MPS execution path. The container is not a verified reproduction of the original Mac demo; its build and end-to-end inference have not been proven here.
