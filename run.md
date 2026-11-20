# Run Ckt-Bot

For Apple Silicon Macs with macOS 14 or later. You need internet access, access to this repository, and your own ChatGPT login with access to `gpt-6.1-terra`. The tested Mac has 24 GB of memory.

To let an agent handle this, open the repository and say **setup**. Browser sign-in still needs your input.

## 1. Install the tools

Install [Homebrew](https://brew.sh/) if needed, then run:

```sh
brew install python@3.12 ngspice git node
```

If Apple's developer tools are missing, run `xcode-select --install` and finish the dialog. Check the required tools:

```sh
/usr/bin/python3 --version
python3.12 --version
/opt/homebrew/bin/ngspice --version
node --version
npm --version
```

The current Mac version requires ngspice at the exact path above.

## 2. Get the code

Skip the clone if you already have this repository. Keep the same Terminal window open throughout setup.

```sh
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
git clone https://github.com/meghamshb/Ckt-Bot.git
cd Ckt-Bot

CKT_BOT_PRIVATE="$HOME/CKT-BOT-PRIVATE"
CKT_BOT_PYTHON="$CKT_BOT_PRIVATE/.venv/bin/python"
mkdir -p "$CKT_BOT_PRIVATE"
```

## 3. Install the Python packages

These two source checkouts are needed by SAM. Skip a clone only if the matching checkout already exists.

```sh
git clone https://github.com/JKc66/CircuitVision.git "$CKT_BOT_PRIVATE/CircuitVision"
git -C "$CKT_BOT_PRIVATE/CircuitVision" checkout --detach fc5b775043b577668906b6d0bbb238a690a82965
git clone https://github.com/facebookresearch/sam2.git "$CKT_BOT_PRIVATE/sam2"
git -C "$CKT_BOT_PRIVATE/sam2" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4

python3.12 -m venv "$CKT_BOT_PRIVATE/.venv"
"$CKT_BOT_PYTHON" -m pip install -r requirements-macos.txt
SAM2_BUILD_CUDA=0 "$CKT_BOT_PYTHON" -m pip install --no-build-isolation --no-deps -e "$CKT_BOT_PRIVATE/sam2"
"$CKT_BOT_PYTHON" -m pip check
```

The last command must report `No broken requirements found`.

## 4. Install the CLI and sign in

```sh
npm install --prefix "$CKT_BOT_PRIVATE/tools" --save-exact @openai/codex@0.169.0-alpha.11.1
CKT_BOT_CODEX="$CKT_BOT_PRIVATE/tools/node_modules/.bin/codex"
"$CKT_BOT_CODEX" --version

export CODEX_HOME="$CKT_BOT_PRIVATE/codex-home"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"
"$CKT_BOT_CODEX" login --device-auth
"$CKT_BOT_CODEX" login status
```

Open the printed link, enter the code and sign in with your own account. The status must say `Logged in using ChatGPT`. Do not copy another person's login files. An API key is not used.

## 5. Download the models

This downloads all three models, about 1.82 GB, from the pinned [Hugging Face archive](https://huggingface.co/Meghamsh111/Ckt-Bot-models). No Hugging Face login is needed.

```sh
CKT_BOT_MODELS="$CKT_BOT_PRIVATE/models" "$CKT_BOT_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Meghamsh111/Ckt-Bot-models",
    revision="8e59f5f03b30cb02eeace43e5df904066a233f2b",
    local_dir=os.environ["CKT_BOT_MODELS"],
    allow_patterns=["*.pt", "*.pth", "ARTIFACTS.json", "README.md", "LICENSE.md", "notices/*"],
    token=False,
)
PY
```

The next step checks every model's checksum. The project notice and upstream model terms still apply.

## 6. Configure and start

Run from the Ckt-Bot source folder:

```sh
"$CKT_BOT_PYTHON" scripts/configure-local.py \
  --output "$CKT_BOT_PRIVATE/runtime" \
  --yolo "$CKT_BOT_PRIVATE/models/yolo26n-final50.pt" \
  --sam-artifacts "$CKT_BOT_PRIVATE/models" \
  --vendor-root "$CKT_BOT_PRIVATE/CircuitVision" \
  --sam2-root "$CKT_BOT_PRIVATE/sam2" \
  --codex "$CKT_BOT_CODEX" \
  --device mps

cd "$CKT_BOT_PRIVATE/runtime"
./check-ready
./run-web --port 8766
```

The output folder must be new. For an existing installation, use the restart commands below.

Open **http://127.0.0.1:8766/**. Upload a circuit, enter a question and click **Run analysis**. Keep Terminal open.

For the known example, upload `docs/images/ac-source.png` from the source repository and ask:

> Find v_o(t) in sinusoidal steady state, using the marked polarity across the 1 ohm resistor. Show the phasor conversion, nodal equation and final time-domain voltage.

## Stop or restart

Press **Control-C** to stop. In a new Terminal window, restart with:

```sh
export CODEX_HOME="$HOME/CKT-BOT-PRIVATE/codex-home"
cd "$HOME/CKT-BOT-PRIVATE/runtime"
./check-ready
./run-web --port 8766
```

If port 8766 is occupied, use `./run-web --port 8768` and open that port instead. If a command fails, keep the error and fix it before continuing.

## What is included

GitHub contains the application, setup commands and selected example images. Models, training data, local environments, login files and run outputs are kept separately. New outputs are saved under the runtime's `UI-RUNS/` folder.

A fresh environment on the original Mac completed one known circuit in **55.8 seconds**, with an independently checked answer. A second physical machine has not been tested. See the [verification record](provenance/HANDOFF_REPRODUCTION_20261119.json).

[Docker](docker/README.md) is optional and has not been validated end to end.
