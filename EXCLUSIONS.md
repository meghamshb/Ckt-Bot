# Excluded artifacts and runtime dependencies

This is a source freeze, not a self-contained model or data release. The active demo is unchanged.

## Model weights not committed

| Artifact path (exact case required) | Bytes | SHA-256 |
| --- | ---: | --- |
| `runtime/models/yolo26n-final50.pt` | 21344349 | `3d05f8ede9c8aa25ab19261fad7df60a1a5fbd61dcd00f8312ee88122c254005` |
| `/Users/meghamshbalantrapu/Documents/Projects/kbot-circuitvision-artifacts-20261113/sam2.1_hiera_large.pt` | 898083611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` |
| `/Users/meghamshbalantrapu/Documents/Projects/kbot-circuitvision-artifacts-20261113/circuitvision_finetuned.pth` | 900710020 | `ff219e3ea853bcd23fef7d22642c3c6341237e2a2a3f11c1f3898c8f48becd01` |

## Other material not committed

- Training datasets, labels, bulk textbook figures/questions, private reference answers, sealed final data and copyrighted course notes. Selected owner-requested documentation images from one saved development run are included under `docs/images/`; see their provenance record.
- Live runs, saved demo outputs, server logs, readiness records, rehearsal outputs and screenshots.
- API keys, CLI login tokens, authentication files, private environment files and user home configuration.
- Local Python environments, installed packages, model caches, old candidates and training checkpoints.
- The three prepared textbook images and their question manifest. The new package uses an empty case manifest; uploads remain the input route.
- Local Qwen weights: current vision and reasoning use the configured hosted CLI provider, not local Qwen.

## Required external installations

The checked-in configuration retains the exact machine-specific paths for provenance. It is not portable without reconfiguration, restoring the approved weights and revalidating the result. Do not delete these dependencies while using the current demo:

- Python environment: `/Users/meghamshbalantrapu/Documents/Projects/kbot-circuitvision-20261113/.venv/bin/python`
- CircuitVision vendor source: `/Users/meghamshbalantrapu/Documents/Projects/kbot-circuitvision-vendor-20261113` at commit `fc5b775043b577668906b6d0bbb238a690a82965`
- SAM2 vendor source: `/Users/meghamshbalantrapu/Documents/Projects/kbot-sam2-vendor-20261113` at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`
- SAM configuration: `/Users/meghamshbalantrapu/Documents/Projects/kbot-sam2-vendor-20261113/sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
- Installed ngspice executable, managed Codex CLI, existing authenticated login and network/model access.

The Python environment uses existing external packages. Environment version metadata and the original source lock are under `provenance/`. These are evidence of the original environment, not a fresh-install success claim.

## Included third-party material

KaTeX JavaScript, CSS and fonts are included with their license. CircuitVision and Ultralytics license notices are preserved. These notices and third-party source remain in their original case; the project notice does not override third-party rights.

## Packaging deltas

Pipeline and web UI source files are byte-for-byte copies. Only the prepared sample manifest is replaced with an empty list. The package readiness hash list excludes omitted files and hashes the empty manifest; the unmodified original freeze record is preserved under `provenance/`. Readiness still requires the excluded YOLO weights and external SAM artifacts.

No training, model inference, data deletion or active server change was performed to make this package.
