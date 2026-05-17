# ChaosEngineAI

**The local AI model runner for serious tinkerers.**

ChaosEngineAI is a desktop control plane for running large language models, image
diffusion pipelines, and video diffusion pipelines locally on your own machine.
It pairs a Tauri + React shell with a Python FastAPI backend that drives
`llama.cpp`, Apple MLX, and (optionally) vLLM — so you get one window for
everything from "I want to try this Hugging Face model" to "show me
tokens-per-second across three quantizations of the same prompt."

!!! warning "Work in progress"
    ChaosEngineAI is under active development. Expect rough edges, breaking
    changes between versions, and features that come and go between releases.
    Issue reports and feedback are welcome.

## What's in this guide

- [Getting started](getting-started/installation.md) — install the desktop
  app, set up the Python backend, and verify your first launch.
- [Usage](usage/chat.md) — walkthroughs for Chat, Compare, HTML Challenge,
  Image Studio, Video Studio, the local server, and Benchmarks.
- [Features](features/mtplx.md) — deep dives into speculative decoding
  (DFlash, DDTree, MTPLX) and cache compression strategies.
- [CLI](cli/overview.md) — automating the entire app surface from the
  terminal with `chaosengine-cli`.
- [Testing](testing/overview.md) — Python + TypeScript unit tests, the
  E2E suite, the pre-build gate, and how to extend each.
- [Architecture](architecture/overview.md) — how the Tauri shell, the
  FastAPI backend, and the inference engines fit together.
- [Troubleshooting](troubleshooting/faq.md) — common failure modes and
  how to recover from them.
- [Contributing](contributing/development-setup.md) — coding guidelines,
  development workflow, and the feature-coverage gate.
- [Reference](reference/api.md) — backend HTTP API, environment
  variables, third-party dependencies, and changelog.

## At a glance

ChaosEngineAI is three cooperating layers:

```
┌─────────────────────────────────────────────────────────┐
│  Tauri shell  (Rust + React + TypeScript)               │
│  ├─ React UI in src/                                    │
│  ├─ In-app updater (signed releases from GitHub)        │
│  └─ Spawns and supervises the Python backend            │
└─────────────────────────────────────────────────────────┘
                          │  HTTP  /  IPC
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Python backend  (backend_service/)                     │
│  ├─ Workspace + library + warm pool state               │
│  ├─ OpenAI-compatible HTTP server                       │
│  ├─ Engine adapters + plugin system                     │
│  └─ DFlash / DDTree / MTPLX speculative decoding        │
└─────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │  llama.cpp   │ │ MLX runtime  │ │   vLLM       │
  │  (GGUF)      │ │ (Apple Si)   │ │ (Linux/CUDA) │
  └──────────────┘ └──────────────┘ └──────────────┘
```

See [Architecture overview](architecture/overview.md) for the details.

## Source of truth

The canonical source for every claim on this site is the repository itself:

- [README.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/README.md)
- [CLAUDE.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CLAUDE.md)
- [CHANGELOG.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/CHANGELOG.md)
- [THIRD_PARTY_NOTICES.md](https://github.com/cryptopoly/ChaosEngineAI/blob/staging/THIRD_PARTY_NOTICES.md)

When the docs disagree with the code, the code wins — please file an issue.
