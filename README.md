# Hermes Music DNA Agent

Neural vocal separation, groove extraction, and sonic reconstruction agent deployed as an isolated Hermes Agent on Hetzner host `62.238.29.81`.

---

## 1. Pipeline Architecture

```
[ Input Audio ] ──> [ Mel-Band RoFormer ] ──> [ Feature Extraction ] ──> [ Reconstruction & Scoring ]
                             │                            │                               │
                      Separates Vocals,            Extracts BPM, swing,            Compares vocal DNA
                      Bass, Drums, Other           spectral centroid, dynamics     against benchmark vault
```

- **Neural Separation Engine**: Mel-Band RoFormer Big Beta 4 (`melband_roformer_big_beta4.ckpt` hosted in `/opt/hermes-data/shared-models/`).
- **Storage Layer**: SQLite storage adapter for vocal identity profiles and groove vectors.
- **Cache & Cache Invalidation**: Dedicated Redis container (`hermes-music-dna-redis`) for state management.

---

## 2. Core Capabilities

1. **Stem Isolation**: Separates master tracks into four isolated stems (vocals, drums, bass, instrumental) with minimal phase cancellation and artifact bleed.
2. **Vocal DNA Profiling**: Extracts formant trajectories, vibrato rate/depth, harmonic excitation, and dynamic breath margins.
3. **Groove Extraction**: Calculates exact micro-timing offsets, swing ratios, and transient envelopes across drum stems.
4. **Reconstruction Benchmarking**: Quantifies model reconstruction fidelity using objective SNR, SDR, and cosine similarity across pairwise vocal vectors.

---

## 3. Directory Layout

```text
hermes-music-dna/
├── hermes_config/
│   ├── SOUL.md                       # Sonic DNA extraction directives
│   └── config.example.yaml           # Runtime configuration
├── src/
│   ├── adapters/
│   │   ├── infra/
│   │   │   └── sqlite_storage.py     # Persistent DNA vector and session storage
│   │   └── pipeline/
│   │       ├── groove_extractor.py   # Transient and swing calculator
│   │       ├── master_bouncer.py     # Master mix render pipeline
│   │       ├── stem_isolator.py      # Mel-Band RoFormer neural interface
│   │       └── vocal_dna_analyzer.py # Formant and vocal acoustic profiler
│   └── domain/                       # Core domain entities and mathematical models
├── benchmark/
│   ├── music/                        # Reconstruction benchmark runs
│   ├── voice_identity/               # Pairwise similarity results
│   └── report.html                   # HTML benchmark visualization
├── Makefile                          # Build and test shortcuts
└── requirements.txt                  # Python dependencies
```

---

## 4. Running Benchmarks

```bash
make benchmark
```\n