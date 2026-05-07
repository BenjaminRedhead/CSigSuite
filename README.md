# Signature-Kernel Based Evaluation Metrics — NeurIPS Anonymous Submission

This repository contains the reference implementation for the
signature-kernel evaluation metrics introduced in the accompanying
NeurIPS submission, together with the five experiments that produce the
paper's tables and figures.

## Contributions reproduced here

- **SigMMD** — biased-V signature-kernel maximum-mean-discrepancy with a
  4-D batched, log-space PDE solver.
- **Censored SigMMD** — three censoring detectors (Kernel, Systemic,
  Sharpe MV) for utility-aligned tail-event evaluation.
- **MAD-based adaptive sigmoid steepness β\*** — robust, scale-invariant
  setting of the soft-censoring sharpness parameter.
- **Foundation-model benchmark** — six time-series foundation models
  (Chronos v1, Chronos-2, Moirai, MoiraiMoE, Sundial, TimesFM) across
  twelve datasets and four prediction horizons.
- **Agent decision experiment** — does ranking by our metric pick the
  forecaster that maximises an agent's realised utility?

## Repository layout

```
neurips_anon_bundle/
├── README.md                     # this file
├── requirements.txt              # Python dependencies
├── LICENSE                       # anonymised MIT
├── ESK/                          # signature-kernel library
│   └── sigkernel_mmd/            # PDE solver, kernels, MMD, preprocessing
├── sigkernel/                    # Salvi et al. baseline (third-party, Apache-2.0)
├── foundation_models/            # foundation-model adapters (Chronos / Moirai / Sundial / TimesFM)
├── src/
│   ├── metrics/                  # eval_metrics.py — SigMMD, CSig, detectors, MAD-β
│   ├── models/                   # adapters.py — Chronos / Moirai / MoiraiMoE wrappers
│   ├── data_provider/            # data_loader.py
│   └── experiments/              # the five entry-point scripts + config_fm.yml
└── data/
    └── cloud/cloud.csv           # bundled — used by ablation, benchmark, agent
                                  # data/long_term_forecast/ must be populated
                                  # by the user (see "Data" below)
```

## Installation

Python 3.10+ and a CUDA-capable GPU (≥ 24 GB recommended for the longer
horizons) are required for the full benchmark; smaller experiments run on
a single 16 GB GPU or CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ESK             # installs the `sigkernel_mmd` package
pip install -e sigkernel       # installs the third-party Salvi baseline
                               # (only required for speed_memory_comparison.py)
```

Foundation-model weights download lazily from HuggingFace on first use.
Set `HF_TOKEN` if you need access to gated checkpoints.

## Data

| Dataset       | Source                    | Target path                                                  |
|---------------|---------------------------|--------------------------------------------------------------|
| `cloud`       | bundled                   | `data/cloud/cloud.csv` (already present)                     |
| `electricity` | Autoformer release        | `data/long_term_forecast/electricity/electricity.csv`        |
| `ETTh1`       | Autoformer release        | `data/long_term_forecast/ETT-small/ETTh1.csv`                |
| `ETTh2`       | Autoformer release        | `data/long_term_forecast/ETT-small/ETTh2.csv`                |
| `ETTm1`       | Autoformer release        | `data/long_term_forecast/ETT-small/ETTm1.csv`                |
| `ETTm2`       | Autoformer release        | `data/long_term_forecast/ETT-small/ETTm2.csv`                |
| `exchange_rate` | Autoformer release      | `data/long_term_forecast/exchange_rate/exchange_rate.csv`    |
| `traffic`     | Autoformer release        | `data/long_term_forecast/traffic/traffic.csv`                |
| `weather`     | Autoformer release        | `data/long_term_forecast/weather/weather.csv`                |
| `illness`     | Autoformer release        | `data/long_term_forecast/illness/national_illness.csv`       |
| `ERA5`        | Copernicus CDS            | `data/ERA5/era5_land_merged.csv`                             |
| `EWELD`       | EWELD repository          | `data/EWELD/Weather Data/EWELD_Merged.csv`                   |

The Autoformer release bundles `electricity`, `ETT-small/`, `exchange_rate`,
`traffic`, `weather`, and `illness` in a single archive that's mirrored in
several public locations. ERA5 requires a free CDS account; EWELD has
public CSV exports.

## Reproducing the experiments

All commands assume `cd <bundle root>`.

### 1. Dependency Experiment (synthetic GP)

Convergence of CSig / SigMMD on Gaussian-Process samples with controlled
spatial / temporal correlation structure across five seeds.

```bash
python -u src/experiments/dependency_final.py
```

Self-contained — no external data needed.

### 2. Steepness Ablation (MAD-based adaptive β\*)

Convergence of the censored signature-MMD as the censoring quantile is
relaxed, with β\* set adaptively from the MAD of the training-score
distribution. Produces a single 1×3 PDF.

```bash
python -u src/experiments/ablation_cquants.py
```

Requires `data/cloud/cloud.csv` (bundled) and
`data/long_term_forecast/exchange_rate/exchange_rate.csv` (download).

### 3. TSFM Inference Benchmark

Six foundation models × twelve datasets × four horizons. Outputs a
machine-readable JSON of per-(dataset, horizon, model) metric values plus
a Wins/Losses/Draws summary across SigMMD, CSig(Kernel), CSig(Systemic),
CSig(Sharpe).

```bash
python -u src/experiments/benchmark_fm_multi.py                       # full sweep
python -u src/experiments/benchmark_fm_multi.py --only_dataset cloud  # single dataset
```

Expected wall-clock on a single A100: ≈ 6–10 hours full sweep.

### 4. Speed Comparison (ESK vs Salvi sigkernel, Laplacian static kernel)

Per-sequence wall-time and peak GPU memory for the two backends across
batch sizes {1, 8, 16, 32, 64} on real Exchange and Weather windows.

```bash
python -u src/experiments/speed_memory_comparison.py
```

Requires `pip install -e sigkernel` and the two CSVs above.

### 5. Agent Decision Experiment

Two scenarios — Markowitz mean-variance on `exchange_rate` and newsvendor
capacity provisioning on `cloud` — × four-point sweep over the agent's
risk aversion γ / cost asymmetry α. Reports Spearman ρ and top-1 hit
rate of each metric (CRPS, ES, QL, VS, SigMMD, CSig) versus the oracle
realised-utility ranking, aggregated across seeds {2021…2025}.

```bash
python -u src/experiments/agent_decision_experiment.py
python -u src/experiments/agent_decision_experiment.py --only_scenario exchange
```

## Output locations

All scripts write to `$WRITEABLE_DIR` if set, otherwise to
`<bundle_root>/results/<experiment_name>/`. Structured JSON dumps and
PDFs land alongside the per-run log file.

## Hardware notes

| Experiment                  | Min GPU VRAM | Approx. wall-clock     |
|-----------------------------|--------------|------------------------|
| Dependency                  | 16 GB        | ~30 min × 5 seeds      |
| Steepness ablation          | 16 GB        | ~45 min                |
| Speed comparison            | 24 GB        | ~15 min                |
| TSFM benchmark (full)       | 80 GB (A100) | 6–10 h                 |
| Agent decision (per seed)   | 80 GB        | ~2 h × 5 seeds         |

The TSFM benchmark and agent decision experiment both call all six FMs
sequentially; they need enough headroom to load the largest weights (≈
1.5–3 GB) plus the float64 PDE solver intermediates (≈ 27 GB at H=336
with the horizon-aware micro-batch).

## Third-party code

- `sigkernel/` is the reference signature-kernel implementation by
  Salvi et al., included verbatim under its Apache-2.0 license. Used as
  a baseline only by `speed_memory_comparison.py`.
- The MoiraiMoE adapter in `src/models/adapters.py` wraps the upstream
  Salesforce MoiraiMoE checkpoint via the `uni2ts` library.
- Foundation-model wrappers in `foundation_models/` call HuggingFace /
  upstream packages (`chronos-forecasting`, `uni2ts`, `transformers`,
  `timesfm`).

## License

MIT — see [LICENSE](LICENSE).
