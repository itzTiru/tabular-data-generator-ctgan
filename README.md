# tabular-synthetic-data-generator-ctgan

This repository builds **privacy-aware synthetic tabular data** from a Superstore-style CSV
using **CTGAN** and **TVAE** (SDV), then evaluates quality with similarity, utility, and
privacy metrics.

It is designed as an end-to-end workflow:

- schema validation and leakage-aware preprocessing
- train/test split by time
- multi-model synthetic generation (`ctgan`, `tvae`)
- post-generation hard constraints + deterministic reconstruction
- reporting (`metrics.json`, figures) + Streamlit dashboard for comparison

## Architecture

### Pipeline Orchestration

![Pipeline Orchestration](docs/images/01_pipeline_orchestration.png)

### Preprocessing and Artifacts

![Preprocessing and Artifacts](docs/images/02_preprocessing_artifacts.png)

### Constraints and Reconstruction

![Constraints and Reconstruction](docs/images/03_constraints_reconstruction.png)

### Evaluation Dependencies

![Evaluation Dependencies](docs/images/04_evaluation_dependencies.png)

Mermaid source: `docs/diagrams/`

## Quickstart (UV)

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync
```

## Run

Pipeline:

```powershell
uv run python scripts/run_pipeline.py --config configs/default.yaml --data-path "C:\Users\Tisha\Desktop\tabular-data-generator-ctgan\data\raw\sample_dataset.csv"
```

Dashboard:

```powershell
uv run streamlit run app/streamlit_app.py
```

## Quality

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## Core Logic (Short)

- Time split: train `< 2017-01-01`, test `>= 2017-01-01`
- Train models: CTGAN + TVAE on train only
- Generate scales: `1x`, `5x`, `10x`
- Enforce constraints: clipping, integer casting, discount snapping, ship-mode delay bounds
- Reconstruct deterministic fields: `Country`, `Category`, `Region`, `Order Date`, `Ship Date`
- Evaluate: similarity, utility (TSTR), privacy indicators

## Key Outputs

- Metrics: `reports/metrics.json`
- Figures: `reports/figures/`
- Synthetic data: `data/synthetic/`
- Models: `artifacts/models/`

## Optional Export

```powershell
uv pip compile pyproject.toml -o requirements.txt
```
