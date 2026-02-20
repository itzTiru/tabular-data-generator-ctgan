# tabular-synthetic-data-generator-ctgan

Privacy-aware synthetic tabular data generation for a Superstore-style dataset using **CTGAN** and **TVAE** (SDV), with constraint-based postprocessing, utility/similarity/privacy evaluation, and a Streamlit comparison dashboard.

## What This Repository Does

- Loads a tabular CSV dataset (`data/raw/sample_dataset.csv` by default, override via `--data-path`).
- Preprocesses data into a leakage-reduced modeling table.
- Trains **CTGAN** and **TVAE** on the train split only.
- Generates synthetic data at `1x`, `5x`, and `10x` scales.
- Applies hard constraints and deterministic reconstruction rules.
- Evaluates synthetic quality across similarity, utility (TSTR), and privacy indicators.
- Produces `reports/metrics.json`, figure outputs, and downloadable synthetic CSVs in Streamlit.

## Architecture (Diagrams)

### 1. Pipeline Orchestration

![Pipeline Orchestration](docs/images/01_pipeline_orchestration.png)

### 2. Preprocessing and Artifact Derivation

![Preprocessing and Artifacts](docs/images/02_preprocessing_artifacts.png)

### 3. Constraints and Reconstruction

![Constraints and Reconstruction](docs/images/03_constraints_reconstruction.png)

### 4. Evaluation Dependencies

![Evaluation Dependencies](docs/images/04_evaluation_dependencies.png)

Mermaid source files are in `docs/diagrams/`.

## Quickstart (UV)

1. Install [uv](https://docs.astral.sh/uv/).
2. Sync environment:

```powershell
uv python install 3.11
uv venv --python 3.11
uv sync
```

## Quality Commands

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest -q
```

## Run Pipeline

```powershell
uv run python scripts/run_pipeline.py --config configs/default.yaml --data-path "C:\Users\Tisha\Desktop\tabular-data-generator-ctgan\data\raw\sample_dataset.csv"
```

Notes:
- `--data-path` is optional.
- Default path is set in `configs/default.yaml` (`data/raw/sample_dataset.csv`).

## Run Streamlit Dashboard

```powershell
uv run streamlit run app/streamlit_app.py
```

The dashboard provides:
- model selector (`ctgan` / `tvae`)
- scale selector (`1x` / `5x` / `10x`)
- output file selector from `data/synthetic/`
- similarity, utility, privacy panels
- download button for selected synthetic CSV

## Core Modeling Logic

### Train/Test Split

- Train: `Order Date < 2017-01-01`
- Test: `Order Date >= 2017-01-01`
- Synthesizers are fitted on train only.

### Final Modeling Columns

- `order_day_index`
- `ship_delay_days`
- `Ship Mode`
- `Segment`
- `State`
- `Sub-Category`
- `Sales`
- `Quantity`
- `Discount`
- `Profit`

### Dropped / Non-Modeled Columns

- Dropped leakage/PII-like fields:
  - `Row ID`, `Order ID`, `Customer Name`, `Product Name`, `Customer ID`, `Product ID`
- Not modeled directly (reconstructed or excluded):
  - `Country`, `Category`, `Region`, `City`, `Postal Code`

### Post-Generation Constraints

- `order_day_index` -> integer, clip `[0, max_order_day_index]`
- `ship_delay_days` -> integer, clip `[0, 7]`
- `Quantity` -> integer, clip `[1, 14]`
- `Sales` -> clip `>= 0`
- `Discount` -> snap to nearest allowed value from train set
- ship mode delay bounds:
  - `Same Day`: `0-1`
  - `First Class`: `1-4`
  - `Second Class`: `1-5`
  - `Standard Class`: `3-7`

### Deterministic Reconstruction

- `Order Date = min_order_date + order_day_index`
- `Ship Date = Order Date + ship_delay_days`
- `Country = "United States"`
- `Category` mapped from `Sub-Category` (train-derived)
- `Region` mapped from `State` (train-derived)

## Evaluation Outputs

Saved to:
- `reports/metrics.json`
- `reports/figures/`

Metrics include:
- Similarity:
  - KS statistic
  - Wasserstein distance
  - Jensen-Shannon divergence
  - correlation drift (Frobenius norm)
  - real-vs-synthetic classifier ROC-AUC
- Utility (TSTR):
  - Logistic Regression and Random Forest
  - scenarios: real->real, synthetic->real, real+synthetic->real
  - ROC-AUC and F1
- Privacy indicators:
  - exact duplicate rate
  - QID collision rates (train and test)
  - nearest-neighbor distance quantiles

## Project Layout

```text
app/                    Streamlit dashboard
artifacts/models/       Trained model artifacts
docs/                   Diagrams and technical docs
configs/default.yaml    Pipeline config
scripts/run_pipeline.py CLI entrypoint
src/                    Pipeline modules
tests/                  Unit and smoke tests
data/raw/               Input datasets
data/processed/         Intermediate outputs
data/synthetic/         Final synthetic outputs
reports/                Metrics JSON and figures
```

## Additional Documentation

- Model deep dive: `docs/models_and_training_deep_dive.md`
- Mermaid diagrams index: `docs/diagrams/README.md`
- Commit plan: `docs/commit_strategy_5_day.md`

## Optional Export

```powershell
uv pip compile pyproject.toml -o requirements.txt
```
