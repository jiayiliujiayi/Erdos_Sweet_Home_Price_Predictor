> This README was proofread with assistance from ChatGPT (OpenAI).

# Multimodal House Price Prediction (New Jersey)

This repository contains an end-to-end project for predicting New Jersey house sale prices using:

- **Tabular data**: property attributes, location, time on market, macro interest rate  
- **Text data**: listing descriptions, formatted address, nearby schools  
- **Deep learning models**: a custom text encoder and TabNet for tabular and multimodal modeling  

The workflow is implemented as a series of Jupyter notebooks with reusable utilities in `src/`.

> **Note on data in GitHub**  
> Due to size limitations, **not all files under `data` / `data` are uploaded**.  
> Only artifacts smaller than ~1 MB are committed. Larger CSVs, model weights, and text corpora are expected to be generated locally by running the notebooks (or kept outside version control).

---

## 1. Repository Structure

The core project layout is:

```text
.
├── data
│   ├── processed
│   │   ├── multimodal_dl_prep_params.json
│   │   ├── multimodal_features
│   │   │   ├── sentencepiece_bpe
│   │   │   │   ├── sp_bpe_16000.model
│   │   │   │   ├── sp_bpe_16000.vocab
│   │   │   │   └── train_text.txt
│   │   │   ├── text_encoder_spbpe_best_vocab16000.pt
│   │   │   ├── text_vocab.json
│   │   │   ├── txt_features_meta_spbpe_vocab16000_d128.json
│   │   │   ├── txt_features_test_spbpe_vocab16000_d128.csv
│   │   │   ├── txt_features_train_spbpe_vocab16000_d128.csv
│   │   │   └── txt_features_val_spbpe_vocab16000_d128.csv
│   │   ├── multimodal_prep_summary.json
│   │   ├── redfin_nj_sold_2016plus_basic_clean.csv
│   │   ├── test_multimodal.csv
│   │   ├── text_encoder_best.pt
│   │   ├── train_multimodal.csv
│   │   └── val_multimodal.csv
│   └── raw
│       ├── FEDFUNDS_151001_to_251001.csv
│       └── redfin_nj_sold_2015-01-01_to_2025-11-24.csv
├── notebooks
│   ├── 01.query.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_prep.ipynb
│   ├── 04_TabNet_tabularONLY.ipynb
│   ├── 04a_LR_baseline.ipynb
│   ├── 05_text_encoder.ipynb
│   └── 06_TabNet_fusion.ipynb
└── src
    ├── __init__.py
    ├── __pycache__/
    │   ├── __init__.cpython-312.pyc
    │   ├── eda_helpers.cpython-312.pyc
    │   ├── io_utils.cpython-312.pyc
    │   ├── metrics.cpython-312.pyc
    │   ├── preprocessing.cpython-312.pyc
    │   └── targets.cpython-312.pyc
    ├── eda_helpers.py
    ├── io_utils.py
    ├── metrics.py
    ├── paths.py
    ├── preprocessing.py
    ├── query.py
    └── targets.py
````

> In the GitHub repository, **large files in `data/processed` and `data/raw` are not fully tracked**.
> Only small examples (typically `< 1 MB`) are included so that the repository remains lightweight.
> To fully reproduce results, you are expected to regenerate the larger CSVs and model artifacts by running the notebooks.

### Key directories

* `data/`
  Working snapshot of data and model artifacts on the local machine.

  * `data/raw/`

    * `redfin_nj_sold_2015-01-01_to_2025-11-24.csv`: Raw Redfin export for NJ residential sales (full history used in this project).
    * `FEDFUNDS_151001_to_251001.csv`: Daily Fed Funds rate (macro feature), used to align each sale with prevailing interest rates.
  * `data/processed/`

    * `redfin_nj_sold_2016plus_basic_clean.csv`: Cleaned tabular dataset produced by `02_eda.ipynb`.
    * `train_multimodal.csv`, `val_multimodal.csv`, `test_multimodal.csv`: Multimodal splits created in `03_data_prep.ipynb`.
    * `multimodal_prep_summary.json`: JSON summary describing splits, feature lists, and target definitions.
    * `multimodal_dl_prep_params.json`: Parameters used when preparing text features for deep learning.
    * `multimodal_features/`

      * `sentencepiece_bpe/`

        * `sp_bpe_16000.model`, `sp_bpe_16000.vocab`: SentencePiece BPE tokenizer artifacts.
        * `train_text.txt`: Text corpus used to train SentencePiece.
      * `text_vocab.json`: Vocabulary metadata for text tokens.
      * `txt_features_meta_spbpe_vocab16000_d128.json`: Metadata describing text encoder output (e.g., embedding dimension).
      * `txt_features_train_spbpe_vocab16000_d128.csv`, `txt_features_val_spbpe_vocab16000_d128.csv`, `txt_features_test_spbpe_vocab16000_d128.csv`: Precomputed text embeddings for each split.
      * `text_encoder_spbpe_best_vocab16000.pt` / `text_encoder_best.pt`: Trained text encoder weights.

* `notebooks/`

  * `01.query.ipynb`
    Redfin scraping orchestration; uses `src/query.py` to generate raw property data and save it to `data/raw/` (or `data/raw/` in a clean setup).
  * `02_eda.ipynb`
    Exploratory data analysis and cleaning of raw Redfin data; merges Fed Funds rate; produces the main cleaned dataset and `multimodal_prep_summary.json`.
  * `03_data_prep.ipynb`
    Multimodal data preparation:

    * Defines targets (`sold_price`, `log_sold_price = log1p(sold_price)`).
    * Builds unified `description_text` from listing description, formatted address, and nearby schools.
    * (Optionally) selects photos and image paths.
    * Creates train/validation/test splits and saves them under `data/processed/`.
  * `04_TabNet_tabularONLY.ipynb`
    TabNet model using **tabular features only**:

    * Loads train/val/test splits.
    * Performs numeric imputation and categorical encoding for TabNet.
    * Trains a `TabNetRegressor` on `log_sold_price`.
    * Evaluates both in log space and back-transformed dollars.
  * `04a_LR_baseline.ipynb`
    Linear regression baseline on tabular features, using the same splits and target.
  * `05_text_encoder.ipynb`
    Text encoder training:

    * Trains SentencePiece BPE tokenizer on `description_text`.
    * Trains a Transformer-based encoder to predict normalized `log_sold_price`.
    * Exports trained weights and text embeddings to `data/processed/multimodal_features/`.
  * `06_TabNet_fusion.ipynb`
    Multimodal fusion model:

    * Loads tabular splits and corresponding text embeddings.
    * Concatenates tabular features with text embeddings.
    * Trains a TabNet fusion model and compares performance to tabular-only baselines.

* `src/`
  Lightweight Python package for reusable components:

  * `query.py`
    Scraping and query utilities (e.g., date chunking and Redfin/homeharvest wrappers).
  * `eda_helpers.py`
    Project-specific EDA helpers (plotting, missingness, etc.).
  * `preprocessing.py`
    Shared preprocessing utilities (e.g., numeric imputation via `fill_numeric_with_train_median`).
  * `metrics.py`
    Shared regression metrics (RMSE, MAE, R²).
  * `targets.py`
    Target transformation and back-transformation utilities (e.g., `log1p` and `exp1m`).
  * `io_utils.py`
    I/O helpers such as JSON loading and “pick latest file” selectors.
  * `paths.py`
    Centralized helper to infer project-level paths (PROJECT_ROOT, data directories).

> The `__pycache__/` directory is simply Python’s compiled bytecode cache and is not logically part of the project API.

---

## 2. Project Workflow

The analytics and modeling pipeline is designed as a sequence of notebooks:

1. **Scraping & Raw Data Acquisition** – `01.query.ipynb`

   * Configure New Jersey counties and date ranges.
   * Query Redfin via `src/query.py`.
   * Store raw listing/sale data under `data/raw/` (or `data/raw/` in a clean setup).

2. **EDA & Cleaning** – `02_eda.ipynb`

   * Load raw Redfin data and Fed Funds CSV.
   * Normalize types, handle missingness, and apply sensible range filters.
   * Derive `sale_date` and `sale_year`.
   * Merge Fed Funds rate aligned to `sale_date`.
   * Save:

     * `redfin_nj_sold_2016plus_basic_clean.csv`
     * `multimodal_prep_summary.json`

3. **Multimodal Data Preparation** – `03_data_prep.ipynb`

   * Load the cleaned tabular dataset.
   * Define target and log target:

     * `sold_price`
     * `log_sold_price = log1p(sold_price)`
   * Build unified `description_text`.
   * Optionally select a single image per property.
   * Filter to rows with valid target and non-empty text.
   * Split into:

     * `train_multimodal.csv`
     * `val_multimodal.csv`
     * `test_multimodal.csv`
   * Update `multimodal_prep_summary.json` with:

     * Split sizes,
     * Numeric/categorical feature lists,
     * Target definitions.

4. **Tabular-Only Baselines**

   * `04a_LR_baseline.ipynb`:

     * Uses train/val/test splits and `src.preprocessing` / `src.metrics`.
     * Trains linear regression on `log_sold_price`.
   * `04_TabNet_tabularONLY.ipynb`:

     * Encodes categoricals for TabNet (embedding indices).
     * imputes numeric features using train medians.
     * Trains TabNet on `log_sold_price`.
     * Uses `src.targets.backtransform(..., "log1p")` for dollar-space metrics.
     * Extracts global feature importances.

5. **Text Encoder & Multimodal Fusion**

   * `05_text_encoder.ipynb`:

     * Trains SentencePiece BPE and a Transformer encoder on `description_text`.
     * Saves tokenizer, trained encoder weights, and text embeddings.
   * `06_TabNet_fusion.ipynb`:

     * Loads tabular features + text embeddings.
     * Trains a TabNet fusion model on the combined feature space.
     * Compares performance against tabular-only TabNet and linear regression baselines.

---

## 3. Setup and Dependencies

This project assumes:

* **Python** ≥ 3.10
* Core packages (non-exhaustive):

  * `pandas`
  * `numpy`
  * `matplotlib`
  * `scikit-learn`
  * `torch`
  * `pytorch-tabnet`
  * `sentencepiece`
  * `tqdm`
  * `jupyter` / `jupyterlab`

Example environment creation:

```bash
conda create -n houseprice python=3.11
conda activate houseprice

pip install pandas numpy matplotlib scikit-learn torch torchvision torchaudio
pip install pytorch-tabnet sentencepiece tqdm
```

If you are running in **Google Colab**, the notebooks include guarded cells that:

* Mount Google Drive,
* Set `PROJECT_ROOT` to your Drive folder for this repo,
* Add `PROJECT_ROOT` to `sys.path` so `src` can be imported.

---

## 4. Reproducing the Pipeline

Because large CSVs and model weights are not fully committed to GitHub, the recommended approach is:

1. **Clone or sync the repository** to your machine or Google Drive.

2. **Rebuild raw data** (if not already present locally):

   * Option A: Use `data/raw/redfin_nj_sold_2015-01-01_to_2025-11-24.csv` and `FEDFUNDS_*.csv` if they exist on your machine.
   * Option B: Run `01.query.ipynb` to regenerate the Redfin export into `data/raw/`.

3. **Run EDA & cleaning**:

   * Execute `02_eda.ipynb` to generate the cleaned dataset and preparation summary.

4. **Prepare multimodal splits**:

   * Execute `03_data_prep.ipynb` to generate `train_multimodal.csv`, `val_multimodal.csv`, and `test_multimodal.csv` plus an updated `multimodal_prep_summary.json`.

5. **Train baselines**:

   * Run `04a_LR_baseline.ipynb` and `04_TabNet_tabularONLY.ipynb`.

6. **Train text encoder and fusion**:

   * Run `05_text_encoder.ipynb` and then `06_TabNet_fusion.ipynb`.

The notebooks are designed to be idempotent with respect to file outputs: re-running them will typically overwrite or reuse existing artifacts.

---

## 5. Design Notes

* **Target Transform**
  The primary modeling target is `log1p(sold_price)`.
  This stabilizes training and improves performance for high-priced outliers.
  All deep learning models operate in log space, and metrics in dollars are obtained via `exp1m` using `src.targets.backtransform(..., "log1p")`.

* **Feature Management**
  `multimodal_prep_summary.json` serves as the single source of truth for:

  * Target and log-target column names,
  * Numeric and categorical feature lists,
  * Split definitions (size, random seed, and criteria).

* **Modularity**
  Shared utilities in `src/` (preprocessing, metrics, targets, IO, paths) minimize duplication across notebooks and keep the project maintainable.

---

## 6. Data Availability (GitHub vs Zenodo)

Because GitHub is not optimized for large, frequently changing data files, this repository only contains **small example artifacts** under `data` / `data copy`:

- **Only small artifacts (typically `< 1 MB`) are committed** under `data` / `data copy`.
- Large CSVs (e.g., full Redfin exports, full multimodal splits, full text embeddings) and model checkpoints are **excluded** from the GitHub repo.

The **full dataset and derived artifacts** for this project (including large CSVs and text feature files) are archived on **Zenodo**:

- Zenodo record: **[10.5281/zenodo.17861674](10.5281/zenodo.17861674)**
- Contents (mirroring the structure described above):
  - Raw Redfin export (`redfin_nj_sold_2015-01-01_to_2025-11-24.csv`)
  - Cleaned dataset (`redfin_nj_sold_2016plus_basic_clean.csv`)
  - `train_multimodal.csv`, `val_multimodal.csv`, `test_multimodal.csv`
  - Text encoder artifacts (SentencePiece model/vocab, encoder checkpoints)
  - Precomputed text embeddings (`txt_features_*_spbpe_vocab16000_d128.csv`)
  - Associated JSON metadata files

To **fully reproduce the results**:

1. Download the Zenodo archive.
2. Extract it so that the directory structure under `data` / `data copy` matches the layout described in this README.
3. Run the notebooks as documented (they will reuse the downloaded artifacts instead of recreating everything from scratch).

If you prefer not to use Zenodo, you can still regenerate the larger artifacts locally by running the notebooks end-to-end, but using the Zenodo snapshot is the recommended path for exact reproducibility.


