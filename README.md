# Climate Stance Detection on Social Media

NLP module research project — extending Lab 4 (week 4) into a full report.

## Research question
*Does a domain-adapted transformer (ClimateBERT) outperform general-purpose pre-trained models (DistilBERT) for climate stance detection on social media, and what linguistic features (hashtags, emojis, tweet length, sarcasm signals) drive the differences?*

## Models
| # | Model | Role |
|---|---|---|
| 0 | Majority class | Sanity baseline |
| 1 | TF-IDF + Logistic Regression | Classical baseline |
| 2 | BiLSTM + GloVe | Pre-transformer neural baseline |
| 3 | Lab's from-scratch transformer | Lab reference |
| 4 | Fine-tuned DistilBERT | Pitched — general pre-trained |
| 5 | Fine-tuned ClimateBERT | Pitched — domain-adapted |

## Folder layout
```
NLP_Project/
├── data/
│   ├── raw/                  # climate_data.csv (100K tweets, balanced)
│   └── processed/            # train/val/test parquet splits
├── notebooks/                # numbered, run in order
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_bilstm_glove.ipynb
│   ├── 04_lab_transformer.ipynb
│   └── 07_error_analysis.ipynb
├── colab/                    # GPU-only notebooks (run on Colab)
│   ├── 05_distilbert.ipynb
│   └── 06_climatebert.ipynb
├── src/                      # shared utilities
├── results/
│   ├── metrics.csv           # all model results in one table
│   └── figures/
├── report/                   # 3000-word report draft
└── requirements.txt
```

## Run order

1. **Mac (already done)**: notebooks `01` (EDA + splits) and `02` (majority + TF-IDF baselines).
2. **School Windows PC with RTX 3070**: clone the repo, install deps, run `scripts/train_all.py`. Trains all four neural models in ~25 min total. Commit and push results back.
3. **Mac**: pull, run notebook `07` to aggregate everything for the report.

The Mac can't train the neural models locally — known TensorFlow 2.21 + pandas 3.0 + macOS abseil deadlock.

## Step 2 — on the school Windows PC

```powershell
git clone <your-github-url> NLP_Project
cd NLP_Project
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts\train_all.py
```

Then push everything back:

```powershell
git add results/
git commit -m "Trained 4 models on RTX 3070"
git push
```

(The `.gitignore` already excludes the huge model weight folders. Only the small `predictions/*.parquet`, `metrics.csv`, and figure PNGs get pushed.)

## Step 3 — back on the Mac

```bash
git pull
source venv/bin/activate
jupyter nbconvert --to notebook --execute --inplace notebooks/07_error_analysis.ipynb
```

This regenerates `results/figures/analysis/` and `results/summary_table.csv` — the figures you'll embed in the report.
