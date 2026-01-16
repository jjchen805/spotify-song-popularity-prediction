## Data

This repo does not include the full raw dataset (to keep the repository small).

### Option A — Use your existing class dataset
1. Place `spotify_dataset.csv` in `data/raw/`
2. Run preprocessing:

```bash
python -m src.data_prep --input data/raw/spotify_dataset.csv --output data/processed/spotify_processed.csv
```

### Option B — Use a different dataset
As long as it contains a `popularity` column plus audio features, you can adapt `src/data_prep.py`.
