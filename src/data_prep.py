from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .config import Config

def mode_as_string(series: pd.Series) -> str:
    """Return a stable string mode for a Series, joining ties to keep deterministic output."""
    try:
        m = series.mode(dropna=True)
        if len(m) == 0:
            return ""
        if len(m) == 1:
            return str(m.iloc[0])
        return "|".join(sorted(map(str, m.tolist())))
    except Exception:
        return str(series.iloc[0]) if len(series) else ""

def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mimics the 'dedupe + aggregate duplicates by track_id' pattern from your notebook."""
    df = df.dropna().copy()
    df = df.loc[~df.duplicated()].copy()

    # cast common boolean-ish columns if present
    for col in ["explicit", "mode"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    if "track_id" in df.columns:
        # Aggregate duplicate tracks by track_id (most fields mode; numeric fields mean)
        agg = {
            "track_name": pd.Series.mode,
            "artists": mode_as_string,
            "album_name": pd.Series.mode,
            "explicit": "max",
            "track_genre": mode_as_string,
            "time_signature": "max",
            "key": "max",
            "mode": "max",
            "popularity": "mean",
            "duration_ms": "mean",
            "danceability": "mean",
            "energy": "mean",
            "loudness": "mean",
            "speechiness": "mean",
            "acousticness": "mean",
            "instrumentalness": "mean",
            "valence": "mean",
            "tempo": "mean",
            "liveness": "mean",
        }
        # only keep keys that exist
        agg = {k: v for k, v in agg.items() if k in df.columns}
        df = df.groupby(["track_id"], as_index=False).agg(agg)

        # Some pandas modes return Series; normalize to string
        for col in ["track_name", "album_name"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: str(x[0]) if isinstance(x, (list, tuple, pd.Series)) else str(x))

    return df

def add_target(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if "popularity" not in df.columns:
        raise ValueError("Expected a 'popularity' column to build the label.")
    q = df["popularity"].quantile(cfg.popularity_quantile)
    df = df.copy()
    df["is_popular"] = (df["popularity"] >= q).astype(int)
    return df

def drop_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = ["Unnamed: 0", "track_id", "album_name", "track_name", "popularity"]
    keep_drop = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=keep_drop)

def preprocess_file(input_path: str | Path, output_path: str | Path, cfg: Config) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    df = pd.read_csv(input_path)

    df = clean_and_aggregate(df)
    df = add_target(df, cfg)
    df = drop_irrelevant(df)

    # explicit to int (after dropping cols)
    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path} (shape={df.shape})")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare Spotify dataset for modeling.")
    p.add_argument("--input", required=True, help="Path to raw spotify_dataset.csv")
    p.add_argument("--output", required=True, help="Path to write processed CSV")
    p.add_argument("--quantile", type=float, default=Config().popularity_quantile, help="Popularity quantile for label")
    return p

def main():
    args = build_argparser().parse_args()
    cfg = Config(popularity_quantile=args.quantile)
    preprocess_file(args.input, args.output, cfg)

if __name__ == "__main__":
    main()
