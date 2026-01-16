# Model Card (Draft)

## Intended use
This model is for educational/portfolio purposes to predict a binary "popular" label from audio features.

## Target definition
`is_popular = 1(popularity >= q)` where `q` is a configurable quantile (default 0.90).

## Metrics
Primary: ROC-AUC  
Secondary: accuracy, precision, recall, F1, confusion matrix

## Limitations
- Popularity is affected by marketing, playlists, and time effects not captured by audio features.
- The quantile-based label is dataset-dependent (not a universal definition of popularity).
