# Urban Noise + Synthetic Vibration ML

This repository contains `mlmodel.py`, a script that trains two models on an urban noise dataset:
- RandomForest classifier to predict noise risk level (Low/Medium/High)
- RandomForest regressor to predict the decibel level

It also synthesizes a `vibration_level` feature from existing columns and includes it in the models.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Recommended Python: 3.8+.

## Usage

Run the script with the dataset path and output directory (defaults shown):

```bash
python mlmodel.py --data-path urban_noise_levels.csv --out-dir ml_with_vibration_models --random-state 42
```

Arguments:
- `--data-path` / `-d`: path to the input CSV (default: `urban_noise_levels.csv`)
- `--out-dir` / `-o`: directory to save artifacts (default: `ml_with_vibration_models`)
- `--random-state`: integer seed for reproducibility (default: `42`)

The script prints training metrics and saves the trained models to the output directory as:
- `noise_vibration_classifier.pkl`
- `noise_vibration_regressor.pkl`

It will also print top feature importances to the console and show a histogram of decibel levels (requires a display).

## Notes

- If running on a headless server, set a non-interactive Matplotlib backend before running, for example:

```bash
MPLBACKEND=Agg python mlmodel.py -d urban_noise_levels.csv -o models
```

- The script expects some common columns in the CSV (e.g. `decibel_level`) — feature selection is automatic and will only use columns present in the data.

## Contact

For changes or customizations (different models, hyperparameters, or to persist additional artifacts), edit `mlmodel.py`.
