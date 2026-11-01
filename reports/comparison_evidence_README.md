# Model-comparison evidence

The [completed results notebook](../notebooks/instant-sprint02-project01-oral-disease.ipynb) preserves the full model-comparison outputs, charts and dataset-inspection images.

The executed notebook is preserved as `../notebooks/instant-sprint02-project01-oral-disease.ipynb`. It includes the final five-model comparison, epoch logs for ten training trials, validation curves, per-class test metrics, and test confusion-matrix figures. The model checkpoint files are separate artifacts and are not embedded in the notebook.

`all_trial_epoch_history.csv` extracts every recorded epoch from the saved notebook output. `validation_trial_summary_from_notebook.csv` reconstructs each trial's best validation epoch using the trainer's improvement tolerance. These are derived from recorded logs, not a rerun or fabricated observations. Summed logged epoch durations exclude some startup/final-report overhead and are not the runner's full training_seconds field.

The original final comparison remains in `../models/fine-tuned/model_comparison.csv`. The original `all_validation_trials.csv`, split manifest, and machine-readable per-class/confusion-matrix/test-prediction files should also be retained from Kaggle when available. Those CSVs preserve exact data and full precision beyond notebook displays.

Selection was based on validation macro-F1 before test evaluation. Results describe this source and split; patient independence and clinical label correctness are unverified.
