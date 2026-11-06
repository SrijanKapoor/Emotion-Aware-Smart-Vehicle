# Emotion-Aware Smart Vehicle Control with Human-in-the-Loop Reinforcement Learning

This repository contains a simulation study of emotion-aware driver alerts. Its primary experiment compares four tabular Q-learning configurations using simulated vehicle dynamics and a classifier trained on the included labeled dataset. The code does not control a real vehicle and has not been validated for road use.

## Reproduce the Main Experiment

The recorded environment used Python 3.13.7. From the repository root, create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the end-to-end pipeline:

```powershell
python run_all.py --csv path/to/authorized_driver_dataset.csv
```

For two back-to-back runs with a hash comparison, use:

```powershell
python run_all.py --csv path/to/authorized_driver_dataset.csv --runs 2 --workers 4
```

Each run writes a timestamped folder under `runs/` with result tables, figures, logs, a manifest, and a reproducibility report. In a fresh checkout, the second run is the first one that can be compared with a prior run. Timing and machine information are recorded for context and are not expected to match across systems.

## Experiment Design

`run_experiments.py` trains a logistic-regression classifier on a stratified train/validation/test split and evaluates four Q-learning configurations: safety-only, reward-only, emotion-aware state and reward, and emotion-aware state only. The simulator consumes classifier predictions from held-out rows. `make_paper_figures.py` plots values from the logged results.

The tracked snapshot under `results/reference-run/` contains the result tables and run manifest from the second of two clean release-copy runs. On 2026-11-06, all 13 data outputs and all eight paper figures were byte-identical across those runs; timing and environment files differed as expected. Run the two-run command above to repeat this check on your machine.

For paper claims, use the result files from a specific run and report the metric definition, configuration, and seeds. Supplementary scripts in `code/` include prototypes and historical presentation utilities; the root-level pipeline above is the authoritative experiment for this repository. Do not treat legacy presentation figures or old summary text as experimental evidence.

## Data and Scope

The raw dataset is intentionally not included in this repository. To rerun the experiments, obtain an authorized copy separately and pass its path to `run_all.py`. The expected schema uses `driver_state` as the emotion label and sensor/context features such as heart rate, steering variability, traffic density, and weather condition. The reference results are simulator-based and do not establish real-world emotion-recognition accuracy, human-subject outcomes, or vehicle safety.

The reference result tables contain aggregate experiment outputs. The held-out feature pool (`pool.npz`) is excluded because it contains rows derived from the input dataset.

## Project Map

- `run_all.py`: runs experiments, creates figures, and records a manifest.
- `run_experiments.py`: trains the classifier and evaluates the Q-learning configurations.
- `make_paper_figures.py`: generates publication figures from logged values.
- `requirements.txt`: pinned dependencies for the main experiment pipeline.
- `code/`: supplementary project modules, including earlier RL demos, ML integration, tests, and optional CARLA code.
- `results/reference-run/`: one execution's result tables, model artifacts, and manifest.
- `figures/reference-run/`: figures produced from the reference run.

The optional CARLA modules require a separately installed CARLA server and a compatible CARLA Python API. They are not required to run the main experiment.

## License and Citation

No code license, dataset license, or preferred citation is included yet. Add the appropriate licensing and citation details once authorship, source, and redistribution terms have been confirmed.