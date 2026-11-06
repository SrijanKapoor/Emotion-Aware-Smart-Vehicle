import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DATA_FILES = (
    "summary.csv",
    "per_emotion.csv",
    "action_dist.csv",
    "eval_episodes.csv",
    "train_curves.csv",
    "classifier_metrics.json",
    "stats.txt",
    "policy_C.csv",
    "qvalues_highrisk.csv",
    "stabilisation.csv",
    "low_conf.csv",
    "confusion_matrix.csv",
    "sweep_rows_paper.tex",
)
INFORMATIONAL_FILES = ("latency.csv", "environment.json")
FIGURE_RENAMES = {
    "fig_confusion_matrix.png": "emotion_classifier_confusion_matrix.png",
    "fig_train_objective.png": "training_return_curves.png",
    "fig_train_alerts.png": "training_alert_escalation_curves.png",
    "fig_eval_alerts.png": "evaluation_mean_alert_escalations.png",
    "fig_relative_change.png": "alert_change_proposed_vs_safety_only.png",
    "fig_return_distribution.png": "safety_return_distribution.png",
    "fig_alerts_by_emotion.png": "alert_rate_by_driver_emotion.png",
    "fig_train_envreturn.png": "training_safety_return_curves.png",
}
FIGURES = tuple(FIGURE_RENAMES.values())
PACKAGE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_directory(runs_root):
    runs_root.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = runs_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        try:
            run_dir.mkdir()
        except FileExistsError:
            time.sleep(1)
            continue
        (run_dir / "results").mkdir()
        (run_dir / "figures").mkdir()
        return run_dir


def run_logged_step(command, cwd, log_path, label):
    command_line = subprocess.list2cmdline([str(part) for part in command])
    print(f"\n{label}: {command_line}", flush=True)
    child_environment = os.environ.copy()
    child_environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout
    log_path.write_text(output, encoding="utf-8")
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        print(
            f"ERROR: {label} failed with return code {completed.returncode}. "
            f"Full output: {log_path}",
            file=sys.stderr,
            flush=True,
        )
    return command_line, completed.returncode


def rename_and_validate_figures(figures_dir):
    missing_sources = [
        source for source in FIGURE_RENAMES if not (figures_dir / source).is_file()
    ]
    if missing_sources:
        raise RuntimeError(
            "Figure generation did not produce expected source file(s): "
            + ", ".join(missing_sources)
        )
    for source, target in FIGURE_RENAMES.items():
        (figures_dir / source).replace(figures_dir / target)

    missing = [
        name
        for name in FIGURES
        if not (figures_dir / name).is_file() or (figures_dir / name).stat().st_size == 0
    ]
    unexpected = sorted(
        path.name
        for path in figures_dir.glob("*.png")
        if path.name not in FIGURES
    )
    if missing:
        print("Missing or empty figures: " + ", ".join(missing), file=sys.stderr)
    if unexpected:
        print("Unexpected PNG files: " + ", ".join(unexpected), file=sys.stderr)
    if missing or unexpected:
        raise RuntimeError("The figures folder does not contain exactly the required 8 PNGs.")
    return len(FIGURES)


def compare_file(previous, current, filename):
    previous_file = previous / filename
    current_file = current / filename
    if not previous_file.is_file() or not current_file.is_file():
        missing = []
        if not previous_file.is_file():
            missing.append("previous run")
        if not current_file.is_file():
            missing.append("current run")
        return "DIFFERENT", f"missing in {', '.join(missing)}", None, None
    previous_hash = sha256_file(previous_file)
    current_hash = sha256_file(current_file)
    if previous_hash == current_hash:
        return "IDENTICAL", "", previous_hash, current_hash
    return "DIFFERENT", "", previous_hash, current_hash


def reproducibility_report(previous, current, report_path):
    lines = ["# Reproducibility report", ""]
    if previous is None:
        lines.extend(
            [
                "first run, nothing to compare",
                "",
                "No earlier run folder was available.",
            ]
        )
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("first run, nothing to compare")
        return "N/A (first run, nothing to compare)"

    lines.extend(
        [
            f"Compared with: `{previous.relative_to(ROOT)}`",
            f"Current run: `{current.relative_to(ROOT)}`",
            "",
            "## Data files",
            "",
            "| File | Result | Previous SHA-256 | Current SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    different = []
    for filename in DATA_FILES:
        status, detail, old_hash, new_hash = compare_file(
            previous / "results", current / "results", filename
        )
        if status == "DIFFERENT":
            different.append(filename)
        old_value = old_hash or detail
        new_value = new_hash or detail
        lines.append(f"| `{filename}` | {status} | {old_value} | {new_value} |")
        print(f"DATA {filename}: {status}" + (f" ({detail})" if detail else ""))

    lines.extend(
        [
            "",
            "## Informational timing files",
            "",
            "| File | Result | Previous SHA-256 | Current SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for filename in INFORMATIONAL_FILES:
        status, detail, old_hash, new_hash = compare_file(
            previous / "results", current / "results", filename
        )
        lines.append(
            f"| `{filename}` | informational: {status} | "
            f"{old_hash or detail} | {new_hash or detail} |"
        )
        print(f"INFORMATIONAL {filename}: {status}" + (f" ({detail})" if detail else ""))

    figure_differences = []
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "| File | SHA-256 result | Previous SHA-256 | Current SHA-256 |",
            "|---|---|---|---|",
        ]
    )
    for filename in FIGURES:
        status, detail, old_hash, new_hash = compare_file(
            previous / "figures", current / "figures", filename
        )
        if status == "DIFFERENT":
            figure_differences.append(filename)
        lines.append(
            f"| `{filename}` | {status} | {old_hash or detail} | {new_hash or detail} |"
        )
        print(
            f"FIGURE {filename}: SHA-256 {status}"
            + (f" ({detail})" if detail else "")
        )

    data_identical = not different
    if data_identical and figure_differences:
        figure_note = "figures differ only in file bytes, data identical"
        lines.extend(["", figure_note])
        print(figure_note)
    if data_identical:
        result = "REPRODUCIBLE: YES"
    else:
        result = "REPRODUCIBLE: NO (different data files: " + ", ".join(different) + ")"
    lines.extend(["", result])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result)
    return result


def collect_file_hashes(directory):
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run_one(csv_path, workers, previous_run):
    started_at = datetime.now().astimezone()
    started_clock = time.perf_counter()
    runs_root = ROOT / "runs"
    run_dir = new_run_directory(runs_root)
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    relative_results = results_dir.relative_to(ROOT)
    relative_figures = figures_dir.relative_to(ROOT)

    experiment_command = [
        sys.executable,
        str(ROOT / "run_experiments.py"),
        "--csv",
        str(csv_path),
        "--out",
        str(relative_results),
        "--sweep",
        "--workers",
        str(workers),
    ]
    experiment_line, return_code = run_logged_step(
        experiment_command, ROOT, run_dir / "console.txt", "Step 1/2"
    )
    if return_code != 0:
        return None

    figures_command = [
        sys.executable,
        str(ROOT / "make_paper_figures.py"),
        "--results",
        str(relative_results),
        "--figdir",
        str(relative_figures),
    ]
    figures_line, return_code = run_logged_step(
        figures_command,
        ROOT,
        run_dir / "figures_console.txt",
        "Step 2/2",
    )
    if return_code != 0:
        return None

    try:
        figure_count = rename_and_validate_figures(figures_dir)
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return None

    latest_dir = ROOT / "latest_figures"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for filename in FIGURES:
        shutil.copy2(figures_dir / filename, latest_dir / filename)

    report_result = reproducibility_report(
        previous_run, run_dir, run_dir / "reproducibility_report.md"
    )

    package_versions = {
        package: importlib.metadata.version(distribution)
        for package, distribution in PACKAGE_DISTRIBUTIONS.items()
    }
    file_hashes = {
        "results": collect_file_hashes(results_dir),
        "figures": collect_file_hashes(figures_dir),
    }
    ended_at = datetime.now().astimezone()
    total_seconds = time.perf_counter() - started_clock
    manifest = {
        "start_time": started_at.isoformat(timespec="seconds"),
        "end_time": ended_at.isoformat(timespec="seconds"),
        "total_seconds": round(total_seconds, 3),
        "commands": [experiment_line, figures_line],
        "csv_path": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "script_sha256": {
            "run_experiments.py": sha256_file(ROOT / "run_experiments.py"),
            "make_paper_figures.py": sha256_file(ROOT / "make_paper_figures.py"),
            "run_all.py": sha256_file(Path(__file__).resolve()),
        },
        "python_version": sys.version,
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "package_versions": package_versions,
        "file_sha256": file_hashes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\nRun folder: {run_dir.relative_to(ROOT)}")
    print(f"Total time: {total_seconds:.2f} seconds")
    print(f"Figures produced: {figure_count}")
    print(report_result)
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run the experiment and generate the paper figures."
    )
    parser.add_argument("--csv", required=True, help="Input driver emotion CSV")
    parser.add_argument("--runs", type=int, default=1, help="Number of full runs (default: 1)")
    parser.add_argument(
        "--workers", type=int, default=4, help="Experiment worker processes (default: 4)"
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"CSV file not found: {csv_path}")
    for script in ("run_experiments.py", "make_paper_figures.py"):
        if not (ROOT / script).is_file():
            parser.error(f"Required script not found: {ROOT / script}")

    earlier_runs = sorted(
        path
        for path in (ROOT / "runs").glob("run_*")
        if path.is_dir()
    ) if (ROOT / "runs").is_dir() else []
    previous_run = earlier_runs[-1] if earlier_runs else None
    for _ in range(args.runs):
        run_dir = run_one(csv_path, args.workers, previous_run)
        if run_dir is None:
            return 1
        previous_run = run_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
