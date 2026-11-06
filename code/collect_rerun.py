#!/usr/bin/env python3
"""
collect_rerun.py -- one command that installs the few missing packages, runs
run_experiments.py, and writes everything needed for the paper into ONE text file
(rerun_bundle.txt) plus a zip of the whole results folder.

It never edits any existing file. It invents nothing: every value is read from a
command or a file; anything it cannot find is written as NOT FOUND.

Usage (from the project folder, with run_experiments.py and the CSV next to it):
    python collect_rerun.py --csv driver_emotion_balanced_15k.csv
Options:
    --out rerun_out      results folder (default rerun_out)
    --workers 4          parallel worker processes
    --skip-run           only collect (use when rerun_out already exists)
"""
import argparse
import ctypes
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import zipfile

PKGS = ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "joblib"]
NEED = ["pandas", "matplotlib", "joblib"]
BUNDLE_FILES = ["results_report.md", "summary.csv", "per_emotion.csv", "action_dist.csv",
                "stats.txt", "classifier_metrics.json", "qvalues_highrisk.csv", "policy_C.csv",
                "stabilisation.csv", "latency.csv", "hyperparams.json", "environment.json",
                "tables.tex", "sweep_rows.tex"]


def run_cmd(cmd, timeout=120):
    """Run a command, return (returncode, combined output). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:                                   # missing tool, timeout, ...
        return -1, f"NOT FOUND ({type(e).__name__}: {e})"


def pkg_versions():
    out = {}
    for p in PKGS:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "NOT INSTALLED"
    return out


def ram_gib():
    try:
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong),
                            ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong),
                            ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
            m = MS()
            m.l = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return f"{m.tp / 2**30:.2f} GiB total"
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
        return f"{kb / 2**20:.2f} GiB total"
    except Exception as e:
        return f"NOT FOUND ({e})"


def cpu_info():
    if os.name == "nt":
        rc, o = run_cmd(["powershell", "-NoProfile", "-Command",
                         "Get-CimInstance Win32_Processor | ForEach-Object "
                         "{ $_.Name + ' | cores=' + $_.NumberOfCores + ' | logical=' + $_.NumberOfLogicalProcessors }"])
        if rc == 0 and o:
            return o
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip() + f" | logical={os.cpu_count()}"
    except Exception:
        pass
    return f"{platform.processor() or 'NOT FOUND'} | logical={os.cpu_count()}"


def os_info():
    if os.name == "nt":
        rc, o = run_cmd(["powershell", "-NoProfile", "-Command",
                         "$o = Get-CimInstance Win32_OperatingSystem; "
                         "$o.Caption + ' | version ' + $o.Version + ' | build ' + $o.BuildNumber"])
        if rc == 0 and o:
            return o
    return platform.platform()


def gpu_info():
    rc, o = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                     "--format=csv,noheader"])
    return o if rc == 0 else ("NOT FOUND (nvidia-smi unavailable)" if o.startswith("NOT FOUND") else "NOT FOUND (" + o[:200] + ")")


def dir_size_mb(path):
    tot = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot / 2**20


def file_size_mb(path):
    return f"{os.path.getsize(path) / 2**20:.3f} MB" if os.path.exists(path) else "NOT FOUND"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="rerun_out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--script", default="run_experiments.py")
    ap.add_argument("--skip-run", action="store_true")
    a = ap.parse_args()

    here = os.getcwd()
    L = []

    def H(t):
        L.append("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

    if not os.path.exists(a.csv):
        sys.exit(f"CSV not found: {a.csv}")
    if not os.path.exists(a.script):
        sys.exit(f"Script not found: {a.script}")

    # ---- 1 environment (before any install) -------------------------------------
    before = pkg_versions()
    H("1. ENVIRONMENT")
    L.append(f"CPU: {cpu_info()}")
    L.append(f"RAM: {ram_gib()}")
    L.append(f"OS: {os_info()}")
    L.append(f"GPU: {gpu_info()}")
    L.append(f"Python: {sys.version}")
    L.append(f"Interpreter: {sys.executable}")

    # ---- 2 install missing packages ---------------------------------------------
    H("2. PACKAGE INSTALL")
    missing = [p for p in NEED if before[p] == "NOT INSTALLED"]
    if missing:
        print("Installing:", missing)
        rc, o = run_cmd([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                         *missing], timeout=900)
        L.append(f"pip install {' '.join(missing)} -> return code {rc}")
        L.append("pip output (last 25 lines):\n" + "\n".join(o.splitlines()[-25:]))
    else:
        L.append("All needed packages already installed; nothing to install.")
    after = pkg_versions()
    L.append("Versions before -> after:")
    for p in PKGS:
        flag = "" if before[p] == after[p] else "   <-- CHANGED"
        L.append(f"  {p}: {before[p]} -> {after[p]}{flag}")
    for p in ("numpy", "scipy", "scikit-learn"):
        if before[p] != after[p] and before[p] != "NOT INSTALLED":
            L.append(f"WARNING: {p} changed during install; report this.")

    # ---- 3 run -------------------------------------------------------------------
    H("3. RUN")
    runtime, rc = None, None
    if a.skip_run:
        L.append("Run skipped (--skip-run).")
    else:
        os.makedirs(a.out, exist_ok=True)
        cmd = [sys.executable, a.script, "--csv", a.csv, "--out", a.out, "--sweep",
               "--workers", str(a.workers)]
        L.append("Command: " + " ".join(cmd))
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        t0 = time.perf_counter()
        tail = []
        with open(os.path.join(a.out, "console.txt"), "w", encoding="utf-8") as log:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env=env)
            for line in p.stdout:
                print(line, end="")
                log.write(line)
                tail.append(line.rstrip())
                tail = tail[-60:]
            rc = p.wait()
        runtime = time.perf_counter() - t0
        L.append(f"Return code: {rc}")
        L.append(f"Wall-clock time: {runtime:.1f} s")
        if rc != 0:
            L.append("RUN FAILED. Last 60 lines of output:\n" + "\n".join(tail))

    # ---- 4 sizes -----------------------------------------------------------------
    H("4. FILE SIZES")
    L.append(f"classifier.joblib: {file_size_mb(os.path.join(a.out, 'classifier.joblib'))}")
    L.append(f"q_tables.npz: {file_size_mb(os.path.join(a.out, 'q_tables.npz'))}")
    L.append("results folder total: " +
             (f"{dir_size_mb(a.out):.2f} MB" if os.path.isdir(a.out) else "NOT FOUND"))

    # ---- 5 GPU usage check -------------------------------------------------------
    H("5. GPU USAGE CHECK")
    hits = []
    with open(a.script, encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if re.match(r"\s*(import|from)\s+(torch|tensorflow|cupy|numba|jax)\b", line):
                hits.append(f"{a.script}:{i}: {line.strip()}")
    L.append("GPU-library imports in " + a.script + ": " + ("; ".join(hits) if hits else "none"))

    # ---- 6 dataset provenance ----------------------------------------------------
    H("6. DATASET PROVENANCE (project .py/.ipynb/.md search)")
    pat = re.compile(r"driver_emotion_balanced|stress_index|voice_?stress|lateral_?accel", re.I)
    skip = {"rerun_out", ".venv", "venv", "__pycache__", ".git", "site-packages", "node_modules"}
    found = 0
    for root, dirs, files in os.walk(here):
        dirs[:] = [d for d in dirs if d not in skip]
        for fn in files:
            if fn.endswith((".py", ".ipynb", ".md", ".tex")) and fn not in (
                    "collect_rerun.py", "run_experiments.py"):
                path = os.path.join(root, fn)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pat.search(line):
                                L.append(f"{os.path.relpath(path, here)}:{i}: {line.strip()[:220]}")
                                found += 1
                                if found >= 150:
                                    break
                except Exception:
                    pass
            if found >= 150:
                break
        if found >= 150:
            L.append("(truncated at 150 matching lines)")
            break
    if not found:
        L.append("NOT FOUND (no project file mentions these names)")

    # ---- 7 bundle files ----------------------------------------------------------
    H("7. RESULT FILES")
    for fn in BUNDLE_FILES:
        path = os.path.join(a.out, fn)
        L.append(f"\n----- {fn} -----")
        if os.path.exists(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                L.append(f.read().rstrip())
        else:
            L.append("NOT FOUND")
    L.append("\n----- PNG files -----")
    pngs = sorted(f for f in os.listdir(a.out) if f.endswith(".png")) if os.path.isdir(a.out) else []
    L.append("\n".join(pngs) if pngs else "NOT FOUND")

    with open("rerun_bundle.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    zip_path = "rerun_out.zip"
    if os.path.isdir(a.out):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for r, _, fs in os.walk(a.out):
                for fn in fs:
                    full = os.path.join(r, fn)
                    z.write(full, os.path.relpath(full, os.path.dirname(a.out) or "."))
    print("\nDONE.")
    print("Send me these two files:")
    print("  ", os.path.abspath("rerun_bundle.txt"))
    print("  ", os.path.abspath(zip_path))
    nf = [ln for ln in L if "NOT FOUND" in ln or "RUN FAILED" in ln or "WARNING" in ln]
    print("Items to look at:" if nf else "No NOT FOUND items.")
    for ln in nf[:10]:
        print("  -", ln[:160])


if __name__ == "__main__":
    main()
