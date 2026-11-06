#!/usr/bin/env python3
"""
run_experiments.py -- reproducible experiments for the emotion-aware alert paper.

Everything reported in the paper must come from files this script writes.
No random generator is used to "draw" results; figures are plotted from logs.

Pipeline
  1. Train a Logistic Regression emotion classifier on the CSV with a stratified
     70/15/15 split (scaler + model fitted on the training part only) and report
     held-out test metrics.
  2. Simulate the driving task. Each step the *true* driver emotion is a hidden
     Markov state; the system sees the classifier's prediction for a real
     held-out sensor row of that emotion class (validation+test rows only, never
     training rows), so classifier errors propagate into the RL state.
  3. Train/evaluate tabular Q-learning agents (4 configurations, 3 seeds,
     2 reward settings) with common random numbers, so A/B/C/D see the same
     exogenous noise in evaluation episode i (valid paired comparison).
  4. Log per-episode results, per-emotion behaviour, safety indicators,
     Q-tables, policies, latency, stabilisation episodes, statistics, tables
     (LaTeX) and figures.

Usage
  pip install numpy pandas scipy scikit-learn matplotlib joblib
  python run_experiments.py --csv driver_emotion_balanced_15k.csv --out results
  python run_experiments.py --csv ... --out results --sweep --workers 4
"""
import argparse
import json
import math
import os
import platform
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Constants (all of these belong in the paper's experimental-setup section)
# --------------------------------------------------------------------------- #
EMO = ["Calm", "Stressed", "Fatigued", "Overloaded"]          # ids 0..3
LABEL2ID = {n: i for i, n in enumerate(EMO)}
ACTIONS = ["Low", "Medium", "High", "Assistance"]             # ids 0..3
SPEED = ["Low", "Medium", "High"]
OBST = ["Near", "Far"]                                         # obstacle 0 = Near

ALPHA, GAMMA = 0.1, 0.9
EPS0, EPS_MIN, EPS_DECAY = 1.0, 0.01, 0.995
TRAIN_EPISODES, STEPS, EVAL_EPISODES = 1000, 100, 100
SEEDS = [42, 123, 999]
SWEEP_SEEDS = [42, 123, 999, 7, 11]
MIN_CONF = 0.6

# Simulator dynamics
BASE_CLEAR = 0.10                    # chance a near obstacle clears with no alert
ALERT_EFF = [0.0, 0.15, 0.30, 0.55]  # extra clearing chance per action
EMO_MULT = [1.0, 0.8, 0.6, 0.4]      # how well a driver in emotion e reacts to an alert (a=1,2)
NEAR_BASE, NEAR_SPEED = 0.10, 0.06   # chance a far obstacle becomes near = base + speed*sv
ASSIST_SLOW = 0.5                    # assistance lowers speed by one level with this prob.
STAY, STAY_OVER = 0.92, 0.97         # emotion persistence (STAY_OVER: overloaded driver + high alert/assist)
OTHERS = [[x for x in range(4) if x != e] for e in range(4)]

# Reward settings: 'paper' = rewards exactly as in the paper; 'cost' adds a per-step
# escalation cost to EVERY configuration (so the baseline is also alert-aware).
ALERT_COST = [0.0, -0.2, -0.4, -0.6]
SETTINGS = {"paper": False, "cost": True}

# Acceptable actions per TRUE emotion (used for "acceptable-action agreement").
# >>> THESE ARE DEFAULTS. Review them and edit before using the metric in the paper. <<<
ACCEPT = {0: {0}, 1: {1, 2}, 2: {3}, 3: {0, 1}}

# A: (speed,obstacle) + safety reward | B: same state + safety+human reward
# C: + predicted emotion in state + safety+human reward  (proposed)
# D: + predicted emotion in state, safety reward only  (isolates state augmentation)
CONFIGS = {
    "A": dict(obs="none", human=False),
    "B": dict(obs="none", human=True),
    "C": dict(obs="pred", human=True),
    "D": dict(obs="pred", human=False),
}
CFG_NAMES = {"A": "Safety-only", "B": "Reward-only", "C": "Proposed",
             "D": "State-only"}


def r_human(e, a):
    if e == 3 and a >= 2:
        return -2.0
    if e == 2 and a == 3:
        return 2.0
    if e == 0 and a == 0:
        return 1.0
    return 0.0


HUM = [[r_human(e, a) for a in range(4)] for e in range(4)]

# --------------------------------------------------------------------------- #
# Sensor pool (real classifier predictions on held-out rows)
# --------------------------------------------------------------------------- #
POOL = None


def init_pool(path):
    global POOL
    d = np.load(path)
    pred, conf, true = d["pred"], d["conf"], d["true"]
    POOL = [(pred[true == k].tolist(), conf[true == k].tolist()) for k in range(4)]


# --------------------------------------------------------------------------- #
# Simulator + Q-learning
# --------------------------------------------------------------------------- #
def dyn_step(sv, so, e, a, u):
    """Environment transition. u = 5 uniforms (exogenous noise)."""
    if so == 0:
        pc = BASE_CLEAR + ALERT_EFF[a] * (1.0 if a == 3 else EMO_MULT[e])
        so2 = 1 if u[0] < pc else 0
    else:
        so2 = 0 if u[0] < NEAR_BASE + NEAR_SPEED * sv else 1
    if a == 3 and u[4] < ASSIST_SLOW:
        sv2 = sv - 1 if sv > 0 else 0
    elif u[1] < 0.2:
        sv2 = sv + 1 if sv < 2 else 2
    elif u[1] < 0.4:
        sv2 = sv - 1 if sv > 0 else 0
    else:
        sv2 = sv
    stay = STAY_OVER if (e == 3 and a >= 2) else STAY
    if u[2] < stay:
        e2 = e
    else:
        j = int((u[2] - stay) / (1.0 - stay) * 3.0)
        e2 = OTHERS[e][j if j < 3 else 2]
    return sv2, so2, e2


def new_det():
    return dict(steps=0, steps_true=[0] * 4, esc_true=[0] * 4, steps_pred=[0] * 4,
                esc_pred=[0] * 4, act_true=[[0] * 4 for _ in range(4)],
                act_pred=[[0] * 4 for _ in range(4)], hpos_true=[0] * 4,
                accept=[0] * 4, mismatch=0, lowconf=0, lowconf_act=[0] * 4,
                visits=[0] * 24, unsafe_dec=0, miss=0, assist_u=0, high_u=0,
                unsafe_out=0, conf_sum=0.0, lowconf_conf_sum=0.0)


def merge_det(ds):
    out = new_det()
    for d in ds:
        for k, v in d.items():
            if isinstance(v, list):
                if v and isinstance(v[0], list):
                    for i in range(len(v)):
                        for j in range(len(v[i])):
                            out[k][i][j] += v[i][j]
                else:
                    for i in range(len(v)):
                        out[k][i] += v[i]
            else:
                out[k] += v
    return out


def run_episode(Q, cfg, rnd, eps, alpha, gamma, U, learn, cost_on, det):
    obs_mode, use_h = cfg["obs"], cfg["human"]
    P = POOL
    u = U[0]
    sv = int(u[0] * 3)
    so = 0 if u[1] < 0.3 else 1
    e = int(u[2] * 4)
    pl, cl = P[e]
    k = int(u[3] * len(pl))
    eh, ch = pl[k], cl[k]
    esc = unsafe_dec = miss = assist_u = unsafe_out = 0
    ret_env = ret_hh = ret_ht = ret_c = ret_tr = 0.0
    for t in range(STEPS):
        u = U[t + 1]
        oe = 0 if obs_mode == "none" else (eh if obs_mode == "pred" else e)
        s = (sv * 2 + so) * 4 + oe
        q = Q[s]
        if learn:
            if rnd.random() < eps:
                a = rnd.randrange(4)
            else:
                m = max(q)
                best = [i for i in (0, 1, 2, 3) if q[i] == m]
                a = best[0] if len(best) == 1 else best[rnd.randrange(len(best))]
        else:
            a = q.index(max(q))                      # ties -> lowest action id
        sv2, so2, e2 = dyn_step(sv, so, e, a, u)
        pl, cl = P[e2]
        k = int(u[3] * len(pl))
        eh2, ch2 = pl[k], cl[k]
        r_env = -3.0 if (sv2 == 2 and so2 == 0) else 1.0
        hh, ht = HUM[eh][a], HUM[e][a]               # decision-time emotion
        c = ALERT_COST[a]
        h_used = ht if obs_mode == "true" else hh
        r = r_env + (h_used if use_h else 0.0) + (c if cost_on else 0.0)
        if learn:
            oe2 = 0 if obs_mode == "none" else (eh2 if obs_mode == "pred" else e2)
            s2 = (sv2 * 2 + so2) * 4 + oe2
            q[a] += alpha * (r + gamma * max(Q[s2]) - q[a])
        is_esc = 1 if a > 0 else 0
        esc += is_esc
        ret_env += r_env
        ret_hh += hh
        ret_ht += ht
        ret_c += c
        ret_tr += r
        if r_env < 0:
            unsafe_out += 1
        if sv == 2 and so == 0:
            unsafe_dec += 1
            if a <= 1:
                miss += 1
            if a == 3:
                assist_u += 1
        if det is not None:
            det["steps"] += 1
            det["steps_true"][e] += 1
            det["esc_true"][e] += is_esc
            det["steps_pred"][eh] += 1
            det["esc_pred"][eh] += is_esc
            det["act_true"][e][a] += 1
            det["act_pred"][eh][a] += 1
            if ht > 0:
                det["hpos_true"][e] += 1
            if a in ACCEPT[e]:
                det["accept"][e] += 1
            if eh != e:
                det["mismatch"] += 1
            det["conf_sum"] += ch
            if ch < MIN_CONF:
                det["lowconf"] += 1
                det["lowconf_conf_sum"] += ch
                det["lowconf_act"][a] += 1
            det["visits"][(sv * 2 + so) * 4 + eh] += 1
            if sv == 2 and so == 0:
                det["unsafe_dec"] += 1
                det["miss"] += 1 if a <= 1 else 0
                det["assist_u"] += 1 if a == 3 else 0
                det["high_u"] += 1 if a >= 2 else 0
            if r_env < 0:
                det["unsafe_out"] += 1
        sv, so, e, eh, ch = sv2, so2, e2, eh2, ch2
    return dict(esc=esc, env=ret_env, h_hat=ret_hh, h_true=ret_ht, cost=ret_c,
                train=ret_tr, unsafe_out=unsafe_out, unsafe_dec=unsafe_dec,
                miss=miss, assist_u=assist_u)


def run_job(job):
    setting, cfg_name, seed, alpha, gamma, eps_decay, n_train, n_eval = job
    cfg, cost_on = CONFIGS[cfg_name], SETTINGS[setting]
    rnd = random.Random(seed * 1000 + ord(cfg_name))
    Q = [[0.0] * 4 for _ in range(24)]
    rng = np.random.default_rng([seed, 1])
    eps, tlog = EPS0, []
    for _ in range(n_train):
        U = rng.random((STEPS + 1, 5)).tolist()
        r = run_episode(Q, cfg, rnd, eps, alpha, gamma, U, True, cost_on, None)
        tlog.append((r["esc"], r["env"], r["train"]))
        eps = max(EPS_MIN, eps * eps_decay)
    det, ev = new_det(), []
    for ep in range(n_eval):
        U = np.random.default_rng([seed, 2, ep]).random((STEPS + 1, 5)).tolist()
        ev.append(run_episode(Q, cfg, rnd, 0.0, alpha, gamma, U, False, cost_on, det))
    return dict(setting=setting, config=cfg_name, seed=seed, train=tlog, eval=ev,
                det=det, Q=Q, params=(alpha, gamma, eps_decay, n_train))


def run_jobs(jobs, workers, pool_path):
    if workers <= 1:
        init_pool(pool_path)
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_pool,
                             initargs=(pool_path,)) as ex:
        return list(ex.map(run_job, jobs, chunksize=1))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
def run_classifier(csv_path, out, seed=42):
    df = pd.read_csv(csv_path)
    y = df["driver_state"].map(LABEL2ID)
    if y.isna().any():
        raise ValueError("Unexpected labels in driver_state: %s" % sorted(df["driver_state"].unique()))
    feat_df = df.drop(columns=["driver_state"])
    cat_cols = [c for c in feat_df.columns if not pd.api.types.is_numeric_dtype(feat_df[c])]
    Xdf = pd.get_dummies(feat_df, columns=cat_cols, dtype=float)
    feats = list(Xdf.columns)
    X, y = Xdf.values.astype(float), y.values.astype(int)
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)

    def fit(cols):
        sc = StandardScaler().fit(X[tr][:, cols])
        m = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr][:, cols]), y[tr])
        return sc, m

    def pred(sc, m, cols, rows):
        p = m.predict_proba(sc.transform(X[rows][:, cols]))
        return p.argmax(1), p.max(1)

    allc = list(range(len(feats)))
    sc, m = fit(allc)
    res = dict(features=feats, n_total=int(len(y)), seed=seed,
               split_sizes=dict(train=len(tr), val=len(va), test=len(te)),
               class_counts={s: {EMO[k]: int((y[ix] == k).sum()) for k in range(4)}
                             for s, ix in [("all", idx), ("train", tr), ("val", va), ("test", te)]})
    for name, rows in [("train", tr), ("val", va), ("test", te)]:
        p, _ = pred(sc, m, allc, rows)
        res["accuracy_" + name] = float(accuracy_score(y[rows], p))
    pte, cte = pred(sc, m, allc, te)
    res["macro_f1_test"] = float(f1_score(y[te], pte, average="macro"))
    res["report_test"] = classification_report(y[te], pte, target_names=EMO, output_dict=True)
    cm = confusion_matrix(y[te], pte)
    res["confusion_test"] = cm.tolist()
    nos = [i for i, f in enumerate(feats) if not f.startswith("stress_index")]
    if len(nos) < len(feats):
        sc2, m2 = fit(nos)
        p2, _ = pred(sc2, m2, nos, te)
        res["accuracy_test_without_stress_index"] = float(accuracy_score(y[te], p2))
    pd.DataFrame(cm, index=EMO, columns=EMO).to_csv(os.path.join(out, "confusion_matrix.csv"))
    with open(os.path.join(out, "features.txt"), "w") as f:
        f.write("\n".join(feats))
    joblib.dump(dict(scaler=sc, model=m, features=feats, label2id=LABEL2ID),
                os.path.join(out, "classifier.joblib"))
    pool_rows = np.concatenate([va, te])
    pp, pc = pred(sc, m, allc, pool_rows)
    np.savez(os.path.join(out, "pool.npz"), pred=pp, conf=pc, true=y[pool_rows], X=X[pool_rows])
    res["pool_accuracy"] = float((pp == y[pool_rows]).mean())
    res["pool_low_confidence_share"] = float((pc < MIN_CONF).mean())
    with open(os.path.join(out, "classifier_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res, cm


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def stabilisation(series, window=50, tol=0.05, abs_tol=0.5):
    """First episode from which the 50-episode moving average stays within
    max(5% of its final value, abs_tol) of the mean of the last 50 episodes."""
    x = np.asarray(series, float)
    if len(x) < 2 * window:
        return None
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    final = x[-window:].mean()
    band = max(tol * abs(final), abs_tol)
    bad = np.where(np.abs(ma - final) > band)[0]
    start = 0 if len(bad) == 0 else bad[-1] + 1
    return None if start >= len(ma) else int(start + window)


def paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    n, md, sd = len(d), d.mean(), d.std(ddof=1)
    t, p = stats.ttest_rel(x, y)
    ci = stats.t.interval(0.95, n - 1, md, sd / math.sqrt(n)) if sd > 0 else (md, md)
    try:
        pw = stats.wilcoxon(x, y).pvalue
    except ValueError:
        pw = float("nan")
    return dict(n=n, mean_diff=md, ci_lo=ci[0], ci_hi=ci[1], t=float(t), p=float(p),
                dz=float(md / sd) if sd > 0 else float("nan"), wilcoxon_p=float(pw))


def fmt(x, nd=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def summarize(res_list):
    esc = np.array([r["esc"] for j in res_list for r in j["eval"]], float)
    envr = np.array([r["env"] for j in res_list for r in j["eval"]], float)
    htr = np.array([r["h_true"] for j in res_list for r in j["eval"]], float)
    trr = np.array([r["train"] for j in res_list for r in j["eval"]], float)
    smeans = [np.mean([r["esc"] for r in j["eval"]]) for j in res_list]
    d = merge_det([j["det"] for j in res_list])
    n = d["steps"]
    acc = [d["accept"][e] / d["steps_true"][e] if d["steps_true"][e] else float("nan") for e in range(4)]
    return dict(
        esc_mean=esc.mean(), esc_sd=esc.std(ddof=1), esc_min=int(esc.min()), esc_max=int(esc.max()),
        esc_seed_mean=float(np.mean(smeans)), esc_seed_sd=float(np.std(smeans, ddof=1)) if len(smeans) > 1 else float("nan"),
        env_mean=envr.mean(), env_sd=envr.std(ddof=1), htrue_mean=htr.mean(), htrue_sd=htr.std(ddof=1),
        train_obj_mean=trr.mean(), unsafe_freq=d["unsafe_out"] / n,
        miss_rate=d["miss"] / d["unsafe_dec"] if d["unsafe_dec"] else float("nan"),
        assist_unsafe=d["assist_u"] / d["unsafe_dec"] if d["unsafe_dec"] else float("nan"),
        unsafe_dec=d["unsafe_dec"], reward_pos_rate=sum(d["hpos_true"]) / n,
        accept_by_class=acc, accept_macro=float(np.nanmean(acc)), mismatch_rate=d["mismatch"] / n,
        lowconf=d["lowconf"], lowconf_act=d["lowconf_act"], steps=n,
    ), d


# --------------------------------------------------------------------------- #
# Latency (live classifier call on real sensor rows; simulator step timed separately)
# --------------------------------------------------------------------------- #
def measure_latency(out, Qc, n=3000):
    b = joblib.load(os.path.join(out, "classifier.joblib"))
    sc, m = b["scaler"], b["model"]
    X = np.load(os.path.join(out, "pool.npz"))["X"]
    rng = random.Random(0)
    comp = {k: [] for k in ["emotion_prediction", "action_selection", "reward", "q_update",
                            "env_transition_sim"]}
    sv, so, e, eh = 1, 1, 0, 0
    pc = time.perf_counter
    for i in range(n):
        u = [rng.random() for _ in range(5)]
        t0 = pc()
        row = X[rng.randrange(len(X))].reshape(1, -1)
        eh_new = int(m.predict_proba(sc.transform(row)).argmax())
        t1 = pc()
        s = (sv * 2 + so) * 4 + eh
        q = Qc[s]
        a = q.index(max(q)) if rng.random() > 0.05 else rng.randrange(4)
        t2 = pc()
        sv2, so2, e2 = dyn_step(sv, so, e, a, u)
        t3 = pc()
        r = (-3.0 if (sv2 == 2 and so2 == 0) else 1.0) + HUM[eh][a]
        t4 = pc()
        s2 = (sv2 * 2 + so2) * 4 + eh_new
        q[a] += ALPHA * (r + GAMMA * max(Qc[s2]) - q[a])
        t5 = pc()
        comp["emotion_prediction"].append(t1 - t0)
        comp["action_selection"].append(t2 - t1)
        comp["env_transition_sim"].append(t3 - t2)
        comp["reward"].append(t4 - t3)
        comp["q_update"].append(t5 - t4)
        sv, so, e, eh = sv2, so2, e2, eh_new
    rows = []
    for k, v in comp.items():
        v = np.array(v) * 1e3
        rows.append(dict(component=k, mean_ms=v.mean(), p95_ms=np.percentile(v, 95),
                         p99_ms=np.percentile(v, 99), max_ms=v.max()))
    loop = sum(np.array(comp[k]) for k in ["emotion_prediction", "action_selection", "reward", "q_update"]) * 1e3
    rows.append(dict(component="decision_loop_total(no sim)", mean_ms=loop.mean(),
                     p95_ms=np.percentile(loop, 95), p99_ms=np.percentile(loop, 99), max_ms=loop.max()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(out, df_train, df_eval, summ, cm):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing -> figures skipped (pip install matplotlib)")
        return
    col = {"A": "tab:red", "B": "tab:orange", "C": "tab:green", "D": "tab:blue"}
    ls = {"A": "-", "B": "-", "C": "-", "D": "--"}
    for setting in SETTINGS:
        for metric, ylabel, fn in [("esc", "Alert escalations per training episode", "train_alerts"),
                                   ("env_return", "Safety (environment) return per episode", "train_envreturn")]:
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for c in CONFIGS:
                g = df_train[(df_train.setting == setting) & (df_train.config == c)]
                piv = g.pivot(index="episode", columns="seed", values=metric)
                mean = piv.mean(axis=1).rolling(20, min_periods=1).mean()
                sd = piv.std(axis=1).rolling(20, min_periods=1).mean()
                ax.plot(mean.index, mean.values, color=col[c], ls=ls[c], lw=1.4, label=f"{c}: {CFG_NAMES[c]}")
                ax.fill_between(mean.index, mean - sd, mean + sd, color=col[c], alpha=0.12)
            ax.set_xlabel("Training episode"); ax.set_ylabel(ylabel)
            ax.legend(fontsize=7); fig.tight_layout()
            fig.savefig(os.path.join(out, f"fig_{fn}_{setting}.png"), dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        for i, c in enumerate(CONFIGS):
            g = df_eval[(df_eval.setting == setting) & (df_eval.config == c)]
            sm = g.groupby("seed").esc.mean()
            ax.bar(i, sm.mean(), yerr=sm.std(ddof=1), color=col[c], alpha=0.7, capsize=3)
            ax.scatter([i] * len(sm), sm.values, color="k", s=10, zorder=3)
        ax.set_xticks(range(len(CONFIGS))); ax.set_xticklabels(list(CONFIGS))
        ax.set_ylabel("Mean escalations / episode"); fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_eval_alerts_{setting}.png"), dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        w = 0.16
        for i, c in enumerate(CONFIGS):
            d = summ[(setting, c)]["det"]
            rate = [d["esc_true"][e] / d["steps_true"][e] if d["steps_true"][e] else 0 for e in range(4)]
            ax.bar(np.arange(4) + (i - (len(CONFIGS) - 1) / 2) * w, rate, w, color=col[c], label=c)
        ax.set_xticks(range(4)); ax.set_xticklabels(EMO)
        ax.set_ylabel("Fraction of steps escalated"); ax.legend(fontsize=7, ncol=len(CONFIGS)); fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_alerts_by_emotion_{setting}.png"), dpi=300); plt.close(fig)

        a = df_eval[(df_eval.setting == setting) & (df_eval.config == "A")].esc.values
        c_ = df_eval[(df_eval.setting == setting) & (df_eval.config == "C")].esc.values
        def pct(new, old):
            return 100 * (new - old) / old if old else 0.0
        vals = [pct(c_.mean(), a.mean()), pct(c_.std(ddof=1), a.std(ddof=1)),
                pct(c_.max() - c_.min(), a.max() - a.min())]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        bars = ax.bar(["Mean", "Std. dev.", "Range span"], vals,
                      color=["tab:green" if v < 0 else "tab:red" for v in vals])
        ax.axhline(0, color="k", lw=0.8)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + (2 if v >= 0 else -6), f"{v:+.1f}%", ha="center", fontsize=8)
        ax.set_ylabel("Change of C vs A (%)"); fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_relative_change_{setting}.png"), dpi=300); plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        data = [df_eval[(df_eval.setting == setting) & (df_eval.config == c)].env_return.values for c in CONFIGS]
        try:
            ax.boxplot(data, tick_labels=list(CONFIGS), showfliers=True)
        except TypeError:                      # older matplotlib
            ax.boxplot(data, labels=list(CONFIGS), showfliers=True)
        ax.set_ylabel("Safety (environment) return / episode"); fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig_return_distribution_{setting}.png"), dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(EMO, rotation=45, fontsize=7); ax.set_yticklabels(EMO, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_confusion_matrix.png"), dpi=300); plt.close(fig)


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
def latex_tables(out, clf, summ, lat, sweep_rows):
    L = []
    rep = clf["report_test"]
    L.append("% tab:class_dist")
    for k in range(4):
        n = clf["class_counts"]["all"][EMO[k]]
        L.append(f"{EMO[k]} & {n} & {100 * n / clf['n_total']:.1f}\\% \\\\")
    L.append(f"\\textbf{{Total}} & \\textbf{{{clf['n_total']}}} & \\textbf{{100\\%}} \\\\")
    L.append("\n%% tab:ml_classwise (held-out test set, n=%d)" % clf["split_sizes"]["test"])
    for k in range(4):
        r = rep[EMO[k]]
        L.append(f"{EMO[k]} & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1-score']:.2f} \\\\")
    r = rep["macro avg"]
    L.append(f"Macro average & {r['precision']:.2f} & {r['recall']:.2f} & {r['f1-score']:.2f} \\\\")
    L.append(f"% overall test accuracy = {clf['accuracy_test']:.4f}")
    for setting in SETTINGS:
        S = {c: summ[(setting, c)] for c in CONFIGS}
        cols = list(CONFIGS)
        L.append(f"\n% ===== setting: {setting} =====\n% columns: " + " , ".join(f"{c}={CFG_NAMES[c]}" for c in cols))

        def row(label, f):
            return label + " & " + " & ".join(f(S[c]) for c in cols) + " \\\\"
        L.append(row("Mean escalations/episode", lambda s: fmt(s["esc_mean"])))
        L.append(row("SD of escalations", lambda s: fmt(s["esc_sd"])))
        L.append(row("Min--max escalations", lambda s: f"{s['esc_min']}--{s['esc_max']}"))
        L.append(row("Seed-wise mean $\\pm$ SD", lambda s: f"{s['esc_seed_mean']:.1f}$\\pm${s['esc_seed_sd']:.1f}"))
        L.append(row("Mean safety return", lambda s: fmt(s["env_mean"])))
        L.append(row("SD of safety return", lambda s: fmt(s["env_sd"])))
        L.append(row("Mean human-reward return (true emotion)", lambda s: fmt(s["htrue_mean"])))
        L.append(row("Unsafe-state frequency (\\%)", lambda s: fmt(100 * s["unsafe_freq"], 1)))
        L.append(row("Critical alert miss rate (\\%)", lambda s: fmt(100 * s["miss_rate"], 1)))
        L.append(row("Assistance in unsafe states (\\%)", lambda s: fmt(100 * s["assist_unsafe"], 1)))
        L.append(row("Reward-positive action rate (\\%)", lambda s: fmt(100 * s["reward_pos_rate"], 1)))
        L.append(row("Acceptable-action agreement, macro (\\%)", lambda s: fmt(100 * s["accept_macro"], 1)))
        L.append(f"% per-emotion escalation rate (fraction of steps escalated), setting {setting}")
        for e in range(4):
            def f(s, e=e):
                d = s["det"]
                return fmt(d["esc_true"][e] / d["steps_true"][e], 3) if d["steps_true"][e] else "--"
            L.append(row(EMO[e], f))
    L.append("\n% tab:overhead (ms per step; env_transition_sim is the simulator, not vehicle hardware)")
    for _, r in lat.iterrows():
        L.append(f"{r['component']} & {r['mean_ms']:.3f} & {r['p95_ms']:.3f} & {r['p99_ms']:.3f} & {r['max_ms']:.3f} \\\\")
    for setting in SETTINGS:
        s = summ[(setting, "C")]
        la = s["lowconf_act"]
        tot = max(s["lowconf"], 1)
        L.append(f"\n% tab:low_conf (config C, setting {setting}, simulator pool)")
        L.append(f"Inference calls & {s['steps']} \\\\")
        L.append(f"Predictions below {MIN_CONF} & {s['lowconf']} ({100 * s['lowconf'] / s['steps']:.2f}\\%) \\\\")
        for a in range(4):
            L.append(f"$a_{a}$ selections in subset & {la[a]}/{s['lowconf']} ({100 * la[a] / tot:.0f}\\%) \\\\")
    if sweep_rows:
        L.append("\n% tab:sensitivity rows: name=value & escalations mean $\\pm$ SD & 95% CI & safety return")
        L += sweep_rows
    with open(os.path.join(out, "tables.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-setting", default="both", choices=list(SETTINGS) + ["both"])
    ap.add_argument("--train-episodes", type=int, default=TRAIN_EPISODES)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    t_start = time.time()

    print("[1/6] classifier ...")
    clf, cm = run_classifier(args.csv, out)
    print(f"      test accuracy = {clf['accuracy_test']:.4f}  (val {clf['accuracy_val']:.4f}, "
          f"train {clf['accuracy_train']:.4f}); macro-F1 = {clf['macro_f1_test']:.4f}")
    pool_path = os.path.join(out, "pool.npz")

    print("[2/6] RL runs ...")
    jobs = [(s, c, sd, ALPHA, GAMMA, EPS_DECAY, args.train_episodes, args.eval_episodes)
            for s in SETTINGS for c in CONFIGS for sd in SEEDS]
    results = run_jobs(jobs, args.workers, pool_path)
    init_pool(pool_path)
    by = {}
    for r in results:
        by.setdefault((r["setting"], r["config"]), []).append(r)

    print("[3/6] logs ...")
    ev_rows, tr_rows = [], []
    for r in results:
        for i, x in enumerate(r["eval"]):
            ev_rows.append(dict(setting=r["setting"], config=r["config"], seed=r["seed"], episode=i,
                                esc=x["esc"], env_return=x["env"], human_true=x["h_true"],
                                human_pred=x["h_hat"], alert_cost=x["cost"], train_objective=x["train"],
                                unsafe_outcome_steps=x["unsafe_out"], unsafe_decisions=x["unsafe_dec"],
                                critical_misses=x["miss"], assist_in_unsafe=x["assist_u"]))
        for i, (e_, v_, t_) in enumerate(r["train"]):
            tr_rows.append(dict(setting=r["setting"], config=r["config"], seed=r["seed"], episode=i + 1,
                                esc=e_, env_return=v_, train_return=t_))
    df_eval, df_train = pd.DataFrame(ev_rows), pd.DataFrame(tr_rows)
    df_eval.to_csv(os.path.join(out, "eval_episodes.csv"), index=False)
    df_train.to_csv(os.path.join(out, "train_curves.csv"), index=False)

    summ, srows, erows, arows = {}, [], [], []
    for (setting, c), rl in by.items():
        s, d = summarize(rl)
        s["det"] = d
        summ[(setting, c)] = s
        srows.append({k: v for k, v in s.items() if k not in ("det", "accept_by_class", "lowconf_act")}
                     | dict(setting=setting, config=c))
        for e in range(4):
            erows.append(dict(setting=setting, config=c, true_emotion=EMO[e], steps=d["steps_true"][e],
                              escalations=d["esc_true"][e],
                              escalation_rate=d["esc_true"][e] / d["steps_true"][e] if d["steps_true"][e] else float("nan"),
                              esc_per_episode=d["esc_true"][e] / (len(SEEDS) * args.eval_episodes),
                              reward_positive_steps=d["hpos_true"][e],
                              acceptable_agreement=s["accept_by_class"][e]))
            tot = max(sum(d["act_true"][e]), 1)
            arows.append(dict(setting=setting, config=c, true_emotion=EMO[e],
                              **{f"frac_{ACTIONS[a]}": d["act_true"][e][a] / tot for a in range(4)}))
    lc_rows = []
    for (setting, c), s_ in summ.items():
        d_ = s_["det"]
        lc_rows.append(dict(setting=setting, config=c, steps=d_["steps"], lowconf=d_["lowconf"],
                            mean_conf_all=d_["conf_sum"] / d_["steps"],
                            mean_conf_low=(d_["lowconf_conf_sum"] / d_["lowconf"]) if d_["lowconf"] else float("nan"),
                            **{f"low_a{a}": d_["lowconf_act"][a] for a in range(4)}))
    pd.DataFrame(lc_rows).to_csv(os.path.join(out, "low_conf.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(out, "summary.csv"), index=False)
    pd.DataFrame(erows).to_csv(os.path.join(out, "per_emotion.csv"), index=False)
    pd.DataFrame(arows).to_csv(os.path.join(out, "action_dist.csv"), index=False)

    # statistics (common random numbers => episode i is paired across configs)
    lines = ["Paired comparisons of escalations/episode (episode i, seed s shares exogenous noise across configs).",
             "Pooled over seeds => episodes within a seed share one trained policy; treat p-values as descriptive,",
             "and rely on per-seed differences + seed-wise mean/SD for the replication claim.\n"]
    for setting in SETTINGS:
        g = df_eval[df_eval.setting == setting].sort_values(["seed", "episode"])
        for treat, base in [("C", "A"), ("C", "B"), ("C", "D"), ("B", "A"), ("D", "A")]:
            x = g[g.config == treat].esc.values
            y = g[g.config == base].esc.values
            p = paired(x, y)
            per = [float(g[(g.config == treat) & (g.seed == sd)].esc.mean() - g[(g.config == base) & (g.seed == sd)].esc.mean())
                   for sd in SEEDS]
            lines.append(f"[{setting}] {treat} - {base}: mean diff {p['mean_diff']:.2f} "
                         f"(95% CI {p['ci_lo']:.2f}, {p['ci_hi']:.2f}), t={p['t']:.2f}, p={p['p']:.2g}, "
                         f"dz={p['dz']:.2f}, Wilcoxon p={p['wilcoxon_p']:.2g}, per-seed diffs={[round(v, 2) for v in per]}")
        lines.append("")
    open(os.path.join(out, "stats.txt"), "w").write("\n".join(lines))

    # stabilisation
    st_rows = []
    for (setting, c), rl in by.items():
        for r in rl:
            st_rows.append(dict(setting=setting, config=c, seed=r["seed"],
                                stab_train_return=stabilisation([t[2] for t in r["train"]]),
                                stab_alerts=stabilisation([t[0] for t in r["train"]])))
    pd.DataFrame(st_rows).to_csv(os.path.join(out, "stabilisation.csv"), index=False)

    # Q-values and policies
    qrows, pol_rows = [], []
    npz = {}
    for (setting, c), rl in by.items():
        for r in rl:
            npz[f"Q_{setting}_{c}_{r['seed']}"] = np.array(r["Q"])
    np.savez(os.path.join(out, "q_tables.npz"), **npz)
    for setting in SETTINGS:
        for c in ["A", "C"]:
            for r in by[(setting, c)]:
                idx = (2 * 2 + 0) * 4 + (3 if c == "C" else 0)
                qrows.append(dict(setting=setting, config=c, seed=r["seed"],
                                  state="High speed, Near obstacle" + (", Overloaded" if c == "C" else ""),
                                  **{f"Q_{ACTIONS[a]}": r["Q"][idx][a] for a in range(4)}))
        vis = summ[(setting, "C")]["det"]["visits"]
        for sid in range(24):
            sv_, so_, e_ = sid // 8, (sid // 4) % 2, sid % 4
            acts = []
            for r in by[(setting, "C")]:
                q = r["Q"][sid]
                acts.append(q.index(max(q)))
            maj = max(set(acts), key=lambda a: (acts.count(a), -a))
            row = dict(setting=setting, state_id=sid, speed=SPEED[sv_], obstacle=OBST[so_], emotion=EMO[e_],
                       eval_visits=vis[sid], majority=ACTIONS[maj])
            for r, a in zip(by[(setting, "C")], acts):
                row[f"seed{r['seed']}"] = ACTIONS[a]
            pol_rows.append(row)
    pd.DataFrame(qrows).to_csv(os.path.join(out, "qvalues_highrisk.csv"), index=False)
    pd.DataFrame(pol_rows).to_csv(os.path.join(out, "policy_C.csv"), index=False)

    print("[4/6] latency ...")
    Qc = [list(row) for row in by[("cost", "C")][0]["Q"]]
    lat = measure_latency(out, Qc)
    lat.to_csv(os.path.join(out, "latency.csv"), index=False)

    sweep_rows = []
    if args.sweep:
        sw_settings = list(SETTINGS) if args.sweep_setting == "both" else [args.sweep_setting]
        print("[5/6] hyper-parameter sweep (settings=%s) ..." % ",".join(sw_settings))
        base = dict(alpha=ALPHA, gamma=GAMMA, eps_decay=EPS_DECAY, episodes=args.train_episodes)
        grid = dict(alpha=[0.05, 0.1, 0.2, 0.3], gamma=[0.7, 0.8, 0.9, 0.99],
                    eps_decay=[0.990, 0.995, 0.999], episodes=[500, 1000, 2000])
        for sw in sw_settings:
            meta, sjobs = [], []
            for name, vals in grid.items():
                for v in vals:
                    p = {**base, name: v}
                    for sd in SWEEP_SEEDS:
                        sjobs.append((sw, "C", sd, p["alpha"], p["gamma"], p["eps_decay"],
                                      p["episodes"], args.eval_episodes))
                        meta.append((name, v, sd))
            sres = run_jobs(sjobs, args.workers, pool_path)
            agg = {}
            for (name, v, sd), r in zip(meta, sres):
                agg.setdefault((name, v), []).append((np.mean([x["esc"] for x in r["eval"]]),
                                                      np.mean([x["env"] for x in r["eval"]])))
            rows_sw = []
            for (name, v), lst in agg.items():
                e_ = np.array([a for a, _ in lst]); s_ = np.array([b for _, b in lst])
                lo, hi = stats.t.interval(0.95, len(e_) - 1, e_.mean(), stats.sem(e_))
                rows_sw.append(f"{name}={v} & {e_.mean():.2f} $\\pm$ {e_.std(ddof=1):.2f} & "
                               f"[{lo:.2f}, {hi:.2f}] & {s_.mean():.2f} \\\\")
            open(os.path.join(out, f"sweep_rows_{sw}.tex"), "w").write("\n".join(rows_sw) + "\n")
            sweep_rows += [f"% setting: {sw}"] + rows_sw
        open(os.path.join(out, "sweep_rows.tex"), "w").write("\n".join(sweep_rows) + "\n")
    else:
        print("[5/6] sweep skipped (use --sweep)")

    print("[6/6] tables, figures, report ...")
    latex_tables(out, clf, summ, lat, sweep_rows)
    make_figures(out, df_train, df_eval, summ, cm)

    env = dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
               numpy=np.__version__, pandas=pd.__version__)
    try:
        import sklearn, scipy
        env.update(sklearn=sklearn.__version__, scipy=scipy.__version__)
    except Exception:
        pass
    try:
        import matplotlib
        env["matplotlib"] = matplotlib.__version__
    except Exception:
        pass
    try:
        import psutil
        env["ram_gib"] = round(psutil.virtual_memory().total / 2 ** 30, 2)
        env["cpu_model_hint"] = platform.processor()
    except Exception:
        pass
    env["runtime_seconds"] = round(time.time() - t_start, 1)
    json.dump(env, open(os.path.join(out, "environment.json"), "w"), indent=2)
    json.dump(dict(alpha=ALPHA, gamma=GAMMA, eps0=EPS0, eps_min=EPS_MIN, eps_decay=EPS_DECAY,
                   train_episodes=args.train_episodes, steps=STEPS, eval_episodes=args.eval_episodes,
                   seeds=SEEDS, alert_cost=ALERT_COST, alert_eff=ALERT_EFF, emo_mult=EMO_MULT,
                   base_clear=BASE_CLEAR, near_base=NEAR_BASE, near_speed=NEAR_SPEED,
                   assist_slow=ASSIST_SLOW, stay=STAY, stay_overload=STAY_OVER,
                   accept={EMO[k]: [ACTIONS[a] for a in sorted(v)] for k, v in ACCEPT.items()}),
              open(os.path.join(out, "hyperparams.json"), "w"), indent=2)

    # console + markdown report
    rpt = ["# Results report (generated by run_experiments.py)\n",
           f"Classifier: test accuracy {clf['accuracy_test']:.4f} (n={clf['split_sizes']['test']}), "
           f"macro-F1 {clf['macro_f1_test']:.4f}; accuracy without stress_index: "
           f"{clf.get('accuracy_test_without_stress_index', float('nan')):.4f}\n"]
    for setting in SETTINGS:
        rpt.append(f"\n## Setting: {setting}\n")
        rpt.append("| Config | Esc/ep (pooled mean ± SD) | seed-wise mean ± SD | safety return | unsafe % | miss % | assist-in-unsafe % | reward-pos % | accept-macro % |")
        rpt.append("|---|---|---|---|---|---|---|---|---|")
        for c in CONFIGS:
            s = summ[(setting, c)]
            rpt.append(f"| {c} {CFG_NAMES[c]} | {s['esc_mean']:.2f} ± {s['esc_sd']:.2f} | "
                       f"{s['esc_seed_mean']:.2f} ± {s['esc_seed_sd']:.2f} | {s['env_mean']:.2f} | "
                       f"{100 * s['unsafe_freq']:.1f} | {100 * s['miss_rate']:.1f} | "
                       f"{100 * s['assist_unsafe']:.1f} | {100 * s['reward_pos_rate']:.1f} | "
                       f"{100 * s['accept_macro']:.1f} |")
    rpt.append("\n## Latency (ms)\n")
    try:
        rpt.append(lat.round(3).to_markdown(index=False))
    except ImportError:                       # 'tabulate' is an optional package
        rpt.append(lat.round(3).to_string(index=False))
    with open(os.path.join(out, "results_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rpt) + "\n")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("\n".join(rpt))
    print(f"\nDone in {env['runtime_seconds']} s. All outputs are in: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()