"""Collect reproducible evidence and fresh simulation results for the paper."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from ablation_no_emotion_final import NoEmotionAgent, SimpleEnv, get_human_reward
from agent import QLearningAgent


BASE = Path(__file__).resolve().parent
OUT = BASE / "paper_info.txt"
SEEDS = [42, 123, 999]
SWEEP_SEEDS = [42, 123, 999, 7, 11]
EPISODES = 1000
STEPS = 100
EVAL_EPISODES = 100
EMOTIONS = ["Calm", "Stressed", "Fatigued", "Overloaded"]
ACTIONS = ["Low", "Medium", "High", "Assistance"]
STATE_ORDER = [
    (speed, obstacle, emotion)
    for speed in range(3)
    for obstacle in range(2)
    for emotion in range(4)
]


def code_quotes(relative_path: str, patterns: list[str], limit: int = 12) -> str:
    path = BASE / relative_path
    if not path.exists():
        return f"{relative_path}: NOT FOUND"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = []
    for index, line in enumerate(lines):
        if any(re.search(pattern, line) for pattern in patterns):
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            block = "\n".join(f"{n + 1}: {lines[n]}" for n in range(start, end))
            if not hits or hits[-1] != block:
                hits.append(block)
            if len(hits) >= limit:
                break
    if not hits:
        return f"{relative_path}: no matching source lines"
    return f"{relative_path}\n" + "\n...\n".join(hits)


def function_quote(relative_path: str, function_name: str, max_lines: int = 42) -> str:
    path = BASE / relative_path
    if not path.exists():
        return f"{relative_path}::{function_name}: NOT FOUND"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    marker = re.compile(rf"^\s*def\s+{re.escape(function_name)}\s*\(")
    for index, line in enumerate(lines):
        if marker.search(line):
            end = index + 1
            while end < len(lines) and end - index < max_lines:
                if re.match(r"^\s*def\s+\w+\s*\(", lines[end]):
                    break
                end += 1
            return "\n".join(f"{n + 1}: {lines[n]}" for n in range(index, end))
    return f"{relative_path}::{function_name}: NOT FOUND"


def make_agent(config: str, seed: int, alpha: float, gamma: float,
               epsilon_decay: float):
    if config in ("A", "B"):
        agent = NoEmotionAgent(seed)
        agent.alpha = alpha
        agent.gamma = gamma
        agent.epsilon_decay = epsilon_decay
        return agent
    random.seed(seed)
    np.random.seed(seed)
    return QLearningAgent(
        learning_rate=alpha,
        discount_factor=gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=epsilon_decay,
    )


def policy_state(config: str, full_state: tuple[int, int, int]):
    return full_state[:2] if config in ("A", "B") else full_state


def train_one(config: str, seed: int, episodes: int = EPISODES,
              alpha: float = 0.1, gamma: float = 0.9,
              epsilon_decay: float = 0.995, collect_eval: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    env = SimpleEnv()
    agent = make_agent(config, seed, alpha, gamma, epsilon_decay)
    training_returns = []

    for _episode in range(episodes):
        full_state = env.reset()
        state = policy_state(config, full_state)
        episode_return = 0.0
        for _step in range(STEPS):
            action = agent.select_action(state)
            next_full_state, env_reward, done = env.step(action)
            next_state = policy_state(config, next_full_state)
            human_reward = get_human_reward(next_full_state[2], action)
            reward = env_reward if config == "A" else env_reward + human_reward
            if config == "C":
                agent.update_q_value(state, action, reward, next_state)
            else:
                agent.update(state, action, reward, next_state, done)
            episode_return += reward
            state = next_state
            if done:
                break
        if config == "C":
            agent.decay_epsilon()
        else:
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        training_returns.append(episode_return)

    if not collect_eval:
        return {"agent": agent, "training_returns": training_returns}

    random.seed(seed + 100000)
    np.random.seed(seed + 100000)
    agent.epsilon = 0
    episode_returns = []
    episode_escalations = []
    events = []
    initial_states = []

    for _episode in range(EVAL_EPISODES):
        full_state = env.reset()
        initial_states.append(full_state)
        state = policy_state(config, full_state)
        episode_return = 0.0
        escalations = 0
        for _step in range(STEPS):
            if config == "C":
                action = agent.get_best_action(state)
            else:
                action = agent.select_action(state)
            unsafe = full_state[0] == 2 and full_state[1] == 0
            next_full_state, env_reward, done = env.step(action)
            human_reward = get_human_reward(next_full_state[2], action)
            reward = env_reward if config == "A" else env_reward + human_reward
            episode_return += reward
            escalations += int(action in (1, 2, 3))
            events.append({
                "state": full_state,
                "emotion": full_state[2],
                "reward_emotion": next_full_state[2],
                "action": action,
                "env_reward": env_reward,
                "human_reward": human_reward,
                "return_reward": reward,
                "unsafe": unsafe,
            })
            full_state = next_full_state
            state = policy_state(config, full_state)
            if done:
                break
        episode_returns.append(episode_return)
        episode_escalations.append(escalations)

    return {
        "agent": agent,
        "training_returns": training_returns,
        "returns": episode_returns,
        "escalations": episode_escalations,
        "events": events,
        "initial_states": initial_states,
    }


def q_values(agent, config: str, state: tuple[int, ...]) -> list[float]:
    if config == "C":
        return [float(agent.q_table.get(state, {}).get(action, 0.0)) for action in range(4)]
    return [float(agent.get_q(state, action)) for action in range(4)]


def greedy_action(agent, state: tuple[int, ...]) -> int:
    return int(np.argmax([agent.get_q_value(state, action) for action in range(4)]))


def moving_average(values: list[float], window: int = 50) -> np.ndarray:
    return np.convolve(np.asarray(values, dtype=float), np.ones(window) / window, mode="valid")


def stabilization_episode(values: list[float]) -> str:
    window = 50
    averages = moving_average(values, window)
    final_value = float(np.mean(values[-window:]))
    tolerance = abs(final_value) * 0.05
    stable = np.abs(averages - final_value) <= tolerance
    for index in range(len(stable)):
        if bool(np.all(stable[index:])):
            return str(index + window)
    return "NOT FOUND"


def environment_details() -> dict[str, str]:
    details = {
        "Python": sys.version.replace("\n", " "),
        "OS": f"{platform.system()} {platform.release()} ({platform.version()})",
        "CPU": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "NOT FOUND"),
        "RAM": "NOT FOUND",
    }
    if platform.system() == "Windows":
        try:
            import ctypes
            import winreg

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                details["RAM"] = f"{status.ullTotalPhys / (1024 ** 3):.6f} GiB total"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                details["CPU"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except Exception as exc:
            details["hardware_probe"] = f"partial: {exc}"
    for package in ("numpy", "scikit-learn", "matplotlib", "scipy", "carla"):
        try:
            details[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            details[package] = "NOT INSTALLED"
    return details


def classifier_evidence() -> list[str]:
    lines = []
    csv_path = BASE / "driver_emotion_balanced_15k.csv"
    bundle_path = BASE / "ml_bundle.pkl"
    lines.append(f"CSV: {csv_path}")
    lines.append(f"CSV rows (excluding header): {len(pd.read_csv(csv_path)) if csv_path.exists() else 'NOT FOUND'}")
    lines.append(f"Bundle: {bundle_path} ({'present' if bundle_path.exists() else 'NOT FOUND'})")
    if bundle_path.exists():
        try:
            import pickle
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with bundle_path.open("rb") as handle:
                    bundle = pickle.load(handle)
            if caught:
                lines.append("Pickle load warnings: " + " | ".join(str(item.message) for item in caught))
            if isinstance(bundle, dict):
                lines.append(f"Bundle keys: {sorted(map(str, bundle.keys()))}")
                model = bundle.get("model")
                scaler = bundle.get("scaler")
                lines.append(f"Model type: {type(model).__module__}.{type(model).__name__}")
                lines.append(f"Scaler type: {type(scaler).__module__}.{type(scaler).__name__}")
                encoder = bundle.get("le_state")
                if encoder is not None and hasattr(encoder, "classes_"):
                    lines.append(f"Saved emotion encoder classes_: {list(map(str, encoder.classes_))}")
                    lines.append("Saved encoder ID mapping: " + repr({str(label): index for index, label in enumerate(encoder.classes_)}))
                accuracy_keys = [key for key in bundle if re.search(r"test.*(acc|accuracy)|(acc|accuracy).*test", str(key), re.I)]
                lines.append(f"Saved test-accuracy metadata: {[(key, bundle[key]) for key in accuracy_keys] if accuracy_keys else 'NOT FOUND'}")
            else:
                lines.append(f"Bundle object type: {type(bundle).__module__}.{type(bundle).__name__}")
        except Exception as exc:
            lines.append(f"Bundle inspection failed: {type(exc).__name__}: {exc}")
    return lines


def package_source_facts() -> list[str]:
    facts = []
    csv_path = BASE / "driver_emotion_balanced_15k.csv"
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        facts.append(f"CSV shape: {frame.shape}; columns: {list(frame.columns)}")
        if "driver_state" in frame:
            facts.append(f"CSV label counts: {frame['driver_state'].value_counts().to_dict()}")
    else:
        facts.append("Dataset CSV: NOT FOUND")
    excluded = {".git", "venv", ".venv", "__pycache__", "site-packages"}
    extensions = {".py", ".md", ".txt", ".tex", ".json", ".jsonl", ".log"}
    files = [
        path for path in BASE.rglob("*")
        if path.is_file()
        and path.suffix.lower() in extensions
        and not any(part in excluded for part in path.relative_to(BASE).parts)
        and path.name not in {"collect_paper_info.py", "paper_info.txt"}
    ]
    for label, pattern in [
        ("test split/training accuracy function", r"train_test_split|accuracy_score|classification_report|\.score\("),
        ("70/15/15 split", r"70\s*[/:-]\s*15\s*[/:-]\s*15|0\.70.{0,40}0\.15"),
        ("Cronbach/rater code", r"Cronbach|rater|inter.?rater|expert ratings"),
        ("paper-specific 5000/4362 calculation", r"4362|5000\s*[,)]|87\.2"),
        ("safety metric definitions", r"critical alert miss|miss rate|unsafe.?state frequency|assistance activation"),
        ("state-index encoding", r"state_to_index|state_index|encode_state"),
    ]:
        matches = [str(p.relative_to(BASE)) for p in files if re.search(pattern, p.read_text(encoding="utf-8", errors="replace"), re.I)]
        facts.append(f"Search {label}: {matches if matches else 'NOT FOUND'}")
    q_artifacts = [
        str(path.relative_to(BASE)) for path in BASE.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".npy", ".npz"}
        and not any(part in excluded for part in path.relative_to(BASE).parts)
        and path.name != "ml_bundle.pkl"
    ]
    facts.append(f"Saved RL Q-table candidates (.pkl/.npy/.npz excluding ML bundle): {q_artifacts if q_artifacts else 'NOT FOUND'}")
    return facts


def main() -> None:
    results = {config: {} for config in ("A", "B", "C")}
    for config in results:
        for seed in SEEDS:
            results[config][seed] = train_one(config, seed)

    lines = []
    missing_items = []
    lines.extend([
        "DISCREPANCIES",
        "==============",
        "- 92.3% and 92.4%: neither is traceable to a test-set evaluation function or saved held-out predictions. The 92.4% figure appears in prose as 'integration accuracy'; no integrated RL evaluation computes classifier accuracy.",
        "- 89.4%: appears as a claim in architecture/report text; no classifier training/test evaluation code was found.",
        "- 87.2% and 4362/5000: 87.2 is hard-coded in presentation material; no 5000-sample selection or numerator calculation was found.",
        "- 159.66: present in the ablation report/summary, but not a committed per-episode result artifact. The ablation evaluator does sum environment + human rewards for B; its reported value is not the chart's 47.40 quantity.",
        "- 47.40: hard-coded as the mean of a synthetic normal distribution in split_charts.py, not computed from Q-learning evaluation episodes.",
        "- 24.42 and 7.98: hard-coded chart/report values; no saved evaluation decision logs or Q-tables support them. Fresh values below are independently rerun results.",
        "- 27.9 ms: no measured latency result/function found in the Python source inspected.",
        "- 15,000 samples: the CSV row count is measured below; this confirms dataset size only, not a training split or held-out evaluation.",
        "- 70/15/15 split: no split implementation/configuration found. Do not describe that split as verified.",
        "- Obstacle encoding conflict: ablation_no_emotion_final.py treats obstacle=0 as near in its unsafe test; config.py labels obstacle=0 as Far. The fresh ablation runs use the former convention to match the user-specified Near=0.",
        "- Emotion encoding conflict: config.py/requested IDs are 0=Calm, 1=Stressed, 2=Fatigued, 3=Overloaded. ml_bundle.pkl's saved LabelEncoder classes are alphabetical, so model IDs are 0=Calm, 1=Fatigued, 2=Overloaded, 3=Stressed; ml_rl_integration.py returns these IDs without remapping. Its prediction docstring also documents that alternate order.",
        "",
        "Evidence boundary: fresh A/B/C numbers below are generated by this collector using the repository's ablation SimpleEnv/get_human_reward and the requested Q-learning parameters. Its emotion is simulator-generated, not an ML prediction. The repository does not contain a sensor-driven training/evaluation loop that could validate those as ML-fed C results.",
        "",
    ])

    lines.extend(["1. ENVIRONMENT", "==============="])
    for name, value in environment_details().items():
        lines.append(f"{name}: {value}")
    lines.append("Source: runtime platform/importlib.metadata queried by collect_paper_info.py::environment_details.")
    lines.append("")

    lines.extend(["2. CLASSIFIER ACCURACY", "======================"])
    lines.append("Exact held-out test accuracy: NOT FOUND. Test-set size: NOT FOUND. No training/test split, held-out labels/predictions, or test-accuracy metadata was found; inspected ml_rl_integration.py::MLEmotionPredictor and ml_bundle.pkl metadata. Recursive text/artifact search results are included below.")
    lines.append("The CSV row count is not a test-set size. The 92.3%/92.4% figures are not verified classifier test-set accuracy. The 92.4% report claim labels it integration accuracy, but no integrated pipeline evaluation function computes either figure. See PROJECT_REPORT.md prose and ml_rl_integration.py::predict_emotion.")
    lines.extend(classifier_evidence())
    lines.append("Source search facts: " + "; ".join(package_source_facts()))
    lines.append("")
    missing_items.extend(["classifier test accuracy and test-set size", "verified 92.3%/92.4% accuracy provenance"])

    lines.extend(["3. PER-EPISODE ALERT COUNTS", "==========================="])
    lines.append("Escalation is defined here exactly as requested: action in {1, 2, 3}. Episodes are capped at 100 steps; SimpleEnv.done is always False. Returns are undiscounted sums of the configuration's training/evaluation reward.")
    lines.append("A=safety-only, 2-state; B=env+human reward, 2-state; C=env+human reward, 24-state. These are fresh simulator-emotion runs, not historical artifact values or verified ML-fed evaluation.")
    for config in results:
        lines.append(f"\nConfig {config}:")
        for seed in SEEDS:
            item = results[config][seed]
            lines.append(f"Seed {seed} escalation counts: {item['escalations']!r}")
            lines.append(f"Seed {seed} episode returns: {item['returns']!r}")
    lines.append("\nPairing/randomness: each run seeds Python random and NumPy with the same seed before training, then reseeds both to seed+100000 before its evaluation. Thus the first reset matches across configurations for a given seed, but complete evaluation streams are NOT paired: action-dependent transition branches consume different numbers of random draws, so later resets/trajectories can diverge. No saved historical A/B/C evaluation traces exist.")
    lines.append("Relevant source quotes:")
    lines.append(code_quotes("ablation_no_emotion_final.py", [r"random.seed\(seed\)", r"np\.random\.seed\(seed\)", r"def reset", r"random.randint", r"random.random", r"def evaluate", r"env.reset\(\)", r"total_reward = r_env", r"episode_reward \+= total_reward"], 14))
    lines.append("Collector reseeding/reset quotes:")
    lines.append(code_quotes("collect_paper_info.py", [r"random.seed\(seed\)", r"np\.random\.seed\(seed\)", r"random.seed\(seed \+ 100000\)", r"np\.random\.seed\(seed \+ 100000\)", r"full_state = env.reset\(\)", r"next_full_state, env_reward, done = env.step"], 10))
    lines.append("")

    lines.extend(["4. REWARD AGGREGATION CHECK", "============================"])
    lines.append("(a) Training logs: ablation_no_emotion_final.py::NoEmotionAgent.train accumulates `episode_reward += total_reward` each step and appends once per episode; its main reports `np.mean(eval_rewards)` after evaluation. The base train.py::TrainingManager.train_hitl adds environment + human reward to training return and logs `np.mean(self.hitl_episode_rewards[-EVALUATION_INTERVAL:])`; baseline training logs the same window mean from env reward only. This project has no original three-config A/B/C training-log function. For the fresh runs, each config's mean training episode return is recomputed from its 1000 returned per-episode totals:")
    for config in results:
        training_values = [value for seed in SEEDS for value in results[config][seed]["training_returns"]]
        lines.append(f"  Config {config}: mean training return={np.mean(training_values):.10f} over {len(training_values)} episodes (collect_paper_info.py::train_one).")
    lines.append("(b) Evaluation: ablation_no_emotion_final.py::NoEmotionAgent.evaluate computes `total_reward = r_env + r_human` and adds it for every one of 100 steps; it does not terminate early because SimpleEnv returns done=False. In contrast, evaluate.py::evaluate_agents adds only env_reward to hitl_rewards (human reward is only used for a separate positive-match count) and evaluates 50 steps. Therefore human reward is added in B-like ablation evaluation, but omitted in legacy evaluate.py HiTL return; C was not implemented in the legacy evaluator.")
    lines.append("The 159.66 ablation-report value is a 100-step combined-return claim. The 47.40 figure is a synthetic mean hard-coded in split_charts.py::split_reward_distribution (`np.random.normal(47.40, 5.04, 100)`), not a return produced by B or C evaluation. This is not an apples-to-apples B/C comparison. In this collector, identical evaluation logic sums 100-step env+human rewards for both B and C; the following are the recomputed means across the 3 seeds x 100 episodes:")
    for config in ("A", "B", "C"):
        pooled = [ret for seed in SEEDS for ret in results[config][seed]["returns"]]
        lines.append(f"Config {config}: mean evaluation return={np.mean(pooled):.10f}; sample SD={np.std(pooled, ddof=1):.10f}; pooled episodes={len(pooled)} (collect_paper_info.py::train_one evaluation loop)")
    lines.append("Code excerpts:")
    lines.append(function_quote("ablation_no_emotion_final.py", "train", 38))
    lines.append(function_quote("ablation_no_emotion_final.py", "evaluate", 42))
    lines.append(function_quote("train.py", "train_baseline", 45))
    lines.append(function_quote("train.py", "train_hitl", 48))
    lines.append(function_quote("evaluate.py", "evaluate_agents", 75))
    lines.append(function_quote("evaluate.py", "print_detailed_analysis", 48))
    lines.append("")

    lines.extend(["5. Q-VALUES", "==========="])
    lines.append("Q-tables were generated afresh by this collector and retained in memory at the end of training episode 1000 (no repository Q-table save/load artifact exists). State index for the requested order is `((speed * 2) + obstacle) * 4 + emotion`; tuple-keyed Q-learning itself stores `(speed, obstacle, emotion)` and does not use an integer index.")
    lines.append("A uses the two-component key `(speed, obstacle)`; C uses `(speed, obstacle, emotion)`. Requested state uses the user-specified Near=0 convention.")
    for seed in SEEDS:
        lines.append(f"Seed {seed} C Q((2,0,3), actions 0..3): {q_values(results['C'][seed]['agent'], 'C', (2, 0, 3))!r}")
        lines.append(f"Seed {seed} A Q((2,0), actions 0..3): {q_values(results['A'][seed]['agent'], 'A', (2, 0))!r}")
    lines.append("Historical Q-table artifacts searched recursively under code (excluding venv/.git and ml_bundle.pkl): " + "; ".join(item for item in package_source_facts() if item.startswith("Saved RL Q-table candidates")) + ". Fresh table values above are final-episode tables generated by collect_paper_info.py::train_one.")
    lines.append("")

    lines.extend(["6. FULL POLICY TABLE", "===================="])
    lines.append("State order: speed Low/Medium/High, then obstacle Near/Far, then emotion Calm/Stressed/Fatigued/Overloaded. Actions: 0 Low, 1 Medium, 2 High, 3 Assistance. Greedy ties use first maximum (lowest action id) in this report; seed-majority ties also use lowest action id.")
    for seed in SEEDS:
        policy = [greedy_action(results["C"][seed]["agent"], state) for state in STATE_ORDER]
        lines.append(f"Seed {seed} action IDs: {policy!r}")
    seed_policies = [[greedy_action(results["C"][seed]["agent"], state) for state in STATE_ORDER] for seed in SEEDS]
    majority = []
    for state_index in range(len(STATE_ORDER)):
        votes = Counter(policy[state_index] for policy in seed_policies)
        majority.append(min(action for action, count in votes.items() if count == max(votes.values())))
    lines.append(f"Seed-majority action IDs: {majority!r}")
    lines.append("Rows as (speed, obstacle, emotion) -> action: " + repr(list(zip(STATE_ORDER, majority))))
    lines.append("")

    lines.extend(["7. STATE AND EMOTION VISIT COUNTS", "================================="])
    lines.append("These are simulator-emotion state visits in the fresh C evaluation, not ML-predicted emotion visits. No evaluation sensor rows are linked to simulator decisions, so true-vs-predicted mismatch is NOT FOUND.")
    for seed in SEEDS:
        events = results["C"][seed]["events"]
        state_counts = Counter(event["state"] for event in events)
        emotion_counts = Counter(event["emotion"] for event in events)
        lines.append(f"Seed {seed} state counts in STATE_ORDER: {[state_counts[state] for state in STATE_ORDER]!r}")
        lines.append(f"Seed {seed} emotion counts Calm/Stressed/Fatigued/Overloaded: {[emotion_counts[index] for index in range(4)]!r}")
        lines.append(f"Seed {seed} decisions: {len(events)}; true/predicted mismatches: NOT FOUND (no paired labels)")
    lines.append("Searched driver_emotion_balanced_15k.csv and ml_rl_integration.py::predict_emotion; CSV labels exist but are not paired to environment decisions by any project function.")
    lines.append("")

    lines.extend(["8. PER-EMOTION ALERT NUMBERS", "============================"])
    lines.append("The exact table 22.1/3.2, 23.8/8.1, 25.0/9.4, 26.9/4.1 was NOT FOUND in source. presentation_charts.py::create_emotional_appropriateness contains unrelated hard-coded percentages, not these alert counts. Values below are from fresh simulator-emotion decisions, pooled across 3 seeds x 100 episodes; mean escalations/episode for emotion = escalations while in emotion / 300 episodes.")
    for config in results:
        all_events = [event for seed in SEEDS for event in results[config][seed]["events"]]
        lines.append(f"Config {config} (emotion | mean escalations/episode | steps | escalation rate):")
        for emotion, name in enumerate(EMOTIONS):
            subset = [event for event in all_events if event["emotion"] == emotion]
            escalations = sum(event["action"] in (1, 2, 3) for event in subset)
            lines.append(f"  {name}: {escalations / (len(SEEDS) * EVAL_EPISODES):.10f} | {len(subset)} | {escalations / len(subset) if subset else float('nan'):.10f}")
    lines.append("Config B's former '~10 for all' claim: no supporting per-emotion calculation found; B values above are recomputed, not historical outputs.")
    lines.append("")

    lines.extend(["9. REWARD-POSITIVE ACTION RATE", "==============================="])
    lines.append("Source for 4362/5000 was NOT FOUND. Recalculation procedure: for each configuration, take the first 5000 evaluation decisions from seed 42 (first 50 episodes x 100 steps); compute the existing get_human_reward using the next simulator emotion and chosen action; count rewards > 0 by that next emotion. This is not the provenance of the paper's 5000 samples.")
    for config in results:
        events = results[config][42]["events"][:5000]
        positive = Counter()
        totals = Counter()
        for event in events:
            reward_emotion = event["reward_emotion"]
            totals[reward_emotion] += 1
            positive[reward_emotion] += int(event["human_reward"] > 0)
        lines.append(f"Config {config}, 5000 decisions; positive counts by emotion: " + repr({EMOTIONS[e]: (positive[e], totals[e]) for e in range(4)}))
    lines.append("")

    lines.extend(["10. ACTION DISTRIBUTION", "======================="])
    lines.append("Pooled fractions across 3 seeds and all evaluation decisions; emotion is the simulator emotion observed in the current decision state (not model prediction).")
    for config in results:
        events = [event for seed in SEEDS for event in results[config][seed]["events"]]
        counts = Counter(event["action"] for event in events)
        lines.append(f"Config {config} overall action fractions 0..3: {[counts[action] / len(events) for action in range(4)]!r} (counts={[counts[action] for action in range(4)]!r})")
        for emotion, emotion_name in enumerate(EMOTIONS):
            subset = [event for event in events if event["emotion"] == emotion]
            action_counts = Counter(event["action"] for event in subset)
            fractions = [action_counts[action] / len(subset) if subset else float("nan") for action in range(4)]
            lines.append(f"  {emotion_name}: fractions={fractions!r}; counts={[action_counts[action] for action in range(4)]!r}; decisions={len(subset)}")
    lines.append("")

    lines.extend(["11. HYPERPARAMETER SWEEP", "========================"])
    lines.append("C-only, one parameter changed at a time from alpha=0.1, gamma=0.9, epsilon_decay=0.995, 1000 training episodes. Each setting uses seeds 42,123,999,7,11; evaluation is 100 episodes x 100 steps with epsilon=0. Outcome is each seed's mean escalations/episode; SD is sample SD across five seed means; CI is two-sided 95% Student-t with df=4.")
    settings = [("alpha", value) for value in (0.05, 0.1, 0.2, 0.3)]
    settings += [("gamma", value) for value in (0.7, 0.8, 0.9, 0.99)]
    settings += [("epsilon_decay", value) for value in (0.990, 0.995, 0.999)]
    settings += [("training_episodes", value) for value in (500, 1000, 2000)]
    for name, value in settings:
        config_values = {"alpha": 0.1, "gamma": 0.9, "epsilon_decay": 0.995}
        train_episodes = EPISODES
        if name == "training_episodes":
            train_episodes = int(value)
        else:
            config_values[name] = float(value)
        per_seed = []
        for seed in SWEEP_SEEDS:
            run = train_one("C", seed, train_episodes, collect_eval=True, **config_values)
            per_seed.append(float(np.mean(run["escalations"])))
        mean = float(np.mean(per_seed))
        sd = float(np.std(per_seed, ddof=1))
        margin = float(t.ppf(0.975, df=len(per_seed) - 1) * sd / np.sqrt(len(per_seed)))
        label = f"{name}={value}"
        lines.append(f"{label} & {mean:.6f} $\\pm$ {sd:.6f} & [{mean - margin:.6f}, {mean + margin:.6f}] \\\\")
    lines.append("")

    lines.extend(["12. CONVERGENCE FIGURE", "======================"])
    lines.append("09a_convergence_q_values.png is generated by split_charts.py::split_convergence_analysis. Y-axis: 'Average Q-Value'; synthetic arrays use 50 + 50*(1-exp(-episodes/400)) plus Gaussian noise for baseline and 50 + 140*(1-exp(-episodes/250)) plus noise for HiTL; smoothed with scipy.ndimage.uniform_filter1d(size=10). It is neither an episodic return nor computed from a Q-table, and it is not a seed average.")
    lines.append("01a_learning_curves_reward.png is generated by split_charts.py::split_learning_curves. Y-axis: 'Cumulative Reward per Episode'; synthetic formulae plus Gaussian noise, smoothed with uniform_filter1d(size=5); no seed average.")
    lines.append("03d_reward_distribution_statistics.png is generated by split_charts.py::split_reward_distribution. Y-axis: generic 'Value' for Mean, Std Dev, Min, Max, Median; baseline/hitl samples drawn from hard-coded normal distributions (48.68,3.25) and (47.40,5.04), length 100; no smoothing and no seed average. Its declared mean/SD bars are hard-coded, while min/max/median come from the synthetic draws.")
    lines.append("Source excerpts:")
    lines.append(code_quotes("split_charts.py", [r"def split_learning_curves", r"uniform_filter1d\(baseline_rewards", r"set_ylabel\('Cumulative Reward per Episode'", r"def split_reward_distribution", r"np.random.normal\(48\.68", r"set_ylabel\('Value'", r"def split_convergence_analysis", r"baseline_q_values =", r"uniform_filter1d\(baseline_q_values", r"set_ylabel\('Average Q-Value'"], 16))
    lines.append("")

    lines.extend(["13. STABILISATION EPISODES", "=========================="])
    lines.append("The final ablation script's find_convergence_episode uses a 50-value trailing moving average, marks an episode stable when one-step improvement is <0.5, and returns after 50 consecutive such comparisons; if absent, it falls back to 70% of training length. The enhanced ablation script uses a different variance/median criterion. Chart annotations (~400/~600) are hard-coded, not detected from experiment traces.")
    lines.append("Fresh recomputation rule: calculate each 50-episode moving average; final value is mean training return over the last 50 episodes; stabilization is the first episode whose moving average and every later moving average remain within 5% of that final value. Results:")
    for config in results:
        for seed in SEEDS:
            lines.append(f"Config {config}, seed {seed}: episode {stabilization_episode(results[config][seed]['training_returns'])}")
    lines.append("Source excerpts:")
    lines.append(function_quote("ablation_no_emotion_final.py", "find_convergence_episode", 35))
    lines.append("")

    lines.extend(["14. SAFETY INDICATORS", "====================="])
    lines.append("Found unsafe-state criterion in ablation_no_emotion_final.py::SimpleEnv.step: speed==2 and obstacle==0. No project definition for 'critical alert' or 'critical alert miss rate' was found, so that metric is NOT FOUND. Assistance-in-unsafe-state below is a transparent recomputation using the found unsafe criterion and action==3; no historical metric function exists.")
    for config in results:
        events = [event for seed in SEEDS for event in results[config][seed]["events"]]
        unsafe_events = [event for event in events if event["unsafe"]]
        unsafe_freq = len(unsafe_events) / len(events)
        assist_count = sum(event["action"] == 3 for event in unsafe_events)
        lines.append(f"Config {config}: unsafe decisions={len(unsafe_events)}/{len(events)}; unsafe-state frequency={unsafe_freq:.10f}; assistance in unsafe states={assist_count}/{len(unsafe_events)} ({assist_count / len(unsafe_events) if unsafe_events else float('nan'):.10f}); critical alert miss rate=NOT FOUND")
        for seed in SEEDS:
            seed_events = results[config][seed]["events"]
            seed_unsafe = [event for event in seed_events if event["unsafe"]]
            seed_assist = sum(event["action"] == 3 for event in seed_unsafe)
            lines.append(f"  Seed {seed}: unsafe decisions={len(seed_unsafe)}/{len(seed_events)}; frequency={len(seed_unsafe) / len(seed_events):.10f}; assistance in unsafe states={seed_assist}/{len(seed_unsafe)} ({seed_assist / len(seed_unsafe) if seed_unsafe else float('nan'):.10f})")
    lines.append("Relevant definition source:")
    lines.append(code_quotes("ablation_no_emotion_final.py", [r"if self\.sv == 2 and self\.so == 0", r"def step", r"done = False"], 6))
    lines.append("")

    lines.extend(["15. EXPERT RATINGS", "=================="])
    lines.append("NOT FOUND. Recursive search of source/docs/data text files under code (excluding .git, venv/.venv, __pycache__, and binaries) found no 5-rater scores for 24 states or Cronbach's alpha computation. Search functions/files: package_source_facts() scans project text extensions and reports matching relative paths.")
    missing_items.extend(["paired true-vs-predicted evaluation labels", "historical per-emotion alert table source", "5000-sample selection/numerator provenance", "critical alert miss-rate definition", "5-rater scores and Cronbach alpha", "saved historical RL Q-tables"])

    lines.extend(["\nNOT FOUND SUMMARY", "================="])
    lines.append("; ".join(missing_items))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print("NOT FOUND: " + "; ".join(missing_items))


if __name__ == "__main__":
    main()