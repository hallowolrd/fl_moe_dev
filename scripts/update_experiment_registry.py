from __future__ import annotations

"""Build a read-only-derived registry and leaderboard from existing outputs."""

import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
PAIR_ROOT = OUTPUT_ROOT / "pair_runs"
REGISTRY_JSON = PROJECT_ROOT / "experiment_registry.json"
REGISTRY_MARKDOWN = PROJECT_ROOT / "EXPERIMENT_REGISTRY.md"

NON_SCIENTIFIC_CONFIG_KEYS = {
    "algorithm_name",
    "output_root",
    "project_root",
    "resolved_device",
    "partition_created_this_run",
    "summary_window",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_metrics(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def last_mean(values: list[float], size: int = 10) -> float | None:
    if not values:
        return None
    return float(fmean(values[-min(size, len(values)) :]))


def final_window_convergence(
    accuracies: list[float],
    *,
    window: int = 10,
    tolerance_pp: float = 0.1,
) -> dict[str, Any]:
    """Apply the project's cumulative-best improvement convergence gate."""
    if len(accuracies) <= window:
        return {
            "converged": False,
            "window_start_round": None,
            "window_end_round": len(accuracies),
            "best_improvement_pp": None,
        }

    cumulative_best: list[float] = []
    current = -math.inf
    for accuracy in accuracies:
        current = max(current, accuracy)
        cumulative_best.append(current)

    end_index = len(accuracies) - 1
    start_index = end_index - window
    improvement_pp = 100.0 * (
        cumulative_best[end_index] - cumulative_best[start_index]
    )
    return {
        "converged": improvement_pp <= tolerance_pp + 1e-12,
        "window_start_round": start_index + 1,
        "window_end_round": end_index + 1,
        "best_improvement_pp": improvement_pp,
    }


def effective_experts(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    distributions = [
        [float(value) for value in json.loads(row["test_route_distribution"])]
        for row in rows[-min(10, len(rows)) :]
    ]
    mean_distribution = [
        fmean(distribution[index] for distribution in distributions)
        for index in range(len(distributions[0]))
    ]
    entropy = -sum(value * math.log(value) for value in mean_distribution if value > 0)
    return float(math.exp(entropy))


def mean_expert_participants(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    participant_rows = [
        [int(value) for value in json.loads(row["expert_participant_counts"])]
        for row in rows[-min(10, len(rows)) :]
    ]
    return [
        float(fmean(row[index] for row in participant_rows))
        for index in range(len(participant_rows[0]))
    ]


def last_kfac_diagnostics(rows: list[dict[str, str]]) -> dict[str, float]:
    diagnostics: list[dict[str, Any]] = []
    for row in rows[-min(10, len(rows)) :]:
        payload = row.get("aggregation_method_metrics", "")
        if payload:
            parsed = json.loads(payload)
            if parsed:
                diagnostics.append(parsed)
    keys = (
        "kfac_coverage_rate",
        "conflict_rate_over_valid",
        "projection_rate_over_active",
        "mean_mapped_correction_ratio",
        "max_mapped_correction_ratio",
        "gain_cap_trigger_rate",
        "max_roundtrip_relative_error",
        "training_active_fisher_zero_client_layer_count",
    )
    return {
        key: float(fmean(float(item.get(key, 0.0)) for item in diagnostics))
        for key in keys
    } if diagnostics else {}


def process_command_lines() -> str:
    try:
        return subprocess.check_output(
            ["ps", "-eo", "args"], text=True, encoding="utf-8"
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def common_signature(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    common = {
        key: value
        for key, value in config.items()
        if key not in NON_SCIENTIFIC_CONFIG_KEYS
    }
    encoded = json.dumps(common, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16], common


def method_record(metrics_path: Path, process_lines: str) -> dict[str, Any]:
    run_dir = metrics_path.parent
    config = load_json(run_dir / "config.json")
    rows = load_metrics(metrics_path)
    accuracies = [float(row["test_accuracy"]) for row in rows]
    losses = [float(row["test_total_loss"]) for row in rows]
    configured_rounds = int(config["num_rounds"])
    summary_path = run_dir / "summary.json"
    signature, common_config = common_signature(config)
    output_root = str(config.get("output_root", ""))
    running = bool(output_root and output_root in process_lines)
    completed = summary_path.exists() and len(rows) == configured_rounds

    if completed:
        status = "completed"
    elif running:
        status = "running"
    elif rows:
        status = "partial_stopped"
    else:
        status = "created_no_rounds"

    return {
        "algorithm_name": config["algorithm_name"],
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "summary_path": (
            str(summary_path.relative_to(PROJECT_ROOT)) if summary_path.exists() else None
        ),
        "status": status,
        "configured_rounds": configured_rounds,
        "completed_rounds": len(rows),
        "common_signature": signature,
        "common_config": common_config,
        "config": config,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "best_accuracy": max(accuracies) if accuracies else None,
        "best_round": (
            accuracies.index(max(accuracies)) + 1 if accuracies else None
        ),
        "last10_accuracy": last_mean(accuracies),
        "last10_loss": last_mean(losses),
        "convergence": final_window_convergence(accuracies),
        "effective_experts_last10": effective_experts(rows),
        "mean_expert_participants_last10": mean_expert_participants(rows),
        "kfac_diagnostics_last10": last_kfac_diagnostics(rows),
        "kfac_config": (
            load_json(run_dir / "kfac_config.json")
            if (run_dir / "kfac_config.json").exists()
            else None
        ),
    }


def pair_metadata(pair_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("git_commit.txt", "git_status.txt", "config.txt"):
        path = pair_dir / "meta" / name
        result[name.removesuffix(".txt")] = (
            path.read_text(encoding="utf-8").strip() if path.exists() else None
        )
    return result


def paired_record(pair_dir: Path, process_lines: str) -> dict[str, Any]:
    methods = [
        method_record(path, process_lines)
        for path in sorted(pair_dir.rglob("metrics.csv"))
    ]
    by_algorithm = {record["algorithm_name"]: record for record in methods}
    kfac = next(
        (record for record in methods if "kfac" in record["algorithm_name"]), None
    )
    uniform = next(
        (record for record in methods if "uniform" in record["algorithm_name"]), None
    )
    pair: dict[str, Any] = {
        "pair_name": pair_dir.name,
        "pair_dir": str(pair_dir.relative_to(PROJECT_ROOT)),
        "metadata": pair_metadata(pair_dir),
        "methods": by_algorithm,
        "common_config_match": bool(
            kfac
            and uniform
            and kfac["common_signature"] == uniform["common_signature"]
        ),
    }
    if kfac and uniform:
        common_rounds = min(kfac["completed_rounds"], uniform["completed_rounds"])
        kfac_rows = load_metrics(PROJECT_ROOT / kfac["metrics_path"])
        uniform_rows = load_metrics(PROJECT_ROOT / uniform["metrics_path"])
        kfac_common_accuracies = [
            float(row["test_accuracy"]) for row in kfac_rows[:common_rounds]
        ]
        uniform_common_accuracies = [
            float(row["test_accuracy"]) for row in uniform_rows[:common_rounds]
        ]
        kfac_common_convergence = final_window_convergence(
            kfac_common_accuracies
        )
        uniform_common_convergence = final_window_convergence(
            uniform_common_accuracies
        )
        window = min(10, common_rounds)
        if window:
            kfac_mean = fmean(
                kfac_common_accuracies[common_rounds - window : common_rounds]
            )
            uniform_mean = fmean(
                uniform_common_accuracies[common_rounds - window : common_rounds]
            )
            gap_last = 100.0 * (kfac_mean - uniform_mean)
        else:
            kfac_mean = None
            uniform_mean = None
            gap_last = None
        pair.update(
            {
                "common_rounds": common_rounds,
                "kfac_last_common_window": kfac_mean,
                "uniform_last_common_window": uniform_mean,
                "gap_last_common_window_pp": gap_last,
                "kfac_common_convergence": kfac_common_convergence,
                "uniform_common_convergence": uniform_common_convergence,
                "gap_best_pp": (
                    100.0
                    * (
                        max(kfac_common_accuracies)
                        - max(uniform_common_accuracies)
                    )
                    if kfac_common_accuracies and uniform_common_accuracies
                    else None
                ),
                "valid_converged": bool(
                    pair["common_config_match"]
                    and kfac["completed_rounds"] == uniform["completed_rounds"]
                    and kfac["status"] != "running"
                    and uniform["status"] != "running"
                    and kfac_common_convergence["converged"]
                    and uniform_common_convergence["converged"]
                    and window == 10
                    and uniform_mean is not None
                    and uniform_mean >= 0.60
                ),
            }
        )
    return pair


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100.0 * value:.3f}%"


def gap(value: float | None) -> str:
    return "-" if value is None else f"{value:+.3f} pp"


def render_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Experiment Registry",
        "",
        f"Generated: `{registry['generated_at_utc']}`",
        "",
        "This file is generated from existing outputs by "
        "`scripts/update_experiment_registry.py`.",
        "",
        "## Pair leaderboard",
        "",
        "| Pair | State | Common rounds | KFAC Last10 | Uniform Last10 | Gap | Valid converged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in registry["pairs"]:
        methods = list(pair["methods"].values())
        kfac = next((item for item in methods if "kfac" in item["algorithm_name"]), None)
        uniform = next(
            (item for item in methods if "uniform" in item["algorithm_name"]), None
        )
        states = "/".join(item["status"] for item in methods) or "empty"
        lines.append(
            "| {name} | {states} | {rounds} | {kfac} | {uniform} | {gap} | {valid} |".format(
                name=pair["pair_name"],
                states=states,
                rounds=pair.get("common_rounds", 0),
                kfac=percent(pair.get("kfac_last_common_window")),
                uniform=percent(pair.get("uniform_last_common_window")),
                gap=gap(pair.get("gap_last_common_window_pp")),
                valid="yes" if pair.get("valid_converged") else "no",
            )
        )

    lines.extend(["", "## Method runs", ""])
    for pair in registry["pairs"]:
        lines.extend([f"### {pair['pair_name']}", ""])
        for record in pair["methods"].values():
            convergence = record["convergence"]
            lines.extend(
                [
                    f"- `{record['algorithm_name']}`: {record['status']}, "
                    f"rounds {record['completed_rounds']}/{record['configured_rounds']}, "
                    f"Last10 {percent(record['last10_accuracy'])}, "
                    f"best {percent(record['best_accuracy'])} at round "
                    f"{record['best_round']}, final-window converged="
                    f"{convergence['converged']} "
                    f"(best improvement {convergence['best_improvement_pp']})."
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    process_lines = process_command_lines()
    pairs = (
        [
            paired_record(path, process_lines)
            for path in sorted(PAIR_ROOT.iterdir())
            if path.is_dir()
        ]
        if PAIR_ROOT.exists()
        else []
    )
    valid = [pair for pair in pairs if pair.get("valid_converged")]
    valid.sort(
        key=lambda pair: pair.get("gap_last_common_window_pp", -math.inf),
        reverse=True,
    )
    registry = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "pairs": pairs,
        "valid_converged_leaderboard": [pair["pair_name"] for pair in valid],
        "best_valid_pair": valid[0]["pair_name"] if valid else None,
    }
    REGISTRY_JSON.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REGISTRY_MARKDOWN.write_text(render_markdown(registry), encoding="utf-8")


if __name__ == "__main__":
    main()
