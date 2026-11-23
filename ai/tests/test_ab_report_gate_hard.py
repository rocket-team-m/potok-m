# SPDX-License-Identifier: GPL-3.0-or-later
"""Hard gates for the closed-loop A/B report artifact (``ab_report.json``).

Five contracts are pinned, all on deterministic synthetic reports shaped like
the shipped artifact, so nothing under ``out/``, no GPU and no simulator is read:

* schema: required shape, typed fields, finite positive metrics, decimal seed keys;
* oracle positive control: every seed carries ``oracle_gtheta`` and it beats its
  own baseline on both metrics, so the instrument can detect an improvement;
* held-out provenance: withheld scenario ids, per-scenario evidence, arm digests
  and seed keys that match the declared seed order;
* no surrogate claims: the measurement names the flown-truth sidecar, voids the
  earlier surrogate A/B and credits the physical oracle, not a proxy;
* exit-code contract: ``exit_code``/``verdict`` are recomputed from the arms and
  the CLI exits 0 (pass), 1 (gate failed) or 2 (unreadable/invocation error).

Nothing here claims a pass for any current model or any flight result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

SCENARIO_COUNT = 20
HELD_OUT = (16, 17, 18, 19)
STUDENTS = ("loopwm", "lrs", "lite")
BASE_LATERAL = 1528.0
BASE_APOGEE = 1735.0
STRING_FIELDS = ("task", "measurement", "sim_source", "supersedes", "seed_note")
HEX = "0123456789abcdefABCDEF"


def _arm(lateral: float, apogee: float) -> dict[str, float]:
    return {"lateral_rms_m": lateral, "apogee_m": apogee}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _report(*, students_win: bool = False, seeds: tuple[int, ...] = (1, 2, 3)) -> dict[str, Any]:
    """One report with three students and an oracle, modeled on the real schema."""
    models_by_seed: dict[str, Any] = {}
    for seed in seeds:
        base_lat = BASE_LATERAL + seed
        base_ap = BASE_APOGEE + seed
        models: dict[str, Any] = {}
        for name in (*STUDENTS, "oracle_gtheta"):
            if name == "oracle_gtheta":
                assisted = _arm(base_lat * 0.68, base_ap * 2.28)
            else:
                delta = 12.0 if students_win else -12.0
                apogee_delta = 30.0 if students_win else -30.0
                assisted = _arm(base_lat - delta, base_ap + apogee_delta)
            per_scenario = []
            for idx in HELD_OUT:
                per_scenario.append(
                    {
                        "scenario": f"scenario_{idx}",
                        "episodes": 5,
                        "baseline": _arm(base_lat + idx, base_ap + idx),
                        "assisted": _arm(
                            assisted["lateral_rms_m"] + idx, assisted["apogee_m"] + idx
                        ),
                    }
                )
            models[name] = {
                "aggregate": {"baseline": _arm(base_lat, base_ap), "assisted": assisted},
                "per_scenario": per_scenario,
                "sha256": {
                    "baseline": _digest(f"{seed}:{name}:baseline"),
                    "assisted": _digest(f"{seed}:{name}:assisted"),
                },
            }
        models_by_seed[str(seed)] = models

    return {
        "task": "closed-loop AI assist vs zero-command PID on OR-aligned held-out scenarios",
        "measurement": "POTOKTRJ v1 flown-truth sidecar, not surrogate",
        "sim_source": "physical actuator mapping; oracle uses run_episode_with_oracle "
        "/ ReferenceOptimizer::fin_accel_command",
        "supersedes": "12 s closed-loop A/B and surrogate A/B; both void",
        "seed_note": "seeds 2/3 use seed-1-trained exports: off-training generalization, "
        "not a training seed sweep",
        "held_out_indices": [idx + SCENARIO_COUNT * k for k in range(4) for idx in HELD_OUT],
        "seed_order": list(seeds),
        "students": [
            {"name": name, "sha256": _digest(name), "kind": "stateful student binary export"}
            for name in STUDENTS
        ],
        "seeds": models_by_seed,
        "verdict": {
            "beats_lateral": students_win,
            "beats_apogee": students_win,
            "pass": students_win,
            "primary": "loopwm",
        },
        "exit_code": 0 if students_win else 1,
        "failure_location": ""
        if students_win
        else "students train on the angle-domain target, "
        "the actuator consumes world-frame acceleration",
    }


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0.0
    )


def _seed_arms(report: dict[str, Any], seed: str, name: str) -> dict[str, Any] | None:
    entry = report["seeds"].get(seed, {}).get(name)
    if isinstance(entry, dict) and isinstance(entry.get("aggregate"), dict):
        return entry["aggregate"]
    return None


def _schema_errors(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["schema: report must be a JSON object"]
    for key in STRING_FIELDS:
        value = report.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"schema: {key} must be a non-empty string")

    for key in ("held_out_indices", "seed_order"):
        value = report.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        ):
            errors.append(f"schema: {key} must be a non-empty list of integers")

    students = report.get("students")
    if (
        not isinstance(students, list)
        or not students
        or any(not isinstance(s, dict) or not isinstance(s.get("name"), str) for s in students)
    ):
        errors.append("schema: students must be a non-empty list of named objects")

    verdict = report.get("verdict")
    if not isinstance(verdict, dict):
        errors.append("schema: verdict must be an object")
    elif any(
        not isinstance(verdict.get(key), bool) for key in ("beats_lateral", "beats_apogee", "pass")
    ) or not isinstance(verdict.get("primary"), str):
        errors.append("schema: verdict needs two beats flags, pass and primary")
    code = report.get("exit_code")
    if isinstance(code, bool) or not isinstance(code, int) or code not in (0, 1):
        errors.append("schema: exit_code must be 0 or 1")
    if not isinstance(report.get("failure_location"), str):
        errors.append("schema: failure_location must be a string")

    seeds = report.get("seeds")
    if not isinstance(seeds, dict) or not seeds:
        errors.append("schema: seeds must be a non-empty object")
        return errors
    for seed, models in seeds.items():
        if not isinstance(seed, str) or not seed.isdecimal() or str(int(seed)) != seed:
            errors.append(f"schema: seed key {seed!r} must be a decimal string")
            continue
        if not isinstance(models, dict):
            errors.append(f"schema: seed {seed} must map models to entries")
            continue
        for name in [s["name"] for s in students if isinstance(s, dict) and "name" in s]:
            arms = _seed_arms(report, seed, name)
            if arms is None:
                errors.append(f"schema: seed {seed} has no aggregate for student {name}")
                continue
            for arm in ("baseline", "assisted"):
                values = arms.get(arm)
                if not isinstance(values, dict):
                    errors.append(f"schema: seed {seed} {name} has no {arm} arm")
                    continue
                for metric in ("lateral_rms_m", "apogee_m"):
                    if not _number(values.get(metric)):
                        errors.append(
                            f"schema: seed {seed} {name} {arm} {metric} is not finite positive"
                        )
    return errors


def _oracle_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for seed in report["seed_order"]:
        seed_key = str(seed)
        if "oracle_gtheta" not in report["seeds"].get(seed_key, {}):
            errors.append(f"oracle: seed {seed} has no oracle_gtheta positive control")
            continue
        arms = _seed_arms(report, seed_key, "oracle_gtheta")
        if arms is None:
            errors.append(f"oracle: seed {seed} has no aggregate")
            continue
        baseline, assisted = arms.get("baseline", {}), arms.get("assisted", {})
        if not all(
            _number(v.get(k)) for v in (baseline, assisted) for k in ("lateral_rms_m", "apogee_m")
        ):
            errors.append(f"oracle: seed {seed} has non-finite positive-control metrics")
            continue
        if not assisted["lateral_rms_m"] < baseline["lateral_rms_m"]:
            errors.append(f"oracle: seed {seed} does not lower lateral RMS vs baseline")
        if not assisted["apogee_m"] > baseline["apogee_m"]:
            errors.append(f"oracle: seed {seed} does not raise apogee vs baseline")
    return errors


def _provenance_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    indices = report["held_out_indices"]
    if not indices:
        errors.append("provenance: held_out_indices must not be empty")
    if len(set(indices)) != len(indices):
        errors.append("provenance: held_out_indices repeats a scenario")
    for idx in indices:
        if idx % SCENARIO_COUNT not in HELD_OUT:
            errors.append(f"provenance: scenario {idx} is a training cell, not held out")
    if {int(seed) for seed in report["seeds"]} != set(report["seed_order"]):
        errors.append("provenance: seed keys do not match seed_order")
    for seed in report["seed_order"]:
        for name, entry in report["seeds"][str(seed)].items():
            scenarios = entry.get("per_scenario")
            if not isinstance(scenarios, list) or not scenarios:
                errors.append(f"provenance: seed {seed} {name} has no per-scenario evidence")
            else:
                for row in scenarios:
                    label = row.get("scenario", "") if isinstance(row, dict) else ""
                    suffix = label.removeprefix("scenario_")
                    if not suffix.isdecimal() or int(suffix) not in indices:
                        errors.append(f"provenance: {label!r} is not a declared held-out scenario")
            digests = entry.get("sha256")
            if not isinstance(digests, dict):
                errors.append(f"provenance: seed {seed} {name} has no arm digests")
                continue
            for arm in ("baseline", "assisted"):
                value = digests.get(arm)
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(char not in HEX for char in value)
                ):
                    errors.append(f"provenance: seed {seed} {name} {arm} digest is not sha256")
    return errors


def _surrogate_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    measurement = report["measurement"].lower()
    if "flown-truth" not in measurement or "sidecar" not in measurement:
        errors.append("surrogate: measurement must name the flown-truth trajectory sidecar")
    if "not surrogate" not in measurement:
        errors.append("surrogate: measurement must explicitly disclaim a surrogate")
    supersedes = report["supersedes"].lower()
    if "surrogate" not in supersedes or "void" not in supersedes:
        errors.append("surrogate: earlier surrogate A/B must be superseded as void")
    if "fin_accel_command" not in report["sim_source"]:
        errors.append("surrogate: sim_source must credit the physical oracle command")
    for key in ("task", "seed_note"):
        if "surrogate" in report[key].lower():
            errors.append(f"surrogate: {key} makes a surrogate claim")
    return errors


def _beats(report: dict[str, Any], metric: str, *, lower: bool) -> bool:
    for seed in report["seed_order"]:
        arms = _seed_arms(report, str(seed), report["verdict"]["primary"])
        if arms is None:
            return False
        student = arms["assisted"][metric]
        baseline = arms["baseline"][metric]
        if not (student < baseline if lower else student > baseline):
            return False
    return True


def _exit_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    beats_lateral = _beats(report, "lateral_rms_m", lower=True)
    beats_apogee = _beats(report, "apogee_m", lower=False)
    passed = beats_lateral and beats_apogee and not _oracle_errors(report)
    expected = 0 if passed else 1
    if report["exit_code"] != expected:
        errors.append(
            f"exit: exit_code {report['exit_code']} disagrees with verdict (expected {expected})"
        )
    for key, value in (
        ("beats_lateral", beats_lateral),
        ("beats_apogee", beats_apogee),
        ("pass", passed),
    ):
        if report["verdict"].get(key) is not value:
            errors.append(f"exit: verdict.{key} disagrees with the arms")
    if report["verdict"].get("primary") not in [s["name"] for s in report["students"]]:
        errors.append("exit: verdict.primary is not a declared student")
    location = report["failure_location"]
    if expected == 1 and len(location.strip()) < 20:
        errors.append("exit: a failing A/B must name its failure location")
    return errors


def _check(report: Any) -> list[str]:
    """Return every violation; schema errors short-circuit gates that need shape."""
    errors = _schema_errors(report)
    if errors:
        return errors
    return (
        _oracle_errors(report)
        + _provenance_errors(report)
        + _surrogate_errors(report)
        + _exit_errors(report)
    )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: test_ab_report_gate_hard.py REPORT.json", file=sys.stderr)
        return 2
    try:
        report = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read A/B report: {exc}", file=sys.stderr)
        return 2
    errors = _check(report)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print("ab report: all hard gates pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


def test_schema_accepts_canonical_report_and_rejects_corruption():
    assert _schema_errors(_report()) == []
    broken = [
        lambda r: r.pop("seeds"),
        lambda r: r.update(seeds=[]),
        lambda r: r["seeds"].update({"01": r["seeds"]["1"]}),
        lambda r: r["seeds"]["1"]["loopwm"].pop("aggregate"),
        lambda r: r["seeds"]["1"]["lrs"]["aggregate"].pop("assisted"),
        lambda r: r["seeds"]["2"]["loopwm"]["aggregate"]["baseline"].update(lateral_rms_m=math.nan),
        lambda r: r["seeds"]["1"]["lite"]["aggregate"]["assisted"].update(apogee_m=-1.0),
        lambda r: r.update(seed_order=[True, 2, 3]),
        lambda r: r.update(held_out_indices="16,17"),
        lambda r: r["students"][0].pop("name"),
        lambda r: r.update(exit_code=True),
        lambda r: r["verdict"].pop("pass"),
    ]
    for mutate in broken:
        report = _report()
        mutate(report)
        assert _schema_errors(report), f"corruption was accepted: {mutate}"


def test_oracle_positive_control_is_required_on_every_seed():
    assert _oracle_errors(_report()) == []
    assert _oracle_errors(_report(students_win=True)) == []
    broken = [
        lambda r: r["seeds"]["2"].pop("oracle_gtheta"),
        lambda r: r["seeds"]["3"]["oracle_gtheta"]["aggregate"]["assisted"].update(
            lateral_rms_m=2000.0
        ),
        lambda r: r["seeds"]["1"]["oracle_gtheta"]["aggregate"]["assisted"].update(apogee_m=1.0),
        lambda r: r["seeds"]["3"]["oracle_gtheta"]["aggregate"]["assisted"].update(
            apogee_m=math.inf
        ),
    ]
    for mutate in broken:
        report = _report()
        mutate(report)
        errors = _oracle_errors(report)
        assert errors and any("oracle" in error for error in errors), mutate


def test_held_out_provenance_is_not_fabricated():
    assert _provenance_errors(_report()) == []
    cases = [
        (lambda r: r["held_out_indices"].append(15), "training cell"),
        (lambda r: r.update(held_out_indices=[]), "must not be empty"),
        (lambda r: r.update(seed_order=[1, 2]), "seed keys"),
        (
            lambda r: r["seeds"]["1"]["loopwm"]["per_scenario"][0].update(scenario="scenario_0"),
            "not a declared held-out",
        ),
        (
            lambda r: r["seeds"]["2"]["oracle_gtheta"]["sha256"].update(baseline="abc"),
            "digest is not sha256",
        ),
        (lambda r: r["seeds"]["1"]["lrs"].pop("per_scenario"), "per-scenario evidence"),
    ]
    for mutate, message in cases:
        report = _report()
        mutate(report)
        errors = _provenance_errors(report)
        assert any(message in error for error in errors), (mutate, errors)


def test_no_surrogate_claims_are_allowed():
    assert _surrogate_errors(_report()) == []
    cases = [
        (lambda r: r.update(measurement="surrogate model fidelity estimate"), "flown-truth"),
        (lambda r: r.update(measurement="POTOKTRJ v1 flown-truth sidecar"), "disclaim"),
        (lambda r: r.update(supersedes="surrogate A/B is the best result"), "void"),
        (lambda r: r.update(sim_source="proxy model predicted the trajectory"), "physical oracle"),
        (lambda r: r.update(task="surrogate fidelity A/B"), "task makes a surrogate claim"),
    ]
    for mutate, message in cases:
        report = _report()
        mutate(report)
        errors = _surrogate_errors(report)
        assert any(message in error for error in errors), (mutate, errors)


def test_exit_code_matches_recomputed_verdict():
    losing = _report()
    assert _check(losing) == []
    assert losing["exit_code"] == 1 and len(losing["failure_location"]) >= 20

    winning = _report(students_win=True)
    assert _check(winning) == []
    assert winning["exit_code"] == 0 and winning["failure_location"] == ""

    broken = [
        lambda r: r.update(exit_code=0),
        lambda r: r["verdict"].__setitem__("pass", True),
        lambda r: r["verdict"].update(beats_lateral=True),
        lambda r: r["verdict"].update(primary="oracle_gtheta"),
        lambda r: r.update(failure_location=""),
    ]
    for mutate in broken:
        report = _report()
        mutate(report)
        assert _exit_errors(report), mutate


def test_gate_is_pure_and_json_roundtrip_stable():
    report = _report()
    snapshot = copy.deepcopy(report)
    assert _check(report) == _check(report) == []
    assert report == snapshot
    assert _check(json.loads(json.dumps(report, sort_keys=True))) == []
    assert json.dumps(report, sort_keys=True) == json.dumps(snapshot, sort_keys=True)


def test_cli_exit_code_contract(tmp_path: Path):
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_report()), encoding="utf-8")
    assert main(["gate", str(valid)]) == 0

    corrupted = _report()
    corrupted["seeds"]["1"].pop("oracle_gtheta")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(corrupted), encoding="utf-8")
    assert main(["gate", str(bad)]) == 1

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    assert main(["gate", str(malformed)]) == 2
    assert main([]) == 2

    script = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(valid)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert script.returncode == 0, script.stderr
    assert "gates pass" in script.stdout
