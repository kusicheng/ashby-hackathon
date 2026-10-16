"""K04: paired, offline probability robustness audit. Author: Kevin-C.

This measures supplied predictions; it does not run an image model, certify a
perturbation as realistic, or establish clinical performance.
"""

from __future__ import annotations

import math

from .safety import _metrics, _number, _schema, _text

AUTHOR = "Kevin-C"


def _rows(value: object, path: str) -> tuple[dict[str, dict], int]:
    if not isinstance(value, list) or not 1 <= len(value) <= 20000:
        raise ValueError(f"{path} must contain between 1 and 20000 rows")
    result, classes = {}, None
    for index, item in enumerate(value):
        here = f"{path}[{index}]"
        row = _schema(item, {"id", "probabilities", "label", "subgroup"}, set(), here)
        row_id = _text(row["id"], f"{here}.id")
        if row_id in result:
            raise ValueError(f"{path} contains a duplicate ID")
        subgroup = _text(row["subgroup"], f"{here}.subgroup")
        probs = row["probabilities"]
        if not isinstance(probs, list) or not 2 <= len(probs) <= 256:
            raise ValueError(f"{here}.probabilities must contain 2 to 256 values")
        probs = [_number(x, f"{here}.probabilities") for x in probs]
        if any(not 0 <= x <= 1 for x in probs):
            raise ValueError(f"{here}.probabilities values must be between 0 and 1")
        total = math.fsum(probs)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{here}.probabilities must sum to 1 within 1e-6")
        probs = [probability / total for probability in probs]
        if classes is not None and len(probs) != classes:
            raise ValueError(f"{path} has inconsistent class dimensions")
        classes = len(probs)
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < classes:
            raise ValueError(f"{here}.label must be an integer within the class range")
        result[row_id] = {"probabilities": probs, "label": label, "subgroup": subgroup}
    return result, classes


def _paired_metrics(clean: list[dict], perturbed: list[dict]) -> dict:
    before = [row["probabilities"] for row in clean]
    after = [row["probabilities"] for row in perturbed]
    clean_metrics = _metrics(clean, before, 0.0)
    perturbed_metrics = _metrics(perturbed, after, 0.0)
    flips, correct_to_wrong, wrong_to_correct, shifts = 0, 0, 0, []
    for row, first, second in zip(clean, before, after, strict=True):
        p = max(range(len(first)), key=first.__getitem__)
        q = max(range(len(second)), key=second.__getitem__)
        flips += p != q
        correct_to_wrong += p == row["label"] and q != row["label"]
        wrong_to_correct += p != row["label"] and q == row["label"]
        shifts.append(math.fsum(abs(a - b) for a, b in zip(first, second, strict=True)) / 2)
    count = len(clean)
    return {
        "count": count,
        "clean": clean_metrics,
        "perturbed": perturbed_metrics,
        "accuracy_change": perturbed_metrics["accuracy"] - clean_metrics["accuracy"],
        "brier_change": perturbed_metrics["brier"] - clean_metrics["brier"],
        "nll_change": perturbed_metrics["nll"] - clean_metrics["nll"],
        "ece_change": perturbed_metrics["ece_10_bins"] - clean_metrics["ece_10_bins"],
        "prediction_flip_rate": flips / count,
        "correct_to_wrong_count": correct_to_wrong,
        "wrong_to_correct_count": wrong_to_correct,
        "mean_total_variation": math.fsum(shifts) / count,
        "maximum_total_variation": max(shifts),
    }


def evaluate(payload: dict) -> dict:
    """Validate complete pairs and return aggregate and supported subgroup metrics."""
    payload = _schema(
        payload,
        {"clean", "perturbed", "provenance"},
        {"subgroup_minimum"},
        "payload",
    )
    provenance = _schema(
        payload["provenance"],
        {"model_id", "dataset_id", "perturbation_id"},
        set(),
        "provenance",
    )
    provenance = {key: _text(value, f"provenance.{key}") for key, value in provenance.items()}
    minimum = payload.get("subgroup_minimum", 5)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 2 <= minimum <= 20000:
        raise ValueError("subgroup_minimum must be an integer between 2 and 20000")
    clean, classes = _rows(payload["clean"], "clean")
    perturbed, other_classes = _rows(payload["perturbed"], "perturbed")
    if classes != other_classes:
        raise ValueError("clean and perturbed class dimensions must match")
    if clean.keys() != perturbed.keys():
        raise ValueError("clean and perturbed must contain exactly the same complete ID set")
    for row_id, row in clean.items():
        if row["label"] != perturbed[row_id]["label"]:
            raise ValueError("paired labels must not change under perturbation")
        if row["subgroup"] != perturbed[row_id]["subgroup"]:
            raise ValueError("paired subgroup membership must not change")
    ids = sorted(clean)
    overall = _paired_metrics([clean[key] for key in ids], [perturbed[key] for key in ids])
    groups = []
    for subgroup in sorted({row["subgroup"] for row in clean.values()}):
        group_ids = [key for key in ids if clean[key]["subgroup"] == subgroup]
        supported = len(group_ids) >= minimum
        groups.append(
            {
                "subgroup": subgroup,
                "count": len(group_ids),
                "evidence": "descriptive_only" if supported else "insufficient_support",
                "metrics": _paired_metrics(
                    [clean[key] for key in group_ids], [perturbed[key] for key in group_ids]
                )
                if supported
                else None,
                "reason": (
                    "Minimum support met; this is not a statistical validity guarantee."
                    if supported
                    else f"Fewer than {minimum} pairs; no subgroup performance conclusion."
                ),
            }
        )
    return {
        "idea_id": "K04",
        "author": AUTHOR,
        "status": "prototype",
        "provenance": provenance,
        "subgroup_minimum": minimum,
        "overall": overall,
        "subgroups": groups,
        "limitations": [
            "Supplied predictions and provenance are not independently verified.",
            "Matched IDs do not establish independent subjects or realistic perturbations.",
            "No raw case IDs are exported; subgroup/provenance text can still be sensitive.",
            "A minimum subgroup count does not establish statistical or clinical validity.",
            "Probability sums within 1e-6 are normalized; argmax ties choose the lowest class.",
            "No clinical readiness, deployment authorization, or safety certification.",
        ],
    }


def demo_payload(unstable: bool = False) -> dict:
    """Return the same fictional cohort under stable or deliberately unstable shifts."""
    if not isinstance(unstable, bool):
        raise ValueError("unstable must be a boolean")
    clean, perturbed = [], []
    for index in range(15):
        label = index % 2
        subgroup = "group-A" if index < 8 else "group-B" if index < 13 else "small-group"
        base = {
            "id": f"blind-{index:03d}",
            "label": label,
            "subgroup": subgroup,
        }
        original = [0.9 if j == label else 0.1 for j in range(2)]
        confidence = 0.2 if unstable and index < 10 else 0.88
        shifted = [confidence if j == label else 1.0 - confidence for j in range(2)]
        clean.append({**base, "probabilities": original})
        perturbed.append({**base, "probabilities": shifted})
    return {
        "clean": clean,
        "perturbed": perturbed,
        "subgroup_minimum": 5,
        "provenance": {
            "model_id": "fictional-paired-predictor-v1",
            "dataset_id": "invented-15-pairs-v1",
            "perturbation_id": "deliberate-score-shift" if unstable else "small-score-shift",
        },
    }


def run_demo() -> dict:
    """Compare stable and unstable supplied predictions on one synthetic cohort."""
    return {
        "idea_id": "K04",
        "author": AUTHOR,
        "stable": evaluate(demo_payload()),
        "unstable": evaluate(demo_payload(unstable=True)),
    }
