"""K01: offline calibration and selective review experiment. Author: Kevin-C.

The target error is an empirical selection constraint, not a safety guarantee.
All fitting uses calibration/selection splits; evaluation never tunes a setting.
"""

from __future__ import annotations

import math

AUTHOR = "Kevin-C"
SCOPES = {"synthetic_research", "authorized_nonclinical_research"}
EPSILON = 1e-15


def _schema(value: object, required: set[str], optional: set[str], path: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path} keys must be strings")
    missing = required - value.keys()
    extra = value.keys() - required - optional
    if missing or extra:
        raise ValueError(
            f"{path}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}"
        )
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{path} must be a nonempty trimmed string")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{path} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be a finite number")
    return result


def _rows(value: object, path: str) -> tuple[list[dict], int]:
    if not isinstance(value, list) or not 1 <= len(value) <= 20000:
        raise ValueError(f"{path} must contain between 1 and 20000 rows")
    rows, seen, classes = [], set(), None
    for index, item in enumerate(value):
        here = f"{path}[{index}]"
        row = _schema(item, {"id", "logits", "label"}, set(), here)
        row_id = _text(row["id"], f"{here}.id")
        if row_id in seen:
            raise ValueError(f"{path} contains a duplicate ID")
        seen.add(row_id)
        logits = row["logits"]
        if not isinstance(logits, list) or not 2 <= len(logits) <= 256:
            raise ValueError(f"{here}.logits must contain 2 to 256 class scores")
        if classes is not None and len(logits) != classes:
            raise ValueError(f"{path} has inconsistent class dimensions")
        classes = len(logits)
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"{here}.label must be an integer class index")
        if not 0 <= label < classes:
            raise ValueError(f"{here}.label is outside the class range")
        rows.append(
            {
                "id": row_id,
                "logits": [_number(x, f"{here}.logits") for x in logits],
                "label": label,
            }
        )
    return rows, classes


def _probabilities(logits: list[float], inverse_temperature: float) -> list[float]:
    largest = max(logits)
    # Subtraction can become -infinity for extreme finite input; exp(-inf)=0.
    weights = [math.exp((x - largest) * inverse_temperature) for x in logits]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _predictions(rows: list[dict], beta: float) -> list[list[float]]:
    return [_probabilities(row["logits"], beta) for row in rows]


def _nll(rows: list[dict], probabilities: list[list[float]]) -> float:
    return math.fsum(
        -math.log(max(EPSILON, probs[row["label"]]))
        for row, probs in zip(rows, probabilities, strict=True)
    ) / len(rows)


def _calibration_objective(rows: list[dict], beta: float) -> float:
    losses = []
    for row in rows:
        logits = row["logits"]
        largest = max(logits)
        difference = largest - logits[row["label"]]
        if math.isfinite(difference):
            penalty = difference * beta
        else:
            # Opposite extreme signs may overflow before a small beta rescales.
            penalty = largest * beta - logits[row["label"]] * beta
        normalizer = math.fsum(math.exp((x - largest) * beta) for x in logits)
        loss = penalty + math.log(normalizer)
        # Divide first so a representable mean does not overflow a running sum.
        losses.append(loss / len(rows))
    try:
        return math.fsum(losses)
    except OverflowError:
        return math.inf


def _fit_temperature(rows: list[dict]) -> float:
    # NLL is convex in inverse temperature. The bounded search includes T=1.
    left, right = 0.01, 20.0
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = right - ratio * (right - left), left + ratio * (right - left)
    f1, f2 = _calibration_objective(rows, x1), _calibration_objective(rows, x2)
    for _ in range(72):
        if f1 < f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - ratio * (right - left)
            f1 = _calibration_objective(rows, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + ratio * (right - left)
            f2 = _calibration_objective(rows, x2)
    candidates = [0.01, 1.0, 20.0, (left + right) / 2]
    beta = min(candidates, key=lambda b: _calibration_objective(rows, b))
    return 1.0 / beta


def _select_threshold(
    rows: list[dict], probabilities: list[list[float]], target_error: float
) -> dict:
    scored = sorted(
        (
            (max(p), max(range(len(p)), key=p.__getitem__) != row["label"])
            for row, p in zip(rows, probabilities, strict=True)
        ),
        reverse=True,
    )
    accepted, errors, best = 0, 0, None
    for index, (confidence, error) in enumerate(scored):
        accepted += 1
        errors += int(error)
        # Equal confidence cases must be retained or rejected together.
        if index + 1 < len(scored) and scored[index + 1][0] == confidence:
            continue
        if errors / accepted <= target_error:
            best = {
                "threshold": confidence,
                "selection_coverage": accepted / len(rows),
                "selection_accepted_error": errors / accepted,
            }
    return best or {
        "threshold": None,
        "selection_coverage": 0.0,
        "selection_accepted_error": None,
    }


def _metrics(rows: list[dict], probabilities: list[list[float]], threshold: float | None) -> dict:
    bins = [[] for _ in range(10)]
    correct, brier, accepted, errors = 0, [], 0, 0
    for row, p in zip(rows, probabilities, strict=True):
        confidence = max(p)
        hit = max(range(len(p)), key=p.__getitem__) == row["label"]
        correct += hit
        brier.append(math.fsum((v - int(j == row["label"])) ** 2 for j, v in enumerate(p)))
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
        if threshold is not None and confidence >= threshold:
            accepted += 1
            errors += not hit
    ece = math.fsum(
        abs(math.fsum(c - hit for c, hit in group)) / len(rows) for group in bins if group
    )
    return {
        "count": len(rows),
        "accuracy": correct / len(rows),
        "brier": math.fsum(brier) / len(rows),
        "nll": _nll(rows, probabilities),
        "ece_10_bins": ece,
        "coverage": accepted / len(rows),
        "accepted_count": accepted,
        "accepted_error": errors / accepted if accepted else None,
    }


def evaluate(payload: dict) -> dict:
    """Validate independent splits, fit on development data, report held-out data."""
    payload = _schema(
        payload,
        {"calibration", "selection", "evaluation", "scope", "provenance"},
        {"target_error"},
        "payload",
    )
    scope = _text(payload["scope"], "scope")
    if scope not in SCOPES:
        raise ValueError("scope must be synthetic_research or authorized_nonclinical_research")
    provenance = _schema(
        payload["provenance"], {"model_id", "dataset_id", "allowed_scope"}, set(), "provenance"
    )
    provenance = {key: _text(value, f"provenance.{key}") for key, value in provenance.items()}
    if provenance["allowed_scope"] != scope:
        raise ValueError("scope is not permitted by provenance.allowed_scope")
    target = _number(payload.get("target_error", 0.1), "target_error")
    if not 0 <= target <= 1:
        raise ValueError("target_error must be between 0 and 1")
    splits, classes, seen = {}, None, set()
    for name in ("calibration", "selection", "evaluation"):
        rows, dimension = _rows(payload[name], name)
        if classes is not None and dimension != classes:
            raise ValueError("all splits must have the same class dimension")
        classes = dimension
        ids = {row["id"] for row in rows}
        if ids & seen:
            raise ValueError("calibration, selection, and evaluation IDs must be disjoint")
        seen.update(ids)
        splits[name] = rows
    temperature = _fit_temperature(splits["calibration"])
    policies, held_out, probabilities = {}, {}, {}
    for name, beta in (("raw", 1.0), ("calibrated", 1.0 / temperature)):
        policies[name] = _select_threshold(
            splits["selection"], _predictions(splits["selection"], beta), target
        )
        probabilities[name] = _predictions(splits["evaluation"], beta)
        held_out[name] = _metrics(
            splits["evaluation"], probabilities[name], policies[name]["threshold"]
        )
    per_row = []
    threshold = policies["calibrated"]["threshold"]
    for index, p in enumerate(probabilities["calibrated"]):
        retained = threshold is not None and max(p) >= threshold
        reasons = []
        if threshold is None:
            reasons.append("No nonempty selection policy met the target error.")
        elif not retained:
            reasons.append("Confidence falls below the threshold fitted on selection data.")
        reasons.append(
            "Research output requires human interpretation; confidence is not correctness."
        )
        per_row.append(
            {
                "case": index + 1,
                "predicted_class": max(range(classes), key=p.__getitem__),
                "confidence": max(p),
                "retain_for_research": retained,
                "action": "inspect_prediction" if retained else "human_review",
                "human_review_reasons": reasons,
            }
        )
    return {
        "idea_id": "K01",
        "author": AUTHOR,
        "status": "prototype",
        "scope": scope,
        "provenance": provenance,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "temperature": temperature,
        "temperature_bounds": [0.05, 100.0],
        "fit_objective": "Unclipped log-sum-exp NLL on calibration rows only",
        "target_selection_error": target,
        "policies": policies,
        "held_out": held_out,
        "rows": per_row,
        "limitations": [
            "Selection error is empirical; held-out error can exceed the target.",
            "Disjoint IDs do not establish independent patients, sites, or distributions.",
            "Provenance is caller-supplied metadata and is not independently verified.",
            "Reported probability NLL clips below 1e-15; fitting uses unclipped log-sum-exp NLL.",
            "ECE uses ten fixed confidence bins; argmax ties choose the lowest class index.",
            "No clinical effectiveness, deployment authorization, or safety certification.",
        ],
    }


def demo_payload() -> dict:
    """Return fictional overconfident predictions with independent split IDs."""

    def row(prefix: str, index: int, gap: float, correct: bool) -> dict:
        label = index % 2
        predicted = label if correct else 1 - label
        return {
            "id": f"{prefix}-{index:03d}",
            "logits": [gap if j == predicted else 0.0 for j in range(2)],
            "label": label,
        }

    return {
        "scope": "synthetic_research",
        "provenance": {
            "model_id": "fictional-overconfident-v1",
            "dataset_id": "invented-split-cases-v1",
            "allowed_scope": "synthetic_research",
        },
        "target_error": 0.1,
        "calibration": [row("cal", i, 4.0, i < 32) for i in range(40)],
        "selection": [row("sel", i, 6.0 if i < 12 else 1.0, i < 12) for i in range(16)],
        "evaluation": [row("eval", i, 6.0 if i < 10 else 1.0, i < 8 or i >= 13) for i in range(16)],
    }


def run_demo() -> dict:
    """Run the deterministic K01 demonstration."""
    return evaluate(demo_payload())
