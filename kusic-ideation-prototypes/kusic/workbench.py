"""Kevin-C's shared input boundary and comparison reports for four prototypes."""

import csv
import hashlib
import importlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

IDEAS = {
    "K01": {"module": "safety", "title": "Calibrated second-look gate"},
    "K02": {"module": "retrieval", "title": "Patient-prior retrieval"},
    "K03": {"module": "hints", "title": "Trainee hint ladder"},
    "K04": {"module": "robustness", "title": "Paired robustness audit"},
}

SCORES = {
    "K01": {
        "complexity": 5,
        "user_friendliness": 6,
        "legal_issues": 7,
        "creativity": 5,
        "rationale": {
            "complexity": "Three disjoint cohorts, calibration, selection and drift controls.",
            "user_friendliness": "Review reasons and coverage are visible; interpreting risk needs training.",
            "legal_issues": "Intended patient-specific confidence display needs institution/device review.",
            "creativity": "Combines calibration and deferral with provenance and held-out controls.",
        },
        "recommendation": "Continue as a research gate; externally validate before patient use.",
    },
    "K02": {
        "complexity": 6,
        "user_friendliness": 7,
        "legal_issues": 6,
        "creativity": 5,
        "rationale": {
            "complexity": "Identity/time/access checks plus optional CONCH runtime, safe weights and image preprocessing.",
            "user_friendliness": "Ranked evidence references and exclusion reasons are actionable.",
            "legal_issues": "Intended real-record retrieval handles sensitive longitudinal records.",
            "creativity": "Familiar retrieval anchored to the patient's own prior records.",
        },
        "recommendation": "Continue after identity/access integration; similarity is not diagnosis.",
    },
    "K03": {
        "complexity": 3,
        "user_friendliness": 7,
        "legal_issues": 3,
        "creativity": 5,
        "rationale": {
            "complexity": "Small authored-content workflow with metadata controls.",
            "user_friendliness": "Three progressive hints with source and review status.",
            "legal_issues": "Educational content still requires permissions and intent review.",
            "creativity": "Recombines hint progression with review and source requirements.",
        },
        "recommendation": "Keep as an education module; obtain expert content and learning evidence.",
    },
    "K04": {
        "complexity": 4,
        "user_friendliness": 6,
        "legal_issues": 3,
        "creativity": 5,
        "rationale": {
            "complexity": "Paired data checks, subgroup support and metric interpretation.",
            "user_friendliness": "Clear stability report; users must understand cohort limitations.",
            "legal_issues": "Offline research audit has limited direct patient-care intent.",
            "creativity": "Combines familiar paired controls and small-group evidence in one audit.",
        },
        "recommendation": "Lead hackathon demonstration; add real licensed exports and external cases.",
    },
}


def module_for(idea_id):
    if idea_id not in IDEAS:
        raise ValueError(f"Unknown idea {idea_id!r}; choose K01, K02, K03 or K04")
    return importlib.import_module(f"kusic.{IDEAS[idea_id]['module']}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_request(text):
    """Do not let duplicate keys or nonstandard NaN silently alter the request."""
    if len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("Input exceeds the 2 MB prototype limit; use a smaller cohort")

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value is not supported: {value}")

    try:
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error
    if not isinstance(result, dict):
        raise ValueError("Input must be a JSON object")
    return result


def read_request(path):
    path = Path(path)
    if path.stat().st_size > 2_000_000:
        raise ValueError("Input exceeds the 2 MB prototype limit")
    return parse_request(path.read_text(encoding="utf-8-sig"))


def image_manifest_sources(payload, base_dir):
    """Collect declared image assets without loading optional model packages."""
    items = [payload.get("query")]
    if isinstance(payload.get("records"), list):
        items.extend(payload["records"])
    paths = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("image_path"), str):
            path = Path(item["image_path"])
            paths.append(path if path.is_absolute() else Path(base_dir) / path)
    return paths


def protect_outputs(outputs, inputs):
    """Refuse output aliases to declared inputs, including existing hard links."""
    resolved_inputs = [Path(path).resolve() for path in inputs]
    resolved_outputs = []
    for output in outputs:
        target = Path(output).resolve()
        for other in resolved_inputs + resolved_outputs:
            same = target == other
            if not same and target.exists() and other.exists():
                same = target.samefile(other)
            if same:
                raise ValueError("Output would overwrite a declared input asset or another output")
        resolved_outputs.append(target)


def run_request(idea_id, payload):
    """Independent of the GUI so all actions can also be exercised in tests/CLI."""
    if not isinstance(payload, dict):
        raise ValueError("Input must be a JSON object")
    started = time.perf_counter()
    result = module_for(idea_id).evaluate(payload)
    return {
        "idea_id": idea_id,
        "author": "Kevin-C",
        "title": IDEAS[idea_id]["title"],
        "result": result,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "input_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "limitations": "Research/education prototype. Exported model values are supplied, not independently verified.",
    }


def effectiveness(idea_id, evidence):
    """Task-specific measurements in a consistent comparison envelope."""
    if idea_id == "K01":
        return {
            "measure": "held_out_nll",
            "direction": "lower",
            "baseline": evidence["held_out"]["raw"]["nll"],
            "prototype": evidence["held_out"]["calibrated"]["nll"],
            "evaluation_cases": evidence["split_counts"]["evaluation"],
            "interpretation": "Probability error on invented logits; accuracy need not improve.",
        }
    if idea_id == "K02":
        return {
            "measure": "returned_metadata_eligibility_precision",
            "direction": "higher",
            "baseline": evidence["baseline"]["eligibility_precision"],
            "prototype": evidence["guarded"]["eligibility_precision"],
            "evaluation_cases": evidence["counts"]["records_supplied"],
            "interpretation": "Metadata-policy compliance, not diagnostic similarity or relevance.",
        }
    if idea_id == "K03":
        return {
            "measure": "blocked_content_exposures",
            "direction": "lower",
            "baseline": evidence["metrics"]["baseline_blocked_content_exposures"],
            "prototype": evidence["metrics"]["guarded_blocked_content_exposures"],
            "evaluation_cases": evidence["metrics"]["scenarios"],
            "interpretation": "Reviewed-content controls only; learning gain remains unmeasured.",
        }
    return {
        "measure": "paired_prediction_flip_rate",
        "direction": "descriptive",
        "baseline": evidence["stable"]["overall"]["prediction_flip_rate"],
        "prototype": evidence["unstable"]["overall"]["prediction_flip_rate"],
        "evaluation_cases": evidence["stable"]["overall"]["count"],
        "interpretation": "Stable versus deliberately unstable control; not a treatment improvement.",
    }


def portfolio():
    """Run every predefined synthetic comparison with one reporting schema."""
    records = []
    for idea_id, definition in IDEAS.items():
        started = time.perf_counter()
        evidence = module_for(idea_id).run_demo()
        scores = SCORES[idea_id]
        index = (
            11
            - scores["complexity"]
            + scores["user_friendliness"]
            + 11
            - scores["legal_issues"]
            + scores["creativity"]
        ) / 4
        records.append(
            {
                "idea_id": idea_id,
                "author": "Kevin-C",
                "title": definition["title"],
                "status": "PROTOTYPED",
                "scores": {
                    key: scores[key]
                    for key in ("complexity", "user_friendliness", "legal_issues", "creativity")
                },
                "score_rationale": scores["rationale"],
                "synthetic_only_legal_issues": 1,
                "priority_index": round(index, 2),
                "recommendation": scores["recommendation"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "evidence": evidence,
                "effectiveness": effectiveness(idea_id, evidence),
                "evidence_path": f"#/ideas/{len(records)}/evidence",
            }
        )
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "schema_version": "1.0",
        "author": "Kevin-C",
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "data": "Invented synthetic fixtures only; no CONCH inference or clinical validation.",
        "score_interpretation": "Complexity, friendliness and creativity assess the scoped product design; legal_issues assesses intended real deployment. Current synthetic-only legal burden is separately 1 for all. No participant usability study or legal determination.",
        "priority_formula": "((11-complexity)+user_friendliness+(11-legal_issues)+creativity)/4",
        "priority_limitation": "Equal-weight screening only; task-specific effectiveness metrics are not interchangeable.",
        "source_sha256": source_hashes,
        "ideas": records,
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_comparison_csv(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "idea_id",
        "author",
        "title",
        "status",
        "complexity",
        "user_friendliness",
        "legal_issues",
        "synthetic_only_legal_issues",
        "creativity",
        "priority_index",
        "elapsed_ms",
        "recommendation",
        "measure",
        "direction",
        "baseline",
        "prototype",
        "evaluation_cases",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in report["ideas"]:
            writer.writerow(
                {
                    **{key: record[key] for key in fields if key in record},
                    **record["scores"],
                    **{
                        key: record["effectiveness"][key]
                        for key in (
                            "measure",
                            "direction",
                            "baseline",
                            "prototype",
                            "evaluation_cases",
                        )
                    },
                }
            )
