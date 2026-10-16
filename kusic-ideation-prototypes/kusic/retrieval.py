"""K02: synthetic, local retrieval with patient and provenance constraints.

The caller supplies authorization claims; this prototype does not authenticate
users. Exact similarity search and synthetic checks do not establish clinical
utility. Outputs contain opaque case/source references, never input identities.
"""

# ruff: noqa: TRY004
# JSON schema failures deliberately share the documented ValueError contract.

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from datetime import UTC, datetime
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?"
    r"(?:Z|[+-](?:[01]\d|2[0-3]):[0-5]\d)\Z"
)
_PURPOSES = {"research", "education"}
_MAX_RECORDS = 10_000
_MAX_DIMENSIONS = 4096


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a nonempty, unpadded ASCII identifier")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        raise ValueError(f"{field} must be an ISO timestamp with seconds and timezone")
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except (ValueError, OverflowError):
        raise ValueError(f"{field} is not a valid timestamp") from None


def _vector(value: Any, field: str, dimensions: int | None = None) -> list[float]:
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_DIMENSIONS:
        raise ValueError(f"{field} must contain 1..{_MAX_DIMENSIONS} numbers")
    if dimensions is not None and len(value) != dimensions:
        raise ValueError(f"{field} dimension does not match the query")
    numbers = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (float, int)):
            raise ValueError(f"{field} must contain finite numbers, excluding booleans")
        try:
            number = float(item)
        except OverflowError:
            raise ValueError(f"{field} must contain finite numbers") from None
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain finite numbers")
        numbers.append(number)
    # Scale first so finite values near float limits cannot overflow the norm.
    scale = max(map(abs, numbers))
    if scale == 0:
        raise ValueError(f"{field} must be nonzero")
    scaled = [number / scale for number in numbers]
    norm = math.sqrt(math.fsum(number * number for number in scaled))
    return [number / norm for number in scaled]


def _unique_identity(seen: dict[tuple[str, str], str], value: str, institution: str) -> None:
    key = (institution.casefold(), value.casefold())
    if key in seen and seen[key] != value:
        raise ValueError("Patient identifiers differ only by case; identity is ambiguous")
    seen[key] = value


def _reference(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _validate(payload: dict) -> tuple[dict, list[dict], int, str]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    query = payload.get("query")
    records = payload.get("records")
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    if not isinstance(records, list) or len(records) > _MAX_RECORDS:
        raise ValueError(f"records must be an array of at most {_MAX_RECORDS} objects")
    top_k = payload.get("top_k", 3)
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k must be an integer from 1 to 50")
    purpose = payload.get("purpose", "research")
    if not isinstance(purpose, str) or purpose not in _PURPOSES:
        raise ValueError("purpose must be research or education")
    q = {
        "patient_id": _identifier(query.get("patient_id"), "query.patient_id"),
        "institution": _identifier(query.get("institution"), "query.institution"),
        "as_of": _timestamp(query.get("as_of"), "query.as_of"),
        "embedding": _vector(query.get("embedding"), "query.embedding"),
    }
    authorized = query.get("authorized_patient_ids")
    if not isinstance(authorized, list):
        raise ValueError("query.authorized_patient_ids must be an array")
    identities: dict[tuple[str, str], str] = {}
    _unique_identity(identities, q["patient_id"], q["institution"])
    auth_ids: set[str] = set()
    for value in authorized:
        value = _identifier(value, "query.authorized_patient_ids")
        _unique_identity(identities, value, q["institution"])
        if value in auth_ids:
            raise ValueError("Duplicate authorized patient identifier")
        auth_ids.add(value)
    q["authorized_patient_ids"] = auth_ids
    normalized: list[dict] = []
    record_ids: set[str] = set()
    institutions = {q["institution"].casefold(): q["institution"]}
    for index, record in enumerate(records):
        field = f"records[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{field} must be an object")
        item = {
            "record_id": _identifier(record.get("record_id"), f"{field}.record_id"),
            "patient_id": _identifier(record.get("patient_id"), f"{field}.patient_id"),
            "institution": _identifier(record.get("institution"), f"{field}.institution"),
            "observed_at": _timestamp(record.get("observed_at"), f"{field}.observed_at"),
            "embedding": _vector(
                record.get("embedding"), f"{field}.embedding", len(q["embedding"])
            ),
        }
        identifier_key = item["record_id"].casefold()
        if identifier_key in record_ids:
            raise ValueError("Duplicate or case-ambiguous record identifier")
        record_ids.add(identifier_key)
        institution_key = item["institution"].casefold()
        previous = institutions.setdefault(institution_key, item["institution"])
        if previous != item["institution"]:
            raise ValueError("Institution identifiers differ only by case")
        _unique_identity(identities, item["patient_id"], item["institution"])
        consent = record.get("consent")
        if type(consent) is not bool:
            raise ValueError(f"{field}.consent must be a boolean")
        source = record.get("source_ref")
        if source is not None and (
            not isinstance(source, str)
            or len(source) > 2048
            or any(ord(char) < 32 or ord(char) == 127 for char in source)
        ):
            raise ValueError(f"{field}.source_ref must be text without control characters")
        permitted_use = record.get("permitted_use")
        if not isinstance(permitted_use, str) or not permitted_use.strip():
            raise ValueError(f"{field}.permitted_use must be nonempty text")
        item.update(consent=consent, source_ref=source, permitted_use=permitted_use)
        normalized.append(item)
    return q, normalized, top_k, purpose


def _exclusions(query: dict, record: dict, purpose: str) -> list[str]:
    reasons = []
    if query["patient_id"] not in query["authorized_patient_ids"]:
        reasons.append("query_patient_not_authorized")
    if record["patient_id"] != query["patient_id"]:
        reasons.append("different_patient")
    if record["patient_id"] not in query["authorized_patient_ids"]:
        reasons.append("record_patient_not_authorized")
    if record["institution"] != query["institution"]:
        reasons.append("different_institution")
    if record["observed_at"] >= query["as_of"]:
        reasons.append("not_strictly_earlier")
    if not record["consent"]:
        reasons.append("consent_absent")
    if not record["source_ref"] or not record["source_ref"].strip():
        reasons.append("source_absent")
    if record["permitted_use"] != purpose:
        reasons.append("use_not_permitted")
    return reasons


def _precision(eligible: int, returned: int) -> float | None:
    return eligible / returned if returned else None


def _cosine(left: list[float], right: list[float]) -> float:
    similarity = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return max(-1.0, min(1.0, similarity))


def evaluate(payload: dict) -> dict:
    """Validate input and return privacy-minimized results and baseline metrics.

    Payload: query(patient_id, institution, as_of, embedding,
    authorized_patient_ids), records(record_id, patient_id, institution,
    observed_at, embedding, source_ref, permitted_use, consent), optional top_k
    (1..50, default 3) and purpose (research/education, default research).
    All timestamps include seconds and a timezone. Malformed input raises
    ValueError. Missing/blank source_ref is a valid but ineligible record.

    Eligibility precision means the fraction satisfying the supplied metadata
    policy, not retrieval relevance or clinical accuracy. Baseline content is
    never returned. Opaque references are pseudonyms, not anonymization.
    """
    query, records, top_k, purpose = _validate(payload)
    exclusions: Counter = Counter()
    evaluated = []
    for record in records:
        reasons = _exclusions(query, record, purpose)
        exclusions.update(reasons)
        evaluated.append((record, reasons))
    # The usable retrieval path filters metadata before calculating similarity.
    # Stable lexicographic tie-break uses the validated unique record identifier.
    eligible = sorted(
        [
            (record, [], _cosine(query["embedding"], record["embedding"]))
            for record, reasons in evaluated
            if not reasons
        ],
        key=lambda row: (-row[2], row[0]["record_id"]),
    )
    guarded = eligible[:top_k]
    # This intentionally unsafe comparison path exports aggregate counts only.
    baseline = sorted(
        [
            (record, reasons, _cosine(query["embedding"], record["embedding"]))
            for record, reasons in evaluated
        ],
        key=lambda row: (-row[2], row[0]["record_id"]),
    )[:top_k]
    baseline_eligible = sum(not reasons for _, reasons, _ in baseline)
    baseline_reasons = Counter(reason for _, reasons, _ in baseline for reason in reasons)
    matches = [
        {
            "case_ref": _reference("case", record["institution"], record["record_id"]),
            "source_ref": _reference("source", record["source_ref"]),
            "observed_at": record["observed_at"].isoformat(),
            "similarity": round(similarity, 12),
        }
        for record, _, similarity in guarded
    ]
    empty_reasons = []
    if not matches:
        if query["patient_id"] not in query["authorized_patient_ids"]:
            empty_reasons.append("query_patient_not_authorized")
        empty_reasons.append("no_records_supplied" if not records else "no_eligible_prior")
    return {
        "idea_id": "K02",
        "author": "Kevin-C",
        "status": "results_available" if matches else "no_eligible_results",
        "purpose": purpose,
        "matches": matches,
        "empty_reasons": empty_reasons,
        "counts": {
            "records_supplied": len(records),
            "eligible": len(eligible),
            "excluded": len(records) - len(eligible),
            "returned": len(matches),
        },
        "excluded_reason_counts": dict(sorted(exclusions.items())),
        "baseline": {
            "method": "unrestricted_cosine_top_k_aggregate_only",
            "returned": len(baseline),
            "eligible": baseline_eligible,
            "violating": len(baseline) - baseline_eligible,
            "eligibility_precision": _precision(baseline_eligible, len(baseline)),
            "violation_reason_counts": dict(sorted(baseline_reasons.items())),
        },
        "guarded": {
            "method": "identity_provenance_filter_then_cosine",
            "returned": len(matches),
            "eligibility_precision": _precision(len(matches), len(matches)),
            "eligible_recall_at_k": _precision(len(matches), len(eligible)),
        },
        "candidate_eligibility_rate": _precision(len(eligible), len(records)),
        "limitations": [
            "Synthetic research demonstration; similarity does not establish diagnosis.",
            "Authorization and consent are caller-supplied claims, not verified access.",
            "Eligibility metrics test metadata rules, not clinical relevance or accuracy.",
            "Opaque references remain pseudonyms; real deployment needs access auditing.",
            "Reason counts overlap; one record can fail several constraints.",
        ],
    }


def resolve_match(payload: dict, case_ref: str) -> dict:
    """Resolve a current match for opt-in local source inspection.

    Revalidates the payload, authorization, provenance and current top-k results.
    Returns the original source_ref and a one-based row in payload['records'];
    it does not open any URL or file. This local detail can contain sensitive
    source paths and is deliberately absent from the default report export.
    """
    if not isinstance(case_ref, str) or not re.fullmatch(r"case-[0-9a-f]{20}", case_ref):
        raise ValueError("case_ref must be an opaque reference from current matches")
    current = evaluate(payload)
    if case_ref not in {match["case_ref"] for match in current["matches"]}:
        raise ValueError("Case reference is not in current eligible retrieval results")
    for index, record in enumerate(payload["records"], start=1):
        reference = _reference("case", record["institution"], record["record_id"])
        if reference == case_ref:
            return {
                "case_ref": case_ref,
                "source_ref": record["source_ref"],
                "input_row": index,
            }
    raise ValueError("Case reference is not in current eligible retrieval results")


def demo_payload() -> dict:
    """Return invented records, including tempting but ineligible near matches."""
    records = []
    changes = [
        ("prior-a", [0.8, 0.6], {}),
        ("prior-b", [0.6, 0.8], {}),
        ("other-patient", [1.0, 0.0], {"patient_id": "SYNTH-P02"}),
        ("future", [0.999, 0.01], {"observed_at": "2026-11-01T10:00:00Z"}),
        ("no-consent", [0.998, 0.02], {"consent": False}),
        ("no-source", [0.997, 0.03], {"source_ref": None}),
        ("other-institution", [0.996, 0.04], {"institution": "SYNTH-LAB-B"}),
        ("education-only", [0.995, 0.05], {"permitted_use": "education"}),
    ]
    for record_id, embedding, updates in changes:
        record = {
            "record_id": record_id,
            "patient_id": "SYNTH-P01",
            "institution": "SYNTH-LAB-A",
            "observed_at": "2026-09-01T10:00:00Z",
            "embedding": embedding,
            "source_ref": f"synthetic://invented-priors/{record_id}",
            "permitted_use": "research",
            "consent": True,
        }
        record.update(updates)
        records.append(record)
    return {
        "query": {
            "patient_id": "SYNTH-P01",
            "institution": "SYNTH-LAB-A",
            "as_of": "2026-10-16T10:00:00Z",
            "embedding": [1.0, 0.0],
            "authorized_patient_ids": ["SYNTH-P01"],
        },
        "records": records,
        "top_k": 3,
        "purpose": "research",
    }


def run_demo() -> dict:
    """Run the reproducible synthetic authorization comparison."""
    return evaluate(demo_payload())
