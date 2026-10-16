"""Kevin-C: reviewed, source-linked educational prompts without generation."""

from datetime import date


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value.strip()


def _day(value, name):
    try:
        return date.fromisoformat(_text(value, name))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from error


def evaluate(payload):
    """Return a single reviewed hint, or an explainable educational-content block."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    intent = payload.get("intent")
    if not isinstance(intent, str) or intent not in {"education", "patient_diagnosis"}:
        raise ValueError("intent must be education or patient_diagnosis")
    lesson_id = _text(payload.get("lesson_id"), "lesson_id")
    step = payload.get("step", 1)
    if type(step) is not int or not 1 <= step <= 3:
        raise ValueError("step must be 1, 2, or 3")
    as_of = _day(payload.get("as_of"), "as_of")
    max_age = payload.get("max_review_age_days", 365)
    if type(max_age) is not int or not 1 <= max_age <= 730:
        raise ValueError("max_review_age_days must be an integer from 1 to 730")
    lessons = payload.get("lessons")
    if not isinstance(lessons, list) or len(lessons) > 1000:
        raise ValueError("lessons must be an array with at most 1000 items")
    ids = set()
    for lesson in lessons:
        if not isinstance(lesson, dict):
            raise ValueError("Each lesson must be an object")
        identifier = _text(lesson.get("lesson_id"), "lesson_id")
        if identifier in ids:
            raise ValueError("Duplicate lesson_id is ambiguous")
        ids.add(identifier)
    matches = [lesson for lesson in lessons if lesson["lesson_id"] == lesson_id]
    reasons = []
    if intent != "education":
        reasons.append("patient_care_request_not_supported")
    if not matches:
        reasons.append("lesson_not_found")
    result = {
        "idea_id": "K03",
        "author": "Kevin-C",
        "purpose": "Fictional educational demonstration; no diagnostic advice.",
        "lesson_id": lesson_id,
        "step": step,
    }
    if reasons:
        return {**result, "status": "blocked", "reasons": reasons, "hint": None}
    lesson = matches[0]
    if lesson.get("use") != "education":
        reasons.append("educational_use_not_documented")
    permission = lesson.get("permission")
    if not isinstance(permission, str) or permission not in {"original_author", "licensed"}:
        reasons.append("content_permission_missing")
    for field in ("source_ref", "attribution", "reviewer", "license_ref"):
        if not isinstance(lesson.get(field), str) or not lesson[field].strip():
            reasons.append(f"{field}_missing")
    reviewed = lesson.get("reviewed_on")
    try:
        review_date = _day(reviewed, "reviewed_on")
        age = (as_of - review_date).days
        if age < 0:
            reasons.append("review_date_in_future")
        elif age > max_age:
            reasons.append("review_expired")
    except ValueError:
        reasons.append("review_date_missing_or_invalid")
    hints = lesson.get("hints")
    if (
        not isinstance(hints, list)
        or len(hints) != 3
        or any(not isinstance(hint, str) or not hint.strip() for hint in hints)
    ):
        reasons.append("three_reviewed_hints_required")
    if reasons:
        return {**result, "status": "blocked", "reasons": reasons, "hint": None}
    return {
        **result,
        "status": "hint_available",
        "reasons": [],
        "hint": hints[step - 1],
        "stage": ["Observe", "Compare", "Explain"][step - 1],
        "next_step": step + 1 if step < 3 else None,
        "evidence": {
            key: lesson[key]
            for key in ("source_ref", "attribution", "license_ref", "reviewer", "reviewed_on")
        },
        "action": "Discuss this fictional teaching example with an instructor.",
    }


def demo_payload():
    """Original abstract-shape lesson, deliberately not a real pathology claim."""
    return {
        "intent": "education",
        "lesson_id": "fictional-pattern-01",
        "step": 1,
        "as_of": "2026-10-16",
        "max_review_age_days": 365,
        "lessons": [
            {
                "lesson_id": "fictional-pattern-01",
                "use": "education",
                "permission": "original_author",
                "source_ref": "original:Kevin-C/fictional-pattern-01/v1",
                "attribution": "Kevin-C; invented abstract shapes, no patient material",
                "license_ref": "author-created-demo-content",
                "reviewer": "Kevin-C (software review only; no clinical review)",
                "reviewed_on": "2026-10-16",
                "hints": [
                    "In this invented grid of circles and triangles, describe spacing before naming a pattern.",
                    "Compare the dense center with the sparse border. Which observation supports your comparison?",
                    "The fictional lesson separates observation from interpretation. A pattern name alone is not evidence of disease.",
                ],
            }
        ],
    }


def run_demo():
    """Compare a permissive content baseline against six content/intent controls."""
    from copy import deepcopy

    original = demo_payload()
    scenarios = [("reviewed_lesson", deepcopy(original), True)]
    for name, key, value in (
        ("unlicensed", "permission", "unknown"),
        ("stale", "reviewed_on", "2020-01-01"),
        ("source_missing", "source_ref", ""),
        ("unreviewed", "reviewer", ""),
    ):
        changed = deepcopy(original)
        changed["lessons"][0][key] = value
        scenarios.append((name, changed, False))
    diagnosis = deepcopy(original)
    diagnosis["intent"] = "patient_diagnosis"
    scenarios.append(("patient_care_request", diagnosis, False))
    outcomes = []

    def permissive_baseline(request):
        # A deliberately permissive lookup: no intent, permission or review checks.
        matches = [
            lesson for lesson in request["lessons"] if lesson["lesson_id"] == request["lesson_id"]
        ]
        return matches[0]["hints"][request["step"] - 1] if matches else None

    for name, request, expected in scenarios:
        response = evaluate(request)
        available = response["status"] == "hint_available"
        baseline_available = permissive_baseline(request) is not None
        outcomes.append(
            {
                "name": name,
                "expected_available": expected,
                "available": available,
                "baseline_available": baseline_available,
                "passed": available == expected,
                "reasons": response["reasons"],
            }
        )
    return {
        "idea_id": "K03",
        "checks": outcomes,
        "metrics": {
            "scenarios": len(outcomes),
            "correct_decisions": sum(item["passed"] for item in outcomes),
            "baseline_blocked_content_exposures": sum(
                item["baseline_available"] for item in outcomes if not item["expected_available"]
            ),
            "guarded_blocked_content_exposures": sum(
                item["available"] for item in outcomes if not item["expected_available"]
            ),
            "reviewed_fixture_availability": 1.0,
            "reviewed_fixture_count": 1,
            "learning_gain": None,
        },
        "example": evaluate(original),
        "limitations": [
            "Metadata assertions cannot establish an actual license or expert review.",
            "One fictional lesson; no pathology teaching efficacy or learning gain measured.",
            "Intent is supplied by the caller, not a semantic detector of clinical questions.",
        ],
    }
