"""Kevin-C: plain-language desktop summaries, separate from evaluation logic."""

import json

from kusic.workbench import IDEAS

GUIDANCE = {
    "K01": (
        "Load a template or JSON export with disjoint calibration, selection and evaluation cohorts. "
        "The tool fits temperature on calibration only and chooses a deferral threshold on selection only. "
        "Evaluation labels measure the policy; they are not used to choose it. Review both coverage and error."
    ),
    "K02": (
        "Load a template or JSON with a query and prior-record embeddings. "
        "Identity, institution, date, consent, source and permitted use are checked before ranking. "
        "Authorization is asserted by the input, not authenticated by this prototype. "
        "Select an opaque case reference and inspect its original local source if needed."
    ),
    "K03": (
        "Run the sample lesson, then choose Next hint to progress from observation to explanation. "
        "Hints are authored text with source/review metadata. The sample is an invented shape exercise. "
        "The tool checks metadata; an expert still needs to establish educational accuracy and rights."
    ),
    "K04": (
        "Load clean and perturbed prediction exports paired by case ID. "
        "Compare accuracy, probability error and prediction flips. "
        "Small subgroups are marked as insufficient evidence. "
        "Synthetic IDs do not establish independent experimental blinding. "
        "Run all comparisons to see deliberately stable and unstable synthetic controls."
    ),
}


def _pretty(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def summary(report):
    if "ideas" in report:
        lines = [
            "FOUR-IDEA COMPARISON",
            "Author: Kevin-C",
            "",
            "Scores are design assessments, not a usability trial or legal determination.",
            "Complexity / friendliness / legal deployment burden / creativity",
            "",
        ]
        for item in report["ideas"]:
            scores = item["scores"]
            lines.extend(
                [
                    f"{item['idea_id']}  {item['title']}",
                    "  "
                    + " / ".join(
                        str(scores[key])
                        for key in ("complexity", "user_friendliness", "legal_issues", "creativity")
                    ),
                    f"  Screening index: {item['priority_index']:.2f}/10",
                    f"  {item['recommendation']}",
                    "",
                ]
            )
        lines.extend(
            [
                "The index uses equal weights. It does not measure clinical value.",
                "Open Evidence JSON for each synthetic experiment and its limitations.",
            ]
        )
        return "\n".join(lines)
    idea_id = report["idea_id"]
    result = report["result"]
    lines = [f"{idea_id}  {IDEAS[idea_id]['title']}", "", "Local research / education result", ""]
    if idea_id == "K02":
        counts = result["counts"]
        lines.extend(
            [
                f"Eligible priors: {counts['eligible']} of {counts['records_supplied']}",
                f"Returned: {counts['returned']}",
                f"Unrestricted baseline violations: {result['baseline']['violating']}",
                "",
                "Eligible matches (similarity does not establish diagnosis):",
            ]
        )
        for match in result["matches"]:
            lines.append(
                f"{match['case_ref']}  |  {match['observed_at']}  |  {match['similarity']:.3f}"
            )
        if result["empty_reasons"]:
            lines.extend(reason.replace("_", " ") for reason in result["empty_reasons"])
        lines.extend(["", "Exclusion counts (reasons can overlap):"])
        lines.extend(
            f"  {key.replace('_', ' ')}: {value}"
            for key, value in result["excluded_reason_counts"].items()
        )
        lines.extend(
            [
                "",
                "Access/consent metadata come from the caller. This is not an authentication system.",
            ]
        )
    elif idea_id == "K03":
        if result["status"] == "hint_available":
            lines.extend(
                [
                    f"Step {result['step']}: {result['stage']}",
                    "",
                    result["hint"],
                    "",
                    "Evidence:",
                    _pretty(result["evidence"]),
                    "",
                    result["action"],
                ]
            )
        else:
            lines.extend(
                ["Hint withheld:"] + [reason.replace("_", " ") for reason in result["reasons"]]
            )
    elif idea_id == "K01":
        raw, calibrated = result["held_out"]["raw"], result["held_out"]["calibrated"]
        error = calibrated["accepted_error"]
        lines.extend(
            [
                f"Temperature fitted on calibration: {result['temperature']:.3f}",
                f"Cohorts: {result['split_counts']}",
                "",
                "Held-out probability error (lower is better):",
                f"  NLL: {raw['nll']:.4f} raw -> {calibrated['nll']:.4f} calibrated",
                f"  Brier: {raw['brier']:.4f} raw -> {calibrated['brier']:.4f} calibrated",
                f"  ECE (10 bins): {raw['ece_10_bins']:.4f} -> {calibrated['ece_10_bins']:.4f}",
                "",
                f"Held-out accuracy: {calibrated['accuracy']:.1%}",
                f"Retained for research: {calibrated['coverage']:.1%} ({calibrated['accepted_count']} cases)",
                "Error among retained cases: "
                + (f"{error:.1%}" if error is not None else "undefined (none retained)"),
                f"Selection target: {result['target_selection_error']:.1%} (an empirical target, not a guarantee)",
                "",
            ]
        )
        if error is not None and error > result["target_selection_error"]:
            lines.extend(
                ["Held-out retained error EXCEEDS the selection target. Review required.", ""]
            )
        lines.append("Per-case disposition (every result still requires human interpretation):")
        for row in result["rows"]:
            lines.append(
                f"  Case {row['case']}: {row['action'].replace('_', ' ')}; confidence {row['confidence']:.1%}"
            )
        lines.extend(["", *result["limitations"]])
    elif idea_id == "K04":
        overall = result["overall"]
        lines.extend(
            [
                f"Complete matched pairs: {overall['count']}",
                f"Accuracy: {overall['clean']['accuracy']:.1%} clean -> {overall['perturbed']['accuracy']:.1%} perturbed",
                f"Prediction flips: {overall['prediction_flip_rate']:.1%}",
                f"Brier change: {overall['brier_change']:+.4f} (positive means worse)",
                f"Mean probability shift (total variation): {overall['mean_total_variation']:.4f}",
                "",
                "Subgroup evidence:",
            ]
        )
        for group in result["subgroups"]:
            lines.extend(
                [
                    f"  {group['subgroup']}: {group['count']} pairs; {group['evidence'].replace('_', ' ')}",
                    "  " + group["reason"],
                ]
            )
        lines.extend(["", *result["limitations"]])
    lines.extend(
        [
            "",
            f"Elapsed: {report['elapsed_ms']:.3f} ms",
            "",
            "Synthetic controls establish software behavior only. No clinical effectiveness measured.",
        ]
    )
    return "\n".join(lines)
