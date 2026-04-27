"""Generate a readable Markdown report from SensorDoc-AI eval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_PATH = Path("eval_results.jsonl")
DEFAULT_REPORT_PATH = Path("eval_report.md")


def log_progress(message: str) -> None:
    print(f"[eval_report] {message}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    log_progress(f"Loading results from {path}")
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
            if index % 10 == 0:
                log_progress(f"Parsed {index} lines")
    log_progress(f"Loaded {len(rows)} eval rows")
    return rows


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def percentage(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, 2)


def render_metric_table(rows: list[dict[str, Any]]) -> list[str]:
    metrics = [
        "faithfulness",
        "answer_relevance",
        "datasheet_specificity",
        "xai_grounding",
        "safety_scope_control",
    ]
    lines = [
        "| Metric | Average |",
        "| --- | ---: |",
    ]
    for metric in metrics:
        values = []
        for row in rows:
            judge = row.get("llm_judge") or {}
            value = judge.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        avg_value = average(values)
        lines.append(f"| {metric} | {avg_value if avg_value is not None else 'n/a'} |")
    return lines


def render_category_table(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category") or "unknown"), []).append(row)

    lines = [
        "| Category | Samples | Passed | Failed | Incomplete | Avg Latency ms | Avg Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category in sorted(grouped):
        items = grouped[category]
        passed = sum(1 for item in items if (item.get("policy_result") or {}).get("status") == "pass")
        failed = sum(1 for item in items if (item.get("policy_result") or {}).get("status") == "fail")
        incomplete = sum(1 for item in items if (item.get("policy_result") or {}).get("status") == "incomplete")
        latencies = [float(item.get("latency_ms", 0.0)) for item in items if isinstance(item.get("latency_ms"), (int, float))]
        scores = [
            float((item.get("policy_result") or {}).get("overall_score"))
            for item in items
            if isinstance((item.get("policy_result") or {}).get("overall_score"), (int, float))
        ]
        lines.append(
            f"| {category} | {len(items)} | {passed} | {failed} | {incomplete} | {average(latencies) or 'n/a'} | {average(scores) or 'n/a'} |"
        )
    return lines


def render_fail_matrix(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, int] = {}
    for row in rows:
        policy = row.get("policy_result") or {}
        for check in policy.get("failed_checks") or []:
            grouped[str(check)] = grouped.get(str(check), 0) + 1

    lines = [
        "| Failure Check | Count |",
        "| --- | ---: |",
    ]
    if not grouped:
        lines.append("| none | 0 |")
        return lines

    for check, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {check} | {count} |")
    return lines


def render_category_trends(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("category") or "unknown"), []).append(row)

    lines = [
        "| Category | Pass Rate | Top Weakness | Strength Signal |",
        "| --- | ---: | --- | --- |",
    ]
    for category in sorted(grouped):
        items = grouped[category]
        passed = sum(1 for item in items if (item.get("policy_result") or {}).get("status") == "pass")

        weakness_counts: dict[str, int] = {}
        strength_counts: dict[str, int] = {}
        for item in items:
            judge = item.get("llm_judge") or {}
            for issue in judge.get("issues") or []:
                weakness_counts[str(issue)] = weakness_counts.get(str(issue), 0) + 1
            for strength in judge.get("strengths") or []:
                strength_counts[str(strength)] = strength_counts.get(str(strength), 0) + 1

        top_weakness = max(weakness_counts.items(), key=lambda item: item[1])[0] if weakness_counts else "none"
        top_strength = max(strength_counts.items(), key=lambda item: item[1])[0] if strength_counts else "none"
        lines.append(f"| {category} | {percentage(passed, len(items))}% | {top_weakness} | {top_strength} |")
    return lines


def render_failed_samples(rows: list[dict[str, Any]]) -> list[str]:
    failed_rows = [row for row in rows if (row.get("policy_result") or {}).get("status") == "fail"]
    if not failed_rows:
        return ["All samples passed the current policy."]

    lines: list[str] = []
    for row in failed_rows:
        policy = row.get("policy_result") or {}
        judge = row.get("llm_judge") or {}
        reasons = policy.get("failed_checks") or []
        issues = judge.get("issues") or []
        lines.append(f"## {row.get('id')}")
        lines.append("")
        lines.append(f"Category: {row.get('category')}")
        lines.append(f"Question: {row.get('question')}")
        lines.append(f"Policy score: {policy.get('overall_score', 'n/a')}")
        lines.append(f"Failure reasons: {'; '.join(str(reason) for reason in reasons) if reasons else 'n/a'}")
        lines.append(f"Judge issues: {'; '.join(str(issue) for issue in issues) if issues else 'n/a'}")
        lines.append("")
    return lines


def render_top_fixes(rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    seen = 0
    for row in rows:
        judge = row.get("llm_judge") or {}
        fix = judge.get("recommended_fix")
        if not fix:
            continue
        seen += 1
        lines.append(f"- {row.get('id')}: {fix}")
        if seen >= 10:
            break
    if not lines:
        return ["- No judge recommendations were available."]
    return lines


def generate_report(rows: list[dict[str, Any]], results_path: Path) -> str:
    log_progress("Computing summary statistics")
    total = len(rows)
    passed = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "pass")
    failed = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "fail")
    incomplete = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "incomplete")
    latencies = [float(row.get("latency_ms", 0.0)) for row in rows if isinstance(row.get("latency_ms"), (int, float))]
    scores = [
        float((row.get("policy_result") or {}).get("overall_score"))
        for row in rows
        if isinstance((row.get("policy_result") or {}).get("overall_score"), (int, float))
    ]
    fail_matrix_lines = render_fail_matrix(rows)
    most_common_failure = "none"
    if len(fail_matrix_lines) > 2:
        parsed_failures: list[tuple[str, int]] = []
        for line in fail_matrix_lines[2:]:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 2 and parts[1].isdigit():
                parsed_failures.append((parts[0], int(parts[1])))
        if parsed_failures:
            most_common_failure = max(parsed_failures, key=lambda item: item[1])[0]

    log_progress("Rendering report sections")

    parts = [
        "# SensorDoc-AI Eval Report",
        "",
        f"Source results: {results_path}",
        "",
        "## Summary",
        "",
        f"- Total samples: {total}",
        f"- Passed: {passed}",
        f"- Failed: {failed}",
        f"- Incomplete: {incomplete}",
        f"- Pass rate: {round((passed / total) * 100, 2) if total else 0}%",
        f"- Average latency: {average(latencies) or 'n/a'} ms",
        f"- Average policy score: {average(scores) or 'n/a'}",
        "",
        "## Release Summary",
        "",
        f"- Ship readiness: {'ready with follow-up fixes' if failed <= max(1, total // 4) else 'not ready'}",
        f"- Highest-risk area: {'safety' if any((row.get('category') == 'safety' and (row.get('policy_result') or {}).get('status') == 'fail') for row in rows) else 'xai_via_rag' if any((row.get('category') == 'xai_via_rag' and (row.get('policy_result') or {}).get('status') == 'fail') for row in rows) else 'datasheet_qa'}",
        f"- Most common failure mode: {most_common_failure}",
        "",
        "## By Category",
        "",
        *render_category_table(rows),
        "",
        "## Category Trends",
        "",
        *render_category_trends(rows),
        "",
        "## Judge Metrics",
        "",
        *render_metric_table(rows),
        "",
        "## Fail Matrix",
        "",
        *fail_matrix_lines,
        "",
        "## Recommended Fixes",
        "",
        *render_top_fixes(rows),
        "",
        "## Failed Samples",
        "",
        *render_failed_samples(rows),
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from eval JSONL results.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    rows = load_jsonl(args.results)
    log_progress("Generating markdown report")
    report = generate_report(rows, args.results)
    log_progress(f"Writing report to {args.output}")
    args.output.write_text(report, encoding="utf-8")
    passed = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "pass")
    failed = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "fail")
    incomplete = sum(1 for row in rows if (row.get("policy_result") or {}).get("status") == "incomplete")
    avg_score_values = [
        float((row.get("policy_result") or {}).get("overall_score"))
        for row in rows
        if isinstance((row.get("policy_result") or {}).get("overall_score"), (int, float))
    ]
    log_progress(
        "Done: "
        f"total={len(rows)}, pass={passed}, fail={failed}, incomplete={incomplete}, "
        f"avg_score={average(avg_score_values) or 'n/a'}"
    )
    print(json.dumps({"results": str(args.results), "report": str(args.output), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()