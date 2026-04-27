"""Batch evaluation runner for SensorDoc-AI.

This script executes questions from a JSONL dataset against the current RAG
pipeline and writes machine-readable results. It can also run an LLM-as-a-judge
pass tailored to datasheet-grounded QA and XAI via RAG behavior.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import RAGPipeline


DEFAULT_DATASET_PATH = Path("eval_dataset.jsonl")
DEFAULT_OUTPUT_PATH = Path("eval_results.jsonl")
DEFAULT_JUDGE_PROMPT_PATH = Path("llm_judge_prompt.md")
DEFAULT_POLICY_PATH = Path("eval_policy.json")


def log_progress(message: str) -> None:
    print(f"[eval_runner] {message}")


@dataclass
class EvalSample:
    sample_id: str
    category: str
    question: str
    prediction_payload: dict[str, Any] | None
    expected_answer: str
    expected_facts: list[str]
    must_include: list[str]
    must_not_include: list[str]
    judge_focus: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], index: int) -> "EvalSample":
        return cls(
            sample_id=str(payload.get("id") or f"sample-{index}"),
            category=str(payload.get("category") or "general"),
            question=str(payload["question"]),
            prediction_payload=payload.get("prediction_payload"),
            expected_answer=str(payload.get("expected_answer") or ""),
            expected_facts=[str(item) for item in payload.get("expected_facts", [])],
            must_include=[str(item) for item in payload.get("must_include", [])],
            must_not_include=[str(item) for item in payload.get("must_not_include", [])],
            judge_focus=str(payload.get("judge_focus") or ""),
        )


def load_policy(path: Path) -> dict[str, Any]:
    log_progress(f"Loading policy from {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    log_progress(f"Loading dataset from {path}")
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples.append(EvalSample.from_dict(payload, index))
            if index % 10 == 0:
                log_progress(f"Parsed {index} dataset lines")
    log_progress(f"Loaded {len(samples)} eval samples")
    return samples


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def contains_phrase(answer: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(answer)


def phrase_is_negated(answer_text: str, phrase: str) -> bool:
    normalized_answer = normalize_text(answer_text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False

    start = 0
    while True:
        index = normalized_answer.find(normalized_phrase, start)
        if index == -1:
            return False

        window_start = max(0, index - 120)
        prefix_window = normalized_answer[window_start:index]
        negation_markers = (
            "no ",
            "not ",
            "cannot conclude",
            "can't conclude",
            "cannot be concluded",
            "can't be concluded",
            "cannot confirm",
            "not enough evidence",
            "insufficient evidence",
            "does not justify",
            "doesn't justify",
            "unsupported",
        )
        if any(marker in prefix_window for marker in negation_markers):
            return True

        start = index + len(normalized_phrase)


def heuristic_metrics(sample: EvalSample, answer: str) -> dict[str, Any]:
    fact_hits = sum(1 for fact in sample.expected_facts if contains_phrase(answer, fact))
    include_hits = sum(1 for phrase in sample.must_include if contains_phrase(answer, phrase))
    forbidden_hits = [phrase for phrase in sample.must_not_include if contains_phrase(answer, phrase)]

    return {
        "expected_fact_hit_rate": round(
            fact_hits / len(sample.expected_facts), 3
        ) if sample.expected_facts else None,
        "must_include_hit_rate": round(
            include_hits / len(sample.must_include), 3
        ) if sample.must_include else None,
        "must_not_include_violations": forbidden_hits,
        "answer_length_chars": len(answer),
        "answer_text": answer,
    }


def compute_overall_judge_score(judge: dict[str, Any] | None) -> float | None:
    if not judge:
        return None

    metric_names = [
        "faithfulness",
        "answer_relevance",
        "datasheet_specificity",
        "xai_grounding",
        "safety_scope_control",
    ]
    values = [float(judge[name]) for name in metric_names if isinstance(judge.get(name), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def evaluate_policy(
    sample: EvalSample,
    heuristics: dict[str, Any],
    judge: dict[str, Any] | None,
    policy: dict[str, Any],
) -> dict[str, Any]:
    global_policy = policy.get("global", {})
    category_policy = (policy.get("categories", {}) or {}).get(sample.category, {})
    failed_checks: list[str] = []

    expected_fact_hit_rate = heuristics.get("expected_fact_hit_rate")
    minimum_expected_fact_hit_rate = category_policy.get(
        "minimum_expected_fact_hit_rate",
        global_policy.get("minimum_expected_fact_hit_rate"),
    )
    if (
        isinstance(expected_fact_hit_rate, (int, float))
        and isinstance(minimum_expected_fact_hit_rate, (int, float))
        and expected_fact_hit_rate < minimum_expected_fact_hit_rate
    ):
        failed_checks.append(f"expected_fact_hit_rate<{minimum_expected_fact_hit_rate}")

    must_include_hit_rate = heuristics.get("must_include_hit_rate")
    minimum_must_include_hit_rate = global_policy.get("minimum_must_include_hit_rate")
    if (
        must_include_hit_rate is not None
        and isinstance(must_include_hit_rate, (int, float))
        and isinstance(minimum_must_include_hit_rate, (int, float))
        and must_include_hit_rate < minimum_must_include_hit_rate
    ):
        failed_checks.append(f"must_include_hit_rate<{minimum_must_include_hit_rate}")

    forbidden_terms = list(heuristics.get("must_not_include_violations") or [])
    answer_text = normalize_text(str(heuristics.get("answer_text") or ""))
    filtered_forbidden_terms: list[str] = []
    for phrase in forbidden_terms:
        normalized_phrase = normalize_text(str(phrase))
        if not normalized_phrase:
            continue
        if phrase_is_negated(answer_text, normalized_phrase):
            continue
        filtered_forbidden_terms.append(str(phrase))
    if global_policy.get("require_zero_forbidden_terms", False) and filtered_forbidden_terms:
        failed_checks.append("must_not_include_violations>0")

    overall_score = compute_overall_judge_score(judge)
    judge_required = bool(global_policy.get("judge_required_for_policy_pass", True))
    if judge_required and not judge:
        return {
            "status": "incomplete",
            "overall_score": overall_score,
            "failed_checks": failed_checks + ["missing_llm_judge"],
        }

    minimum_overall_score = category_policy.get("minimum_overall_score")
    if (
        judge
        and isinstance(overall_score, (int, float))
        and isinstance(minimum_overall_score, (int, float))
        and overall_score < minimum_overall_score
    ):
        failed_checks.append(f"overall_score<{minimum_overall_score}")

    for field_name, minimum_value in (category_policy.get("minimum_judge_fields") or {}).items():
        value = None if not judge else judge.get(field_name)
        if isinstance(value, (int, float)):
            if value < minimum_value:
                failed_checks.append(f"{field_name}<{minimum_value}")
        else:
            failed_checks.append(f"{field_name}_missing")

    return {
        "status": "pass" if not failed_checks else "fail",
        "overall_score": overall_score,
        "failed_checks": failed_checks,
    }


def load_judge_prompt(path: Path) -> str:
    raw_text = path.read_text(encoding="utf-8")
    match = re.search(r"```text\n(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def parse_judge_json(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        fenced_match = re.search(r"```(?:json)?\n(.*?)```", raw_text, re.DOTALL)
        candidate = fenced_match.group(1).strip() if fenced_match else raw_text.strip()

        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            raise

        extracted = match.group(0)
        sanitized = re.sub(r":\s*N/A(?=[,\n}])", ": null", extracted)
        sanitized = re.sub(r":\s*None(?=[,\n}])", ": null", sanitized)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(sanitized)
            if not isinstance(parsed, dict):
                raise ValueError("Judge output did not parse into a dictionary.")
            return parsed


def run_judge(
    pipeline: RAGPipeline,
    judge_prompt: str,
    sample: EvalSample,
    answer: str,
) -> dict[str, Any]:
    prompt = ChatPromptTemplate.from_template(
        "{judge_prompt}\n\n"
        "Sample metadata:\n"
        "- id: {sample_id}\n"
        "- category: {category}\n"
        "- judge_focus: {judge_focus}\n\n"
        "User question:\n{question}\n\n"
        "Prediction payload:\n{prediction_payload}\n\n"
        "Retrieved answer:\n{answer}\n\n"
        "Expected answer guidance:\n{expected_answer}\n\n"
        "Expected facts:\n{expected_facts}\n\n"
        "Must-include terms:\n{must_include}\n\n"
        "Must-not-include terms:\n{must_not_include}\n"
    )
    chain = prompt | pipeline.llm | StrOutputParser()
    raw_result = chain.invoke(
        {
            "judge_prompt": judge_prompt,
            "sample_id": sample.sample_id,
            "category": sample.category,
            "judge_focus": sample.judge_focus,
            "question": sample.question,
            "prediction_payload": json.dumps(sample.prediction_payload, ensure_ascii=True, indent=2),
            "answer": answer,
            "expected_answer": sample.expected_answer,
            "expected_facts": json.dumps(sample.expected_facts, ensure_ascii=True),
            "must_include": json.dumps(sample.must_include, ensure_ascii=True),
            "must_not_include": json.dumps(sample.must_not_include, ensure_ascii=True),
        }
    )
    try:
        return parse_judge_json(raw_result)
    except Exception:
        repair_prompt = ChatPromptTemplate.from_template(
            "Convert the following evaluation output into valid JSON only. "
            "Keep the same meaning. Replace unsupported values like N/A with null.\n\n"
            "Malformed evaluation output:\n{raw_result}"
        )
        repair_chain = repair_prompt | pipeline.llm | StrOutputParser()
        repaired_result = repair_chain.invoke({"raw_result": raw_result})
        return parse_judge_json(repaired_result)


def run_eval(
    dataset_path: Path,
    output_path: Path,
    judge_enabled: bool,
    judge_prompt_path: Path,
    policy_path: Path,
) -> dict[str, Any]:
    samples = load_jsonl(dataset_path)
    policy = load_policy(policy_path)
    log_progress("Initializing RAG pipeline")
    pipeline = RAGPipeline()
    pipeline.load_existing()
    log_progress("Loaded existing retrieval index")

    judge_prompt = load_judge_prompt(judge_prompt_path) if judge_enabled else ""
    if judge_enabled:
        log_progress(f"Loaded judge prompt from {judge_prompt_path}")

    summary = {
        "total_samples": len(samples),
        "judge_enabled": judge_enabled,
        "policy_path": str(policy_path),
        "category_counts": {},
        "policy_status_counts": {"pass": 0, "fail": 0, "incomplete": 0},
        "results_path": str(output_path),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            log_progress(
                f"Running sample {index}/{len(samples)}: id={sample.sample_id}, category={sample.category}"
            )
            start_time = time.perf_counter()
            answer = pipeline.query_sensor_info(
                sample.question,
                prediction_payload=sample.prediction_payload,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_progress(
                f"Finished answer for {sample.sample_id} in {latency_ms} ms"
            )

            heuristics = heuristic_metrics(sample, answer)
            result: dict[str, Any] = {
                "id": sample.sample_id,
                "category": sample.category,
                "question": sample.question,
                "prediction_payload": sample.prediction_payload,
                "answer": answer,
                "latency_ms": latency_ms,
                "heuristics": heuristics,
            }

            if judge_enabled:
                try:
                    log_progress(f"Judging sample {sample.sample_id}")
                    result["llm_judge"] = run_judge(pipeline, judge_prompt, sample, answer)
                    log_progress(f"Judge completed for {sample.sample_id}")
                except Exception as exc:
                    result["llm_judge_error"] = str(exc)
                    log_progress(f"Judge failed for {sample.sample_id}: {exc}")

            result["policy_result"] = evaluate_policy(
                sample=sample,
                heuristics=heuristics,
                judge=result.get("llm_judge"),
                policy=policy,
            )
            log_progress(
                f"Policy status for {sample.sample_id}: {result['policy_result']['status']}"
            )

            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            summary["category_counts"][sample.category] = summary["category_counts"].get(sample.category, 0) + 1
            policy_status = result["policy_result"]["status"]
            summary["policy_status_counts"][policy_status] = summary["policy_status_counts"].get(policy_status, 0) + 1

    log_progress(
        "Done: "
        f"total={summary['total_samples']}, "
        f"pass={summary['policy_status_counts']['pass']}, "
        f"fail={summary['policy_status_counts']['fail']}, "
        f"incomplete={summary['policy_status_counts']['incomplete']}"
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval samples against SensorDoc-AI.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--judge", action="store_true", help="Run the LLM-as-a-judge pass.")
    parser.add_argument("--judge-prompt", type=Path, default=DEFAULT_JUDGE_PROMPT_PATH)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    args = parser.parse_args()

    summary = run_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        judge_enabled=args.judge,
        judge_prompt_path=args.judge_prompt,
        policy_path=args.policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()