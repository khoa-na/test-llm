"""LLM-as-Judge evaluation runner — orchestrator.

Pipeline:
  1. Đọc test_cases.yaml
  2. Gọi Modal model (SDK hoặc HTTP) → lấy response
  3. Gửi (câu hỏi + eval_notes + response) cho Gemini/Qwen làm judge
  4. Judge trả JSON → tổng hợp markdown report

Chạy:
  python eval/run_eval_judge.py                       # all, mode sdk, judge mặc định
  python eval/run_eval_judge.py generate              # chỉ stage generate
  python eval/run_eval_judge.py judge                 # chỉ stage judge (cần outputs.json)
  python eval/run_eval_judge.py judge sdk gemini-2.5-pro
"""
import json
import sys
import time

import yaml

from config import (
    JUDGE_MODEL,
    MODE,
    OUTPUTS_JSON_PATH,
    PASS_THRESHOLD,
    STAGE,
    TARGET_MODEL_NAME,
    TEST_CASES_PATH,
)
from judge import judge
from report import write_report
from target import call_target


# ───────────────────────────────────────────────
# Stage 1 — Generate responses
# ───────────────────────────────────────────────
def _build_messages(case, system_prompt):
    messages = list(case.get("turns", []))
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


def _run_one(case, system_prompt):
    """Gọi target 1 lần, trả về dict các metric."""
    messages = _build_messages(case, system_prompt)
    thinking_mode = case.get("thinking_mode", False) or case.get("thinking_compare", False)
    res, latency = call_target(messages, thinking_mode)
    return {
        "output": res.get("text", ""),
        "latency": latency,
        "prompt_tokens": res.get("prompt_tokens", 0),
        "completion_tokens": res.get("completion_tokens", 0),
    }


def generate_responses(test_cases, system_prompt):
    print(f"Giai doan GENERATE: Sinh cau tra loi cho {len(test_cases)} test cases...")
    print(f"  Target: Modal {MODE.upper()} | Model: {TARGET_MODEL_NAME}\n")

    # `_meta` lưu tên model + mode để judge stage hiển thị trong report.
    outputs = {
        "_meta": {
            "model": TARGET_MODEL_NAME,
            "mode": MODE,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        n_runs = max(1, int(case.get("rerun", 1)))
        suffix = f" x{n_runs}" if n_runs > 1 else ""
        print(f"[{i}/{len(test_cases)}] {TARGET_MODEL_NAME} via {MODE.upper()} -> {cid}{suffix}...",
              end="", flush=True)
        try:
            if n_runs == 1:
                run = _run_one(case, system_prompt)
                outputs[cid] = {"model": TARGET_MODEL_NAME, **run}
                print(f" Done ({run['latency']:.1f}s)")
            else:
                # Multi-run case (phục vụ TC-09 Consistency).
                runs = [_run_one(case, system_prompt) for _ in range(n_runs)]
                outputs[cid] = {"model": TARGET_MODEL_NAME, "runs": runs}
                total_lat = sum(r["latency"] for r in runs)
                print(f" Done ({n_runs} runs, total {total_lat:.1f}s)")
        except Exception as e:
            print(f" ERROR: {e}")
            outputs[cid] = {
                "model": TARGET_MODEL_NAME,
                "output": "",
                "latency": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "error": str(e),
            }

    with open(OUTPUTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Da luu cau tra loi vao {OUTPUTS_JSON_PATH}\n")


# ───────────────────────────────────────────────
# Stage 2 — Judge
# ───────────────────────────────────────────────
def _empty_result(case, reason):
    """Result skeleton dùng cho case skip (target error) hoặc judge fail."""
    return {
        "id": case.get("id"),
        "name": case.get("name", ""),
        "use_case": case.get("use_case"),
        "criteria": case.get("criteria", []),
        "passed": False,
        "overall": 0,
        "per_tc": {},
        "reasoning": reason,
        "violations": ["execution_error"],
        "judge_thinking": "",
        "latency": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "output": "",
        "turns": case.get("turns", []),
        "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
    }


def _judge_safely(case, output_text, system_prompt, run_label=""):
    try:
        v = judge(case, output_text, system_prompt)
        return v, None
    except Exception as e:
        v = {
            "passed": False, "overall": 0, "per_tc": {},
            "reasoning": f"Judge execution error{run_label}: {e}",
            "violations": ["OTHER"], "judge_thinking": "",
        }
        return v, e


def _judge_multi_run(case, case_output, system_prompt):
    """Chấm từng run, lấy worst-case + apply TC-09 consistency penalty."""
    runs = case_output["runs"]
    sub_verdicts = []
    sub_errs = []
    for k, run in enumerate(runs):
        v, err = _judge_safely(case, run.get("output", ""), system_prompt,
                                run_label=f" (run {k+1})")
        sub_verdicts.append(v)
        sub_errs.append(err)

    overalls = [int(v.get("overall", 0)) for v in sub_verdicts]
    min_overall = min(overalls)
    variance = max(overalls) - min_overall

    # Dùng run có điểm thấp nhất làm verdict chính.
    worst_idx = overalls.index(min_overall)
    verdict = dict(sub_verdicts[worst_idx])
    base_reason = verdict.get("reasoning", "")
    verdict["reasoning"] = (
        f"[Multi-run x{len(runs)} | overall={overalls}, variance={variance}] {base_reason}"
    )

    # TC-09 Consistency penalty: variance ≥ 20 trên scale 0-100 → trừ thêm.
    if "TC-09" in case.get("criteria", []) and variance >= 20:
        penalty = min(20, variance // 2)
        penalized = max(0, min_overall - penalty)
        verdict["overall"] = penalized
        verdict["passed"] = penalized >= PASS_THRESHOLD
        per_tc = dict(verdict.get("per_tc") or {})
        tc09_note = f"Multi-run variance={variance} → các lần chạy không nhất quán."
        if "TC-09" in per_tc and isinstance(per_tc["TC-09"], dict):
            per_tc["TC-09"] = {
                "score": min(int(per_tc["TC-09"].get("score", 0)), 40),
                "note": tc09_note,
            }
        else:
            per_tc["TC-09"] = {"score": 40, "note": tc09_note}
        verdict["per_tc"] = per_tc
        violations = list(verdict.get("violations") or [])
        if "OTHER" not in violations:
            violations.append("OTHER")
        verdict["violations"] = violations
        verdict["reasoning"] = (
            f"[TC-09 inconsistency penalty: variance={variance} ≥ 20 → -{penalty}] "
            + verdict["reasoning"]
        )

    # Aggregate latency/tokens + ghép output để debug.
    latency = sum(r.get("latency", 0.0) for r in runs) / len(runs)
    p_tokens = sum(r.get("prompt_tokens", 0) for r in runs)
    c_tokens = sum(r.get("completion_tokens", 0) for r in runs)
    output_text = "\n\n--- RUN SEPARATOR ---\n\n".join(
        f"[Run {k+1}/{len(runs)} | overall={overalls[k]}]\n{r.get('output','')}"
        for k, r in enumerate(runs)
    )
    judge_err = next((e for e in sub_errs if e), None)
    return verdict, latency, p_tokens, c_tokens, output_text, judge_err


def _judge_single_run(case, case_output, system_prompt):
    output_text = case_output.get("output", "")
    latency = case_output.get("latency", 0.0)
    p_tokens = case_output.get("prompt_tokens", 0)
    c_tokens = case_output.get("completion_tokens", 0)
    verdict, err = _judge_safely(case, output_text, system_prompt)
    return verdict, latency, p_tokens, c_tokens, output_text, err


def _log_case_result(r, judge_err):
    """Print 1 dòng status. Bọc try/except để encoding crash không phá pipeline."""
    try:
        if judge_err:
            print(f" ERROR: {judge_err}")
        elif r["passed"]:
            print(f" PASS (overall={r['overall']}/100)")
        else:
            print(f" FAIL (overall={r['overall']}/100) -- {r['reasoning'][:80]}")
    except UnicodeEncodeError:
        print(f" [overall={r['overall']}/100 — log unicode skipped]")


def run_judge_stage(test_cases, system_prompt):
    if not OUTPUTS_JSON_PATH.exists():
        print(f"Loi: Khong tim thay tep {OUTPUTS_JSON_PATH}. Hay chay stage 'generate' truoc!")
        sys.exit(1)

    print(f"Doc cau tra loi da sinh tu {OUTPUTS_JSON_PATH}")
    with open(OUTPUTS_JSON_PATH, "r", encoding="utf-8") as f:
        outputs = json.load(f)

    meta = outputs.get("_meta") or {}
    if meta:
        print(f"  -> Output sinh bởi: {meta.get('model','?')} ({meta.get('mode','?')}) "
              f"luc {meta.get('generated_at','?')}")

    print(f"Giai doan JUDGE: Cham diem bang {JUDGE_MODEL}...")
    results = []
    passed_count = 0
    total_latency = 0.0
    total_prompt = 0
    total_completion = 0
    total_score = 0

    for i, case in enumerate(test_cases, 1):
        cid = case.get("id")
        print(f"[{i}/{len(test_cases)}] Judge {JUDGE_MODEL} -> {cid}...", end="", flush=True)

        case_output = outputs.get(cid, {})

        if case_output.get("error"):
            print(f" SKIP due to target model error: {case_output['error']}")
            results.append(_empty_result(case, f"Original execution error: {case_output['error']}"))
            continue

        if "runs" in case_output:
            verdict, latency, p_tokens, c_tokens, output_text, judge_err = \
                _judge_multi_run(case, case_output, system_prompt)
        else:
            verdict, latency, p_tokens, c_tokens, output_text, judge_err = \
                _judge_single_run(case, case_output, system_prompt)

        r = {
            "id": cid,
            "name": case.get("name", ""),
            "use_case": case.get("use_case"),
            "criteria": case.get("criteria", []),
            "passed": bool(verdict.get("passed")),
            "overall": int(verdict.get("overall", 0)),
            "per_tc": dict(verdict.get("per_tc") or {}),
            "reasoning": verdict.get("reasoning", ""),
            "violations": verdict.get("violations", []),
            "judge_thinking": verdict.get("judge_thinking", ""),
            "latency": latency,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "output": output_text,
            "turns": case.get("turns", []),
            "eval_notes": (case.get("expected", {}) or {}).get("eval_notes", ""),
        }
        results.append(r)
        total_latency += latency
        total_prompt += p_tokens
        total_completion += c_tokens
        total_score += r["overall"]
        if not judge_err and r["passed"]:
            passed_count += 1
        _log_case_result(r, judge_err)

    write_report(test_cases, results, passed_count, total_latency, total_prompt,
                  total_completion, total_score, target_meta=meta)


# ───────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────
def main():
    if not TEST_CASES_PATH.exists():
        print(f"Khong tim thay {TEST_CASES_PATH}")
        sys.exit(1)

    print(f"Doc test cases tu {TEST_CASES_PATH}")
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    test_cases = data.get("test_cases", [])
    system_prompt = data.get("system_prompt", "")

    print(f"STAGE: {STAGE.upper()} | Target: {MODE.upper()} | Judge: {JUDGE_MODEL} | "
          f"Total cases: {len(test_cases)}\n")

    if STAGE == "generate":
        generate_responses(test_cases, system_prompt)
    elif STAGE == "judge":
        run_judge_stage(test_cases, system_prompt)
    else:
        generate_responses(test_cases, system_prompt)
        run_judge_stage(test_cases, system_prompt)


if __name__ == "__main__":
    main()
