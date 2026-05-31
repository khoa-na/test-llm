"""Chẩn đoán & tune EMBEDDING ROUTER của RAG — gói cả quy trình vào 1 file.

Thay cho việc chạy lẻ từng đoạn inline, script này gọi `retrieve_only` trên Modal
một lần cho mỗi case trong test_cases_rag_live.yaml rồi in:

  1. BẢNG ROUTE SCORES   — cosine query↔anchor mỗi intent (calendar/task/email),
                           kèm intent nào FIRE ở ngưỡng hiện tại + recall của case.
  2. TỔNG KẾT RECALL/REFUSE — recall@k (case có data) + refuse clean (case không data).
  3. QUÉT NGƯỠNG          — thử nhiều RAG_ROUTE_MIN_SCORE, đếm:
                              • miss   = intent CẦN có nhưng không fire (routing hụt)
                              • bẩn    = intent THỪA fire trên case có data (over-route)
                              • leak   = refuse case bị calendar/task fire (rò data)
                           → gợi ý ngưỡng tốt nhất (đủ intent, ít bẩn/leak nhất).

Cách dùng:
  python eval/rag_diagnose.py                 # đo trên deployment hiện tại
  python eval/rag_diagnose.py --deploy        # deploy modal_app.py trước rồi đo
  python eval/rag_diagnose.py --top-k 6
  python eval/rag_diagnose.py --sweep 0.45 0.70 0.01

Lưu ý: `email` fire trên refuse case KHÔNG tính leak vì _add_emails lọc theo tên
người → hỏi người không có vẫn trả rỗng. Chỉ calendar/task fire mới rò data thật.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# Windows console cp1252 → crash khi print tiếng Việt. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

APP_NAME = "test-llm-chatbot-thuky"
CLS_NAME = "LLMServer"
_ROOT = Path(__file__).parent.parent
_YAML = Path(__file__).parent / "test_cases_rag_live.yaml"
_INTENTS = ("calendar", "tasks", "emails")

# Mặc định khớp modal_app/.env — chỉ để hiển thị "fire ở ngưỡng hiện tại".
DEFAULT_ROUTE_MIN = 0.58


def _last_user_query(case) -> str:
    for t in reversed(case.get("turns", [])):
        if t.get("role") == "user":
            return t.get("content", "") or ""
    return ""


def _expected_intents(case) -> set[str]:
    """Suy intent structured CẦN có từ expected_sources. docs/ = semantic, bỏ qua."""
    out = set()
    for s in case.get("expected_sources", []):
        if s.startswith("calendar/"):
            out.add("calendar")
        elif s.startswith("task/"):
            out.add("tasks")
        elif s.startswith("email/"):
            out.add("emails")
    return out


def _recall(expected, got):
    if not expected:
        return None  # refuse case
    hits = sum(1 for e in expected if any(e in s for s in got))
    return hits / len(expected)


def _fired(scores: dict, thr: float) -> set[str]:
    return {i for i in _INTENTS if scores.get(i, 0.0) >= thr}


def _collect(cases, top_k):
    """Gọi Modal 1 lần/case, cache lại (sources + route_scores) để sweep offline."""
    import modal
    inst = modal.Cls.from_name(APP_NAME, CLS_NAME)()
    rows = []
    for c in cases:
        q = _last_user_query(c)
        try:
            r = inst.retrieve_only.remote(q, top_k=top_k)
            sources = r.get("sources", [])
            scores = r.get("route_scores", {}) or {}
            err = r.get("error")
        except Exception as e:
            sources, scores, err = [], {}, str(e)
        rows.append({
            "id": c.get("id"), "query": q, "sources": sources, "scores": scores,
            "expected": c.get("expected_sources", []),
            "want_intents": _expected_intents(c),
            "is_refuse": not c.get("expected_sources", []),
            "err": err,
        })
    return rows


def _print_scores_table(rows, thr):
    print(f"\n{'='*78}\n1) ROUTE SCORES  (fire khi ≥ ngưỡng hiện tại {thr})\n{'='*78}")
    print(f"{'ID':<6}{'cal':>7}{'task':>7}{'email':>7}  {'fire':<22} {'recall':>7}  query")
    print("-" * 78)
    for r in rows:
        s = r["scores"]
        fire = _fired(s, thr)
        rc = _recall(r["expected"], r["sources"])
        rc_s = "refuse" if rc is None else f"{rc*100:.0f}%"
        fire_s = ",".join(sorted(fire)) or "—"
        print(f"{r['id']:<6}{s.get('calendar',0):>7}{s.get('tasks',0):>7}"
              f"{s.get('emails',0):>7}  {fire_s:<22} {rc_s:>7}  {r['query'][:34]}")


def _print_recall_summary(rows):
    print(f"\n{'='*78}\n2) RECALL / REFUSE\n{'='*78}")
    recalls = [_recall(r["expected"], r["sources"]) for r in rows]
    data = [x for x in recalls if x is not None]
    if data:
        print(f"  Recall@k trung bình (case có data): {sum(data)/len(data)*100:.1f}%  (n={len(data)})")
    refuse = [r for r in rows if r["is_refuse"]]
    clean = sum(1 for r in refuse if not r["sources"])
    if refuse:
        print(f"  Refuse clean: {clean}/{len(refuse)}")
        for r in refuse:
            if r["sources"]:
                print(f"    LEAK {r['id']}: {r['sources']}")


def _sweep(rows, lo, hi, step):
    print(f"\n{'='*78}\n3) QUÉT NGƯỠNG  ({lo} → {hi}, bước {step})\n{'='*78}")
    print(f"{'thr':>6}{'miss':>6}{'bẩn':>6}{'leak':>6}   ghi chú")
    print("-" * 60)
    best = None
    t = lo
    results = []
    while t <= hi + 1e-9:
        thr = round(t, 3)
        miss = bad = leak = 0
        miss_ids, bad_ids, leak_ids = [], [], []
        for r in rows:
            fire = _fired(r["scores"], thr)
            # miss: intent cần có nhưng không fire
            for want in r["want_intents"]:
                if want not in fire:
                    miss += 1; miss_ids.append(f"{r['id']}:{want}")
            # bẩn: case có data, intent thừa fire (calendar/task mới gây nhiễu nội dung)
            if not r["is_refuse"]:
                for extra in fire - r["want_intents"]:
                    if extra in ("calendar", "tasks"):
                        bad += 1; bad_ids.append(f"{r['id']}:{extra}")
            # leak: refuse case bị calendar/task fire (email name-filtered → bỏ qua)
            if r["is_refuse"]:
                for lk in fire & {"calendar", "tasks"}:
                    leak += 1; leak_ids.append(f"{r['id']}:{lk}")
        results.append((thr, miss, bad, leak, miss_ids, bad_ids, leak_ids))
        t += step

    for thr, miss, bad, leak, mi, bi, li in results:
        note = []
        if mi: note.append("miss=" + ",".join(mi))
        if li: note.append("leak=" + ",".join(li))
        print(f"{thr:>6}{miss:>6}{bad:>6}{leak:>6}   {'; '.join(note)[:40]}")

    # Gợi ý: ưu tiên miss=0 & leak=0, rồi bẩn nhỏ nhất, rồi ngưỡng thấp nhất (an toàn recall).
    feasible = [r for r in results if r[1] == 0 and r[3] == 0]
    pool = feasible or [r for r in results if r[1] == 0] or results
    best = min(pool, key=lambda r: (r[3], r[2], r[0]))
    print("-" * 60)
    if feasible:
        print(f"  ✓ GỢI Ý RAG_ROUTE_MIN_SCORE = {best[0]}  (miss=0, leak=0, bẩn={best[2]})")
    else:
        print(f"  ⚠ Không có ngưỡng nào miss=0 & leak=0. Tốt nhất tương đối: {best[0]} "
              f"(miss={best[1]}, leak={best[3]}, bẩn={best[2]}) — cân nhắc chỉnh anchor.")


def main():
    ap = argparse.ArgumentParser(description="Chẩn đoán & tune embedding router của RAG.")
    ap.add_argument("--file", default=str(_YAML))
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--thr", type=float, default=DEFAULT_ROUTE_MIN,
                    help="Ngưỡng hiện tại để hiển thị cột 'fire' (mặc định 0.55).")
    ap.add_argument("--sweep", nargs=3, type=float, metavar=("LO", "HI", "STEP"),
                    default=[0.45, 0.70, 0.01])
    ap.add_argument("--deploy", action="store_true", help="modal deploy trước khi đo.")
    args = ap.parse_args()

    if args.deploy:
        print("→ Deploy modal_app.py ...")
        subprocess.run(["modal", "deploy", str(_ROOT / "modal_app.py")], check=True)

    path = Path(args.file)
    cases = yaml.safe_load(path.read_text(encoding="utf-8")).get("test_cases", [])
    print(f"Chẩn đoán router trên {len(cases)} case ({path.name}), top_k={args.top_k}")

    rows = _collect(cases, args.top_k)
    errs = [r for r in rows if r["err"]]
    if errs:
        for r in errs:
            print(f"  ERR {r['id']}: {r['err']}")
    _print_scores_table(rows, args.thr)
    _print_recall_summary(rows)
    _sweep(rows, args.sweep[0], args.sweep[1], args.sweep[2])


if __name__ == "__main__":
    main()
