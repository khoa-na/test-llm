"""Đo chất lượng RETRIEVAL tách khỏi generation (recall@k).

Gọi method `retrieve_only` trên Modal cho mỗi case trong test_cases_rag_live.yaml,
đối chiếu `sources` trả về với `expected_sources` khai báo trong case.

  python eval/eval_retrieval.py                 # mặc định test_cases_rag_live.yaml
  python eval/eval_retrieval.py --top-k 6

`expected_sources` là list SUBSTRING (vd 'docs/bao_cao_tai_chinh', 'task/Soạn hợp đồng khách C',
'calendar/2026-05-27'). Một expected được tính HIT nếu khớp substring với BẤT KỲ source nào
retriever trả về → robust với heading-suffix / giờ cụ thể.

  - Case có expected_sources rỗng = REFUSE (corpus không có data): kỳ vọng retriever
    trả về RỖNG. Báo "clean" nếu đúng không lấy nguồn nào.

Vì sao tách: khi generation FAIL, biết được là do retrieval miss (recall thấp) hay do model.
"""
import argparse
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
_EVAL_DIR = Path(__file__).parent


def _last_user_query(case) -> str:
    for t in reversed(case.get("turns", [])):
        if t.get("role") == "user":
            return t.get("content", "") or ""
    return ""


def _retrieve(cls, query, top_k):
    res = cls().retrieve_only.remote(query, top_k=top_k)
    return res.get("sources", []), res.get("error")


def _recall(expected, got):
    """Mỗi expected (substring) HIT nếu khớp bất kỳ source got nào."""
    if not expected:
        return None  # refuse case
    hits = sum(1 for e in expected if any(e in s for s in got))
    return hits / len(expected)


def main():
    ap = argparse.ArgumentParser(description="Đo recall@k của RAG retrieval (Modal SDK).")
    ap.add_argument("--file", default=str(_EVAL_DIR / "test_cases_rag_live.yaml"),
                    help="YAML test set (mặc định test_cases_rag_live.yaml).")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Khong tim thay {path}")
        sys.exit(1)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = data.get("test_cases", [])

    import modal
    cls = modal.Cls.from_name(APP_NAME, CLS_NAME)

    print(f"Do retrieval recall@{args.top_k} tren {len(cases)} case ({path.name})\n")
    print(f"{'ID':<8} {'recall':>7}  {'#got':>4}  ket qua")
    print("-" * 70)

    recalls = []
    refuse_total = refuse_clean = 0
    for case in cases:
        cid = case.get("id")
        query = _last_user_query(case)
        expected = case.get("expected_sources", [])
        try:
            got, err = _retrieve(cls, query, args.top_k)
        except Exception as e:
            got, err = [], str(e)
        if err:
            print(f"{cid:<8} {'ERR':>7}        {err}")
            continue

        r = _recall(expected, got)
        if r is None:  # refuse case: kỳ vọng rỗng
            refuse_total += 1
            ok = len(got) == 0
            refuse_clean += int(ok)
            print(f"{cid:<8} {'refuse':>7}  {len(got):>4}  {'clean (rỗng đúng)' if ok else 'LEAK: ' + str(got)}")
        else:
            recalls.append(r)
            miss = [e for e in expected if not any(e in s for s in got)]
            note = "đủ" if not miss else f"thiếu {miss}"
            print(f"{cid:<8} {r*100:>6.0f}%  {len(got):>4}  {note}")

    print("-" * 70)
    if recalls:
        print(f"Recall@{args.top_k} trung binh (case co data): {sum(recalls)/len(recalls)*100:.1f}%  "
              f"(n={len(recalls)})")
    if refuse_total:
        print(f"Refuse clean: {refuse_clean}/{refuse_total} "
              f"(retriever dung khong tra nguon khi corpus khong co data)")


if __name__ == "__main__":
    main()
