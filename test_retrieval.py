"""Unit test local cho retrieval.py — chạy KHÔNG cần GPU/model thật.

    python test_retrieval.py        # tự chạy, in PASS/FAIL
    pytest test_retrieval.py        # nếu có pytest

Phần semantic dùng FakeEmbedder (bag-of-words hashing) để kiểm plumbing +
xác nhận chunk có từ trùng query được xếp hạng cao — không cần bge-m3 thật.
"""
import hashlib
import sys
from datetime import date

import numpy as np

import retrieval as R

# Windows console cp1252 → crash khi in tiếng Việt. Force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, Exception):
    pass

REF = date(2026, 5, 27)  # thứ Tư


# ───────────────────────────────────────────────
# FakeEmbedder: bag-of-words hashing → vector cố định theo text
# ───────────────────────────────────────────────
class FakeEmbedder:
    DIM = 256

    def encode(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in R.strip_accents(t).split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM
                out[i, h] += 1.0
        return out


# ───────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────
def test_strip_accents():
    assert R.strip_accents("Lịch hôm nay") == "lich hom nay"
    assert R.strip_accents("Đặt lịch") == "dat lich"


def test_date_range():
    assert R.parse_date_range("lịch hôm nay", REF) == (REF, REF)
    assert R.parse_date_range("sáng mai có gì", REF) == (date(2026, 5, 28), date(2026, 5, 28))
    assert R.parse_date_range("tuần này", REF) == (date(2026, 5, 25), date(2026, 5, 31))
    assert R.parse_date_range("tuần sau", REF) == (date(2026, 6, 1), date(2026, 6, 7))
    assert R.parse_date_range("tháng sau", REF) == (date(2026, 6, 1), date(2026, 6, 30))
    assert R.parse_date_range("việc gì gấp", REF) is None


def test_time_of_day():
    assert R.parse_time_of_day("lịch chiều nay") == ("12:00", "17:59")
    assert R.parse_time_of_day("sáng nay") == ("00:00", "11:59")
    assert R.parse_time_of_day("có gì không") is None


def test_chunk_markdown():
    md = "# Tiêu đề\nmở đầu\n## Phần 1\nnội dung 1\n## Phần 2\nnội dung 2"
    chunks = R.chunk_markdown(md, "docs/x.md")
    assert len(chunks) == 3
    assert any("Phần 1" in c.source for c in chunks)


def test_structured_calendar_today():
    r = R.Retriever(reference_date=REF)
    res = r.retrieve("lịch hôm nay")
    assert not res.is_empty
    cal = next(b for b in res.blocks if b.startswith("[Lịch"))
    assert "Họp giao ban" in cal and "Họp ngân sách Q3" in cal
    # đúng 4 sự kiện ngày 27/5
    assert cal.count("\n- ") == 4


def test_structured_calendar_afternoon():
    r = R.Retriever(reference_date=REF)
    res = r.retrieve("lich chieu nay co gi")  # không dấu
    cal = next(b for b in res.blocks if b.startswith("[Lịch"))
    assert "Họp ngân sách Q3" in cal   # 14:00
    assert "Call khách hàng X" in cal  # 16:00
    assert "Họp giao ban" not in cal   # 09:00 bị lọc


def test_structured_tasks_overdue():
    r = R.Retriever(reference_date=REF)
    res = r.retrieve("task nào đang trễ")
    tasks = next(b for b in res.blocks if b.startswith("[Công việc"))
    assert "Soạn hợp đồng khách C" in tasks and "QUÁ HẠN" in tasks


def test_structured_email():
    r = R.Retriever(reference_date=REF)
    res = r.retrieve("email anh Tuấn nói gì")
    assert any("12%" in b and "80 triệu" in b for b in res.blocks)


def test_semantic_doc_retrieval():
    # min_score=0 để test thuần plumbing/ranking (tách khỏi việc tune ngưỡng).
    old = R.SEMANTIC_MIN_SCORE
    R.SEMANTIC_MIN_SCORE = 0.0
    try:
        r = R.Retriever(reference_date=REF)
        r.build_index(FakeEmbedder())
        # token ABC/MOU/Tan/Lee chỉ có trong biên bản ABC → phải kéo đúng doc đó.
        res = r.retrieve("biên bản ghi nhớ MOU đối tác ABC ông Tan bà Lee")
        doc_sources = [s for s in res.sources if s.startswith("docs/")]
        assert any("bien_ban_hop_abc" in s for s in doc_sources), res.sources
    finally:
        R.SEMANTIC_MIN_SCORE = old


def test_empty_when_no_match():
    # query không khớp keyword structured + không build_index → semantic skip
    r = R.Retriever(reference_date=REF)
    res = r.retrieve("xin chào bạn khỏe không")
    assert res.is_empty


# ───────────────────────────────────────────────
# Runner standalone (không cần pytest)
# ───────────────────────────────────────────────
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
