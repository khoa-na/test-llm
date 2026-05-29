"""Hybrid RAG retrieval cho chatbot thư ký — server-side, dùng được cả local.

KHUNG (Phase A). Hai nguồn dữ liệu, gộp kết quả theo kiểu always-hybrid:

  1. STRUCTURED (calendar / task / email metadata)
     → SQLite `:memory:` build từ seed JSON trong `rag_corpus/`.
     → query bằng SQL, lọc theo ngày/trạng thái. Ngày tương đối ("hôm nay",
       "tuần này", "chiều nay"...) neo vào REFERENCE_DATE (KHÔNG dùng now() vì
       container Modal chạy UTC và test cases neo ngày cố định 27/05/2026).

  2. SEMANTIC (nội dung docs `.md` + body email)
     → numpy cosine trên embedding (bge-m3). Embedder được TIÊM VÀO
       (`build_index(embedder)`), bất kỳ object nào có `.encode(list[str]) -> ndarray`
       — nhờ vậy unit test local chạy được không cần GPU/model thật (mock embedder).

Module này KHÔNG import modal/torch → import nhanh, test local dễ.
"""
from __future__ import annotations

import functools
import json
import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# ───────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────
_CORPUS_DIR = Path(os.environ.get("RAG_CORPUS_DIR", Path(__file__).parent / "rag_corpus"))


def _resolve_reference_date() -> date:
    """Ngày 'hôm nay' của trợ lý. Neo qua env RAG_REFERENCE_DATE (YYYY-MM-DD)."""
    raw = os.environ.get("RAG_REFERENCE_DATE", "").strip()
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            pass
    return date.today()


REFERENCE_DATE: date = _resolve_reference_date()
DEFAULT_TOP_K: int = int(os.environ.get("RAG_TOP_K", "4"))
# Ngưỡng cosine tối thiểu để 1 chunk được coi là liên quan (lọc nhiễu).
SEMANTIC_MIN_SCORE: float = float(os.environ.get("RAG_MIN_SCORE", "0.35"))

# ───────────────────────────────────────────────
# Router keywords (so khớp sau khi BỎ DẤU — để input không dấu "lich" vẫn khớp "lịch")
# ───────────────────────────────────────────────
_CAL_KW = ["lich", "hop", "cuoc hop", "calendar", "may gio", "gio nao", "gap", "hen", "su kien"]
_TASK_KW = ["task", "viec", "cong viec", "deadline", "han", "tre", "qua han", "gap rut",
            "uu tien", "ton dong", "to-do", "to do", "nhiem vu",
            # trạng thái/tiến độ + báo cáo (vd "Minh gửi báo cáo Q2 chưa")
            "bao cao", "trang thai", "tien do", "hoan thanh", "da xong", "da gui", "da nop"]
_EMAIL_KW = ["email", "mail", "thu dien tu", "noi gi", "gui gi", "viet gi"]

# Honorific/chức danh — bỏ khi trích TÊN người từ trường sender (để lọc email đúng người).
_HONORIFIC = {"anh", "chi", "em", "ong", "ba", "co", "chu", "bac", "ngai", "sep",
              "ke", "toan", "truong", "phong", "nhan", "su", "giam", "doc",
              "cfo", "cto", "ceo", "hr"}


# ───────────────────────────────────────────────
# Data classes
# ───────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    source: str          # nhãn nguồn để trích dẫn (vd "docs/bao_cao_tai_chinh_q1_2026.md#Dòng tiền")
    doc_id: str          # id thô của tài liệu gốc


@dataclass
class RetrievalResult:
    blocks: list[str] = field(default_factory=list)   # các khối text đã format để chèn vào prompt
    sources: list[str] = field(default_factory=list)  # id nguồn (phục vụ đo recall@k)

    @property
    def is_empty(self) -> bool:
        return not self.blocks


# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def strip_accents(s: str) -> str:
    """Bỏ dấu tiếng Việt + lowercase, để so khớp keyword robust với input không dấu."""
    s = s.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


@functools.lru_cache(maxsize=None)
def _kw_re(kw: str) -> "re.Pattern":
    # Khớp theo RANH GIỚI TỪ (sau khi bỏ dấu → \\w là ascii) để 'han' KHÔNG khớp
    # bên trong 'nhan' (nhân), 'thanh'... Cụm nhiều từ ('qua han') vẫn khớp đúng.
    return re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)")


def _any_kw(norm_query: str, kws) -> bool:
    return any(_kw_re(kw).search(norm_query) for kw in kws)


def _name_tokens(sender: str) -> set[str]:
    """Token TÊN người từ trường sender (bỏ honorific/chức danh). Dùng để khớp người
    được hỏi trong query. Vd 'chị Lan (Kế toán trưởng)' → {'lan'}; 'anh Tuấn' → {'tuan'}."""
    toks = re.findall(r"[a-z0-9]+", strip_accents(sender))
    return {t for t in toks if t not in _HONORIFIC and len(t) > 1}


# ───────────────────────────────────────────────
# Chunking (docs .md — heading-aware)
# ───────────────────────────────────────────────
def chunk_markdown(text: str, doc_id: str, max_chars: int = 1200) -> list[Chunk]:
    """Tách .md theo heading (##/###). Section quá dài → cắt tiếp theo đoạn.

    Giữ heading gần nhất vào đầu mỗi chunk làm ngữ cảnh.
    """
    chunks: list[Chunk] = []
    current_heading = ""
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if not body:
            return
        label = f"{doc_id}#{current_heading}" if current_heading else doc_id
        # cắt theo max_chars nếu section dài (overlap nhẹ ở ranh giới đoạn)
        if len(body) <= max_chars:
            chunks.append(Chunk(text=_with_heading(current_heading, body), source=label, doc_id=doc_id))
        else:
            for piece in _split_long(body, max_chars):
                chunks.append(Chunk(text=_with_heading(current_heading, piece), source=label, doc_id=doc_id))

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            flush()
            buf = []
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        else:
            buf.append(line)
    flush()
    return chunks


def _with_heading(heading: str, body: str) -> str:
    return f"## {heading}\n{body}" if heading else body


def _split_long(body: str, max_chars: int) -> list[str]:
    paras = re.split(r"\n\s*\n", body)
    out, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > max_chars:
            out.append(cur.strip())
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur.strip():
        out.append(cur.strip())
    return out


# ───────────────────────────────────────────────
# Date-range parser (ngày tương đối → range cụ thể, neo REFERENCE_DATE)
# ───────────────────────────────────────────────
def parse_date_range(query: str, ref: date | None = None) -> tuple[date, date] | None:
    """Trả (start, end) inclusive cho biểu thức thời gian trong query, hoặc None.

    Hỗ trợ: hôm nay, mai, hôm qua, tuần này, tuần sau, tháng này, tháng sau.
    (Khung — mở rộng dần: 'thứ 4 tới', 'cuối tháng', 'đầu tháng sau'...)
    """
    ref = ref or REFERENCE_DATE
    q = strip_accents(query)

    if "hom nay" in q or "today" in q or "chieu nay" in q or "sang nay" in q or "toi nay" in q:
        return ref, ref
    if "ngay mai" in q or "sang mai" in q or re.search(r"\bmai\b", q):
        d = ref + timedelta(days=1)
        return d, d
    if "hom qua" in q:
        d = ref - timedelta(days=1)
        return d, d
    if "tuan nay" in q or "this week" in q:
        return _week_bounds(ref)
    if "tuan sau" in q or "tuan toi" in q or "next week" in q:
        return _week_bounds(ref + timedelta(days=7))
    if "thang sau" in q or "thang toi" in q:
        return _month_bounds(_first_of_next_month(ref))
    if "thang nay" in q:
        return _month_bounds(ref)
    return None


def parse_time_of_day(query: str) -> tuple[str, str] | None:
    """Lọc theo buổi: trả (start_hhmm, end_hhmm) hoặc None."""
    q = strip_accents(query)
    if "sang" in q:
        return "00:00", "11:59"
    if "chieu" in q:
        return "12:00", "17:59"
    if "toi" in q:
        return "18:00", "23:59"
    return None


def _week_bounds(d: date) -> tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


def _month_bounds(d: date) -> tuple[date, date]:
    first = d.replace(day=1)
    nxt = _first_of_next_month(d)
    return first, nxt - timedelta(days=1)


def _first_of_next_month(d: date) -> date:
    return d.replace(day=28) + timedelta(days=4) if d.month == 12 \
        else date(d.year + (d.month // 12), (d.month % 12) + 1, 1)


# ───────────────────────────────────────────────
# Retriever
# ───────────────────────────────────────────────
class Retriever:
    """Hybrid retriever. Vòng đời:

        r = Retriever()                 # load seed JSON → SQLite, chunk docs
        r.build_index(embedder)         # encode chunks (cần ở server, sau khi có GPU)
        res = r.retrieve("lịch hôm nay")  # -> RetrievalResult
    """

    def __init__(self, corpus_dir: Path | str = _CORPUS_DIR, reference_date: date | None = None):
        self.corpus_dir = Path(corpus_dir)
        self.reference_date = reference_date or REFERENCE_DATE
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None
        self._embedder = None
        self._load_structured()
        self._load_docs()

    # ----- build phase -----
    def _load_structured(self):
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE calendar(date TEXT, start TEXT, "end" TEXT, title TEXT,
                                  location TEXT, attendees TEXT);
            CREATE TABLE tasks(title TEXT, due_date TEXT, status TEXT,
                               priority TEXT, assignee TEXT);
            CREATE TABLE emails(sender TEXT, date TEXT, subject TEXT, body TEXT);
            """
        )
        self._seed_table("calendar.json", "calendar",
                         ["date", "start", "end", "title", "location", "attendees"])
        self._seed_table("tasks.json", "tasks",
                         ["title", "due_date", "status", "priority", "assignee"])
        self._seed_table("emails.json", "emails",
                         ["sender", "date", "subject", "body"])
        self.conn.commit()

    def _seed_table(self, fname: str, table: str, cols: list[str]):
        path = self.corpus_dir / fname
        if not path.exists():
            return
        rows = json.loads(path.read_text(encoding="utf-8"))
        placeholders = ", ".join("?" for _ in cols)
        quoted = ", ".join(f'"{c}"' for c in cols)
        for row in rows:
            vals = [row.get(c, "") for c in cols]
            # email body cũng đẩy vào semantic index
            self.conn.execute(
                f'INSERT INTO {table}({quoted}) VALUES ({placeholders})', vals
            )

    def _load_docs(self):
        docs_dir = self.corpus_dir / "docs"
        if docs_dir.exists():
            for p in sorted(docs_dir.glob("*.md")):
                self.chunks += chunk_markdown(p.read_text(encoding="utf-8"), f"docs/{p.name}")
        # body email → semantic chunk (để hỏi "email X nói gì" bắt được nội dung)
        for r in self.conn.execute("SELECT sender, date, subject, body FROM emails"):
            txt = f"Email từ {r['sender']} ({r['date']}) — {r['subject']}: {r['body']}"
            self.chunks.append(Chunk(text=txt, source=f"email/{r['sender']}", doc_id=f"email/{r['sender']}"))

    def build_index(self, embedder):
        """Encode toàn bộ chunk. `embedder.encode(list[str]) -> ndarray (N, D)`."""
        self._embedder = embedder
        if not self.chunks:
            self._embeddings = None
            return
        emb = np.asarray(embedder.encode([c.text for c in self.chunks]), dtype=np.float32)
        self._embeddings = _l2_normalize(emb)

    # ----- query phase -----
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        """Always-hybrid: gộp structured (theo router keyword) + semantic."""
        res = RetrievalResult()
        norm = strip_accents(query)

        if _any_kw(norm, _CAL_KW):
            self._add_calendar(query, res)
        if _any_kw(norm, _TASK_KW):
            self._add_tasks(query, res)
        if _any_kw(norm, _EMAIL_KW):
            self._add_emails(query, res)

        self._add_semantic(query, res, top_k)
        return res

    def _add_calendar(self, query: str, res: RetrievalResult):
        rng = parse_date_range(query, self.reference_date) or (self.reference_date, self.reference_date)
        rows = self.conn.execute(
            'SELECT * FROM calendar WHERE date BETWEEN ? AND ? ORDER BY date, start',
            (rng[0].isoformat(), rng[1].isoformat()),
        ).fetchall()
        tod = parse_time_of_day(query)
        if tod:
            rows = [r for r in rows if tod[0] <= (r["start"] or "") <= tod[1]]
        if not rows:
            return
        lines = [f"- {r['date']} {r['start']}–{r['end']} {r['title']}"
                 + (f" @ {r['location']}" if r["location"] else "")
                 + (f" (thành phần: {r['attendees']})" if r["attendees"] else "")
                 for r in rows]
        res.blocks.append("[Lịch — dữ liệu truy xuất]\n" + "\n".join(lines))
        res.sources += [f"calendar/{r['date']}/{r['start']}" for r in rows]

    def _add_tasks(self, query: str, res: RetrievalResult):
        rows = self.conn.execute(
            'SELECT * FROM tasks ORDER BY due_date'
        ).fetchall()
        if not rows:
            return
        ref = self.reference_date.isoformat()
        lines = []
        for r in rows:
            overdue = (r["due_date"] or "") < ref and (r["status"] or "").lower() not in ("done", "đã xong", "hoàn thành")
            flag = " ⚠️QUÁ HẠN" if overdue else ""
            lines.append(f"- {r['title']} | hạn {r['due_date']} | {r['status']} | ưu tiên {r['priority']}"
                         + (f" | {r['assignee']}" if r["assignee"] else "") + flag)
        res.blocks.append("[Công việc/Task — dữ liệu truy xuất]\n" + "\n".join(lines))
        res.sources += [f"task/{r['title']}" for r in rows]

    def _add_emails(self, query: str, res: RetrievalResult):
        rows = self.conn.execute('SELECT * FROM emails ORDER BY date DESC').fetchall()
        if not rows:
            return
        # Lọc theo TÊN người được hỏi (vd "email anh Tuấn" → chỉ email của Tuấn).
        # Nếu query KHÔNG nêu tên khớp sender nào → trả rỗng (không dump cả hộp thư).
        # Nhờ vậy hỏi người KHÔNG có trong store ("email chị Hương") → rỗng → bot refuse.
        qn = strip_accents(query)
        matched = [r for r in rows if any(tok in qn for tok in _name_tokens(r["sender"]))]
        if not matched:
            return
        lines = [f"- Từ {r['sender']} ({r['date']}) — {r['subject']}: {r['body']}" for r in matched]
        res.blocks.append("[Email — dữ liệu truy xuất]\n" + "\n".join(lines))
        res.sources += [f"email/{r['sender']}" for r in matched]

    def _add_semantic(self, query: str, res: RetrievalResult, top_k: int):
        if self._embeddings is None or self._embedder is None or not self.chunks:
            return
        q = np.asarray(self._embedder.encode([query]), dtype=np.float32)
        q = _l2_normalize(q)
        scores = (self._embeddings @ q.T).ravel()
        order = np.argsort(-scores)[:top_k]
        for i in order:
            if scores[i] < SEMANTIC_MIN_SCORE:
                continue
            c = self.chunks[i]
            if c.source in res.sources:  # tránh trùng email đã có từ structured
                continue
            res.blocks.append(f"[Tài liệu truy xuất: {c.source} | score={scores[i]:.2f}]\n{c.text}")
            res.sources.append(c.source)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norm, 1e-12, None)
