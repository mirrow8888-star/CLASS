# ═══ 多策略 RAG 文件問答（網頁版）· Streamlit + Groq + ChromaDB + LINE / Telegram 投放 ═══
# 執行：streamlit run line_rag_web.py
# LINE：側欄貼上 Channel Access Token，問答結果（500 字內）可 Broadcast 給所有好友
# Telegram：側欄貼上 Bot Token 與 Your Chat ID，問答結果（text）送到你的 Telegram
# 需要：Python 3.9+（建議 3.12）
from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chromadb
import httpx
import numpy as np
import streamlit as st
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from linebot.v3.messaging.exceptions import ApiException
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

# ── 設定 ──
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")   # 可換 multilingual-e5-base、BAAI/bge-m3
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
TG_LIMIT, TG_HOST = 500, "https://api.telegram.org"   # Telegram 單則上限 4096 字，這裡預設 500，可自行調高
LINE_LIMIT, LINE_HOST = 500, "https://api.line.me"   # LINE 訊息字數上限（可自行調整）

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """嵌入模型 + 向量庫 + 目前載入的單一文件。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體模式，關閉程式即清空
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self.doc: Doc | None = None

    @property
    def ready(self):
        return self._embedder is not None

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 文件載入 ──
    def load(self, path):
        t0, path = time.time(), Path(path)
        if path.suffix.lower() not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        segs = self._pdf(path) if path.suffix.lower() == ".pdf" else self._docx(path)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, path.name, [p[0] for p in pairs], [p[1] for p in pairs])
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids], [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.drop(doc)
            raise
        self.drop(self.doc)  # 新文件成功後才釋放舊文件
        self.doc = doc
        return f"✓ {doc.name}\n{len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def drop(self, doc):
        if doc is None:
            return
        for c in (self.col, self.ccol):
            try:
                c.delete(where={"sid": doc.sid})
            except Exception:
                pass

    def _pdf(self, path):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(str(path)).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, path):
        doc, segs, head, buf, nt = Document(str(path)), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=self._enc(docs[i:i + b]))

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        res = (self.ccol if child else self.col).query(query_embeddings=self._enc(qs, True), n_results=max(1, min(k, total)),
                                                       where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。"""
        doc = self.doc
        if doc is None:
            return "請先載入 PDF 或 DOCX 文件。", "", ""
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（檢索結果請看「檢索片段」分頁）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


# ── 匯出：Word / CSV ──
def export_csv(records, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def export_docx(records, path, with_src=True):
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Microsoft JhengHei", Pt(11)
    st.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    doc.save(path)


def plain(md: str) -> str:
    """Tk 文字框不渲染 Markdown：移除粗體標記，標題轉成一般行。"""
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


# ── LINE Bot 投放 ──
def line_text(q, ans, limit=LINE_LIMIT):
    """組成 LINE 訊息：問題 + 回答（去除 Markdown 與 [片段n] 標註），總長不超過 limit 字。"""
    body = re.sub(r"[ \t]*\[片段[^\]]*\]", "", plain(ans))
    body = re.sub(r"[ \t]+(?=\n|$)", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    q = " ".join(q.split())
    msg = f"❓ {q[:59] + '…' if len(q) > 60 else q}\n\n{body}"
    if len(msg) > limit:
        cut = msg[:limit - 1]
        end = max(cut.rfind(c) for c in "\n。！？；")  # 盡量在句尾截斷
        msg = (cut[:end + 1] if end > limit * 0.6 else cut).rstrip() + "…"
    return msg


def line_send(token, text, host=LINE_HOST):
    """Broadcast 給所有加入此官方帳號的好友（會消耗每月訊息額度）。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


def line_check(token, host=LINE_HOST):
    """驗證 Token，回傳官方帳號名稱。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_err(e):
    if isinstance(e, ApiException):
        hint = {400: "請求格式錯誤", 401: "Token 無效或已過期", 403: "權限不足", 429: "超過訊息額度或請求過於頻繁"}.get(e.status, "")
        return f"HTTP {e.status} {hint}".strip()
    return f"{type(e).__name__}: {e}"


# ── Telegram Bot 投放 ──
class TgError(Exception):
    def __init__(self, code, desc):
        super().__init__(f"{code} {desc}")
        self.code, self.desc = code, desc


def tg_api(token, method, **payload):
    """呼叫 Telegram Bot API，失敗時拋出 TgError。"""
    r = httpx.post(f"{TG_HOST}/bot{token}/{method}", json=payload, timeout=15)
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code != 200 or not data.get("ok"):
        raise TgError(data.get("error_code", r.status_code), data.get("description", r.reason_phrase))
    return data.get("result")


def tg_check(token, chat_id=""):
    """驗證 Bot Token（並在有填 Chat ID 時一併確認可送達），回傳 Bot 名稱。"""
    me = tg_api(token, "getMe")
    who = "@" + me["username"] if me.get("username") else me.get("first_name", "Telegram Bot")
    if chat_id:
        tg_api(token, "getChat", chat_id=chat_id)
    else:
        who += "（尚未填 Chat ID）"
    return who


def tg_send(token, chat_id, text):
    """把 text 送到指定 Chat ID（個人、群組或頻道）。純文字，不使用 parse_mode 以免特殊符號出錯。"""
    tg_api(token, "sendMessage", chat_id=chat_id, text=text)


def tg_err(e):
    if isinstance(e, TgError):
        d = e.desc.lower()
        if "chat not found" in d:
            return "找不到此 Chat ID（請先在 Telegram 對 Bot 按 /start，或確認 ID 正確）"
        hint = {400: "請求格式錯誤", 401: "Bot Token 無效", 403: "Bot 被封鎖或無權限發言", 404: "Bot Token 格式不正確",
                429: "請求過於頻繁，請稍後再試"}.get(e.code, "")
        return f"HTTP {e.code} {hint or e.desc}".strip()
    if isinstance(e, httpx.HTTPError):
        return f"網路連線失敗（{type(e).__name__}）"
    return f"{type(e).__name__}: {e}"


# ═══════════════ 網頁介面 ═══════════════
st.set_page_config(page_title="RAG 文件問答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="⏳ 載入嵌入模型（首次需下載，約 470 MB）…")
def get_embedder():
    """嵌入模型全站共用一份，避免每位使用者各載入一次。"""
    return RAG().embedder()


@st.cache_data(ttl=600, show_spinner=False)
def cached_models(key):
    return list_models(key)


def init_state():
    S = st.session_state
    if "rag" not in S:
        rag = RAG()
        rag._embedder = get_embedder()
        S.rag, S.records, S.last_qa, S.doc_sig, S.last_src = rag, [], None, None, ""
        S.msgs = {"line": ("", ""), "tg": ("", "")}   # 各渠道最近一次狀態 (ok/err, 文字)
    return S


def tmp_bytes(writer, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = f.name
    try:
        writer(path)
        return Path(path).read_bytes()
    finally:
        os.unlink(path)


def deliver(ch, label, fn, errfmt, text):
    """執行投放並把結果存進 session，供側欄顯示。"""
    try:
        fn(text)
        st.session_state.msgs[ch] = ("ok", f"✓ 已投放到 {label}（{len(text)} 字）· {datetime.now():%H:%M:%S}")
    except Exception as e:  # noqa: BLE001
        st.session_state.msgs[ch] = ("err", f"✗ {label} 投放失敗：{errfmt(e)}")


def show_msg(ch, empty_hint):
    kind, msg = st.session_state.msgs[ch]
    if msg:
        (st.success if kind == "ok" else st.error)(msg)
    else:
        st.caption(empty_hint)


def main():
    S = init_state()

    # ── 側欄 ──
    with st.sidebar:
        st.title("📚 RAG 文件問答")
        st.caption("Groq × ChromaDB × LINE × Telegram")

        with st.container(border=True):
            st.markdown("**🔑 Groq 模型**")
            key = st.text_input("Groq API Key", type="password", placeholder="gsk_…", label_visibility="collapsed")
            models = cached_models(key.strip()) if key.strip() else []
            if key.strip():
                st.caption(f"✓ Key 有效 · {len(models)} 個模型" if models else "✗ Key 無效或無法取得模型")
            model = st.selectbox("模型", models or MODELS, label_visibility="collapsed")

        with st.container(border=True):
            st.markdown("**📄 文件**")
            up = st.file_uploader("選擇 PDF / DOCX", type=["pdf", "docx"], label_visibility="collapsed")
            if up is not None and S.doc_sig != (up.name, up.size):
                with st.spinner("⏳ 解析並建立索引中…"):
                    p = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{Path(up.name).name}"
                    p.write_bytes(up.getvalue())
                    try:
                        S.rag.load(p)
                        S.doc_sig = (up.name, up.size)
                    except Exception as e:  # noqa: BLE001
                        st.error(f"✗ {e}" if isinstance(e, ValueError) else f"✗ 載入失敗：{type(e).__name__}: {e}")
                    finally:
                        p.unlink(missing_ok=True)
            if S.rag.doc:
                st.success(f"✓ {S.rag.doc.name}\n\n{len(S.rag.doc.chunks)} 片段")
            else:
                st.caption("尚未載入文件")

        with st.container(border=True):
            st.markdown("**🔎 檢索設定**")
            strategy = st.selectbox("策略", list(STRATEGIES), format_func=lambda k: STRATEGIES[k][0], label_visibility="collapsed")
            st.caption(STRATEGIES[strategy][1])
            topk = st.slider("片段數量 Top-K", 1, 10, 4)

        with st.container(border=True):
            st.markdown("**📤 投放回答**")
            t_line, t_tg = st.tabs(["💬 LINE", "✈️ Telegram"])

            with t_line:
                ltok = st.text_input("Channel Access Token", type="password", key="line_tok",
                                     placeholder="貼上 LINE Channel Access Token").strip()
                lauto = st.checkbox(f"回答後自動投放（≤{LINE_LIMIT} 字）", key="line_auto",
                                    help="Broadcast 會群發給所有好友，並消耗每月訊息額度。")
                if lauto:
                    st.warning("自動投放已啟用：每次回答都會群發給所有好友。")
                c1, c2 = st.columns(2)
                if c1.button("✔ 驗證", key="line_check", use_container_width=True, disabled=not ltok):
                    try:
                        S.msgs["line"] = ("ok", f"✓ 驗證成功 · {line_check(ltok) or 'LINE 官方帳號'}")
                    except Exception as e:  # noqa: BLE001
                        S.msgs["line"] = ("err", f"✗ 驗證失敗：{line_err(e)}")
                if c2.button("📤 投放", key="line_send", use_container_width=True, disabled=not (ltok and S.last_qa)):
                    deliver("line", "LINE", lambda t: line_send(ltok, t), line_err, line_text(*S.last_qa))
                show_msg("line", "⚠ 尚未設定 Token" if not ltok else "按「驗證」確認 Token")

            with t_tg:
                ttok = st.text_input("Bot Token", type="password", key="tg_tok",
                                     placeholder="向 @BotFather 取得 Bot Token").strip()
                tchat = st.text_input("Your Chat ID", key="tg_chat", placeholder="例如 123456789").strip()
                tauto = st.checkbox(f"回答後自動投放（≤{TG_LIMIT} 字）", key="tg_auto")
                c1, c2 = st.columns(2)
                if c1.button("✔ 驗證", key="tg_check", use_container_width=True, disabled=not ttok):
                    try:
                        S.msgs["tg"] = ("ok", f"✓ 驗證成功 · {tg_check(ttok, tchat)}")
                    except Exception as e:  # noqa: BLE001
                        S.msgs["tg"] = ("err", f"✗ 驗證失敗：{tg_err(e)}")
                if c2.button("📤 投放", key="tg_send", use_container_width=True, disabled=not (ttok and tchat and S.last_qa)):
                    deliver("tg", "Telegram", lambda t: tg_send(ttok, tchat, t), tg_err, line_text(*S.last_qa, TG_LIMIT))
                show_msg("tg", "⚠ 尚未設定 Token / Chat ID" if not (ttok and tchat) else "按「驗證」確認設定")
                st.caption("請先對你的 Bot 按一次 /start，才能收到訊息。Chat ID 可用 @userinfobot 查詢。")

        with st.container(border=True):
            st.markdown("**💾 匯出問答紀錄**")
            with_src = st.checkbox("Word 包含參考片段", value=True)
            if S.records:
                c1, c2 = st.columns(2)
                stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
                c1.download_button("📄 Word", tmp_bytes(lambda p: export_docx(S.records, p, with_src), ".docx"),
                                   f"RAG問答紀錄_{stamp}.docx", use_container_width=True)
                c2.download_button("📊 CSV", tmp_bytes(lambda p: export_csv(S.records, p), ".csv"),
                                   f"RAG問答紀錄_{stamp}.csv", use_container_width=True)
            else:
                st.caption("目前沒有問答紀錄可匯出")

        if st.button("🗑 清除對話", use_container_width=True):
            S.records, S.last_qa, S.last_src = [], None, ""
            S.msgs = {"line": ("", ""), "tg": ("", "")}
            st.rerun()

    # ── 主區 ──
    tab_chat, tab_src = st.tabs(["💬 對話", "🔎 檢索片段"])
    with tab_chat:
        if not S.records:
            st.info("請先在左側載入 PDF / DOCX，再於下方輸入問題。例如：這份文件的重點是什麼？")
        for r in S.records:
            with st.chat_message("user"):
                st.write(r["問題"])
            with st.chat_message("assistant"):
                st.markdown(r["回答"])
                st.caption(f"{r['策略']} · {r['模型']} · {r['時間']}")
    with tab_src:
        st.text(S.last_src or "尚無檢索結果")

    q = st.chat_input("輸入問題，Enter 送出")
    if q:
        q = q.strip()
        with st.spinner("思考中…"):
            try:
                ans, src, timing = S.rag.answer(q, strategy, topk, model, key)
            except Exception as e:  # noqa: BLE001
                ans, src, timing = f"✗ {type(e).__name__}: {e}", "", ""
        S.last_src = src
        S.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          "文件": S.rag.doc.name if S.rag.doc else "", "問題": q, "回答": ans,
                          "策略": STRATEGIES[strategy][0], "模型": model or MODELS[0],
                          "參考片段": src.split("─" * 40, 1)[-1].strip()})
        if timing:  # timing 非空 = 模型確實產生了回答（排除未載入文件／無 Key／無結果等提示）
            S.last_qa = (q, ans)
            if lauto and ltok:
                deliver("line", "LINE", lambda t: line_send(ltok, t), line_err, line_text(q, ans))
            if tauto and ttok and tchat:
                deliver("tg", "Telegram", lambda t: tg_send(ttok, tchat, t), tg_err, line_text(q, ans, TG_LIMIT))
        st.rerun()


main()
