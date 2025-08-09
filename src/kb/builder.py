import re
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def read_txt(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def read_pdf(p: Path) -> str:
    r = PdfReader(str(p))
    return "\n".join((page.extract_text() or "") for page in r.pages)


def chunk_text(text: str, max_chars=900, overlap=200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i + max_chars]
        cut = chunk.rfind(".")
        if cut > max_chars * 0.6:
            chunk = chunk[:cut + 1]
        chunks.append(chunk.strip())
        i += max(1, len(chunk) - overlap)
    return [c for c in chunks if c]


def load_docs(raw_dir: Path) -> List[Dict]:
    docs: List[Dict] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".txt":
            text = read_txt(p)
        elif ext == ".md":
            text = read_md(p)
        elif ext == ".pdf":
            text = read_pdf(p)
        else:
            continue
        text = re.sub(r"\s+\n", "\n", text).strip()
        if text:
            docs.append({"path": p, "text": text})
    return docs


def build_index(root: Path, emb_model: str) -> None:
    raw_dir = root / "kb" / "raw"
    idx_dir = root / "kb" / "index"
    chunks_csv = idx_dir / "chunks.csv"
    index_file = idx_dir / "faiss.index"

    idx_dir.mkdir(parents=True, exist_ok=True)
    docs = load_docs(raw_dir)
    if not docs:
        print(f"[warn] No docs found in {raw_dir}. Put .txt/.md/.pdf files there.")
        return

    rows = []
    for d in docs:
        for j, c in enumerate(chunk_text(d["text"])):
            rows.append({"doc_path": str(d["path"].relative_to(root)), "chunk_id": j, "content": c})
    df = pd.DataFrame(rows)
    print(f"[build] {len(docs)} docs → {len(df)} chunks")

    model = SentenceTransformer(emb_model)
    emb = model.encode(df["content"].tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine (since vectors normalized)
    index.add(emb)
    faiss.write_index(index, str(index_file))
    df.assign(model=emb_model).to_csv(chunks_csv, index=False)

    print(f"[done] index:   {index_file}")
    print(f"[done] chunks:  {chunks_csv}")
    print(f"[done] model:   {emb_model}")
