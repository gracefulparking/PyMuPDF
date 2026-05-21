"""Load tagged PDF chunks into a ChromaDB collection for semantic search.

Input:  parsed_pdfs/tagged_chunks.json
Output: corpus/chroma_db  (persistent ChromaDB)

Uses all-MiniLM-L6-v2 (local, free, 384-dim) for embeddings.
To switch to OpenAI embeddings, set OPENAI_API_KEY and pass --openai.

Usage:
    python load_to_chroma.py [--db corpus/chroma_db] [--collection higher_ed_corpus]
    python load_to_chroma.py --openai          # use text-embedding-3-small
    python load_to_chroma.py --reset           # drop and rebuild collection
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Metadata flattening
# ChromaDB only accepts str / int / float / bool — no lists or dicts.
# Lists are stored as "|"-joined strings for easy $contains filtering.
# ---------------------------------------------------------------------------

def _join(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return "|".join(str(v) for v in val if v is not None)
    return str(val)


def flatten_metadata(chunk: dict) -> dict:
    """Return a ChromaDB-safe metadata dict (all scalars)."""
    src = chunk.get("source") or {}
    return {
        # source fields
        "source_authors":        _join(src.get("authors")),
        "source_year":           int(src.get("year") or 0),
        "source_title":          str(src.get("title") or ""),
        "source_venue":          str(src.get("venue") or ""),
        "source_type":           str(src.get("type") or ""),
        "source_doi":            str(src.get("doi") or ""),
        # chunk position
        "doc_id":                str(chunk.get("doc_id") or ""),
        "section_title":         str(chunk.get("section_title") or ""),
        "section_idx":           int(chunk.get("section_idx") or 0),
        "sub_idx":               int(chunk.get("sub_idx") or 0),
        # taxonomy — lists stored as pipe-delimited strings
        "section":               str(chunk.get("section") or ""),
        "tradition":             _join(chunk.get("tradition")),
        "topics":                _join(chunk.get("topics")),
        "populations":           _join(chunk.get("populations")),
        "methods":               _join(chunk.get("methods")),
        "key_constructs":        _join(chunk.get("key_constructs")),
        "builds_on":             _join(chunk.get("builds_on")),
        "cited_by_canonical":    _join(chunk.get("cited_by_canonical")),
        "field_position":        str(chunk.get("field_position") or ""),
        "era":                   str(chunk.get("era") or ""),
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load(
    input_file: str = "parsed_pdfs/tagged_chunks.json",
    db_path: str = "corpus/chroma_db",
    collection_name: str = "higher_ed_corpus",
    use_openai: bool = False,
    reset: bool = False,
    batch_size: int = 64,
):
    import chromadb
    from chromadb.utils import embedding_functions

    chunks = json.load(open(input_file, encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {input_file}")

    # Embedding function
    if use_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY not set")
        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        )
        print("Using OpenAI text-embedding-3-small")
    else:
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        print("Using local all-MiniLM-L6-v2 (384-dim)")

    Path(db_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"Dropped existing collection '{collection_name}'")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()
    print(f"Collection '{collection_name}': {existing} docs already present")

    # Skip IDs already loaded
    if existing > 0 and not reset:
        existing_ids = set(collection.get(include=[])["ids"])
    else:
        existing_ids = set()

    to_add = [c for c in chunks if c["chunk_id"] not in existing_ids]
    print(f"Adding {len(to_add)} new chunks…")

    for i in range(0, len(to_add), batch_size):
        batch = to_add[i: i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[flatten_metadata(c) for c in batch],
        )
        print(f"  {min(i + batch_size, len(to_add))}/{len(to_add)}", end="\r", flush=True)

    print(f"\nDone. Collection now has {collection.count()} documents.")
    print(f"DB path: {Path(db_path).resolve()}")
    return collection


# ---------------------------------------------------------------------------
# Quick smoke-test query
# ---------------------------------------------------------------------------

def demo_query(collection, query: str = "student identity development meaning making"):
    results = collection.query(query_texts=[query], n_results=3)
    print(f"\nTop 3 results for: '{query}'")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        print(f"\n[{i+1}] dist={dist:.3f}  {meta['source_title'][:60]}")
        print(f"     section={meta['section']}  topics={meta['topics'][:60]}")
        print(f"     {doc[:200].replace(chr(10), ' ')}…")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="parsed_pdfs/tagged_chunks.json")
    parser.add_argument("--db", default="corpus/chroma_db")
    parser.add_argument("--collection", default="higher_ed_corpus")
    parser.add_argument("--openai", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Drop and rebuild collection")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-demo", action="store_true", help="Skip demo query")
    args = parser.parse_args()

    col = load(
        input_file=args.input,
        db_path=args.db,
        collection_name=args.collection,
        use_openai=args.openai,
        reset=args.reset,
        batch_size=args.batch_size,
    )
    if not args.no_demo:
        demo_query(col)
