"""Knowledge base vector search tool using ChromaDB with hybrid keyword validation."""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import chromadb
from chromadb.api import ClientAPI

from core.config import KB_DIR

logger = logging.getLogger(__name__)

STOPWORDS = {
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'what', 'which', 'who', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'get'
}


def _extract_keywords(text: str) -> List[str]:
    """Extract significant lowercase keywords for validation."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]


class KnowledgeBaseStore:
    """Singleton ChromaDB-backed knowledge base retriever."""

    _instance: Optional["KnowledgeBaseStore"] = None

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = kb_dir
        self.client: ClientAPI = chromadb.Client()
        self.collection_name = "customer_support_kb"
        self._init_collection()

    @classmethod
    def get_instance(cls) -> "KnowledgeBaseStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_collection(self) -> None:
        """Read all markdown documents from knowledge_base/ and index chunks."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(self.collection_name)

        docs: List[str] = []
        metadatas: List[dict] = []
        ids: List[str] = []
        chunk_id = 0

        md_files = sorted(self.kb_dir.glob("*.md"))
        if not md_files:
            logger.warning("No markdown files found in %s", self.kb_dir)
            return

        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            doc_title = md_file.stem.replace("_", " ").title()

            # Split by markdown headers
            sections = content.split("\n## ")
            for idx, sec in enumerate(sections):
                chunk = sec.strip()
                if not chunk:
                    continue

                if idx > 0:
                    chunk = f"## {chunk}"

                header_prefix = f"[{doc_title}] "
                full_chunk = f"{header_prefix}\n{chunk}"

                docs.append(full_chunk)
                metadatas.append({
                    "source": md_file.name,
                    "title": doc_title,
                })
                ids.append(f"chunk_{chunk_id}")
                chunk_id += 1

        if docs:
            self.collection.add(
                documents=docs,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Indexed %d knowledge base chunks into ChromaDB.", len(docs))

    def search(self, query: str, top_k: int = 3) -> str:
        """Search the knowledge base for relevant passages."""
        clean_query = query.strip()
        if not clean_query:
            return "No relevant documents found in knowledge base."

        try:
            results = self.collection.query(
                query_texts=[clean_query],
                n_results=min(top_k, self.collection.count()),
            )
        except Exception as e:
            logger.error("ChromaDB query error: %s", e)
            return "No relevant documents found in knowledge base."

        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant documents found in knowledge base."

        query_keywords = _extract_keywords(clean_query)
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]
        retrieved_dists = results["distances"][0]

        valid_chunks: List[Tuple[str, str, float]] = []

        for doc, meta, dist in zip(retrieved_docs, retrieved_metas, retrieved_dists):
            doc_lower = doc.lower()
            matched_kws = [k for k in query_keywords if k in doc_lower]

            # Semantic match criteria:
            # 1. Very strong vector match (distance < 1.05) OR
            # 2. Acceptable vector distance (distance < 1.25) with at least 1 keyword match
            if dist < 1.05 or (dist < 1.25 and len(matched_kws) >= 1):
                valid_chunks.append((doc, meta.get("source", "unknown"), dist))

        if not valid_chunks:
            return "No relevant documents found in knowledge base."

        # Deduplicate and format output
        formatted_passages: List[str] = []
        seen_passages = set()

        for doc, source, _ in valid_chunks[:2]:
            clean_text = doc.strip()
            if clean_text in seen_passages:
                continue
            seen_passages.add(clean_text)
            formatted_passages.append(f"[Source: {source}]\n{clean_text}")

        return "\n\n---\n\n".join(formatted_passages)


def search_kb(query: str) -> str:
    """Searches company knowledge base documents for relevant policies and FAQs.

    Args:
        query: The customer question, summary, or search keywords.

    Returns:
        Relevant passages with [Source: filename.md] or 'No relevant documents found in knowledge base.'
    """
    store = KnowledgeBaseStore.get_instance()
    return store.search(query)
