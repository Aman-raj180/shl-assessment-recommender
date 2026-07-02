"""
Retrieval layer over the SHL Individual Test Solutions catalog.

Design choice (see approach doc): BM25 keyword retrieval instead of dense
embeddings. The catalog is small (370 items) and queries tend to contain
exact/technical tokens (".NET", "OPQ32r", "Java", "GSA") that BM25 handles
very well, sometimes better than semantic embeddings which can blur exact
product-name matches. It also means zero external model downloads at
container start -> faster, more reliable cold starts on free hosting
within the 2-minute wake-up window, and no dependency on a third-party
model host being reachable at runtime.
"""
import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CATALOG_PATH = Path(__file__).parent / "catalog_clean.json"

_TOKEN_RE = re.compile(r"[a-z0-9\+\.#]+")


def _tokenize(text: str):
    return _TOKEN_RE.findall(text.lower())


def _compose_text(item: dict) -> str:
    parts = [
        item["name"],
        item["description"],
        " ".join(item["categories"]),
        " ".join(item["job_levels"]),
        " ".join(item["test_types"]),
    ]
    return " ".join(p for p in parts if p)


class CatalogRetriever:
    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog = json.load(open(catalog_path, encoding="utf-8"))
        self._by_id = {item["id"]: item for item in self.catalog}
        self._by_name_lower = {item["name"].lower(): item for item in self.catalog}

        corpus = [_compose_text(item) for item in self.catalog]
        self._tokenized_corpus = [_tokenize(t) for t in corpus]
        self.bm25 = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: int = 15):
        """Return top_k catalog items ranked by BM25 relevance to query."""
        if not query.strip():
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked_idx[:top_k]:
            if scores[i] <= 0:
                continue
            results.append({**self.catalog[i], "_score": float(scores[i])})
        return results

    def find_by_name(self, name: str):
        """Exact/near-exact name lookup, used for compare-feature grounding."""
        key = name.strip().lower()
        if key in self._by_name_lower:
            return self._by_name_lower[key]
        # fallback: substring match either direction
        for n, item in self._by_name_lower.items():
            if key in n or n in key:
                return item
        return None

    def find_many_by_names(self, names: list[str]):
        found = []
        for n in names:
            item = self.find_by_name(n)
            if item:
                found.append(item)
        return found

    def filter_by_test_type(self, items: list[dict], test_types: list[str]):
        if not test_types:
            return items
        wanted = set(test_types)
        return [i for i in items if wanted & set(i["test_types"])]


# module-level singleton, loaded once per process
retriever = CatalogRetriever()
