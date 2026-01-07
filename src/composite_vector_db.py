"""Composite Vector DB that queries multiple VectorDB backends"""
from typing import List, Tuple, Optional, Dict

from vector_db import VectorDB


class CompositeVectorDB:
    """Query across multiple VectorDBs and merge results"""

    def __init__(self, stores: List[VectorDB], store_policy: str = "private"):
        # Order convention: [private, shared]
        self.stores = stores or []
        self.store_policy = store_policy  # private | shared | both

    def add_entry(self, *args, **kwargs):
        """Add according to policy: private/shared/both"""
        if not self.stores:
            return False, "No stores configured"
        results = []
        if self.store_policy in ("private", "both") and len(self.stores) >= 1:
            results.append(self.stores[0].add_entry(*args, **kwargs))
        if self.store_policy in ("shared", "both") and len(self.stores) >= 2:
            results.append(self.stores[1].add_entry(*args, **kwargs))
        # Return success if any succeeds
        if not results:
            return False, "No target store for policy"
        for ok, rid in results:
            if ok:
                return ok, rid
        return results[0]

    def find_similar(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for store in self.stores:
            try:
                res = store.find_similar(text, top_k=top_k)
                results.extend(res)
            except Exception:
                continue
        # Deduplicate by content, keep highest similarity
        best: Dict[str, float] = {}
        for content, sim in results:
            if content not in best or sim > best[content]:
                best[content] = sim
        # Sort and take top_k
        merged = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(c, s) for c, s in merged]

    def get_stats(self):
        totals = [s.get_stats() for s in self.stores]
        total_entries = sum(t["total_entries"] for t in totals)
        index_size = sum(t["index_size"] for t in totals)
        return {
            "stores": totals,
            "total_entries": total_entries,
            "index_size": index_size
        }

    def clear(self):
        for s in self.stores:
            s.clear()

