import json
import os
from threading import Lock

class VirtualColumnCache:
    def __init__(self, cache_file="/data/virtual_cache/virtual_cache.json"):
        self.cache_file = cache_file
        print(f"[CACHE INIT] Cache file path: {os.path.abspath(self.cache_file)}")
        self.cache = self._load()
        self.hits = 0
        self.misses = 0

    def get(self, row_id, column_name):
        key = f"{row_id}|{column_name}"
        return self.cache.get(key)

    def set(self, row_id, column_name, value):
        key = f"{row_id}|{column_name}"
        self.cache[key] = value
        self._save()

    def get_or_compute(self, row_id, column_name, compute_fn):
        existing = self.get(row_id, column_name)
        if existing is not None:
            print(f"[CACHE HIT] {row_id} / {column_name}")
            return existing
        print(f"[CACHE MISS] {row_id} / {column_name}")
        val = compute_fn()
        self.set(row_id, column_name, val)
        return val


    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("[WARNING] Cache file is corrupted. Starting fresh.")
                return {}
        return {}

    def _save(self):
        print(f"[CACHE SAVE] Writing cache to {self.cache_file} with {len(self.cache)} entries")
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"[CACHE ERROR] Failed to write cache: {e}")

    def report_stats(self):
        print(f"[CACHE STATS] Hits: {self.hits}, Misses: {self.misses}")
