from __future__ import annotations
import hashlib
import json

def stable_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()