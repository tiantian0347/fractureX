# fracturex/postprocess/history.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class HistoryLogger:
    path: str
    fieldnames: Optional[list] = None

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def _ensure_header(self, keys):
        if self.fieldnames is None:
            self.fieldnames = list(keys)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def append(self, row: Dict[str, Any]):
        self._ensure_header(row.keys())
        # 补齐缺失字段
        for k in self.fieldnames:
            row.setdefault(k, "")
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writerow(row)
