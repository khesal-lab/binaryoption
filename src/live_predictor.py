"""
اجرای زندهٔ مدل آموزش‌دیده برای تصمیم‌گیری خودکار CALL/PUT
=============================================================

مدل روی ویژگی‌هایی آموزش دیده که یکی از آن‌ها خودِ جهت معامله است
(direction_call). یعنی مدل یاد گرفته «با این وضعیت بازار + این جهت، احتمال
برد چقدر است»، نه این‌که خودش جهت را انتخاب کند. پس برای تصمیم‌گیری زنده،
یک اسنپ‌شات واحد از وضعیت لحظه‌ای بازار را دو بار به مدل می‌دهیم — یک‌بار با
فرض CALL و یک‌بار با فرض PUT — و جهتی که مدل برایش احتمال برد بالاتری
پیش‌بینی می‌کند انتخاب می‌شود (اگر از آستانهٔ اطمینان config عبور کند).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import xgboost as xgb

from src.ml_features import flatten_snapshot_for_model


class LivePredictor:
    def __init__(self, model_path: Path, feature_names_path: Path):
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        with open(feature_names_path, "r", encoding="utf-8") as f:
            self.feature_names: list[str] = json.load(f)
        print(f"[LivePredictor] مدل بارگذاری شد ({len(self.feature_names)} ویژگی).")

    def _build_row(self, base_snapshot: dict, direction: str) -> pd.DataFrame:
        row = dict(base_snapshot)
        row["direction"] = direction
        flat = flatten_snapshot_for_model(row)
        aligned = {name: flat.get(name, 0.0) for name in self.feature_names}
        return pd.DataFrame([aligned], columns=self.feature_names)

    def predict_direction(self, base_snapshot: dict) -> tuple[Optional[str], float]:
        """
        بهترین جهت (CALL یا PUT) و احتمال برد پیش‌بینی‌شدهٔ مدل برای آن را
        برمی‌گرداند. اگر خطایی رخ دهد، (None, 0.0) برمی‌گردد تا صدا‌زننده هرگز
        بر اساس دادهٔ نامعتبر معامله نکند.
        """
        best_direction: Optional[str] = None
        best_prob = -1.0
        for direction in ("CALL", "PUT"):
            X = self._build_row(base_snapshot, direction)
            prob_win = float(self.model.predict_proba(X)[0][1])
            if prob_win > best_prob:
                best_prob = prob_win
                best_direction = direction
        return best_direction, max(best_prob, 0.0)
