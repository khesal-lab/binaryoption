"""
آموزش مدل XGBoost از روی دیتاست جمع‌آوری‌شده در SQLite
==========================================================

نسبت به یک اسکریپت سادهٔ آموزش که فقط ستون‌های متنی/JSON را حذف می‌کند، این
نسخه سه تفاوت مهم دارد:

    ۱. ستون‌های JSON (chart_shape_json و recent_tick_velocities_json) به‌جای
       حذف کامل، Flatten می‌شوند (چند ستون عددی مجزا) — چون این‌ها دقیقاً
       همان اطلاعاتی هستند که شما با چشم از روی شکل چارت و سرعت لحظه‌ای
       تشخیص می‌دهید؛ حذفشان یعنی مهم‌ترین ورودی‌های تصمیم را از مدل گرفته‌اید.

    ۲. ستون direction حذف نمی‌شود بلکه به یک ویژگی عددی (direction_call)
       تبدیل می‌شود. اگر direction اصلاً به مدل داده نشود، مدل هنگام معاملهٔ
       زنده هیچ راهی برای تشخیص «این‌جا باید CALL بزنم یا PUT» نخواهد داشت —
       فقط می‌تواند بگوید «این وضعیت به‌طور کلی خوب است یا نه»، بدون این‌که
       بداند کدام جهت را باید انتخاب کند.

    ۳. ستون price_change_pct قطعاً حذف می‌شود. این ستون از روی قیمت خروج
       محاسبه شده که فقط ۳ ثانیه *بعد* از ورود به معامله معلوم می‌شود — یعنی
       در لحظهٔ واقعی تصمیم‌گیری (قبل از کلیک BUY/SELL) اصلاً وجود ندارد. اگر
       این ستون در دادهٔ آموزشی بماند، دقتی که مدل روی دادهٔ تست نشان می‌دهد
       گمراه‌کننده است (Data Leakage) — مدل چیزی را «پیش‌بینی» می‌کند که در
       واقعیت هرگز در لحظهٔ تصمیم در دسترسش نیست.

نتیجه در قالب دو فایل ذخیره می‌شود:
    - data/models/pocket_option_xgb_model.json  (خودِ مدل)
    - data/models/feature_names.json            (لیست دقیق و ترتیب ستون‌ها)
هر دو فایل توسط src/live_predictor.py برای معاملهٔ زنده لازم‌اند.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import config
from src.ml_features import flatten_snapshot_for_model


def main() -> None:
    print("Loading data from database...")
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    print(f"Total trades loaded: {len(df)}")

    df = df.dropna(subset=["result"])
    y_all = df["result"].astype(int)

    records = df.drop(columns=["result"]).to_dict(orient="records")
    flattened_rows = [flatten_snapshot_for_model(row) for row in records]

    flat_df = pd.DataFrame(flattened_rows).fillna(0.0)
    flat_df["result"] = y_all.values

    X = flat_df.drop(columns=["result"])
    y = flat_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("==========================================")
    print(f"📊 Model Real Win-Rate (Accuracy on Test Data): {accuracy * 100:.2f}%")
    print("==========================================")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_JSON_PATH))
    with open(config.MODEL_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

    print(f"\nModel saved to {config.MODEL_JSON_PATH}")
    print(f"Feature list saved to {config.MODEL_FEATURES_PATH}")


if __name__ == "__main__":
    main()
