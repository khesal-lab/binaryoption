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

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
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

    # تشخیص زودهنگام دو مشکل رایج قبل از آموزش: تعداد نمونهٔ خیلی کم برای یکی
    # از دو جهت (CALL/PUT)، و وین‌ریت خیلی متفاوت بین آن دو - هرکدام باعث
    # می‌شود مدل نتواند جهت را درست یاد بگیرد.
    print("\n--- توزیع جهت معاملات در دادهٔ آموزشی ---")
    direction_counts = df["direction"].value_counts()
    print(direction_counts.to_dict())
    for direction in ("CALL", "PUT"):
        subset = df[df["direction"] == direction]
        if len(subset) > 0:
            winrate = subset["result"].astype(int).mean()
            print(f"  {direction}: {len(subset)} معامله، وین‌ریت {winrate:.1%}")
    min_direction_count = direction_counts.min() if len(direction_counts) == 2 else 0
    if min_direction_count < 0.2 * len(df):
        print(f"⚠️  هشدار: یکی از دو جهت (CALL/PUT) کمتر از ۲۰٪ کل داده را دارد "
              f"({min_direction_count} از {len(df)}). مدل احتمالاً نمی‌تواند برای آن جهت "
              f"سیگنال معتبری یاد بگیرد - سعی کنید معاملات هر دو جهت را متعادل‌تر جمع‌آوری کنید.")

    records = df.drop(columns=["result"]).to_dict(orient="records")
    flattened_rows = [flatten_snapshot_for_model(row) for row in records]

    flat_df = pd.DataFrame(flattened_rows).fillna(0.0)
    flat_df["result"] = y_all.values

    X = flat_df.drop(columns=["result"])
    y = flat_df["result"]

    # --- اعتبارسنجی K-Fold ---------------------------------------------------
    # یک تقسیم تصادفی train/test (پایین‌تر) به اندازهٔ کافی برای دیتاست‌های کوچک
    # (چند صد/چند هزار ردیف) قابل‌اعتماد نیست - با جابه‌جا شدن ۲۰٪ داده که در
    # تست قرار می‌گیرد، عدد دقت می‌تواند به‌راحتی ۴-۵٪ نوسان کند و به اشتباه به‌نظر
    # برسد مدل بهتر/بدتر شده، درحالی‌که فقط نویز آماری همان تقسیم است. در
    # K-Fold، دیتاست به K بخش مساوی تقسیم می‌شود؛ مدل K بار آموزش می‌بیند
    # (هر بار با یک بخش متفاوت به‌عنوان تست و بقیه به‌عنوان آموزش) و در آخر
    # میانگین و انحراف‌معیار دقت روی این K اجرا گزارش می‌شود - تخمینی
    # پایدارتر از یک تقسیم تکی.
    print("\n--- اعتبارسنجی K-Fold (۵ بخش) برای تخمین پایدارتر دقت مدل ---")
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        fold_model = xgb.XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
        )
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fold_model.predict(X.iloc[test_idx])
        fold_acc = accuracy_score(y.iloc[test_idx], fold_pred)
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold_idx}/{k_folds}: دقت = {fold_acc * 100:.2f}%")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"میانگین دقت K-Fold: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    print("(این عدد تخمین قابل‌اعتمادتری از توانایی واقعی مدل است تا عدد "
          "«دقت روی تست» که پایین‌تر چاپ می‌شود و فقط از یک تقسیم تکی می‌آید.)")

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

    # هشدار مهم: اگر مدل اصلاً روی direction_call یا فیچرهای «هم‌جهت با
    # تصمیم» Split نزده باشد، یعنی در معاملهٔ زنده نمی‌تواند بین CALL و PUT
    # فرق بگذارد و همیشه یک جهت را انتخاب می‌کند (دقیقاً همان باگی که قبلاً
    # دیده شد). این‌جا زودتر و واضح هشدار می‌دهیم.
    direction_related_cols = [
        c for c in X.columns
        if c == "direction_call" or c.endswith("_in_direction") or c.startswith("distance_to_")
        or c == "trend_aligned_with_direction" or c == "candle_color_aligned_with_direction"
    ]
    importances = dict(zip(X.columns, model.feature_importances_))
    direction_signal_total = sum(importances.get(c, 0.0) for c in direction_related_cols)
    print(f"\nمجموع اهمیت فیچرهای مرتبط با جهت (direction_call + هم‌جهت‌ها): {direction_signal_total:.4f}")
    if direction_signal_total == 0.0:
        print("⚠️  هشدار جدی: مدل هیچ‌کدام از فیچرهای مرتبط با جهت معامله را استفاده نکرده "
              "— یعنی در معاملهٔ زنده احتمالاً همیشه یک جهت ثابت (مثلاً همیشه CALL) انتخاب "
              "خواهد کرد، نه این‌که واقعاً بین دو جهت تصمیم بگیرد. راه‌حل: دادهٔ بیشتر/متنوع‌تر "
              "جمع‌آوری کنید (به‌خصوص از جهتی که کمتر معامله شده)، یا max_depth را کمی افزایش دهید.")

    # مهم‌ترین فیچرها از دید مدل (بر اساس اهمیت XGBoost - چند بار و چقدر مؤثر
    # در تصمیم‌های درخت‌ها استفاده شده‌اند)، برای این‌که مشخص شود مدل واقعاً
    # روی کدام ویژگی‌ها تکیه می‌کند - نه فقط جمع فیچرهای جهت‌دار.
    top_n = 25
    sorted_importances = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n--- {top_n} فیچر با بیشترین تأثیر ---")
    for rank, (feature_name, importance) in enumerate(sorted_importances[:top_n], start=1):
        print(f"  {rank:>2}. {feature_name:<45} {importance:.4f}")

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.MODEL_JSON_PATH))
    with open(config.MODEL_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)

    print(f"\nModel saved to {config.MODEL_JSON_PATH}")
    print(f"Feature list saved to {config.MODEL_FEATURES_PATH}")


if __name__ == "__main__":
    main()
