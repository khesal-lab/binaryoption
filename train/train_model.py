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

    ۳. ستون‌هایی که فقط بعد از پایان معامله معلوم می‌شوند (مثل
       price_change_pct)، یا فقط بازتاب موقعیت زمانی/تجمعی ردیف در دیتاست‌اند
       (مثل total_trades_so_far، overall_winrate) قطعاً حذف می‌شوند - این‌ها
       یا نشتِ اطلاعات از آینده‌اند، یا در تقسیم تصادفی K-Fold/train_test_split
       می‌توانند «بازهٔ زمانی این ردیف» را به‌جای یک الگوی بازار واقعی به مدل
       نشان دهند (سیگنالی که در معاملهٔ زنده تعمیم پیدا نمی‌کند).

علاوه بر مدل اصلی (با همهٔ فیچرها)، یک مدل دوم هم فقط با N فیچر مهم‌تر (طبق
اهمیت فیچر مدل اول) آموزش داده می‌شود - تا با مقایسهٔ دقت این دو مدل، نقش
واقعی «بقیهٔ فیچرها» (غیر از N تای برتر) در دقت کلی روشن شود. کدام یک از این
دو مدل واقعاً در معاملهٔ زندهٔ main.py استفاده می‌شود را config.LIVE_MODEL_VARIANT
مشخص می‌کند (نه این اسکریپت).

نتیجه در چهار فایل ذخیره می‌شود:
    - data/models/pocket_option_xgb_model.json               (مدل اصلی)
    - data/models/feature_names.json                         (فیچرهای مدل اصلی)
    - data/models/pocket_option_xgb_model_top_features.json  (مدل مقایسه‌ای)
    - data/models/feature_names_top_features.json            (فیچرهای مدل مقایسه‌ای)
دو فایل اول توسط src/live_predictor.py برای معاملهٔ زنده لازم‌اند.
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
from src.ml_features import (
    flatten_snapshot_for_model,
    DIRECTION_INTERACTION_COLUMNS,
    MICRO_SWING_COLUMNS,
    chart_shape_long_current_candle_columns,
)

TOP_N_FEATURES = 25
K_FOLDS = 5


def _new_classifier() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
    )


def run_kfold_cv(X: pd.DataFrame, y: pd.Series, label: str) -> tuple[float, float]:
    """
    یک تقسیم تصادفی train/test به‌تنهایی برای دیتاست‌های چند‌هزارتایی
    قابل‌اعتماد نیست - عدد دقت می‌تواند به‌راحتی چند درصد فقط به‌خاطر نویز
    همان تقسیم نوسان کند. این‌جا دیتاست به K_FOLDS بخش مساوی تقسیم می‌شود؛
    مدل K بار آموزش می‌بیند (هر بار با یک بخش متفاوت به‌عنوان تست) و میانگین/
    انحراف‌معیار دقت روی این K اجرا برگردانده می‌شود - تخمینی پایدارتر از یک
    تقسیم تکی.
    """
    print(f"\n--- اعتبارسنجی K-Fold ({K_FOLDS} بخش) {label} ---")
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        fold_model = _new_classifier()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fold_model.predict(X.iloc[test_idx])
        fold_acc = accuracy_score(y.iloc[test_idx], fold_pred)
        fold_accuracies.append(fold_acc)
        print(f"  Fold {fold_idx}/{K_FOLDS}: دقت = {fold_acc * 100:.2f}%")

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    print(f"میانگین دقت K-Fold {label}: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)")
    return mean_acc, std_acc


def train_final_model(X: pd.DataFrame, y: pd.Series, label: str) -> tuple[xgb.XGBClassifier, float]:
    """آموزش نهایی روی یک تقسیم تکی train/test، برای گزارش تفصیلی و ذخیرهٔ مدل."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining XGBoost model {label}...")
    model = _new_classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("==========================================")
    print(f"📊 دقت روی دادهٔ تست {label}: {accuracy * 100:.2f}%")
    print("==========================================")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))
    return model, accuracy


def print_direction_signal(X: pd.DataFrame, model: xgb.XGBClassifier, label: str) -> dict[str, float]:
    """
    هشدار مهم: اگر مدل اصلاً روی direction_call یا فیچرهای «هم‌جهت با تصمیم»
    Split نزده باشد، یعنی در معاملهٔ زنده نمی‌تواند بین CALL و PUT فرق بگذارد
    و همیشه یک جهت را انتخاب می‌کند. این‌جا زودتر و واضح هشدار می‌دهیم.
    برمی‌گرداند: دیکشنری کامل اهمیت فیچرها (برای استفادهٔ فراخوان).
    """
    direction_related_cols = [c for c in X.columns if c in DIRECTION_INTERACTION_COLUMNS]
    importances = dict(zip(X.columns, model.feature_importances_))
    direction_signal_total = sum(importances.get(c, 0.0) for c in direction_related_cols)
    print(f"\nمجموع اهمیت فیچرهای مرتبط با جهت (direction_call + هم‌جهت‌ها) {label}: "
          f"{direction_signal_total:.4f}")
    if direction_signal_total == 0.0:
        print("⚠️  هشدار جدی: مدل هیچ‌کدام از فیچرهای مرتبط با جهت معامله را استفاده نکرده "
              "— یعنی در معاملهٔ زنده احتمالاً همیشه یک جهت ثابت (مثلاً همیشه CALL) انتخاب "
              "خواهد کرد، نه این‌که واقعاً بین دو جهت تصمیم بگیرد. راه‌حل: دادهٔ بیشتر/متنوع‌تر "
              "جمع‌آوری کنید (به‌خصوص از جهتی که کمتر معامله شده)، یا max_depth را کمی افزایش دهید.")
    return importances


def print_top_features(importances: dict[str, float], top_n: int, label: str) -> list[str]:
    """
    مهم‌ترین فیچرها از دید مدل (بر اساس اهمیت XGBoost)، برای این‌که مشخص شود
    مدل واقعاً روی کدام ویژگی‌ها تکیه می‌کند. لیست نام فیچرها را برمی‌گرداند.
    """
    sorted_importances = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\n--- {top_n} فیچر با بیشترین تأثیر {label} ---")
    for rank, (feature_name, importance) in enumerate(sorted_importances[:top_n], start=1):
        print(f"  {rank:>2}. {feature_name:<45} {importance:.4f}")
    return [name for name, _ in sorted_importances[:top_n]]


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

    # --- توزیع منبع معاملات (دستی / level_strategy / bot) ---------------------
    # معاملات bot را خودِ یک مدل قبلی (با آستانهٔ اطمینان) انتخاب کرده، نه یک
    # نمونهٔ بی‌طرف از بازار - پس می‌توانند باعث شوند مدل جدید صرفاً باور مدل
    # قبلی را در خودش تقویت کند (Selection Bias / حلقهٔ بازخورد)، نه یک الگوی
    # مستقل و جدید یاد بگیرد. این‌جا سهم هر منبع را همیشه قابل‌مشاهده می‌کنیم.
    source_series = (
        df["meta_trade_source"].fillna("manual") if "meta_trade_source" in df.columns
        else pd.Series("manual", index=df.index)
    )
    print("\n--- توزیع منبع معاملات (دستی/level_strategy/bot) ---")
    for source, count in source_series.value_counts().items():
        source_winrate = y_all[source_series == source].mean()
        print(f"  {source}: {count} معامله، وین‌ریت {source_winrate:.1%}")

    records = df.drop(columns=["result"]).to_dict(orient="records")
    flattened_rows = [flatten_snapshot_for_model(row) for row in records]

    flat_df = pd.DataFrame(flattened_rows).fillna(0.0)
    flat_df["result"] = y_all.values

    X = flat_df.drop(columns=["result"])
    y = flat_df["result"]

    # =========================================================================
    # مدل اصلی: با همهٔ فیچرها
    # =========================================================================
    full_kfold_mean, full_kfold_std = run_kfold_cv(X, y, label="(کامل - همهٔ فیچرها)")
    full_model, full_test_accuracy = train_final_model(X, y, label="(کامل - همهٔ فیچرها)")
    full_importances = print_direction_signal(X, full_model, label="(کامل)")
    top_feature_names = print_top_features(full_importances, TOP_N_FEATURES, label="(کامل)")

    # سه دسته فیچر صرف‌نظر از رتبهٔ اهمیتشان به مدل مقایسه‌ای اضافه می‌شوند:
    #   ۱. فیچرهای جهت‌دار (direction_call و تعامل‌هایش) - چون توانایی تفکیک
    #      CALL از PUT یک نیاز عملکردی است، نه یک انتخاب اختیاری بر اساس رتبه.
    #      بدون این‌ها، مدل فقط می‌تواند بگوید «این وضعیت کلی خوب است یا نه»،
    #      نه «کدام جهت را باید انتخاب کنم».
    #   ۲. فیچرهای «نوسان ریز» (Micro-Swing) - یک خانوادهٔ به‌هم‌مرتبط که در
    #      ترین‌های مختلف، اعضای متفاوتی از آن در فیچرهای برتر ظاهر شده‌اند.
    #   ۳. آخرین کندل در پنجرهٔ بلندِ شکل چارت - در آخرین ترین ثابت شد سیگنال
    #      واقعی دارد (رتبهٔ ۲ و ۷ اهمیت).
    # هدف: نوسان رتبهٔ خام اهمیت بین اجراهای مختلف، این فیچرهای اثبات‌شده را
    # اتفاقی از مدل مقایسه‌ای حذف نکند.
    forced_columns = DIRECTION_INTERACTION_COLUMNS | MICRO_SWING_COLUMNS | chart_shape_long_current_candle_columns()
    forced_extra_cols = [c for c in forced_columns if c in X.columns and c not in top_feature_names]
    if forced_extra_cols:
        print(f"\n(فیچرهای زیر صرف‌نظر از رتبهٔ اهمیت - جهت‌دار/نوسان ریز/کندل جاری در پنجرهٔ بلند - "
              f"به مدل مقایسه‌ای اضافه شدند: {', '.join(forced_extra_cols)})")
        top_feature_names = top_feature_names + forced_extra_cols

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    full_model.save_model(str(config.MODEL_JSON_PATH))
    with open(config.MODEL_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)
    print(f"\nModel saved to {config.MODEL_JSON_PATH}")
    print(f"Feature list saved to {config.MODEL_FEATURES_PATH}")

    # =========================================================================
    # آزمایش تشخیصی: مدل کامل بدون معاملات bot
    # هدف: معاملات bot را خودِ یک مدل قبلی (با آستانهٔ اطمینان) انتخاب کرده،
    # نه یک نمونهٔ بی‌طرف از بازار - پس ممکن است دقتِ بالای مدل کامل (نسبت به
    # baseline خام) صرفاً بازتاب/تقویتِ باور همان مدل قبلی باشد، نه یک الگوی
    # مستقل و جدید بازار. این مدل فقط برای مقایسه چاپ می‌شود؛ ذخیره یا در
    # معاملهٔ زنده استفاده نمی‌شود.
    # =========================================================================
    non_bot_mask = source_series.values != "bot"
    baseline_winrate = float(y.mean())
    print(f"\n\nوین‌ریت خام کل داده (baseline - فرض «همیشه برنده است»): {baseline_winrate * 100:.2f}%")

    if 0 < non_bot_mask.sum() < len(y) and non_bot_mask.sum() >= 50:
        print("\n==========================================================")
        print(f"آزمایش تشخیصی: مدل کامل بدون معاملات bot ({int(non_bot_mask.sum())} از {len(y)} ردیف)")
        print("==========================================================")
        X_non_bot = X[non_bot_mask]
        y_non_bot = y[non_bot_mask]
        non_bot_baseline = float(y_non_bot.mean())
        print(f"وین‌ریت خام دادهٔ بدون bot: {non_bot_baseline * 100:.2f}%")

        non_bot_kfold_mean, non_bot_kfold_std = run_kfold_cv(
            X_non_bot, y_non_bot, label="(کامل - بدون bot)"
        )
        gap = (non_bot_kfold_mean - full_kfold_mean) * 100
        print(f"\nمقایسه با مدل کامل (با همهٔ منابع، {full_kfold_mean * 100:.2f}%): {gap:+.2f} واحد درصد")
        if gap < -1.0:
            print("⚠️  دقت مدل بدون معاملات bot به‌طور محسوسی پایین‌تر است - یعنی بخشی از دقت "
                  "دیده‌شده در مدل کامل ممکن است ناشی از حلقهٔ بازخورد با یک مدل قبلی باشد، نه "
                  "یک الگوی مستقل و جدید بازار. برای اطمینان، فعلاً دادهٔ بیشتری با collect_data.py "
                  "(یا معاملهٔ دستی) جمع کنید تا سهم bot در دیتاست کم‌رنگ‌تر شود.")
        else:
            print("تفاوت محسوس منفی دیده نمی‌شود - دقت مدل کامل به دادهٔ bot وابسته به‌نظر نمی‌رسد.")
    else:
        print("\n(داده‌ای برای آزمایش «بدون bot» کافی نیست یا اصلاً معاملهٔ bot ثبت نشده.)")

    # =========================================================================
    # مدل مقایسه‌ای: TOP_N_FEATURES فیچر مهم‌تر + فیچرهای اجباری (جهت‌دار/
    # نوسان ریز/کندل جاری در پنجرهٔ بلند)
    # هدف: با مقایسهٔ دقت این مدل با مدل کامل، نقش واقعی بقیهٔ فیچرها (غیر از
    # این مجموعه) در دقت کلی مشخص شود.
    # =========================================================================
    top_count = len(top_feature_names)
    top_label = f"({top_count} فیچر: {TOP_N_FEATURES} برتر + اجباری)"
    print("\n\n==========================================================")
    print(f"مدل مقایسه‌ای: {top_label}")
    print("==========================================================")
    X_top = X[top_feature_names]

    top_kfold_mean, top_kfold_std = run_kfold_cv(X_top, y, label=top_label)
    top_model, top_test_accuracy = train_final_model(X_top, y, label=top_label)
    print_direction_signal(X_top, top_model, label=top_label)

    top_model.save_model(str(config.MODEL_TOP_FEATURES_JSON_PATH))
    with open(config.MODEL_TOP_FEATURES_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(top_feature_names, f, ensure_ascii=False, indent=2)
    print(f"\nModel saved to {config.MODEL_TOP_FEATURES_JSON_PATH}")
    print(f"Feature list saved to {config.MODEL_TOP_FEATURES_LIST_PATH}")

    # =========================================================================
    # مقایسهٔ نهایی
    # =========================================================================
    print("\n\n--- مقایسهٔ مدل کامل در برابر مدل با فقط فیچرهای برتر ---")
    print(f"مدل کامل ({len(X.columns)} فیچر):")
    print(f"  K-Fold: {full_kfold_mean*100:.2f}% ± {full_kfold_std*100:.2f}%   |   "
          f"دقت روی تست: {full_test_accuracy*100:.2f}%")
    print(f"مدل مقایسه‌ای {top_label}:")
    print(f"  K-Fold: {top_kfold_mean*100:.2f}% ± {top_kfold_std*100:.2f}%   |   "
          f"دقت روی تست: {top_test_accuracy*100:.2f}%")
    diff = (top_kfold_mean - full_kfold_mean) * 100
    if abs(diff) < 1.0:
        print(f"\nتفاوت دقت K-Fold بین دو مدل ناچیز است ({diff:+.2f} واحد درصد) - یعنی بقیهٔ "
              f"{len(X.columns) - top_count} فیچر (غیر از مجموعهٔ بالا) عملاً سیگنال اضافی "
              "معناداری به مدل نمی‌دهند.")
    elif diff > 0:
        print(f"\nمدل مقایسه‌ای حتی کمی بهتر است (+{diff:.2f} واحد درصد) - احتمالاً بقیهٔ فیچرها "
              "بیشتر نویز اضافه می‌کنند تا سیگنال.")
    else:
        print(f"\nمدل کامل {abs(diff):.2f} واحد درصد بهتر است - یعنی بقیهٔ فیچرها هم مقداری سیگنال "
              "واقعی (هرچند کوچک) به مدل اضافه می‌کنند.")


if __name__ == "__main__":
    main()
