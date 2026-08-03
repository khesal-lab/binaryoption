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
from typing import Optional

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

# فقط این منبع(ها) برای آموزش استفاده می‌شوند. هدف فعلی این است که مدل مستقیماً
# با دادهٔ collect_data.py (استراتژی سطوح level_strategy، نمونه‌ای نسبتاً بی‌طرف
# از رفتار بازار) ترین شود - نه با معاملات bot (که خودِ یک مدل قبلی با آستانهٔ
# اطمینان انتخابشان کرده و ریسک حلقهٔ بازخورد دارند) یا manual (نمونهٔ خیلی کم
# و بدون الگوی معاملاتی ثابت). برای برگشت به استفاده از همهٔ منابع، None کنید.
TRAIN_INCLUDE_SOURCES: Optional[tuple[str, ...]] = ("level_strategy",)


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


# ترتیب مهم است: پیشوندهای خاص‌تر (مثل chart_shape_long_) باید قبل از پیشوند
# عمومی‌ترشان (chart_shape_) بررسی شوند وگرنه هیچ‌وقت به آن‌ها نمی‌رسد.
_FEATURE_FAMILY_PREFIXES = [
    ("chart_shape_long_", "chart_shape_long (شکل چارت - پنجرهٔ بلند ۲۵ کندلی)"),
    ("chart_shape_", "chart_shape (شکل چارت - پنجرهٔ کوتاه ۱۰ کندلی)"),
    ("micro_chart_shape_", "micro_chart_shape (شکل چارت ریز - ۵ثانیه‌ای)"),
    ("micro_swing_", "micro_swing (نوسان ریز تیک‌ها)"),
    ("micro_market_", "micro_market_regime (رژیم بازار ۵ثانیه‌ای)"),
    ("level_strategy_", "level_strategy (سیگنال استراتژی سطوح)"),
    ("candle_", "candle_structure (ساختار کندل‌های اخیر)"),
    ("tick_velocity_", "tick_velocity (توالی سرعت تیک)"),
    ("current_symbol_payout", "payout (پی‌آوت لحظه‌ای/رتبه در لیست)"),
    ("hour_of_day", "time_of_day (ساعت/روز هفته)"),
    ("day_of_week", "time_of_day (ساعت/روز هفته)"),
]


def _feature_family(name: str) -> str:
    """فیچرهای جهت‌دار (DIRECTION_INTERACTION_COLUMNS) خانوادهٔ خودشان را دارند - چون print_direction_signal همین‌ها را جدا هم گزارش می‌کند."""
    if name in DIRECTION_INTERACTION_COLUMNS:
        return "direction_interaction (تعامل با جهت معامله)"
    for prefix, family_label in _FEATURE_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return family_label
    return "other (روند/سوئینگ/حمایت-مقاومت/سایر)"


def print_feature_family_importance(importances: dict[str, float], label: str) -> None:
    """
    مجموع اهمیت فیچرها به تفکیک «خانواده» (مثلاً همهٔ ۴۰+ ستون chart_shape_long_*
    با هم). یک لیست تخت از ۲۵ نام فیچر تک‌تک (اکثراً پیکسل‌های شکل چارت که به‌تنهایی
    قابل‌تفسیر نیستند) نمی‌گوید کدام دسته از اطلاعات واقعاً برای مدل مهم است؛ این
    خلاصه دقیقاً همان را نشان می‌دهد - برای تصمیم‌گیری دربارهٔ این‌که کدام خانواده
    را باید بیشتر توسعه داد یا کدام را می‌شود حذف کرد.
    """
    family_totals: dict[str, float] = {}
    for feature_name, importance in importances.items():
        family_totals[_feature_family(feature_name)] = family_totals.get(_feature_family(feature_name), 0.0) + importance
    print(f"\n--- اهمیت به تفکیک خانوادهٔ فیچر {label} ---")
    for family_label, total in sorted(family_totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {family_label:<55} {total:.4f}")


def print_result_source_sanity_check(df: pd.DataFrame) -> None:
    """
    بررسی سلامت لیبل‌گذاری: نتیجهٔ گزارش‌شدهٔ خودِ پلتفرم (ws_deal) را با محاسبهٔ
    داخلیِ مستقل از تیک‌های خودمان (tick_fallback_result) مقایسه می‌کند. این دو
    منبع معمولاً باید تقریباً همیشه هم‌رأی باشند؛ اختلاف زیاد (به‌خصوص وقتی
    ws_deal=WIN ولی tick_fallback=LOSS) نشانهٔ احتمالی یک باگ در تفسیر پیام‌های
    WebSocket است - دقیقاً شبیه باگ successopenOrder/successcloseOrder که قبلاً
    پیدا و رفع شد (پیام «باز شدن معامله» با سود پیش‌بینی‌شدهٔ همیشه-مثبت، به‌جای
    نتیجهٔ واقعیِ بسته‌شدن معامله اشتباه گرفته می‌شد). این بررسی برای این است که
    اگر مشکل مشابهی دوباره پیش بیاید، همین‌جا زود دیده شود - نه بعد از هفته‌ها
    ترین روی دادهٔ آلوده.
    """
    if "result_source" not in df.columns or "tick_fallback_result" not in df.columns:
        return
    ws_deal_rows = df[df["result_source"] == "ws_deal"].dropna(subset=["tick_fallback_result"])
    if len(ws_deal_rows) == 0:
        return

    result_int = ws_deal_rows["result"].astype(int)
    tick_int = ws_deal_rows["tick_fallback_result"].astype(int)
    disagreement = result_int != tick_int
    disagreement_rate = float(disagreement.mean())

    print("\n--- بررسی سلامت لیبل (ws_deal در برابر tick_fallback) ---")
    print(f"  {len(ws_deal_rows)} معامله نتیجه‌شان از خودِ پلتفرم (ws_deal) گرفته شده.")
    print(f"  اختلاف با محاسبهٔ داخلی (tick_fallback): {int(disagreement.sum())} مورد ({disagreement_rate:.1%})")
    if disagreement.sum() > 0:
        ws_win_tick_loss = int(((result_int == 1) & (tick_int == 0)).sum())
        ws_loss_tick_win = int(((result_int == 0) & (tick_int == 1)).sum())
        print(f"    - ws_deal=WIN ولی tick_fallback=LOSS: {ws_win_tick_loss} مورد")
        print(f"    - ws_deal=LOSS ولی tick_fallback=WIN: {ws_loss_tick_win} مورد")
        if disagreement_rate > 0.05:
            print("  ⚠️ نرخ اختلاف بالاست - احتمال یک باگ در تفسیر پیام نتیجهٔ معامله وجود دارد؛ "
                  "بررسی کنید (شبیه باگ successopenOrder/successcloseOrder که قبلاً پیدا شد).")


def main() -> None:
    print("Loading data from database...")
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    print(f"Total trades loaded: {len(df)}")

    # معاملات ریفاندی نه برد واقعی‌اند نه باخت واقعی (اصل مبلغ برگشته) - وارد
    # کردنشان با یک لیبل دوتایی اجباری فقط نویز اضافه می‌کند. قبل از هر چیز
    # دیگری از دادهٔ آموزشی کنار گذاشته می‌شوند.
    if "meta_is_refund" in df.columns:
        refund_mask = df["meta_is_refund"].fillna(False).astype(bool)
        if refund_mask.any():
            print(f"{int(refund_mask.sum())} معاملهٔ ریفاندی (نه برد نه باخت واقعی) از دادهٔ آموزشی حذف شد.")
            df = df[~refund_mask]

    df = df.dropna(subset=["result"])

    # بررسی سلامت لیبل - قبل از هر فیلتر دیگری، روی کل دادهٔ باقی‌مانده.
    print_result_source_sanity_check(df)

    # --- توزیع منبع معاملات (دستی / level_strategy / bot) ----------------------
    # معاملات bot را خودِ یک مدل قبلی (با آستانهٔ اطمینان) انتخاب کرده، نه یک
    # نمونهٔ بی‌طرف از بازار - پس می‌توانند باعث شوند مدل جدید صرفاً باور مدل
    # قبلی را در خودش تقویت کند (Selection Bias / حلقهٔ بازخورد)، نه یک الگوی
    # مستقل و جدید یاد بگیرد. این‌جا سهم هر منبع همیشه قابل‌مشاهده می‌ماند،
    # حتی وقتی TRAIN_INCLUDE_SOURCES فقط بخشی از آن‌ها را برای آموزش انتخاب می‌کند.
    source_series_all = (
        df["meta_trade_source"].fillna("manual") if "meta_trade_source" in df.columns
        else pd.Series("manual", index=df.index)
    )
    print("\n--- توزیع منبع معاملات (دستی/level_strategy/bot) ---")
    for source, count in source_series_all.value_counts().items():
        source_winrate = df.loc[source_series_all == source, "result"].astype(int).mean()
        print(f"  {source}: {count} معامله، وین‌ریت {source_winrate:.1%}")

    if TRAIN_INCLUDE_SOURCES is not None:
        include_mask = source_series_all.isin(TRAIN_INCLUDE_SOURCES)
        excluded_count = len(df) - int(include_mask.sum())
        if excluded_count > 0:
            print(f"فقط منبع(های) {TRAIN_INCLUDE_SOURCES} برای آموزش استفاده می‌شود "
                  f"({excluded_count} معامله از منابع دیگر کنار گذاشته شد - train/train_model.py:TRAIN_INCLUDE_SOURCES).")
        df = df[include_mask]

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

    # =========================================================================
    # مدل اصلی: با همهٔ فیچرها
    # =========================================================================
    full_kfold_mean, full_kfold_std = run_kfold_cv(X, y, label="(کامل - همهٔ فیچرها)")
    full_model, full_test_accuracy = train_final_model(X, y, label="(کامل - همهٔ فیچرها)")
    full_importances = print_direction_signal(X, full_model, label="(کامل)")
    top_feature_names = print_top_features(full_importances, TOP_N_FEATURES, label="(کامل)")
    print_feature_family_importance(full_importances, label="(کامل)")

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

    baseline_winrate = float(y.mean())
    print(f"\n\nوین‌ریت خام دادهٔ آموزشی (baseline - فرض «همیشه برنده است»): {baseline_winrate * 100:.2f}%")

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
    top_importances = print_direction_signal(X_top, top_model, label=top_label)
    print_feature_family_importance(top_importances, label=top_label)

    top_model.save_model(str(config.MODEL_TOP_FEATURES_JSON_PATH))
    with open(config.MODEL_TOP_FEATURES_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(top_feature_names, f, ensure_ascii=False, indent=2)
    print(f"\nModel saved to {config.MODEL_TOP_FEATURES_JSON_PATH}")
    print(f"Feature list saved to {config.MODEL_TOP_FEATURES_LIST_PATH}")

    # =========================================================================
    # مقایسهٔ نهایی: baseline خام (وین‌ریت واقعی level_strategy روی همین داده)
    # در برابر مدل کامل در برابر مدل فقط-فیچر-برتر. این دقیقاً پاسخ به سؤال
    # «آیا مدل یادگیری‌شده واقعاً از قانون سادهٔ level_strategy بهتر است؟» است -
    # نه فقط دقت خام مدل، بلکه مقایسه‌اش با همان چیزی که collect_data.py
    # بدون هیچ مدلی به‌دست می‌آورد.
    # =========================================================================
    print("\n\n--- مقایسهٔ نهایی: baseline خام (level_strategy) / مدل کامل / مدل فقط-فیچر-برتر ---")
    print(f"وین‌ریت خام level_strategy روی همین داده (baseline، بدون هیچ مدلی): {baseline_winrate * 100:.2f}%")
    print(f"مدل کامل ({len(X.columns)} فیچر):")
    print(f"  K-Fold: {full_kfold_mean*100:.2f}% ± {full_kfold_std*100:.2f}%   |   "
          f"دقت روی تست: {full_test_accuracy*100:.2f}%")
    print(f"مدل مقایسه‌ای {top_label}:")
    print(f"  K-Fold: {top_kfold_mean*100:.2f}% ± {top_kfold_std*100:.2f}%   |   "
          f"دقت روی تست: {top_test_accuracy*100:.2f}%")

    def _compare_to_baseline(model_name: str, kfold_mean: float) -> None:
        gap = (kfold_mean - baseline_winrate) * 100
        if gap > 1.0:
            print(f"  ✅ {model_name} حدود {gap:.2f} واحد درصد از baseline خام (level_strategy) بهتر است - "
                  "یعنی مدل واقعاً سیگنالی فراتر از قانون سادهٔ level_strategy یاد گرفته.")
        elif gap < -1.0:
            print(f"  ⚠️ {model_name} حدود {abs(gap):.2f} واحد درصد از baseline خام (level_strategy) "
                  "پایین‌تر است - یعنی فعلاً چیزی بهتر از خودِ قانون ساده یاد نگرفته؛ اعتماد به این "
                  "مدل به‌جای level_strategy در معاملهٔ زنده می‌تواند حتی نتیجه را بدتر کند.")
        else:
            print(f"  ~ {model_name} عملاً تفاوت محسوسی با baseline خام (level_strategy) ندارد.")

    print()
    _compare_to_baseline("مدل کامل", full_kfold_mean)
    _compare_to_baseline("مدل فقط-فیچر-برتر", top_kfold_mean)

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
