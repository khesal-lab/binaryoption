"""
شبیه‌سازی حجم/وین‌ریت معاملهٔ خودکار برای ترکیب‌های مختلف آستانهٔ اطمینان و
حداقل مارجین بین دو جهت
========================================================================

پاسخ به سؤال «اگر AUTO_TRADE_CONFIDENCE_THRESHOLD و AUTO_TRADE_MIN_DIRECTION_MARGIN
را روی مقادیر دیگری بگذارم، چند معامله و با چه وین‌ریتی نتیجه می‌گیرم؟» -
بدون این‌که واقعاً config.py را تغییر بدهید و منتظر دادهٔ جدید بمانید.

برخلاف analyze_confidence_calibration.py (که فقط اطمینانِ جهتِ واقعاً
معامله‌شده را بازسازی می‌کند)، این اسکریپت هر دو فرضیه (CALL و PUT) را
مستقل با مدلِ فعلاً ذخیره‌شده ارزیابی می‌کند - دقیقاً مثل
src/live_predictor.predict_direction - و جهتِ «شبیه‌سازی‌شده» را خودش تعیین
می‌کند؛ ممکن است با جهتِ واقعاً معامله‌شدهٔ همان ردیف فرق داشته باشد (مثلاً
برای معاملات level_strategy که اصلاً از مدل پرسیده نشده بودند). چون نتیجهٔ
یک معاملهٔ باینری قطعی است (اگر یک جهت می‌برد، جهت مخالف می‌بازد)، نتیجهٔ
شبیه‌سازی‌شده مستقیم از روی همان ستون result واقعی استنتاج می‌شود - بدون
نیاز به هیچ دادهٔ جدید یا معاملهٔ واقعی تازه.

⚠️ همان محدودیتِ Overfitting در analyze_confidence_calibration.py این‌جا هم
صادق است: مدل احتمالاً بخشی از دادهٔ ترینِ خودش را حفظ کرده، پس شبیه‌سازی
روی آن دادهٔ درون‌نمونه خوش‌بینانه است. به همین دلیل معاملات بر اساس زمان
ذخیره‌شدن مدل (mtime فایل مدل) به درون‌نمونه/برون‌نمونه تقسیم می‌شوند و
گزارش برون‌نمونه (تست واقعی) جدا و اول نشان داده می‌شود.

⚠️ نکتهٔ دیگر: این شبیه‌سازی روی مدلِ فعلاً ذخیره‌شده کار می‌کند، نه لزوماً
مدلی که با دادهٔ کاملاً تمیز (بعد از فیکس closeTimestamp) دوباره ترین
شده باشد.

این اسکریپت فقط خواندنی است؛ هیچ تغییری در دیتابیس، مدل، یا config.py نمی‌دهد.

اجرا:
    python simulate_auto_trade_thresholds.py
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import xgboost as xgb

import config
from src.ml_features import flatten_snapshot_for_model

CONFIDENCE_THRESHOLDS_TO_SCAN = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
MARGIN_THRESHOLDS_TO_SCAN = [0.00, 0.05, 0.10, 0.15, 0.20]
MIN_N_TO_REPORT = 20


def _load_model() -> tuple[xgb.XGBClassifier, list[str]]:
    model = xgb.XGBClassifier()
    model.load_model(str(config.LIVE_MODEL_JSON_PATH))
    with open(config.LIVE_MODEL_FEATURES_PATH, "r", encoding="utf-8") as f:
        feature_names: list[str] = json.load(f)
    return model, feature_names


def _load_all_trades() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    df = df.dropna(subset=["result", "direction"])
    if "meta_is_refund" in df.columns:
        df = df[~df["meta_is_refund"].fillna(False).astype(bool)]
    df["result"] = df["result"].astype(int)
    return df.reset_index(drop=True)


def _split_in_out_of_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    تفکیک معاملات به «درون‌نمونه» (قبل/هم‌زمان با آخرین ذخیره‌شدن مدل - مدل
    ممکن است این‌ها را در ترین حفظ کرده باشد) و «برون‌نمونه» (بعد از آن -
    مدل هرگز ندیده). شبیه‌سازی روی دادهٔ درون‌نمونه می‌تواند به‌خاطر Overfitting
    خوش‌بینانه باشد؛ فقط عدد برون‌نمونه واقعاً نشان می‌دهد این ترکیب آستانه‌ها
    در عمل (روی موقعیت‌های ندیده) چه نتیجه‌ای می‌دهد.
    """
    if "meta_entry_timestamp" not in df.columns:
        return df, df.iloc[0:0]
    model_mtime = config.LIVE_MODEL_JSON_PATH.stat().st_mtime
    in_sample = df[df["meta_entry_timestamp"] <= model_mtime]
    out_of_sample = df[df["meta_entry_timestamp"] > model_mtime]
    return in_sample, out_of_sample


def _predict_both_directions(
    df: pd.DataFrame, model: xgb.XGBClassifier, feature_names: list[str]
) -> tuple[pd.Series, pd.Series]:
    call_probs: list[float] = []
    put_probs: list[float] = []
    for row in df.to_dict(orient="records"):
        for direction, out in (("CALL", call_probs), ("PUT", put_probs)):
            base = dict(row)
            base["direction"] = direction
            flat = flatten_snapshot_for_model(base)
            aligned = {name: flat.get(name, 0.0) for name in feature_names}
            X = pd.DataFrame([aligned], columns=feature_names)
            out.append(float(model.predict_proba(X)[0][1]))
    return pd.Series(call_probs, index=df.index), pd.Series(put_probs, index=df.index)


def _simulate(df: pd.DataFrame) -> pd.DataFrame:
    """
    جهت، اطمینان و مارجینِ شبیه‌سازی‌شده را می‌سازد - دقیقاً همان منطق
    src/live_predictor.py:predict_direction. نتیجهٔ شبیه‌سازی‌شده هم از روی
    نتیجهٔ واقعیِ همان ردیف استنتاج می‌شود: اگر جهتِ شبیه‌سازی‌شده با جهتِ
    واقعاً معامله‌شده یکی بود همان result، وگرنه مکمل آن (۱-result) - چون
    یک معاملهٔ باینری همیشه دقیقاً یکی از دو جهت را می‌برد.
    """
    df = df.copy()
    df["sim_direction"] = df.apply(lambda r: "CALL" if r["call_prob"] >= r["put_prob"] else "PUT", axis=1)
    df["sim_confidence"] = df[["call_prob", "put_prob"]].max(axis=1)
    df["sim_margin"] = (df["call_prob"] - df["put_prob"]).abs()
    same_direction = df["sim_direction"] == df["direction"]
    df["sim_result"] = df["result"].where(same_direction, 1 - df["result"])
    return df


def _report_grid(df: pd.DataFrame) -> None:
    print(f"\n{'آستانهٔ اطمینان':>15} {'حداقل مارجین':>15} {'تعداد':>8} {'وین‌ریت':>10}")
    for conf_t in CONFIDENCE_THRESHOLDS_TO_SCAN:
        for margin_t in MARGIN_THRESHOLDS_TO_SCAN:
            subset = df[(df["sim_confidence"] >= conf_t) & (df["sim_margin"] >= margin_t)]
            if len(subset) < MIN_N_TO_REPORT:
                continue
            winrate = subset["sim_result"].mean()
            print(f"{conf_t:>14.0%} {margin_t:>14.0%} {len(subset):>8} {winrate:>9.1%}")


def _report_specific(df: pd.DataFrame, confidence: float, margins: list[float]) -> None:
    print(f"\n--- نتیجهٔ دقیق برای اطمینان >= {confidence:.0%} ---")
    for margin_t in margins:
        subset = df[(df["sim_confidence"] >= confidence) & (df["sim_margin"] >= margin_t)]
        if len(subset) == 0:
            print(f"  مارجین >= {margin_t:.0%}: هیچ معامله‌ای با این ترکیب باقی نمی‌ماند.")
            continue
        winrate = subset["sim_result"].mean()
        note = "" if len(subset) >= MIN_N_TO_REPORT else "  ⚠️ نمونه خیلی کم است، به این عدد چندان اعتماد نکنید"
        print(f"  مارجین >= {margin_t:.0%}: {len(subset)} معامله، وین‌ریت {winrate:.1%}{note}")


def main() -> None:
    if not config.LIVE_MODEL_JSON_PATH.exists():
        print(f"فایل مدل پیدا نشد: {config.LIVE_MODEL_JSON_PATH} - اول train/train_model.py را اجرا کنید.")
        return

    df = _load_all_trades()
    if df.empty:
        print("هیچ معامله‌ای در دیتابیس نیست.")
        return

    print(f"مدل درحال استفاده: {config.LIVE_MODEL_JSON_PATH.name} (نوع: {config.LIVE_MODEL_VARIANT})")
    print(f"تعداد کل معاملات قابل‌بررسی (بعد از حذف ریفاند): {len(df)}")

    model, feature_names = _load_model()
    call_probs, put_probs = _predict_both_directions(df, model, feature_names)
    df["call_prob"] = call_probs
    df["put_prob"] = put_probs
    df = _simulate(df)

    changed = int((df["sim_direction"] != df["direction"]).sum())
    print(f"در {changed} ردیف از {len(df)} ({changed / len(df):.1%})، جهتِ شبیه‌سازی‌شده با جهتِ واقعاً "
          f"معامله‌شده فرق داشت (طبیعی برای معاملات level_strategy که اصلاً از مدل پرسیده نشده بودند).")

    in_sample, out_of_sample = _split_in_out_of_sample(df)
    model_saved_at = pd.Timestamp(config.LIVE_MODEL_JSON_PATH.stat().st_mtime, unit="s")
    print(f"\nمدل در {model_saved_at} (به‌وقت سیستم) ذخیره شده.")
    print(f"  درون‌نمونه (قبل/هم‌زمان با ترین - ممکن است مدل حفظشان کرده باشد): {len(in_sample)}")
    print(f"  برون‌نمونه (بعد از ترین - مدل هرگز ندیده، تست واقعی): {len(out_of_sample)}")

    print("\n" + "=" * 70)
    if len(out_of_sample) >= MIN_N_TO_REPORT:
        print("شبیه‌سازی روی معاملات برون‌نمونه (تست واقعی - این بخش قابل‌اعتماد است)")
        print("=" * 70)
        _report_grid(out_of_sample)
        _report_specific(out_of_sample, confidence=0.75, margins=[0.15, 0.20])
    else:
        print("شبیه‌سازی روی معاملات برون‌نمونه")
        print("=" * 70)
        print(f"⚠️ فقط {len(out_of_sample)} معاملهٔ برون‌نمونه موجود است - نتیجه هنوز قابل‌اعتماد نیست.")

    print("\n" + "=" * 70)
    print("شبیه‌سازی روی معاملات درون‌نمونه (⚠️ ممکن است به‌خاطر Overfitting خوش‌بینانه باشد - فقط برای مقایسه)")
    print("=" * 70)
    _report_grid(in_sample)
    _report_specific(in_sample, confidence=0.75, margins=[0.15, 0.20])


if __name__ == "__main__":
    main()
