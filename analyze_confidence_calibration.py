"""
بررسی رابطهٔ «درصد اطمینان» مدل با نرخ برد واقعی (کالیبراسیون)
==================================================================

هدف: پاسخ به این سؤال - آیا وقتی مدل می‌گوید «۸۰٪ مطمئنم»، واقعاً چیزی
نزدیک ۸۰٪ از این معاملات می‌برند؟ و کدام آستانهٔ اطمینان کمترین خطا /
بهترین وین‌ریت را می‌دهد؟

چون درصد اطمینان به‌صورت دائمی در دیتاست ذخیره نمی‌شود (فقط لحظهٔ معاملهٔ
زنده در ترمینال چاپ می‌شود)، این اسکریپت آن را *بازسازی* می‌کند: برای هر
معاملهٔ ذخیره‌شده (از هر منبعی - level_strategy یا bot)، همان اسنپ‌شات
فیچرها را با همان جهتی که واقعاً معامله شده (ستون direction) دوباره به
مدل *فعلاً ذخیره‌شده* می‌دهد - دقیقاً همان محاسبه‌ای که src/live_predictor.py
در لحظهٔ معاملهٔ زنده انجام می‌دهد (نگاه کنید به predict_direction).

فقط معاملات bot به‌خودی‌خود کافی نیستند، چون AUTO_TRADE از قبل فیلتر شده
(فقط >= AUTO_TRADE_CONFIDENCE_THRESHOLD معامله می‌کند) - یعنی بازهٔ پایین
اطمینان اصلاً در آن‌ها دیده نمی‌شود. بازسازی روی *همهٔ* معاملات، کل طیف
اطمینان را نشان می‌دهد.

⚠️ نکتهٔ مهم دربارهٔ Overfitting: کالیبراسیون را روی *همهٔ* معاملات حساب
کردن گمراه‌کننده است - چون مدل احتمالاً بخشی از دادهٔ ترینِ خودش را حفظ
کرده، وین‌ریتِ «به‌ظاهر عالی» روی آن دادهٔ درون‌نمونه می‌تواند فقط بازتاب
حفظ‌کردن باشد، نه توانایی پیش‌بینی واقعی (دقیقاً همان چیزی که یک‌بار در
عمل دیده شد: کالیبراسیون بک‌تست‌شده عالی به‌نظر می‌رسید، ولی وین‌ریت
معاملهٔ زندهٔ بعدی زیر ۵۰٪ افتاد). به همین دلیل این اسکریپت معاملات را بر
اساس زمان ذخیره‌شدن مدل (mtime فایل مدل) به دو گروه تقسیم می‌کند:
درون‌نمونه (قبل از ترین - قابل‌اعتماد نیست) و برون‌نمونه (بعد از ترین -
مدل هرگز ندیده، تنها معیار واقعی).

⚠️ نکتهٔ دیگر: این اسکریپت با مدلِ *فعلاً ذخیره‌شده* در data/models/ کار
می‌کند (همان فایلی که main.py هم طبق config.LIVE_MODEL_VARIANT استفاده
می‌کند). اگر این مدل با دادهٔ قدیمی‌تر (قبل از رفع باگ زمان دقیق بسته‌شدن
معامله) ترین شده باشد، این تحلیل هم به همان محدودیت آغشته است.

این اسکریپت فقط خواندنی است؛ هیچ تغییری در دیتابیس یا مدل نمی‌دهد.

اجرا:
    python analyze_confidence_calibration.py
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

import config
from src.ml_features import flatten_snapshot_for_model

CONFIDENCE_BUCKET_WIDTH = 0.05
MIN_N_FOR_BUCKET_FLAG = 20
MIN_N_FOR_THRESHOLD = 30


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
    تفکیک معاملات به «درون‌نمونه» (قبل/هم‌زمان با آخرین ذخیره‌شدن مدل - یعنی
    مدل احتمالاً این‌ها را در ترین دیده) و «برون‌نمونه» (بعد از آن - یعنی مدل
    هرگز این‌ها را ندیده، دقیقاً مثل معاملهٔ زندهٔ واقعی بعد از ترین).

    کالیبراسیون روی دادهٔ درون‌نمونه گمراه‌کننده است: با ده‌ها فیچر روی چند
    هزار ردیف، مدل به‌راحتی می‌تواند بخشی از دادهٔ ترین را حفظ کند
    (Overfitting) - پس وین‌ریت بالا در آن گروه لزوماً توانایی پیش‌بینی واقعی
    نیست، فقط یادآوری. عدد برون‌نمونه تنها معیاری است که واقعاً نشان می‌دهد
    مدل روی موقعیت‌های ندیده چطور عمل می‌کند - دقیقاً همان چیزی که K-Fold در
    train/train_model.py هم اندازه می‌گیرد (و همیشه پایین‌تر از عدد ساده‌لوحانهٔ
    «دقت روی کل داده» بوده است).
    """
    if "meta_entry_timestamp" not in df.columns:
        return df, df.iloc[0:0]
    model_mtime = config.LIVE_MODEL_JSON_PATH.stat().st_mtime
    in_sample = df[df["meta_entry_timestamp"] <= model_mtime]
    out_of_sample = df[df["meta_entry_timestamp"] > model_mtime]
    return in_sample, out_of_sample


def _compute_confidence(df: pd.DataFrame, model: xgb.XGBClassifier, feature_names: list[str]) -> pd.Series:
    confidences = []
    for row in df.to_dict(orient="records"):
        flat = flatten_snapshot_for_model(row)
        aligned = {name: flat.get(name, 0.0) for name in feature_names}
        X = pd.DataFrame([aligned], columns=feature_names)
        confidences.append(float(model.predict_proba(X)[0][1]))
    return pd.Series(confidences, index=df.index)


def _report_bucketed_calibration(df: pd.DataFrame) -> None:
    print("\n--- کالیبراسیون: نرخ برد واقعی به تفکیک بازهٔ اطمینان مدل ---")
    bins = np.arange(0, 1.0001, CONFIDENCE_BUCKET_WIDTH)
    bucketed = pd.cut(df["confidence"], bins=bins, include_lowest=True)
    for bucket, group in df.groupby(bucketed, observed=True):
        if len(group) == 0:
            continue
        winrate = group["result"].mean()
        mid = (bucket.left + bucket.right) / 2
        gap = winrate - mid
        flag = ""
        if len(group) >= MIN_N_FOR_BUCKET_FLAG:
            if gap > 0.07:
                flag = "  (کم‌برآورد - واقعیت بهتر از ادعای مدل)"
            elif gap < -0.07:
                flag = "  (بیش‌برآورد - واقعیت بدتر از ادعای مدل)"
        print(f"  {bucket}: n={len(group):>4}, وین‌ریت واقعی={winrate:.1%}{flag}")


def _report_threshold_scan(df: pd.DataFrame) -> None:
    print(f"\n--- وین‌ریت روی معاملاتی که اطمینان >= آستانه بود (حداقل {MIN_N_FOR_THRESHOLD} نمونه) ---")
    thresholds = np.arange(0.50, 0.96, 0.05)
    best = None
    for t in thresholds:
        subset = df[df["confidence"] >= t]
        if len(subset) < MIN_N_FOR_THRESHOLD:
            continue
        winrate = subset["result"].mean()
        print(f"  >= {t:.2f}: n={len(subset):>4}, وین‌ریت={winrate:.1%}")
        if best is None or winrate > best[1]:
            best = (t, winrate, len(subset))
    if best:
        print(f"\nبهترین آستانه (با حداقل {MIN_N_FOR_THRESHOLD} نمونه): "
              f">= {best[0]:.2f} با وین‌ریت {best[1]:.1%} (روی {best[2]} معامله)")
        print("توجه: این «بهترین در این نمونهٔ خاص» است، نه لزوماً بهینهٔ واقعی - "
              "با نمونهٔ کوچک ممکن است فقط شانسی بالا آمده باشد؛ به بخش کالیبراسیون بالا هم نگاه کنید.")
    else:
        print("  داده برای هیچ آستانه‌ای به حداقل نمونه نرسید.")


def _report_brier_score(df: pd.DataFrame) -> None:
    brier = float(((df["confidence"] - df["result"]) ** 2).mean())
    naive_brier = float(((df["result"].mean() - df["result"]) ** 2).mean())
    print(f"\n--- Brier Score (هرچه کمتر بهتر؛ ۰=کالیبراسیون کامل، baseline زیر=حدس ثابتِ میانگین) ---")
    print(f"  مدل: {brier:.4f}")
    print(f"  baseline (فرض ثابت میانگین وین‌ریت کل به‌جای اطمینان لحظه‌ای): {naive_brier:.4f}")
    if brier < naive_brier:
        print("  مدل کالیبرهٔ بهتری از فرض ثابت دارد.")
    else:
        print("  ⚠️ مدل حتی از فرض ثابتِ میانگین وین‌ریت هم کالیبرهٔ بهتری ندارد - عدد اطمینانش عملاً اطلاعات اضافه نمی‌دهد.")


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
    df["confidence"] = _compute_confidence(df, model, feature_names)

    print(f"میانگین اطمینان محاسبه‌شده: {df['confidence'].mean():.1%}   |   "
          f"کمینه: {df['confidence'].min():.1%}   |   بیشینه: {df['confidence'].max():.1%}")

    in_sample, out_of_sample = _split_in_out_of_sample(df)
    model_saved_at = pd.Timestamp(config.LIVE_MODEL_JSON_PATH.stat().st_mtime, unit="s")
    print(f"\nمدل در {model_saved_at} (به‌وقت سیستم) ذخیره شده.")
    print(f"  معاملات درون‌نمونه (قبل/هم‌زمان با ترین - مدل ممکن است حفظشان کرده باشد): {len(in_sample)}")
    print(f"  معاملات برون‌نمونه (بعد از ترین - مدل هرگز ندیده، تست واقعی): {len(out_of_sample)}")

    print("\n" + "=" * 70)
    if len(out_of_sample) >= MIN_N_FOR_THRESHOLD:
        print("گزارش برون‌نمونه (تست واقعی روی داده‌ای که مدل هرگز ندیده - این بخش قابل‌اعتماد است)")
        print("=" * 70)
        _report_bucketed_calibration(out_of_sample)
        _report_threshold_scan(out_of_sample)
        _report_brier_score(out_of_sample)
    else:
        print("گزارش برون‌نمونه")
        print("=" * 70)
        print(f"⚠️ فقط {len(out_of_sample)} معاملهٔ برون‌نمونه موجود است (کمتر از {MIN_N_FOR_THRESHOLD}) - "
              "برای نتیجهٔ قابل‌اعتماد باید صبر کرد تا معاملات بیشتری بعد از این ترین جمع شود. "
              "تا آن زمان، به گزارش درون‌نمونهٔ پایین فقط برای مقایسه نگاه کنید، نه تصمیم‌گیری.")

    print("\n" + "=" * 70)
    print("گزارش درون‌نمونه (⚠️ ممکن است به‌خاطر Overfitting خوش‌بینانه باشد - فقط برای مقایسه با بالا)")
    print("=" * 70)
    _report_bucketed_calibration(in_sample)
    _report_threshold_scan(in_sample)
    _report_brier_score(in_sample)


if __name__ == "__main__":
    main()
