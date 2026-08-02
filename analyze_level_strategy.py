"""
تحلیل زنجیره‌های برد/باخت پشت‌سرهم استراتژی سطوح (level_strategy)
===================================================================

هدف: پیدا کردن تفاوت بین موقعیت‌هایی که استراتژی سطوح (collect_data.py) پشت‌سرهم
برنده بوده و موقعیت‌هایی که پشت‌سرهم بازنده بوده - مخصوصاً در شرایطی که قیمت با
نوسان بین کندل‌ها ولی در نهایت در یک مسیر اصلی (روند) حرکت می‌کند.

فرضیهٔ اصلی که این اسکریپت آزمایش می‌کند:
    level_strategy.py فقط به اکسترمم‌های محلی *داخل همان کندل جاری* نسبت به
    open آن کندل نگاه می‌کند (تابع check_signal در src/level_strategy.py) و
    هیچ آگاهی‌ای از روند/استریک چندکندلی بزرگ‌تر ندارد. وقتی نوسان داخل کندل
    هر دو طرف open را لمس کند («spans_open»)، استراتژی همیشه یک سیگنال
    بازگشتی (Mean-Reversion) می‌زند: نزدیک سقف -> PUT، نزدیک کف -> CALL - حتی
    اگر روند بزرگ‌تر (چند کندل قبل) قویاً در یک جهت باشد و همان لمس سقف/کف در
    واقع فقط یک مکث کوچک قبل از ادامهٔ روند بوده باشد، نه بازگشت واقعی.

    اگر این فرضیه درست باشد، باید ببینیم: معاملاتی که جهتشان با روند بزرگ‌تر
    هم‌جهت بوده (trend_aligned_with_direction > 0) به‌طور محسوسی وین‌ریت
    بالاتری از معاملاتی دارند که خلاف روند بزرگ‌تر بوده‌اند
    (trend_aligned_with_direction < 0) - همین سه فیچر که از قبل در پروژه
    محاسبه و ذخیره می‌شوند (src/ml_features.py:_add_direction_interaction_features):

        trend_aligned_with_direction        sign(direction) * trend_regime
        streak_aligned_with_direction       sign(direction) * candle_color_streak
        candle_color_aligned_with_direction sign(direction) * (کندل جاری صعودی؟)

    sign = +1 برای CALL و -1 برای PUT. یعنی مثبت = هم‌جهت با روند/استریک/رنگ
    کندل جاری، منفی = خلاف آن‌ها.

این اسکریپت مستقیماً از همان دیتابیس SQLite که main.py/collect_data.py
می‌سازند (config.SQLITE_DB_PATH) می‌خواند - نیازی به هیچ دادهٔ اضافی نیست و
هیچ تغییری هم در دیتابیس نمی‌دهد (فقط خواندنی).

اجرا:
    python analyze_level_strategy.py
"""

from __future__ import annotations

import sqlite3

import pandas as pd

import config
from src.ml_features import flatten_snapshot_for_model

MIN_STREAK_LEN = 2  # حداقل طول زنجیره برای این‌که «پشت‌سرهم» حساب شود


def _load_level_strategy_trades() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM trades", conn)
    finally:
        conn.close()

    df = df[df["meta_trade_source"] == "level_strategy"].copy()
    df = df.sort_values("meta_entry_timestamp").reset_index(drop=True)
    return df


def _add_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """برای هر ردیف، همان فیچرهای هم‌جهت‌بودن را با flatten_snapshot_for_model
    (دقیقاً همان مسیری که train_model.py استفاده می‌کند) محاسبه می‌کند."""
    cols = ["trend_aligned_with_direction", "streak_aligned_with_direction",
            "candle_color_aligned_with_direction"]
    extracted = {c: [] for c in cols}
    for row in df.to_dict("records"):
        flat = flatten_snapshot_for_model(row)
        for c in cols:
            extracted[c].append(flat.get(c))
    for c in cols:
        df[c] = extracted[c]
    return df


def _mark_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """به هر ردیف streak_id (شمارهٔ زنجیرهٔ نتایج یکسان متوالی) و
    streak_len (طول کل همان زنجیره) اضافه می‌کند."""
    streak_id = (df["result"] != df["result"].shift()).cumsum()
    df["streak_id"] = streak_id
    df["streak_len"] = df.groupby("streak_id")["streak_id"].transform("size")
    return df


def _report_split(df: pd.DataFrame, column: str, label: str) -> None:
    print(f"\n--- وین‌ریت بر اساس {label} ({column}) ---")
    aligned = df[df[column] > 0]
    opposed = df[df[column] < 0]
    neutral = df[df[column] == 0]
    for name, subset in (("هم‌جهت (>0)", aligned), ("خلاف‌جهت (<0)", opposed), ("خنثی (=0)", neutral)):
        if len(subset) == 0:
            print(f"  {name}: هیچ معامله‌ای نبود")
            continue
        winrate = subset["result"].mean()
        print(f"  {name}: {len(subset)} معامله، وین‌ریت {winrate:.1%}")


def main() -> None:
    df = _load_level_strategy_trades()
    if df.empty:
        print("هیچ معاملهٔ level_strategy در دیتابیس پیدا نشد.")
        return

    print(f"تعداد کل معاملات level_strategy: {len(df)}")
    print(f"وین‌ریت کلی: {df['result'].mean():.1%}")

    df = _add_interaction_columns(df)
    df = _mark_streaks(df)

    ready_mask = df["trend_regime_ready"] == 1 if "trend_regime_ready" in df.columns else pd.Series(True, index=df.index)
    print(f"\nتعداد معاملاتی که هنگام ورود، روند (trend_regime) قابل‌تشخیص بوده: "
          f"{int(ready_mask.sum())} از {len(df)} ({ready_mask.mean():.1%})")

    # --- آزمایش اصلی فرضیه: وین‌ریت هم‌جهت با روند در برابر خلاف روند ---
    _report_split(df[ready_mask], "trend_aligned_with_direction", "هم‌جهتی با روند بزرگ‌تر (trend_regime)")
    _report_split(df, "streak_aligned_with_direction", "هم‌جهتی با استریک رنگ کندل‌ها")
    _report_split(df, "candle_color_aligned_with_direction", "هم‌جهتی با رنگ کندل جاری")

    # --- مقایسهٔ زنجیره‌های برد پشت‌سرهم در برابر باخت پشت‌سرهم ---
    streaks = df[df["streak_len"] >= MIN_STREAK_LEN]
    win_streak_trades = streaks[streaks["result"] == 1]
    loss_streak_trades = streaks[streaks["result"] == 0]
    print(f"\n--- مقایسهٔ معاملات داخل زنجیره‌های برد/باخت پشت‌سرهم (طول >= {MIN_STREAK_LEN}) ---")
    print(f"تعداد معامله داخل زنجیرهٔ برد: {len(win_streak_trades)}")
    print(f"تعداد معامله داخل زنجیرهٔ باخت: {len(loss_streak_trades)}")
    for col in ["trend_aligned_with_direction", "streak_aligned_with_direction", "candle_color_aligned_with_direction"]:
        w_mean = win_streak_trades[col].mean()
        l_mean = loss_streak_trades[col].mean()
        print(f"  {col}: میانگین در زنجیرهٔ برد = {w_mean:+.3f}  |  میانگین در زنجیرهٔ باخت = {l_mean:+.3f}")

    print("\nراهنمای خواندن نتیجه: اگر وین‌ریت «هم‌جهت با روند» به‌وضوح بالاتر از "
          "«خلاف‌جهت» باشد (مثلاً بیش از ۱۰-۱۵ واحد درصد فاصله)، یعنی سیگنال "
          "بازگشتیِ level_strategy.py (نزدیک سقف->PUT / نزدیک کف->CALL) وقتی با "
          "روند بزرگ‌تر می‌جنگد عملکرد بدی دارد - همان الگویی که توصیف کردید.")


if __name__ == "__main__":
    main()
