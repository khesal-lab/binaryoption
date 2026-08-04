"""
بررسی نتیجهٔ معاملهٔ خودکار زنده (main.py / AUTO_TRADE با مدل ترین‌شده)
========================================================================

هدف این اسکریپت متفاوت از analyze_level_strategy.py است: آن یکی دیتای
جمع‌آوری‌شده با استراتژی ثابت (collect_data.py) را برای پیدا کردن فیچر مؤثر
تحلیل می‌کند؛ این‌جا فقط معاملاتی که خودِ ربات با مدل ترین‌شده
(source="bot"، یعنی main.py با AUTO_TRADE_ENABLED=True) واقعاً باز کرده
بررسی می‌شود - برای این‌که ببینیم مدل در معاملهٔ زنده واقعاً چطور عمل کرده.

این اسکریپت فقط خواندنی است (هیچ تغییری در دیتابیس نمی‌دهد) و مستقیماً از
همان دیتابیس SQLite که main.py می‌سازد (config.SQLITE_DB_PATH) می‌خواند.

اجرا:
    python analyze_bot_results.py
"""

from __future__ import annotations

import math
import sqlite3

import pandas as pd

import config

MIN_N_FOR_TEST = 30  # حداقل تعداد نمونه برای این‌که آزمایش آماری معنادار باشد
MIN_TRADES_PER_SYMBOL = 15  # حداقل تعداد معامله روی یک نماد تا در گزارش تک‌تک دیده شود


def _load_bot_trades() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    if "meta_trade_source" not in df.columns:
        return df.iloc[0:0]

    df = df[df["meta_trade_source"] == "bot"].copy()
    df = df.dropna(subset=["result"])
    df["result"] = df["result"].astype(int)
    if "meta_entry_timestamp" in df.columns:
        df = df.sort_values("meta_entry_timestamp").reset_index(drop=True)
    return df


def _two_sided_p_value(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _two_proportion_z_test(wins1: float, n1: int, wins2: float, n2: int) -> tuple[float, float]:
    """z-test دو-نسبتی استاندارد (pooled). خروجی: (z, p-value دوطرفه)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = wins1 / n1
    p2 = wins2 / n2
    pooled_p = (wins1 + wins2) / (n1 + n2)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, _two_sided_p_value(z)


def _format_z_test(wins1: float, n1: int, wins2: float, n2: int, min_n: int = MIN_N_FOR_TEST) -> str:
    if n1 < min_n or n2 < min_n:
        return "نمونه برای آزمایش آماری کافی نیست"
    z, pval = _two_proportion_z_test(wins1, n1, wins2, n2)
    flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
    return f"z={z:+.2f}, p={pval:.3f}{flag}"


def _report_refunds(df_with_refunds: pd.DataFrame) -> pd.DataFrame:
    """معاملات ریفاندی را از دادهٔ اصلی جدا می‌کند و تعدادشان را گزارش می‌دهد."""
    if "meta_is_refund" not in df_with_refunds.columns:
        return df_with_refunds
    refund_mask = df_with_refunds["meta_is_refund"].fillna(False).astype(bool)
    n_refund = int(refund_mask.sum())
    if n_refund > 0:
        print(f"{n_refund} معامله ریفاندی بود (نه برد نه باخت واقعی) - از آمار وین‌ریت زیر کنار گذاشته شد.")
    return df_with_refunds[~refund_mask]


def _report_overall(df: pd.DataFrame) -> None:
    n = len(df)
    wins = int(df["result"].sum())
    winrate = df["result"].mean()
    print(f"\n--- خلاصهٔ کلی معاملات بات (source=bot) ---")
    print(f"تعداد کل: {n}   |   برد: {wins}   |   باخت: {n - wins}   |   وین‌ریت: {winrate:.1%}")


def _report_vs_level_strategy_baseline(bot_df: pd.DataFrame) -> None:
    """
    مقایسهٔ وین‌ریت بات با وین‌ریت level_strategy (روی همان دیتابیس) - چون
    level_strategy همان سیگنال ساده‌ایست که مدل قرار است از آن بهتر عمل کند.
    """
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    full_df = pd.read_sql_query("SELECT meta_trade_source, meta_is_refund, result FROM trades", conn)
    conn.close()
    if "meta_trade_source" not in full_df.columns:
        return
    level_df = full_df[full_df["meta_trade_source"] == "level_strategy"].dropna(subset=["result"])
    if "meta_is_refund" in level_df.columns:
        level_df = level_df[~level_df["meta_is_refund"].fillna(False).astype(bool)]
    if level_df.empty:
        return

    level_winrate = level_df["result"].astype(int).mean()
    level_n = len(level_df)
    print(f"\n--- مقایسه با baseline (وین‌ریت level_strategy روی همین دیتابیس) ---")
    print(f"level_strategy: {level_n} معامله، وین‌ریت {level_winrate:.1%}")
    print(f"bot:            {len(bot_df)} معامله، وین‌ریت {bot_df['result'].mean():.1%}")
    print(f"آزمایش آماری: {_format_z_test(bot_df['result'].sum(), len(bot_df), level_df['result'].astype(int).sum(), level_n)}")


def _report_by_direction(df: pd.DataFrame) -> None:
    print("\n--- وین‌ریت به تفکیک جهت (CALL/PUT) ---")
    if "direction" not in df.columns:
        print("  ستون direction موجود نیست.")
        return
    for direction, group in df.groupby("direction"):
        print(f"  {direction}: {len(group)} معامله، وین‌ریت {group['result'].mean():.1%}")


def _report_by_symbol(df: pd.DataFrame) -> None:
    print(f"\n--- وین‌ریت به تفکیک نماد (فقط نمادهایی با حداقل {MIN_TRADES_PER_SYMBOL} معامله) ---")
    if "meta_symbol" not in df.columns:
        print("  ستون meta_symbol موجود نیست.")
        return
    counts = df.groupby("meta_symbol").size().sort_values(ascending=False)
    shown_any = False
    for symbol, n in counts.items():
        if n < MIN_TRADES_PER_SYMBOL:
            continue
        shown_any = True
        group = df[df["meta_symbol"] == symbol]
        winrate = group["result"].mean()
        z_text = _format_z_test(
            group["result"].sum(), len(group),
            df["result"].sum() - group["result"].sum(), len(df) - len(group),
        )
        print(f"  {symbol:<18} {n:>4} معامله، وین‌ریت {winrate:.1%}   ({z_text})")
    if not shown_any:
        print(f"  هیچ نمادی به‌تنهایی {MIN_TRADES_PER_SYMBOL} معامله نداشت.")


def _report_label_sanity_check(df: pd.DataFrame) -> None:
    """
    همان بررسی سلامت لیبل که در train/train_model.py هست، این‌جا هم تکرار
    می‌شود - چون معاملات بات هم می‌توانند دچار همان اختلاف ws_deal/tick_fallback
    باشند (که معمولاً به‌خاطر نوسان واقعی نزدیک لحظهٔ انقضا است، نه یک باگ - به
    شرحی که قبلاً بررسی شد؛ عدد بالا صرفاً یعنی ارزش دارد دوباره نگاه شود).
    """
    if "result_source" not in df.columns or "tick_fallback_result" not in df.columns:
        return
    ws_deal_rows = df[df["result_source"] == "ws_deal"].dropna(subset=["tick_fallback_result"])
    if len(ws_deal_rows) == 0:
        return
    disagreement = ws_deal_rows["result"].astype(int) != ws_deal_rows["tick_fallback_result"].astype(int)
    rate = float(disagreement.mean())
    print(f"\n--- بررسی سلامت لیبل (ws_deal در برابر tick_fallback) ---")
    print(f"  {len(ws_deal_rows)} معامله نتیجه‌شان از خودِ پلتفرم (ws_deal) گرفته شده.")
    print(f"  اختلاف با محاسبهٔ داخلی (tick_fallback): {int(disagreement.sum())} مورد ({rate:.1%})")
    if rate > 0.05:
        print("  ⚠️ نرخ اختلاف بالاست - قبل از هر نتیجه‌گیری از این گزارش، این را بررسی کنید.")


def _report_payout(df: pd.DataFrame) -> None:
    if "meta_payout_percent" not in df.columns:
        return
    payouts = df["meta_payout_percent"].dropna()
    if payouts.empty:
        return
    print(f"\n--- پی‌آوت معاملات بات ---")
    print(f"  میانگین: {payouts.mean():.1f}٪   |   کمینه: {payouts.min():.1f}٪   |   بیشینه: {payouts.max():.1f}٪")
    below_min = (payouts < config.MIN_PAYOUT_PERCENT).sum()
    if below_min > 0:
        print(f"  {below_min} معامله پی‌آوتی زیر آستانهٔ {config.MIN_PAYOUT_PERCENT}٪ داشت.")


def _report_trend_over_time(df: pd.DataFrame) -> None:
    """
    وین‌ریت نیمهٔ اول در برابر نیمهٔ دوم معاملات بات (به ترتیب زمانی) - برای
    این‌که ببینیم عملکرد مدل در طول اجرا ثابت مانده، بهتر شده، یا بدتر شده.
    """
    n = len(df)
    if n < 2 * MIN_N_FOR_TEST:
        print(f"\n(برای بررسی روند زمانی حداقل {2 * MIN_N_FOR_TEST} معامله لازم است - فعلاً {n} معامله موجود است.)")
        return
    mid = n // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]
    print("\n--- روند زمانی (نیمهٔ اول در برابر نیمهٔ دوم معاملات بات) ---")
    print(f"  نیمهٔ اول  ({len(first_half)} معامله): وین‌ریت {first_half['result'].mean():.1%}")
    print(f"  نیمهٔ دوم  ({len(second_half)} معامله): وین‌ریت {second_half['result'].mean():.1%}")
    print(f"  آزمایش آماری: {_format_z_test(first_half['result'].sum(), len(first_half), second_half['result'].sum(), len(second_half))}")


def main() -> None:
    df = _load_bot_trades()
    if df.empty:
        print("هیچ معاملهٔ source=bot در دیتابیس پیدا نشد (یعنی main.py هنوز با AUTO_TRADE معامله‌ای نبسته).")
        return

    print(f"تعداد کل معاملات بات (پیش از حذف ریفاند): {len(df)}")
    df = _report_refunds(df)
    if df.empty:
        print("بعد از حذف ریفاندها، معامله‌ای برای بررسی باقی نماند.")
        return

    _report_overall(df)
    _report_vs_level_strategy_baseline(df)
    _report_by_direction(df)
    _report_by_symbol(df)
    _report_label_sanity_check(df)
    _report_payout(df)
    _report_trend_over_time(df)


if __name__ == "__main__":
    main()
