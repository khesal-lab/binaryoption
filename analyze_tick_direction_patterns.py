"""
کشف الگو در توالی خام جهت تیک‌ها (مستقل از سطح قیمت)
========================================================

پاسخ به این پیشنهاد: «آیا در توالی خام تغییرات جهت قیمت (بدون وابستگی به
خودِ قیمت) الگویی وجود دارد که بشود از آن برای جهت‌یابی معامله استفاده کرد؟»

این اسکریپت روی جدول tick_directions (که TickDirectionLogger هنگام اجرای
main.py/collect_data.py پر می‌کند - فقط علامت +۱/-۱/۰ هر تیک نسبت به تیک
قبلیِ همان نماد، نه خودِ قیمت) کار می‌کند و سه سؤال را جواب می‌دهد:

    ۱. آیا جهت یک تیک به جهت تیک(های) قبلی‌اش وابسته است (اتوکورلیشن/احتمال
       تداوم)، یا بازار در این مقیاس زمانی عملاً یک random walk است؟
    ۲. توزیع طول رشته‌های هم‌جهت (streak) با چیزی که یک سکهٔ منصفانه پیش‌بینی
       می‌کند (میانگین ۲) چقدر فرق دارد؟
    ۳. آیا الگوهای N-گرم خاصی (مثلاً «سه تیک بالا پشت‌سرهم») وجود دارند که
       احتمال تیک بعدی را به‌طور آماری معنادار از میانگین کل جدا کنند؟

⚠️ نکتهٔ مهم دربارهٔ آزمایش‌های چندگانه (Multiple Testing): در بخش N-گرم
دهها الگوی مختلف همزمان تست می‌شوند؛ با آستانهٔ p<0.05 به‌طور طبیعی حدود ۵٪
از الگوهای کاملاً بی‌اثر هم «معنادار» به نظر می‌رسند. فقط به الگوهایی که هم
p-value خیلی کوچک‌تر از ۰.۰۵ دارند و هم در چند اجرای مستقل (چند روزهٔ متفاوت)
تکرار می‌شوند اعتماد کنید.

این اسکریپت فقط خواندنی است؛ هیچ تغییری در دیتابیس یا config.py نمی‌دهد.

اجرا:
    python analyze_tick_direction_patterns.py
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict

import pandas as pd

import config
from src.tick_direction_logger import TABLE_NAME

MAX_GAP_SECONDS = 5.0  # فاصلهٔ زمانی بیشتر از این بین دو تیک متوالی = پایان یک توالی پیوسته
LAGS_TO_SCAN = [1, 2, 3, 5, 10]
NGRAM_CONTEXT_LENGTHS = [1, 2, 3]
MIN_N_TO_REPORT = 50
TOP_N_PATTERNS = 15


def _two_sided_p_value(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _one_proportion_z_test(successes: float, n: int, null_p: float = 0.5) -> tuple[float, float]:
    """z-test یک‌نسبتی در برابر یک مقدار ثابت (پیش‌فرض ۰.۵ - انتظار سکهٔ منصفانه)."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    se = math.sqrt(null_p * (1 - null_p) / n)
    z = (p - null_p) / se if se > 0 else 0.0
    return z, _two_sided_p_value(z)


def _two_proportion_z_test(wins1: float, n1: int, wins2: float, n2: int) -> tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = wins1 / n1
    p2 = wins2 / n2
    pooled_p = (wins1 + wins2) / (n1 + n2)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return z, _two_sided_p_value(z)


def _load_tick_directions() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT timestamp, symbol, direction FROM {TABLE_NAME} ORDER BY symbol, timestamp", conn
        )
    except pd.errors.DatabaseError:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction"])
    finally:
        conn.close()
    return df


def _segment_by_gap(df: pd.DataFrame, max_gap_seconds: float) -> pd.DataFrame:
    """
    یک ستون segment_id اضافه می‌کند: هر بار نماد عوض شود یا فاصلهٔ زمانی بین
    دو تیک متوالیِ همان نماد از max_gap_seconds بیشتر شود (یعنی قطعی/ری‌استارت
    اسکریپت بین این دو تیک رخ داده)، توالی جدیدی شروع می‌شود - وگرنه تحلیل
    الگو دو لحظهٔ کاملاً بی‌ربط را به‌عنوان «تیک متوالی» در نظر می‌گیرد.
    """
    df = df.copy()
    symbol_changed = df["symbol"] != df["symbol"].shift()
    gap_too_large = df.groupby("symbol")["timestamp"].diff() > max_gap_seconds
    new_segment = symbol_changed | gap_too_large.fillna(False)
    df["segment_id"] = new_segment.cumsum()
    return df


def _report_overall_distribution(df: pd.DataFrame) -> None:
    n = len(df)
    up = int((df["direction"] == 1).sum())
    down = int((df["direction"] == -1).sum())
    flat = int((df["direction"] == 0).sum())
    print(f"\n--- توزیع کلی جهت تیک‌ها ---")
    print(f"  کل تیک‌های ثبت‌شده (بعد از اولین تیکِ هر توالی): {n}")
    print(f"  صعودی: {up} ({up / n:.1%})  |  نزولی: {down} ({down / n:.1%})  |  بدون تغییر: {flat} ({flat / n:.1%})")


def _continuation_sequences(df: pd.DataFrame) -> list[list[int]]:
    """
    برای هر توالی پیوسته (segment)، فقط تیک‌های غیرصفر (واقعاً صعودی/نزولی) را
    به لیستی از +۱/-۱ تبدیل می‌کند - تیک‌های بدون تغییر برای الگوی جهت اطلاعاتی
    ندارند و فقط نویز اضافه می‌کنند.
    """
    sequences = []
    for _, group in df[df["direction"] != 0].groupby("segment_id"):
        seq = group["direction"].tolist()
        if len(seq) >= 2:
            sequences.append(seq)
    return sequences


def _report_lag_continuation(sequences: list[list[int]]) -> None:
    """
    برای هر فاصلهٔ k در LAGS_TO_SCAN: چند درصد از تیک‌ها همان جهتِ تیکِ k قدم
    قبل از خودشان را دارند؟ اگر بازار واقعاً random walk باشد، این عدد باید
    برای هر k نزدیک ۵۰٪ بماند (بدون اهمیت آماری).
    """
    print(f"\n--- احتمال تداوم جهت (تیک با جهتِ همان یا مخالفِ k تیک قبل از خودش) ---")
    print(f"{'k (فاصله)':>10} {'تعداد':>10} {'تداوم':>10} {'z':>8} {'p-value':>10}")
    for k in LAGS_TO_SCAN:
        same = 0
        total = 0
        for seq in sequences:
            for i in range(k, len(seq)):
                total += 1
                if seq[i] == seq[i - k]:
                    same += 1
        if total < MIN_N_TO_REPORT:
            print(f"{k:>10} {total:>10}   نمونهٔ کافی نیست")
            continue
        z, pval = _one_proportion_z_test(same, total, null_p=0.5)
        flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
        print(f"{k:>10} {total:>10} {same / total:>9.1%} {z:>8.2f} {pval:>10.3f}{flag}")


def _report_streak_lengths(sequences: list[list[int]]) -> None:
    """
    طول رشته‌های هم‌جهت پشت‌سرهم (streak) را می‌شمارد و میانگینش را با ۲.۰ (چیزی
    که یک سکهٔ منصفانه پیش‌بینی می‌کند) مقایسه می‌کند. میانگین بزرگ‌تر از ۲ یعنی
    بازار در این مقیاس گرایش به تداوم (مومنتوم) دارد؛ کوچک‌تر از ۲ یعنی گرایش
    به برگشت سریع (نوسان/mean-reversion).
    """
    lengths: list[int] = []
    for seq in sequences:
        current_len = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                current_len += 1
            else:
                lengths.append(current_len)
                current_len = 1
        lengths.append(current_len)

    if not lengths:
        print("\n--- توزیع طول رشته‌های هم‌جهت ---\n  داده‌ای موجود نیست.")
        return

    n = len(lengths)
    mean_len = sum(lengths) / n
    print(f"\n--- توزیع طول رشته‌های هم‌جهت (streak) ---")
    print(f"  تعداد رشته: {n}  |  میانگین طول واقعی: {mean_len:.3f}  |  "
          f"میانگین موردانتظار (سکهٔ منصفانه): 2.000")
    if n >= MIN_N_TO_REPORT:
        # تقریب نرمال برای میانگین توزیع هندسی (واریانس تقریبی p(1-p)/p^2 با p=0.5 یعنی ۲)
        se = math.sqrt(2.0) / math.sqrt(n)
        z = (mean_len - 2.0) / se
        pval = _two_sided_p_value(z)
        flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
        print(f"  z={z:+.2f}, p={pval:.3f}{flag}")


def _report_ngram_patterns(sequences: list[list[int]]) -> None:
    """
    برای هر طول زمینه k در NGRAM_CONTEXT_LENGTHS: توالیِ k تیکِ آخر را به‌عنوان
    یک الگو در نظر می‌گیرد و احتمال تجربی این‌که تیک بعدی صعودی باشد را برای هر
    الگو حساب می‌کند؛ سپس آن را با نسبت پایهٔ کل داده (همان دادهٔ همین طول
    زمینه) با z-test دو-نسبتی مقایسه می‌کند و بیشترین انحراف‌ها را نشان می‌دهد.
    """
    for k in NGRAM_CONTEXT_LENGTHS:
        pattern_next: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for seq in sequences:
            for i in range(k, len(seq)):
                pattern = tuple(seq[i - k:i])
                pattern_next[pattern].append(1 if seq[i] == 1 else 0)

        total_next = [v for values in pattern_next.values() for v in values]
        if len(total_next) < MIN_N_TO_REPORT:
            print(f"\n--- الگوهای {k}-گرم ---\n  نمونهٔ کافی نیست.")
            continue
        baseline_up = sum(total_next)
        baseline_n = len(total_next)
        baseline_rate = baseline_up / baseline_n

        rows = []
        for pattern, values in pattern_next.items():
            n = len(values)
            if n < MIN_N_TO_REPORT:
                continue
            up = sum(values)
            z, pval = _two_proportion_z_test(up, n, baseline_up, baseline_n)
            rows.append((pattern, n, up / n, z, pval))

        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        print(f"\n--- الگوهای {k}-گرم (نرخ پایهٔ صعودی بودنِ تیک بعدی: {baseline_rate:.1%}, "
              f"از روی {baseline_n} نمونه) ---")
        if not rows:
            print(f"  هیچ الگویی حداقل {MIN_N_TO_REPORT} نمونه نداشت.")
            continue
        print(f"{'الگو (۱=صعودی/-۱=نزولی)':>28} {'تعداد':>8} {'نرخ صعودیِ بعدی':>18} {'z':>8} {'p-value':>10}")
        significant_count = 0
        for pattern, n, rate, z, pval in rows[:TOP_N_PATTERNS]:
            flag = ""
            if pval < 0.05:
                flag = " ⚠️"
                significant_count += 1
            pattern_str = ",".join("+" if d == 1 else "-" for d in pattern)
            print(f"{pattern_str:>28} {n:>8} {rate:>17.1%} {z:>8.2f} {pval:>10.3f}{flag}")
        total_significant = sum(1 for r in rows if r[4] < 0.05)
        print(f"  از مجموع {len(rows)} الگوی قابل‌بررسی، {total_significant} تا در سطح p<0.05 "
              f"معنادار بودند (با آستانهٔ ۰.۰۵ روی این‌همه الگو، حدود "
              f"{len(rows) * 0.05:.1f} مورد را انتظار تصادفی هم توضیح می‌دهد).")


def main() -> None:
    df = _load_tick_directions()
    if df.empty:
        print(f"جدول {TABLE_NAME} خالی یا موجود نیست - اول با "
              f"config.TICK_DIRECTION_LOGGING_ENABLED=True یک بار main.py یا "
              f"collect_data.py را اجرا کنید تا داده جمع شود.")
        return

    df = _segment_by_gap(df, MAX_GAP_SECONDS)
    print(f"تعداد کل ردیف خام: {len(df)}  |  تعداد نماد متفاوت: {df['symbol'].nunique()}  |  "
          f"تعداد توالی پیوسته (بعد از تفکیک قطعی/تعویض نماد): {df['segment_id'].nunique()}")

    _report_overall_distribution(df)

    sequences = _continuation_sequences(df)
    total_signal_ticks = sum(len(s) for s in sequences)
    print(f"\nتعداد تیک‌های غیرصفر قابل‌استفاده برای تحلیل الگو (صعودی/نزولی، نه بدون‌تغییر): {total_signal_ticks}")
    if total_signal_ticks < MIN_N_TO_REPORT:
        print("داده برای تحلیل الگو کافی نیست - بعداً با دادهٔ بیشتر دوباره اجرا کنید.")
        return

    _report_lag_continuation(sequences)
    _report_streak_lengths(sequences)
    _report_ngram_patterns(sequences)

    print("\n" + "=" * 70)
    print("جمع‌بندی: اگر هیچ‌کدام از جدول‌های بالا انحراف معنادار و تکرارشونده‌ای")
    print("(نه فقط یکی-دو الگوی تصادفی) از ۵۰٪ نشان ندهند، یعنی توالی خام جهت")
    print("تیک‌ها در این مقیاس زمانی عملاً یک random walk است و اضافه‌کردنش به")
    print("مدل به‌عنوان فیچر بعید است سیگنال جهت‌داری بدهد.")
    print("=" * 70)


if __name__ == "__main__":
    main()
