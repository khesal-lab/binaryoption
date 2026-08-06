"""
الگوی تیک‌ها هنگام تستِ یک سطح حمایت/مقاومت: عبور یا بازگشت؟
================================================================

پیگیری این سؤال: تا حالا بررسی کردیم آیا خودِ جهتِ implied level_strategy
(نزدیک سقف/کف کندل → انتظار بازگشت) درست است یا نه. این اسکریپت یک سؤال
دقیق‌تر می‌پرسد: در لحظه‌ای که قیمت واقعاً به یک سطح حمایت/مقاومت
(همان اکسترمم‌هایی که level_strategy هم از آن‌ها استفاده می‌کند) می‌رسد،
آیا الگوی تیک‌های منتهی به آن لحظه می‌تواند مشخص کند که قیمت از آن سطح
**عبور** می‌کند یا **بازمی‌گردد**؟

هر «تست سطح» یکی از دو نتیجه دارد (config.TRADE_EXPIRY_SECONDS ثانیه بعد
اندازه‌گیری می‌شود - همان افق زمانیِ یک معاملهٔ واقعی):
    - "bounce"   (بازگشت): قیمت هنوز در همان سمتِ قبلیِ سطح است (سطح را
      نشکسته) - همان فرضِ level_strategy (near_high → PUT / near_low → CALL).
    - "breakout" (عبور): قیمت به آن‌طرفِ سطح رسیده - برخلافِ فرضِ
      level_strategy.

برای هر «تست»، الگوی جهتِ چند تیکِ منتهی به آن (سرعت/شتاب نزدیک‌شدن) به‌عنوان
زمینه در نظر گرفته می‌شود و با z-test بررسی می‌شود آیا زمینه‌های خاصی نسبت
عبور/بازگشت را از نرخ پایه به‌طور معنادار جدا می‌کنند.

علاوه بر الگوی گسستهٔ جهت (+۱/-۱)، دو فیچر پیوسته هم در لحظهٔ تست محاسبه و
با میانه به دو گروه تقسیم می‌شوند (همان روش analyze_level_strategy.py برای
فیچرهای پیوسته):
    - سرعتِ درصدی (Momentum) - دقیقاً همان تعریف TickBuffer.get_velocity_pct
      (درصد تغییر قیمت بین دو تیک آخر، تقسیم بر تغییر زمان)
    - شتابِ درصدی (Acceleration) - همان تعریف TickBuffer.get_acceleration_pct

⚠️ حجم معاملات (Volume) در این تحلیل نیست چون اصلاً در دسترس نیست: پیام‌های
خام وب‌ساکت Pocket Option برای هر تیک فقط [symbol, timestamp, price] دارند
(نگاه کنید به src/browser_session.py:extract_ticks_from_payload) - هیچ
فیلد حجمی وجود ندارد، چون این یک فیدِ قیمتِ synthetic/OTC است، نه یک
اوردربوکِ صرافیِ واقعی با حجم معامله.

⚠️ فاصلهٔ دقیقِ قیمت از سطح (gap) عمداً از این گزارش حذف شده: در اعتبارسنجی
روی دادهٔ synthetic بدون هیچ سیگنال واقعی (random walk خالص)، تفکیک بر
اساس این فیچر به‌طور مکرر (روی هر ۶ seed تصادفیِ متفاوتی که امتحان شد)
اختلاف کاذب و «معنادار» نشان می‌داد - نه یک false positive تصادفی یک‌باره.
علتش این بود: gap=0 تقریباً همیشه یعنی همین تیک دارد یک اکسترمم *تازه* را
می‌سازد (سرعت/جهت لحظه‌ای‌اش طبعاً هم‌جهت است)، در حالی که gap>0 معمولاً
یعنی قیمت دارد یک سطحِ *قبلاً شکل‌گرفته* را دوباره تست می‌کند - این دو
موقعیت پویایی کاملاً متفاوتی دارند و مقایسه‌شان زیر یک فیچر خطی گمراه‌کننده
است. تفکیک درست‌تر (تازه/مجدد) خودش هم روی همان دادهٔ بدون سیگنال انحراف
نشان داد، پس آن هم قابل‌اعتماد نبود؛ طراحی یک فیچر «فاصله از سطح» درست به
بررسی بیشتری نیاز دارد و به همین دلیل فعلاً از این گزارش کنار گذاشته شده.

دقیقاً مثل analyze_future_price_prediction.py، فقط اولین تیکِ هر «تست»
(نه همهٔ تیک‌های یک توقفِ واحد) شمرده می‌شود تا نمونه‌ها واقعاً مستقل
باشند - همان فاصلهٔ حداقلیِ config.LEVEL_STRATEGY_MIN_CLICK_INTERVAL_SECONDS.

نیازمند ستون price در جدول tick_directions (از PR اضافه‌شدن لاگ قیمت به
بعد). این اسکریپت فقط خواندنی است؛ هیچ تغییری در دیتابیس یا config.py
نمی‌دهد.

اجرا:
    python analyze_level_test_patterns.py
"""

from __future__ import annotations

import bisect
import math
import sqlite3
import statistics
from collections import defaultdict

import pandas as pd

import config
from src.browser_session import Tick
from src.feature_engineering import CandleAggregator, TickHistory, compute_level_strategy_context
from src.tick_direction_logger import TABLE_NAME

MAX_GAP_SECONDS = 5.0
OUTCOME_HORIZON_SECONDS = config.TRADE_EXPIRY_SECONDS
MAX_FUTURE_SEARCH_SECONDS = OUTCOME_HORIZON_SECONDS + 2.0
APPROACH_CONTEXT_LENGTHS = [1, 2, 3]
MIN_N_TO_REPORT = 30
TOP_N_PATTERNS = 15


def _two_sided_p_value(z: float) -> float:
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def _one_proportion_z_test(successes: float, n: int, null_p: float = 0.5) -> tuple[float, float]:
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


def _load_ticks_with_price() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    try:
        df = pd.read_sql_query(
            f"SELECT timestamp, symbol, direction, price FROM {TABLE_NAME} "
            f"WHERE price IS NOT NULL ORDER BY symbol, timestamp",
            conn,
        )
    except pd.errors.DatabaseError:
        return pd.DataFrame(columns=["timestamp", "symbol", "direction", "price"])
    finally:
        conn.close()
    return df


def _segment_by_gap(df: pd.DataFrame, max_gap_seconds: float) -> pd.DataFrame:
    df = df.copy()
    symbol_changed = df["symbol"] != df["symbol"].shift()
    gap_too_large = df.groupby("symbol")["timestamp"].diff() > max_gap_seconds
    new_segment = symbol_changed | gap_too_large.fillna(False)
    df["segment_id"] = new_segment.cumsum()
    return df


def _detect_level_tests(symbol: str, timestamps: list[float], prices: list[float]) -> list[dict]:
    """
    توالی را بازپخش می‌کند (با کلاس‌های تولیدی CandleAggregator/TickHistory/
    compute_level_strategy_context) و هر «تست سطح» مستقل (اولین تیکِ هر
    توقف نزدیک سقف/کف کندل جاری) را با نوع سطح و قیمتِ دقیق آن برمی‌گرداند.
    """
    candle_aggregator = CandleAggregator()
    tick_history = TickHistory()
    last_signal_timestamp: float | None = None
    tests: list[dict] = []

    for i, (t, price) in enumerate(zip(timestamps, prices)):
        tick = Tick(symbol=symbol, price=price, timestamp=t)
        candle_aggregator.add_tick(tick)
        tick_history.add(tick)

        current_candle = candle_aggregator.current
        if current_candle is None or current_candle.range <= 0:
            continue

        level_context = compute_level_strategy_context(tick_history, current_candle)
        if level_context is None:
            continue

        price_position = (price - current_candle.low) / current_candle.range
        near_high_gap = abs(price_position - level_context["level_strategy_near_high_position"])
        near_low_gap = abs(price_position - level_context["level_strategy_near_low_position"])
        is_near_high = near_high_gap <= config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO
        is_near_low = near_low_gap <= config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO
        if not (is_near_high or is_near_low):
            continue

        if (last_signal_timestamp is not None
                and t - last_signal_timestamp < config.LEVEL_STRATEGY_MIN_CLICK_INTERVAL_SECONDS):
            continue
        last_signal_timestamp = t

        if is_near_high and (not is_near_low or near_high_gap <= near_low_gap):
            level_type = "resistance"
            level_price = (level_context["level_strategy_near_high_position"] * current_candle.range
                            + current_candle.low)
            gap = near_high_gap
        else:
            level_type = "support"
            level_price = (level_context["level_strategy_near_low_position"] * current_candle.range
                            + current_candle.low)
            gap = near_low_gap

        tests.append({"index": i, "level_type": level_type, "level_price": level_price, "gap": gap})

    return tests


def _velocity_pct_at(timestamps: list[float], prices: list[float], i: int) -> float | None:
    """درصد تغییر قیمت بین دو تیک آخر (تا شاخص i)، تقسیم بر تغییر زمان - دقیقاً مثل TickBuffer.get_velocity_pct."""
    if i < 1:
        return None
    dt = timestamps[i] - timestamps[i - 1]
    if dt <= 0 or prices[i - 1] == 0:
        return None
    return (prices[i] - prices[i - 1]) / prices[i - 1] / dt


def _acceleration_pct_at(timestamps: list[float], prices: list[float], i: int) -> float | None:
    """تغییر سرعتِ درصدی بین دو مقدار آخر، تقسیم بر تغییر زمان - دقیقاً مثل TickBuffer.get_acceleration_pct."""
    if i < 2:
        return None
    v_curr = _velocity_pct_at(timestamps, prices, i)
    v_prev = _velocity_pct_at(timestamps, prices, i - 1)
    if v_curr is None or v_prev is None:
        return None
    dt = timestamps[i] - timestamps[i - 1]
    if dt <= 0:
        return None
    return (v_curr - v_prev) / dt


def _outcome_for_test(timestamps: list[float], prices: list[float], test: dict) -> str | None:
    """"breakout"، "bounce"، یا None (نتیجه در بازهٔ جست‌وجو پیدا نشد یا دقیقاً روی سطح ایستاد)."""
    i = test["index"]
    target_time = timestamps[i] + OUTCOME_HORIZON_SECONDS
    j = bisect.bisect_left(timestamps, target_time, lo=i + 1)
    if j >= len(timestamps):
        return None
    if timestamps[j] - timestamps[i] > MAX_FUTURE_SEARCH_SECONDS:
        return None

    future_price = prices[j]
    level_price = test["level_price"]
    if test["level_type"] == "resistance":
        if future_price > level_price:
            return "breakout"
        if future_price < level_price:
            return "bounce"
    else:
        if future_price < level_price:
            return "breakout"
        if future_price > level_price:
            return "bounce"
    return None


def _report_base_rates(all_tests: list[tuple[dict, str]]) -> None:
    print(f"\n--- نرخ پایهٔ عبور/بازگشت (افق {OUTCOME_HORIZON_SECONDS:.0f} ثانیه) ---")
    total = len(all_tests)
    if total < MIN_N_TO_REPORT:
        print(f"  نمونهٔ کافی نیست (n={total}).")
        return

    breakout_n = sum(1 for _, outcome in all_tests if outcome == "breakout")
    print(f"  کل تست‌های مستقل: {total}  |  عبور: {breakout_n} ({breakout_n / total:.1%})  |  "
          f"بازگشت: {total - breakout_n} ({(total - breakout_n) / total:.1%})")

    for level_type, label in (("resistance", "تست مقاومت (near_high)"), ("support", "تست حمایت (near_low)")):
        subset = [o for t, o in all_tests if t["level_type"] == level_type]
        if len(subset) < MIN_N_TO_REPORT:
            print(f"  {label}: نمونهٔ کافی نیست (n={len(subset)}).")
            continue
        breakout = sum(1 for o in subset if o == "breakout")
        n = len(subset)
        z, pval = _one_proportion_z_test(breakout, n, null_p=0.5)
        flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
        print(f"  {label}: n={n}, نرخ عبور={breakout / n:.1%} (در برابر ۵۰٪ تصادفی) z={z:+.2f}, p={pval:.3f}{flag}")


def _report_approach_patterns(all_tests: list[tuple[dict, str]], directions_by_test_key: dict) -> None:
    print(f"\n{'=' * 70}\nآیا الگوی تیک‌های منتهی به تست، عبور/بازگشت را پیش‌بینی می‌کند؟\n{'=' * 70}")

    for k in APPROACH_CONTEXT_LENGTHS:
        pattern_outcomes: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for test, outcome in all_tests:
            context = directions_by_test_key.get(id(test))
            if context is None or len(context) < k:
                continue
            ctx_k = tuple(context[-k:])
            if 0 in ctx_k:
                continue
            pattern_outcomes[ctx_k].append(1 if outcome == "breakout" else 0)

        total = [v for values in pattern_outcomes.values() for v in values]
        if len(total) < MIN_N_TO_REPORT:
            print(f"\n--- زمینهٔ {k} تیکِ آخر قبل از تست ---\n  نمونهٔ کافی نیست (n={len(total)}).")
            continue
        baseline_breakout = sum(total)
        baseline_n = len(total)

        rows = []
        for pattern, values in pattern_outcomes.items():
            n = len(values)
            if n < MIN_N_TO_REPORT:
                continue
            breakout = sum(values)
            z, pval = _two_proportion_z_test(breakout, n, baseline_breakout, baseline_n)
            rows.append((pattern, n, breakout / n, z, pval))
        rows.sort(key=lambda r: abs(r[3]), reverse=True)

        print(f"\n--- زمینهٔ {k} تیکِ آخر قبل از تست (نرخ پایهٔ عبور: "
              f"{baseline_breakout / baseline_n:.1%}, از روی {baseline_n} نمونه) ---")
        if not rows:
            print(f"  هیچ الگویی حداقل {MIN_N_TO_REPORT} نمونه نداشت.")
            continue
        print(f"{'الگو':>20} {'تعداد':>8} {'نرخ عبور':>12} {'z':>8} {'p-value':>10}")
        for pattern, n, rate, z, pval in rows[:TOP_N_PATTERNS]:
            flag = " ⚠️" if pval < 0.05 else ""
            pattern_str = ",".join("+" if d == 1 else "-" for d in pattern)
            print(f"{pattern_str:>20} {n:>8} {rate:>11.1%} {z:>8.2f} {pval:>10.3f}{flag}")
        total_significant = sum(1 for r in rows if r[4] < 0.05)
        print(f"  از مجموع {len(rows)} الگوی قابل‌بررسی، {total_significant} تا در سطح p<0.05 معنادار بودند "
              f"(انتظار تصادفی حدود {len(rows) * 0.05:.1f} مورد).")


def _report_continuous_split(
    all_tests: list[tuple[dict, str]], continuous_features: dict, feature_key: str, label: str
) -> None:
    """
    یک فیچر پیوسته (فاصله/سرعت/شتاب) را با میانه به دو گروه تقسیم می‌کند و
    نرخ عبور هر گروه را با z-test مقایسه می‌کند - همان روشی که
    analyze_level_strategy.py برای فیچرهای پیوسته استفاده می‌کند.
    """
    print(f"\n--- {label} ---")
    values: list[tuple[float, str]] = []
    for test, outcome in all_tests:
        v = continuous_features.get(id(test), {}).get(feature_key)
        if v is not None:
            values.append((v, outcome))

    if len(values) < MIN_N_TO_REPORT * 2:
        print(f"  نمونهٔ کافی نیست (n={len(values)}).")
        return

    median = statistics.median(v for v, _ in values)
    above = [o for v, o in values if v > median]
    below = [o for v, o in values if v <= median]
    if len(above) < MIN_N_TO_REPORT or len(below) < MIN_N_TO_REPORT:
        print(f"  بعد از تفکیک با میانه، نمونهٔ کافی نیست (بالای میانه={len(above)}, پایین/مساوی میانه={len(below)}).")
        return

    above_breakout = sum(1 for o in above if o == "breakout")
    below_breakout = sum(1 for o in below if o == "breakout")
    z, pval = _two_proportion_z_test(above_breakout, len(above), below_breakout, len(below))
    flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
    print(f"  میانه: {median:.6g}  |  n کل: {len(values)}")
    print(f"  بالای میانه: n={len(above)}, نرخ عبور={above_breakout / len(above):.1%}")
    print(f"  پایین/مساوی میانه: n={len(below)}, نرخ عبور={below_breakout / len(below):.1%}")
    print(f"  z={z:+.2f}, p={pval:.3f}{flag}")


def main() -> None:
    df = _load_ticks_with_price()
    if df.empty:
        print(f"هیچ ردیفی با ستون price در جدول {TABLE_NAME} پیدا نشد. باید main.py یا "
              f"collect_data.py را (با نسخهٔ جدید) مدتی اجرا کنید تا داده جمع شود.")
        return

    df = _segment_by_gap(df, MAX_GAP_SECONDS)

    all_tests: list[tuple[dict, str]] = []
    directions_by_test_key: dict = {}
    continuous_features: dict = {}

    for _, group in df.groupby("segment_id"):
        if len(group) < 5:
            continue
        symbol = group["symbol"].iloc[0]
        timestamps = group["timestamp"].tolist()
        prices = group["price"].tolist()
        directions = group["direction"].tolist()

        tests = _detect_level_tests(symbol, timestamps, prices)
        for test in tests:
            outcome = _outcome_for_test(timestamps, prices, test)
            if outcome is None:
                continue
            all_tests.append((test, outcome))
            directions_by_test_key[id(test)] = directions[:test["index"]]
            continuous_features[id(test)] = {
                "velocity_pct": _velocity_pct_at(timestamps, prices, test["index"]),
                "acceleration_pct": _acceleration_pct_at(timestamps, prices, test["index"]),
            }

    print(f"تعداد کل ردیف قابل‌استفاده (با price): {len(df)}  |  تعداد نماد: {df['symbol'].nunique()}  |  "
          f"تعداد تست‌های سطحِ مستقل با نتیجهٔ مشخص: {len(all_tests)}")

    if len(all_tests) < MIN_N_TO_REPORT:
        print("داده برای تحلیل کافی نیست - بعداً با دادهٔ بیشتر دوباره اجرا کنید.")
        return

    _report_base_rates(all_tests)
    _report_approach_patterns(all_tests, directions_by_test_key)

    print(f"\n{'=' * 70}\nآیا سرعت یا شتاب لحظهٔ تست، عبور/بازگشت را پیش‌بینی می‌کند؟\n{'=' * 70}")
    _report_continuous_split(all_tests, continuous_features, "velocity_pct",
                              "سرعتِ درصدیِ لحظهٔ تست (Momentum)")
    _report_continuous_split(all_tests, continuous_features, "acceleration_pct",
                              "شتابِ درصدیِ لحظهٔ تست (Acceleration)")

    print(f"\n{'=' * 70}")
    print("جمع‌بندی: نرخ پایه نشان می‌دهد آیا خودِ سطوح حمایت/مقاومت به‌طور کلی")
    print("بیشتر باعث بازگشت می‌شوند یا عبور. بخش‌های بعدی نشان می‌دهند آیا از")
    print("روی جهتِ چند تیکِ آخر، سرعت، یا شتاب لحظهٔ تست می‌شود حدس زد کدام‌یک")
    print("(عبور یا بازگشت) رخ خواهد داد. حجم معامله در این بررسی نیست چون در")
    print("دادهٔ این بروکر اصلاً ثبت نمی‌شود. فاصلهٔ دقیق از سطح (gap) عمداً از")
    print("این گزارش حذف شده - نگاه کنید به توضیح بالای فایل.")
    print("=" * 70)


if __name__ == "__main__":
    main()
