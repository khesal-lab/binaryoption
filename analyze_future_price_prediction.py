"""
آیا از روی الگوی تیک‌های اخیر می‌شود قیمتِ چند ثانیه بعد را پیش‌بینی کرد؟
================================================================================

پیگیری این سؤال: analyze_tick_direction_patterns.py نشان داد تیک‌ها یک
گرایش کوتاه‌مدت به برگشت دارند (بعد از یک تیک صعودی، تیک بعدی بیشتر نزولی
است)، ولی این گرایش تا فاصلهٔ ۵-۱۰ تیک کاملاً محو می‌شود. آن‌چه واقعاً برای
یک معاملهٔ باینری مهم است این نیست که «تیک بعدی» کدام جهت است، بلکه این
است که در پایان انقضای معامله (config.TRADE_EXPIRY_SECONDS ثانیه بعد از
ورود) قیمت بالاتر بوده یا پایین‌تر - در آن بازه معمولاً چندین تیک اتفاق
می‌افتد، نه فقط یکی.

این اسکریپت روی همان جدول tick_directions کار می‌کند (اما نیاز به ستون
price دارد - فقط از این به بعد ثبت می‌شود، ردیف‌های قدیمی‌تر NULL هستند و
نادیده گرفته می‌شوند) و دو سؤال را جواب می‌دهد:

    ۱. به‌طور کلی (در هر لحظه‌ای)، آیا الگوی جهتِ چند تیکِ اخیر می‌تواند
       علامتِ (قیمت در t+HORIZON − قیمت الان) را پیش‌بینی کند؟
    ۲. مخصوصاً در لحظاتی که level_strategy واقعاً سیگنال ورود می‌دهد (قیمت
       نزدیک سقف یا کف کندل جاری) - همان پیشنهاد شما - آیا خودِ جهتِ
       implied این استراتژی (بدون هیچ مدلی) در همان لحظات واقعاً می‌برد؟
       این دقیقاً backtest خودِ قانون level_strategy روی دادهٔ تیکِ خام است
       (نه فقط روی معاملات واقعی که تعدادشان خیلی کمتر است).

برای بخش ۲، از همان کلاس‌های تولیدی CandleAggregator/TickHistory/
compute_level_strategy_context استفاده می‌شود (نه پیاده‌سازی جداگانه) تا
دقیقاً همان منطقی که main.py زنده اجرا می‌کند این‌جا هم بازپخش شود.

⚠️ همان هشدار Multiple Testing و نمونهٔ کم که در analyze_tick_direction_
patterns.py بود، این‌جا هم صادق است.

این اسکریپت فقط خواندنی است؛ هیچ تغییری در دیتابیس یا config.py نمی‌دهد.

اجرا:
    python analyze_future_price_prediction.py
"""

from __future__ import annotations

import bisect
import math
import sqlite3
from collections import defaultdict

import pandas as pd

import config
from src.browser_session import Tick
from src.feature_engineering import CandleAggregator, TickHistory, compute_level_strategy_context
from src.tick_direction_logger import TABLE_NAME

MAX_GAP_SECONDS = 5.0
HORIZON_SECONDS = config.TRADE_EXPIRY_SECONDS
MAX_FUTURE_SEARCH_SECONDS = HORIZON_SECONDS + 2.0  # اگر تیکِ بعد از این فاصله هم پیدا نشد، یعنی توالی همین‌جا تمام شده
NGRAM_CONTEXT_LENGTHS = [1, 2, 3]
MIN_N_TO_REPORT = 50
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


class _Segment:
    """یک توالی پیوستهٔ (timestamp, price, direction) برای یک نماد، بدون قطعی/تعویض نماد در وسط."""

    def __init__(self, symbol: str, timestamps: list[float], prices: list[float], directions: list[int]):
        self.symbol = symbol
        self.timestamps = timestamps
        self.prices = prices
        self.directions = directions

    def future_target(self, i: int) -> int | None:
        """
        علامتِ (قیمت در t_i+HORIZON − قیمت در t_i): +۱=بالاتر (CALL می‌برد)،
        -۱=پایین‌تر (PUT می‌برد)، ۰=مساوی (رد می‌شود، شبیه ریفاند). اگر تیکی
        در بازهٔ [t_i+HORIZON, t_i+MAX_FUTURE_SEARCH] پیدا نشود، None.
        """
        target_time = self.timestamps[i] + HORIZON_SECONDS
        j = bisect.bisect_left(self.timestamps, target_time, lo=i + 1)
        if j >= len(self.timestamps):
            return None
        if self.timestamps[j] - self.timestamps[i] > MAX_FUTURE_SEARCH_SECONDS:
            return None
        future_price = self.prices[j]
        current_price = self.prices[i]
        if future_price > current_price:
            return 1
        if future_price < current_price:
            return -1
        return 0

    def replay_level_strategy(self) -> list[int | None]:
        """
        این توالی را از اول با CandleAggregator/TickHistory واقعی بازپخش
        می‌کند و برای هر نقطه، جهتِ implied level_strategy را برمی‌گرداند
        اگر آن لحظه «نزدیک سطح کلیدی» (سقف یا کف کندل جاری) بوده - دقیقاً
        همان معیاری که main.py برای gating استفاده می‌کند - وگرنه None.

        نکتهٔ مهم دربارهٔ استقلال نمونه‌ها: وقتی قیمت نزدیک سقف/کف کندل
        «می‌ایستد» (stall)، این توقف معمولاً چند تیک پشت‌سرهم طول می‌کشد -
        همهٔ آن‌ها یک اتفاق واحدند، نه چند نمونهٔ مستقل (پنجرهٔ ۳-ثانیه‌ایِ
        آینده‌شان هم روی هم می‌افتد). اگر همهٔ این تیک‌ها جداگانه شمرده شوند،
        z-test بعدی (که فرض نمونه‌های مستقل دارد) عدد معناداریِ کاذب و
        اغراق‌شده نشان می‌دهد. برای همین، فقط اولین تیکِ هر توقف - با همان
        فاصلهٔ حداقلی‌ای که خودِ ربات زنده بین دو کلیک واقعی رعایت می‌کند
        (config.LEVEL_STRATEGY_MIN_CLICK_INTERVAL_SECONDS) - به‌عنوان یک
        نمونهٔ مستقل حساب می‌شود؛ بقیهٔ تیک‌های همان توقف None می‌مانند.
        """
        candle_aggregator = CandleAggregator()
        tick_history = TickHistory()
        implied_directions: list[int | None] = []
        last_signal_timestamp: float | None = None

        for t, price in zip(self.timestamps, self.prices):
            tick = Tick(symbol=self.symbol, price=price, timestamp=t)
            candle_aggregator.add_tick(tick)
            tick_history.add(tick)

            current_candle = candle_aggregator.current
            direction: int | None = None
            if current_candle is not None and current_candle.range > 0:
                level_context = compute_level_strategy_context(tick_history, current_candle)
                if level_context is not None:
                    price_position = (price - current_candle.low) / current_candle.range
                    near_high_gap = abs(price_position - level_context["level_strategy_near_high_position"])
                    near_low_gap = abs(price_position - level_context["level_strategy_near_low_position"])
                    is_near_level = (near_high_gap <= config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO
                                      or near_low_gap <= config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO)
                    if is_near_level and (
                            last_signal_timestamp is None
                            or t - last_signal_timestamp >= config.LEVEL_STRATEGY_MIN_CLICK_INTERVAL_SECONDS):
                        if near_high_gap <= near_low_gap:
                            direction = level_context["level_strategy_near_high_implied_direction"]
                        else:
                            direction = level_context["level_strategy_near_low_implied_direction"]
                        last_signal_timestamp = t
            implied_directions.append(direction)

        return implied_directions


def _build_segments(df: pd.DataFrame) -> list[_Segment]:
    segments = []
    for _, group in df.groupby("segment_id"):
        if len(group) < 5:
            continue
        segments.append(_Segment(
            symbol=group["symbol"].iloc[0],
            timestamps=group["timestamp"].tolist(),
            prices=group["price"].tolist(),
            directions=group["direction"].tolist(),
        ))
    return segments


def _report_unconditional_ngrams(segments: list[_Segment]) -> None:
    print(f"\n{'=' * 70}\nبخش ۱: پیش‌بینیِ کلی (بدون شرط) - آیا الگوی تیک‌های اخیر، قیمتِ "
          f"{HORIZON_SECONDS:.0f} ثانیه بعد را پیش‌بینی می‌کند؟\n{'=' * 70}")

    for k in NGRAM_CONTEXT_LENGTHS:
        pattern_targets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for seg in segments:
            for i in range(k, len(seg.timestamps)):
                context = tuple(seg.directions[i - k:i])
                if 0 in context:
                    continue  # فقط تیک‌های واقعاً صعودی/نزولی به‌عنوان زمینه
                target = seg.future_target(i)
                if target is None or target == 0:
                    continue
                pattern_targets[context].append(1 if target == 1 else 0)

        total = [v for values in pattern_targets.values() for v in values]
        if len(total) < MIN_N_TO_REPORT:
            print(f"\n--- الگوهای {k}-گرم ---\n  نمونهٔ کافی نیست (n={len(total)}).")
            continue
        baseline_up = sum(total)
        baseline_n = len(total)

        rows = []
        for pattern, values in pattern_targets.items():
            n = len(values)
            if n < MIN_N_TO_REPORT:
                continue
            up = sum(values)
            z, pval = _two_proportion_z_test(up, n, baseline_up, baseline_n)
            rows.append((pattern, n, up / n, z, pval))
        rows.sort(key=lambda r: abs(r[3]), reverse=True)

        print(f"\n--- الگوهای {k}-گرم (نرخ پایهٔ صعودی‌بودنِ قیمت {HORIZON_SECONDS:.0f} ثانیه بعد: "
              f"{baseline_up / baseline_n:.1%}, از روی {baseline_n} نمونه) ---")
        if not rows:
            print(f"  هیچ الگویی حداقل {MIN_N_TO_REPORT} نمونه نداشت.")
            continue
        print(f"{'الگو':>20} {'تعداد':>8} {'نرخ صعودی آینده':>18} {'z':>8} {'p-value':>10}")
        for pattern, n, rate, z, pval in rows[:TOP_N_PATTERNS]:
            flag = " ⚠️" if pval < 0.05 else ""
            pattern_str = ",".join("+" if d == 1 else "-" for d in pattern)
            print(f"{pattern_str:>20} {n:>8} {rate:>17.1%} {z:>8.2f} {pval:>10.3f}{flag}")
        total_significant = sum(1 for r in rows if r[4] < 0.05)
        print(f"  از مجموع {len(rows)} الگوی قابل‌بررسی، {total_significant} تا در سطح p<0.05 معنادار بودند "
              f"(انتظار تصادفی حدود {len(rows) * 0.05:.1f} مورد).")


def _report_level_strategy_backtest(segments: list[_Segment]) -> None:
    print(f"\n{'=' * 70}\nبخش ۲: بک‌تست خودِ قانون level_strategy روی دادهٔ خام تیک - آیا در "
          f"لحظاتی که این استراتژی سیگنال می‌دهد، جهتِ implied آن واقعاً "
          f"{HORIZON_SECONDS:.0f} ثانیه بعد می‌برد؟ (فقط اولین تیکِ هر توقف شمرده "
          f"می‌شود - نگاه کنید به توضیح replay_level_strategy)\n{'=' * 70}")

    up_count = 0
    total = 0
    for seg in segments:
        implied_directions = seg.replay_level_strategy()
        for i, direction in enumerate(implied_directions):
            if direction is None:
                continue
            target = seg.future_target(i)
            if target is None or target == 0:
                continue
            total += 1
            predicted_up = direction == 1
            actual_up = target == 1
            if predicted_up == actual_up:
                up_count += 1

    print(f"\nتعداد لحظاتی که level_strategy سیگنال می‌داد و نتیجهٔ {HORIZON_SECONDS:.0f} "
          f"ثانیه بعد هم مشخص بود: {total}")
    if total < MIN_N_TO_REPORT:
        print(f"  نمونهٔ کافی نیست (حداقل {MIN_N_TO_REPORT} لازم است) - بعداً با دادهٔ بیشتر دوباره اجرا کنید.")
        return

    winrate = up_count / total
    z, pval = _one_proportion_z_test(up_count, total, null_p=0.5)
    flag = " ⚠️ معنادار (p<0.05)" if pval < 0.05 else ""
    print(f"  وین‌ریتِ جهتِ implied level_strategy (در برابر تصادفی ۵۰٪): {winrate:.1%}")
    print(f"  z={z:+.2f}, p={pval:.3f}{flag}")


def main() -> None:
    df = _load_ticks_with_price()
    if df.empty:
        print(f"هیچ ردیفی با ستون price در جدول {TABLE_NAME} پیدا نشد. این ستون تازه اضافه شده - "
              f"باید main.py یا collect_data.py را (با نسخهٔ جدید) مدتی اجرا کنید تا داده جمع شود.")
        return

    df = _segment_by_gap(df, MAX_GAP_SECONDS)
    segments = _build_segments(df)
    total_ticks = sum(len(s.timestamps) for s in segments)
    print(f"تعداد کل ردیف قابل‌استفاده (با price): {len(df)}  |  تعداد نماد: {df['symbol'].nunique()}  |  "
          f"تعداد توالی پیوستهٔ قابل‌استفاده (حداقل ۵ تیک): {len(segments)}  |  "
          f"جمع تیک در این توالی‌ها: {total_ticks}")

    if total_ticks < MIN_N_TO_REPORT:
        print("داده برای تحلیل کافی نیست - بعداً با دادهٔ بیشتر دوباره اجرا کنید.")
        return

    _report_unconditional_ngrams(segments)
    _report_level_strategy_backtest(segments)

    print(f"\n{'=' * 70}")
    print("جمع‌بندی: بخش ۱ نشان می‌دهد آیا صرفِ توالیِ جهتِ تیک‌های اخیر (بدون هیچ")
    print("قانونی) پیش‌بینی‌کنندهٔ قیمتِ آینده است. بخش ۲ نشان می‌دهد آیا خودِ")
    print("قاعدهٔ level_strategy (که فقط در لحظات خاص - نزدیک سقف/کف کندل - وارد")
    print("عمل می‌شود) روی دادهٔ خام تیک (نه فقط معاملات واقعی محدود) واقعاً برتری")
    print("آماری معنادار دارد یا نه.")
    print("=" * 70)


if __name__ == "__main__":
    main()
