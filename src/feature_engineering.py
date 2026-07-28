"""
بخش دوم: پردازش داده‌ها و محاسبات فنی لحظه‌ای (Real-time Feature Engineering)
=============================================================================

نکتهٔ کلیدی این نسخه: **هیچ مقدار خام قیمتی در خروجی نهایی ذخیره نمی‌شود.**
مدل باید فقط با مفاهیم نسبی آموزش ببیند (نسبت، درصد، z-score، پرچم ۰/۱) تا روی
هر جفت‌ارز و هر بازهٔ قیمتی تعمیم پیدا کند — دقیقاً همان‌طور که یک تریدر با
نگاه به شکل چارت تصمیم می‌گیرد، نه با نگاه به عدد خام قیمت.

این ماژول شامل سه بخش است:

    ۱. TickBuffer:
       بافر کوتاه (۱۰ تیک) برای محاسبهٔ سرعت/شتاب **درصدی** (Percentage Velocity/
       Acceleration) — یعنی نرخ تغییر قیمت به‌صورت درصد، نه مقدار مطلق، تا بین
       نمادهای مختلف قابل مقایسه باشد.

    ۲. TickHistory:
       بافر بلندتر (۲۰۰ تیک) برای دو محاسبهٔ آماری:
           - Spike Z-Score: آیا حرکت لحظهٔ فعلی نسبت به نوسان عادی اخیر غیرعادی است؟
           - Stall Detection: نقاط توقف/برگشت قیمت داخل کندل در حال شکل‌گیری.

    ۳. CandleAggregator:
       کندل‌های یک‌دقیقه‌ای را می‌سازد و ویژگی‌های **شکلی نسبی** آن‌ها (نسبت بدنه
       به سایه، جایگاه قیمت داخل کندل و ...) را استخراج می‌کند — بدون OHLC خام.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.browser_session import Tick
import config


# ---------------------------------------------------------------------------
# بخش ۲.۱ — بافر تیک و محاسبهٔ سرعت/شتاب درصدی
# ---------------------------------------------------------------------------
class TickBuffer:
    """
    نگه‌دارندهٔ آخرین N تیک قیمت (پیش‌فرض ۱۰ تیک، طبق config.TICK_BUFFER_SIZE).
    سرعت و شتاب همیشه به‌صورت **درصد تغییر قیمت** محاسبه می‌شوند، نه مقدار مطلق،
    تا مستقل از سطح قیمت نماد باشند.
    """

    def __init__(self, maxlen: int = config.TICK_BUFFER_SIZE):
        self.buffer: deque[Tick] = deque(maxlen=maxlen)

    def add(self, tick: Tick) -> None:
        self.buffer.append(tick)

    def is_ready(self, min_points: int = 2) -> bool:
        return len(self.buffer) >= min_points

    def latest(self) -> Optional[Tick]:
        return self.buffer[-1] if self.buffer else None

    def latest_price_after(self, timestamp: float) -> Optional[float]:
        """
        نزدیک‌ترین قیمت ثبت‌شده بعد (یا مساوی) یک timestamp مشخص را برمی‌گرداند.
        این تابع برای بررسی نتیجهٔ معامله بعد از ۳ ثانیه از لحظهٔ ورود استفاده می‌شود
        (فقط داخلی؛ خودِ قیمت هرگز در دیتاست نهایی ذخیره نمی‌شود).
        """
        candidates = [t for t in self.buffer if t.timestamp >= timestamp]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.timestamp - timestamp).price

    # -- سرعت درصدی (Percentage Velocity) ------------------------------------
    def get_velocity_pct(self) -> float:
        """
        درصد تغییر قیمت بین دو تیک آخر، تقسیم بر تغییر زمان.
        واحد: درصد بر ثانیه. مقدار مثبت=صعودی، منفی=نزولی.
        """
        if len(self.buffer) < 2:
            return 0.0
        t_prev, t_curr = self.buffer[-2], self.buffer[-1]
        dt = t_curr.timestamp - t_prev.timestamp
        if dt <= 0 or t_prev.price == 0:
            return 0.0
        pct_change = (t_curr.price - t_prev.price) / t_prev.price
        return pct_change / dt

    def _velocity_pct_series(self) -> list[float]:
        ticks = list(self.buffer)
        velocities = []
        for i in range(1, len(ticks)):
            dt = ticks[i].timestamp - ticks[i - 1].timestamp
            if dt > 0 and ticks[i - 1].price != 0:
                pct_change = (ticks[i].price - ticks[i - 1].price) / ticks[i - 1].price
                velocities.append(pct_change / dt)
        return velocities

    # -- شتاب درصدی (Percentage Acceleration) --------------------------------
    def get_acceleration_pct(self) -> float:
        """
        تغییر سرعتِ درصدی بین دو مقدار آخر، تقسیم بر تغییر زمان.
        شتاب منفی معمولاً نشانهٔ نزدیک‌شدن به نقطهٔ توقف/بازگشت قیمت است.
        """
        velocities = self._velocity_pct_series()
        if len(velocities) < 2:
            return 0.0
        ticks = list(self.buffer)
        dt = ticks[-1].timestamp - ticks[-2].timestamp
        if dt <= 0:
            return 0.0
        return (velocities[-1] - velocities[-2]) / dt

    # -- نسخهٔ هموارشده با رگرسیون (اختیاری، برای کاهش نویز) -----------------
    def get_velocity_pct_smoothed(self) -> float:
        """
        شیب خط رگرسیون خطی روی بازده‌های درصدی تجمعی بافر (نسبت به اولین تیک
        بافر)، به‌عنوان سرعت هموارشده. نویز کمتر از get_velocity_pct.
        """
        if len(self.buffer) < 3:
            return self.get_velocity_pct()
        ticks = list(self.buffer)
        base_price = ticks[0].price
        if base_price == 0:
            return self.get_velocity_pct()
        times = np.array([t.timestamp for t in ticks])
        pct_returns = np.array([(t.price - base_price) / base_price for t in ticks])
        times = times - times[0]
        slope, _ = np.polyfit(times, pct_returns, 1)
        return float(slope)

    def get_acceleration_pct_smoothed(self) -> float:
        """شتاب هموارشده با برازش سهمی روی بازده‌های درصدی تجمعی بافر."""
        if len(self.buffer) < 4:
            return self.get_acceleration_pct()
        ticks = list(self.buffer)
        base_price = ticks[0].price
        if base_price == 0:
            return self.get_acceleration_pct()
        times = np.array([t.timestamp for t in ticks])
        pct_returns = np.array([(t.price - base_price) / base_price for t in ticks])
        times = times - times[0]
        a, _, _ = np.polyfit(times, pct_returns, 2)
        return float(2 * a)


# ---------------------------------------------------------------------------
# بخش ۲.۲ — بافر بلند برای Spike Z-Score و Stall Detection
# ---------------------------------------------------------------------------
class TickHistory:
    """
    بافر بلندتر (پیش‌فرض ۲۰۰ تیک) که مبنای آماری برای دو ویژگی مهم فراهم می‌کند:

        - Spike Z-Score: حرکت لحظهٔ فعلی چند برابر انحراف‌معیار حرکات عادی اخیر است؟
        - Stall Detection: نقاط توقف/برگشت قیمت در بین تیک‌های کندل جاری.
    """

    def __init__(self, maxlen: int = config.TICK_HISTORY_SIZE):
        self.buffer: deque[Tick] = deque(maxlen=maxlen)

    def add(self, tick: Tick) -> None:
        self.buffer.append(tick)

    def _pct_returns(self) -> list[float]:
        ticks = list(self.buffer)
        returns = []
        for i in range(1, len(ticks)):
            if ticks[i - 1].price != 0:
                returns.append((ticks[i].price - ticks[i - 1].price) / ticks[i - 1].price)
        return returns

    def get_spike_zscore(self) -> float:
        """
        z-score بازدهٔ درصدی آخرین تیک نسبت به میانگین/انحراف‌معیار کل بافر.
        قدرمطلق بزرگ یعنی این حرکت لحظه‌ای نسبت به رفتار عادی اخیر «غیرعادی» است.
        """
        returns = self._pct_returns()
        if len(returns) < 5:
            return 0.0
        mean = statistics.fmean(returns)
        std = statistics.pstdev(returns)
        if std == 0:
            return 0.0
        return (returns[-1] - mean) / std

    def get_stall_features(self, candle_start_time: float, candle_high: float, candle_low: float) -> dict:
        """
        تعداد نقاط توقف/برگشت قیمت (تغییر جهت تیک به تیک) در بازهٔ کندل جاری، و
        موقعیت نرمال‌شدهٔ (۰ تا ۱ نسبت به های/لوی همان کندل) آخرین نقطهٔ توقف.
        """
        ticks_in_candle = [t for t in self.buffer if t.timestamp >= candle_start_time]
        if len(ticks_in_candle) < 3:
            return {"stall_count_in_candle": 0, "last_stall_position_in_candle": None}

        stall_count = 0
        last_stall_price = None
        prev_direction = None
        for i in range(1, len(ticks_in_candle)):
            diff = ticks_in_candle[i].price - ticks_in_candle[i - 1].price
            if diff == 0:
                continue
            direction = 1 if diff > 0 else -1
            if prev_direction is not None and direction != prev_direction:
                stall_count += 1
                last_stall_price = ticks_in_candle[i - 1].price
            prev_direction = direction

        candle_range = candle_high - candle_low
        position = None
        if last_stall_price is not None and candle_range > 0:
            position = (last_stall_price - candle_low) / candle_range

        return {"stall_count_in_candle": stall_count, "last_stall_position_in_candle": position}


# ---------------------------------------------------------------------------
# بخش ۲.۳ — ساخت کندل یک‌دقیقه‌ای و استخراج ویژگی‌های شکلی نسبی آن
# ---------------------------------------------------------------------------
@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float  # آخرین قیمت ثبت‌شده (اگر کندل هنوز در حال شکل‌گیری باشد، close = قیمت جاری)
    start_time: float

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def body_to_wick_ratio(self) -> float:
        """نسبت طول بدنه به مجموع سایه‌ها. عدد بزرگ=کندل روند‌دار؛ نزدیک صفر=دوجی."""
        total_wick = self.upper_wick + self.lower_wick
        if total_wick <= 0:
            return float(self.body > 0) * 999.0
        return self.body / total_wick

    def as_dict(self, prefix: str) -> dict:
        """
        فقط ویژگی‌های شکلیِ نسبی کندل — بدون هیچ مقدار خام OHLC. نسبت‌های بدنه/سایه
        به‌طور ذاتی مستقل از سطح قیمت‌اند (چون صورت و مخرج هر دو به همان واحد قیمتند).
        """
        rng = self.range
        return {
            f"{prefix}_is_bullish": int(self.is_bullish),
            f"{prefix}_body_to_wick_ratio": self.body_to_wick_ratio,
            f"{prefix}_upper_wick_ratio": (self.upper_wick / rng) if rng > 0 else 0.0,
            f"{prefix}_lower_wick_ratio": (self.lower_wick / rng) if rng > 0 else 0.0,
        }


def compare_recent_candles(history: "deque[Candle]", n: int = 3) -> dict:
    """
    مقایسهٔ نسبی هر کندل تکمیل‌شده با کندل قبل از خودش، برای n کندل اخیر:
    نسبت اندازه (Range) و هم‌رنگ بودن یا نبودن. اندیس ۱=جدیدترین جفت.
    """
    features: dict = {}
    candles = list(history)
    for i in range(1, n + 1):
        key_size = f"candle_size_ratio_prev{i}"
        key_color = f"candle_color_match_prev{i}"
        if len(candles) > i:
            newer, older = candles[-i], candles[-i - 1]
            features[key_size] = (newer.range / older.range) if older.range > 0 else None
            features[key_color] = int(newer.is_bullish == older.is_bullish)
        else:
            features[key_size] = None
            features[key_color] = None
    return features


class CandleAggregator:
    """
    از روی جریان تیک‌های ورودی، کندل‌های OHLC با تایم‌فریم مشخص (پیش‌فرض ۱ دقیقه)
    می‌سازد. همیشه یک کندل «در حال شکل‌گیری» (current) و تاریخچهٔ کندل‌های تکمیل‌شده
    (history) در دسترس است. هر بار که یک کندل تمام شود، همان کندلِ بسته‌شده از
    add_tick برگردانده می‌شود تا ماژول‌های دیگر (مثل ساختار بازار) از آن مطلع شوند.
    """

    def __init__(
        self,
        timeframe_seconds: int = config.CANDLE_TIMEFRAME_SECONDS,
        history_size: int = config.CANDLE_HISTORY_SIZE,
    ):
        self.timeframe_seconds = timeframe_seconds
        self.history: deque[Candle] = deque(maxlen=history_size)
        self.current: Optional[Candle] = None
        self._current_bucket: Optional[int] = None

    def add_tick(self, tick: Tick) -> Optional[Candle]:
        bucket = int(tick.timestamp // self.timeframe_seconds)

        if self._current_bucket is None:
            self._start_new_candle(tick, bucket)
            return None

        if bucket != self._current_bucket:
            closed_candle = self.current
            self.history.append(closed_candle)
            self._start_new_candle(tick, bucket)
            return closed_candle
        else:
            self.current.high = max(self.current.high, tick.price)
            self.current.low = min(self.current.low, tick.price)
            self.current.close = tick.price
            return None

    def _start_new_candle(self, tick: Tick, bucket: int) -> None:
        self._current_bucket = bucket
        self.current = Candle(
            open=tick.price,
            high=tick.price,
            low=tick.price,
            close=tick.price,
            start_time=bucket * self.timeframe_seconds,
        )

    def get_previous_candle(self) -> Optional[Candle]:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# بخش ۲.۴ — ترکیب ویژگی‌های تیک/کندل در یک دیکشنری واحد (بدون قیمت خام)
# ---------------------------------------------------------------------------
def build_feature_snapshot(
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
) -> dict:
    """
    تمام ویژگی‌های نسبیِ سطح تیک/کندل را در یک دیکشنری جمع می‌کند. ویژگی‌های
    ساختار بازار (سوئینگ/لگ/حمایت‌مقاومت/روند) جداگانه توسط
    market_structure.MarketStructureTracker.get_features() ساخته و در
    DataLogger با این خروجی ترکیب می‌شوند.
    """
    latest = tick_buffer.latest()
    snapshot: dict = {
        "velocity_pct": tick_buffer.get_velocity_pct(),
        "acceleration_pct": tick_buffer.get_acceleration_pct(),
        "velocity_pct_smoothed": tick_buffer.get_velocity_pct_smoothed(),
        "acceleration_pct_smoothed": tick_buffer.get_acceleration_pct_smoothed(),
        "spike_zscore": tick_history.get_spike_zscore(),
    }

    current_candle = candle_aggregator.current
    if current_candle and latest:
        rng = current_candle.range
        snapshot["price_position_in_candle"] = (
            (latest.price - current_candle.low) / rng if rng > 0 else 0.5
        )
        snapshot["distance_from_open_ratio"] = (
            (latest.price - current_candle.open) / rng if rng > 0 else 0.0
        )
        snapshot.update(current_candle.as_dict("candle_curr"))
        snapshot.update(
            tick_history.get_stall_features(current_candle.start_time, current_candle.high, current_candle.low)
        )

        # هم‌جهتی حرکت لحظه‌ای (تیک) با جهت کلی کندل در حال شکل‌گیری:
        # اگر مخالف هم باشند، ممکن است نشانهٔ خستگی/برگشت حرکت باشد.
        velocity_sign = 0
        if snapshot["velocity_pct"] > 0:
            velocity_sign = 1
        elif snapshot["velocity_pct"] < 0:
            velocity_sign = -1
        candle_sign = 1 if current_candle.is_bullish else -1
        snapshot["tick_vs_candle_alignment"] = (
            1 if velocity_sign == candle_sign else (-1 if velocity_sign != 0 else 0)
        )
    else:
        snapshot["price_position_in_candle"] = None
        snapshot["distance_from_open_ratio"] = None
        snapshot["stall_count_in_candle"] = None
        snapshot["last_stall_position_in_candle"] = None
        snapshot["tick_vs_candle_alignment"] = None

    if candle_aggregator.get_previous_candle():
        snapshot.update(candle_aggregator.get_previous_candle().as_dict("candle_prev1"))

    snapshot.update(compare_recent_candles(candle_aggregator.history, n=3))

    return snapshot
