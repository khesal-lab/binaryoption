"""
بخش دوم: پردازش داده‌ها و محاسبات فنی لحظه‌ای (Real-time Feature Engineering)
=============================================================================

این ماژول شامل دو بخش اصلی است:

    ۱. TickBuffer:
       صف (Buffer) آخرین ۱۰ تیک قیمت که سرعت (Velocity) و شتاب (Acceleration)
       تغییرات قیمت را از روی آن‌ها محاسبه می‌کند.

    ۲. CandleAggregator:
       از روی همان تیک‌ها، کندل‌های یک‌دقیقه‌ای (Open/High/Low/Close) را در لحظه
       می‌سازد و ویژگی‌های کندل جاری و کندل قبلی (نسبت بدنه به سایه و غیره) را
       استخراج می‌کند.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.browser_session import Tick
import config


# ---------------------------------------------------------------------------
# بخش ۲.۱ — بافر تیک و محاسبهٔ سرعت/شتاب
# ---------------------------------------------------------------------------
class TickBuffer:
    """
    نگه‌دارندهٔ آخرین N تیک قیمت (پیش‌فرض ۱۰ تیک، طبق config.TICK_BUFFER_SIZE).
    از روی این بافر، سرعت و شتاب لحظه‌ای تغییر قیمت محاسبه می‌شود.
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
        این تابع برای بررسی نتیجهٔ معامله بعد از ۳ ثانیه از لحظهٔ ورود استفاده می‌شود.
        """
        candidates = [t for t in self.buffer if t.timestamp >= timestamp]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.timestamp - timestamp).price

    # -- سرعت (Velocity) -----------------------------------------------------
    def get_velocity(self) -> float:
        """
        سرعت لحظه‌ای = تغییر قیمت / تغییر زمان، بین دو تیک آخر.
        واحد: قیمت بر ثانیه.
        """
        if len(self.buffer) < 2:
            return 0.0
        t_prev, t_curr = self.buffer[-2], self.buffer[-1]
        dt = t_curr.timestamp - t_prev.timestamp
        if dt <= 0:
            return 0.0
        return (t_curr.price - t_prev.price) / dt

    def _velocity_series(self) -> list[float]:
        """سرعت بین هر دو تیک متوالی داخل بافر؛ برای محاسبهٔ شتاب لازم است."""
        ticks = list(self.buffer)
        velocities = []
        for i in range(1, len(ticks)):
            dt = ticks[i].timestamp - ticks[i - 1].timestamp
            if dt > 0:
                velocities.append((ticks[i].price - ticks[i - 1].price) / dt)
        return velocities

    # -- شتاب (Acceleration) -------------------------------------------------
    def get_acceleration(self) -> float:
        """
        شتاب لحظه‌ای = تغییر سرعت / تغییر زمان، بین دو مقدار سرعت آخر.
        شتاب مثبت یعنی حرکت صعودی/نزولی در حال «شدت گرفتن» است.
        شتاب منفی معمولاً نشانهٔ نزدیک شدن به نقطهٔ توقف/بازگشت قیمت است —
        دقیقاً همان چیزی که شما در تحلیل دستی خود دنبال می‌کنید.
        """
        velocities = self._velocity_series()
        if len(velocities) < 2:
            return 0.0
        ticks = list(self.buffer)
        dt = ticks[-1].timestamp - ticks[-2].timestamp
        if dt <= 0:
            return 0.0
        return (velocities[-1] - velocities[-2]) / dt

    # -- نسخهٔ هموارشده با رگرسیون (اختیاری، برای کاهش نویز) -----------------
    def get_velocity_regression(self) -> float:
        """
        به‌جای استفاده از فقط دو تیک آخر، شیب خط رگرسیون خطی روی کل بافر
        (۱۰ تیک) را به‌عنوان سرعت هموارشده برمی‌گرداند. نویز کمتر، تأخیر بیشتر.
        """
        if len(self.buffer) < 3:
            return self.get_velocity()
        times = np.array([t.timestamp for t in self.buffer])
        prices = np.array([t.price for t in self.buffer])
        times = times - times[0]  # برای پایداری عددی
        slope, _ = np.polyfit(times, prices, 1)
        return float(slope)

    def get_acceleration_regression(self) -> float:
        """شتاب هموارشده با برازش سهمی (درجهٔ دوم) روی کل بافر."""
        if len(self.buffer) < 4:
            return self.get_acceleration()
        times = np.array([t.timestamp for t in self.buffer])
        prices = np.array([t.price for t in self.buffer])
        times = times - times[0]
        a, _, _ = np.polyfit(times, prices, 2)
        return float(2 * a)  # مشتق دوم تابع درجه دو: 2a


# ---------------------------------------------------------------------------
# بخش ۲.۲ — ساخت کندل یک‌دقیقه‌ای و استخراج ویژگی‌های آن
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
        """
        نسبت طول بدنه به مجموع سایه‌ها. عدد بزرگ یعنی کندل قدرتمند و روند‌دار
        (بدنهٔ بزرگ، سایهٔ کوچک)؛ عدد نزدیک صفر یعنی بلاتکلیفی/دوجی.
        """
        total_wick = self.upper_wick + self.lower_wick
        if total_wick <= 0:
            return float(self.body > 0) * 999.0  # بدنه بدون هیچ سایه‌ای
        return self.body / total_wick

    def as_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}_open": self.open,
            f"{prefix}_high": self.high,
            f"{prefix}_low": self.low,
            f"{prefix}_close": self.close,
            f"{prefix}_is_bullish": int(self.is_bullish),
            f"{prefix}_body": self.body,
            f"{prefix}_upper_wick": self.upper_wick,
            f"{prefix}_lower_wick": self.lower_wick,
            f"{prefix}_body_to_wick_ratio": self.body_to_wick_ratio,
        }


class CandleAggregator:
    """
    از روی جریان تیک‌های ورودی، کندل‌های OHLC با تایم‌فریم مشخص (پیش‌فرض ۱ دقیقه)
    می‌سازد. همیشه یک کندل «در حال شکل‌گیری» (current) و چند کندل تکمیل‌شدهٔ
    قبلی (history) در دسترس است.
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

    def add_tick(self, tick: Tick) -> None:
        bucket = int(tick.timestamp // self.timeframe_seconds)

        if self._current_bucket is None:
            self._start_new_candle(tick, bucket)
            return

        if bucket != self._current_bucket:
            # کندل جاری تمام شده؛ آن را به تاریخچه منتقل می‌کنیم و کندل جدید می‌سازیم
            self.history.append(self.current)
            self._start_new_candle(tick, bucket)
        else:
            self.current.high = max(self.current.high, tick.price)
            self.current.low = min(self.current.low, tick.price)
            self.current.close = tick.price

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
# بخش ۲.۳ — ترکیب همهٔ ویژگی‌ها در یک دیکشنری واحد
# ---------------------------------------------------------------------------
def build_feature_snapshot(tick_buffer: TickBuffer, candle_aggregator: CandleAggregator) -> dict:
    """
    تمام ویژگی‌های فنی لحظه را در یک دیکشنری جمع می‌کند. این خروجی مستقیماً
    به‌عنوان ستون‌های یک ردیف از دیتاست آموزشی استفاده می‌شود.
    """
    latest = tick_buffer.latest()
    snapshot = {
        "price": latest.price if latest else None,
        "timestamp": latest.timestamp if latest else None,
        "symbol": latest.symbol if latest else None,
        "velocity": tick_buffer.get_velocity(),
        "acceleration": tick_buffer.get_acceleration(),
        "velocity_smoothed": tick_buffer.get_velocity_regression(),
        "acceleration_smoothed": tick_buffer.get_acceleration_regression(),
    }

    if candle_aggregator.current:
        snapshot.update(candle_aggregator.current.as_dict("candle_curr"))
    if candle_aggregator.get_previous_candle():
        snapshot.update(candle_aggregator.get_previous_candle().as_dict("candle_prev"))

    return snapshot
