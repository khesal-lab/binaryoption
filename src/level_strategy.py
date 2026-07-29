"""
استراتژی ثابت (Rule-Based) برای ربات مخصوص جمع‌آوری داده
================================================================

استراتژی: کندل جاری در حال شکل‌گیری معیار است.

داخل این کندل، قیمت بین سقف (HIGH) و کف (LOW) نوسان می‌کند و در نقاطی
متوقف/برمی‌گردد (اکسترمم‌های محلی = نقاط توقف). از میان این نقاط دو سطح
کلیدی انتخاب می‌شوند:

    • نزدیک‌ترین نقطه توقف به سقف کندل جاری  ->  PUT
      (نزدیک مقاومت؛ هنگامی که قیمت این سطح را لمس کند)

    • نزدیک‌ترین نقطه توقف به کف کندل جاری  ->  CALL
      (نزدیک حمایت؛ هنگامی که قیمت این سطح را لمس کند)

هر سطح فقط یک‌بار در طول عمر کندل جاری فعال می‌شود. با بسته‌شدن کندل
و شروع کندل جدید، سطوح پاک شده و از نو از روی داده‌های کندل جدید
محاسبه می‌شوند.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.browser_session import Tick
from src.feature_engineering import Candle, TickHistory
import config


@dataclass
class LevelStrategyTracker:
    match_tolerance_ratio: float = config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO

    _traded_levels: list = field(default_factory=list, init=False)

    def on_new_candle(self, closed_candle: Candle) -> None:
        """با شروع کندل جدید، تمام سطوح معامله‌شده پاک می‌شوند."""
        self._traded_levels = []

    def check_signal(
        self,
        prev_tick: Optional[Tick],
        latest_tick: Tick,
        current_candle: Candle,
        tick_history: TickHistory,
    ) -> Optional[str]:
        """
        سطوح کلیدی کندل جاری را بررسی می‌کند. اگر قیمت به نزدیک‌ترین نقطه
        توقف به سقف یا کف کندل برسد، سیگنال PUT یا CALL برمی‌گرداند.
        """
        return self._check_edge_stalls(prev_tick, latest_tick, current_candle, tick_history)

    def _check_edge_stalls(
        self,
        prev_tick: Optional[Tick],
        latest_tick: Tick,
        current_candle: Candle,
        tick_history: TickHistory,
    ) -> Optional[str]:
        if prev_tick is None:
            return None

        candle_range = current_candle.range
        if candle_range <= 0:
            return None

        ticks_in_candle = [
            t for t in tick_history.buffer if t.timestamp >= current_candle.start_time
        ]
        if len(ticks_in_candle) < 3:
            return None

        # استخراج همهٔ اکسترمم‌های محلی (نقاط توقف) داخل کندل جاری
        extrema: list[float] = []
        prev_dir: Optional[int] = None
        for i in range(1, len(ticks_in_candle)):
            diff = ticks_in_candle[i].price - ticks_in_candle[i - 1].price
            if diff == 0:
                continue
            d = 1 if diff > 0 else -1
            if prev_dir is not None and d != prev_dir:
                extrema.append(ticks_in_candle[i - 1].price)
            prev_dir = d

        if not extrema:
            return None

        tolerance = candle_range * self.match_tolerance_ratio

        def to_bucket(price: float) -> int:
            return round(price / tolerance) if tolerance > 0 else round(price * 1_000_000)

        # نزدیک‌ترین نقطه توقف به سقف کندل -> نزدیک مقاومت -> PUT
        near_high = min(extrema, key=lambda p: abs(current_candle.high - p))
        # نزدیک‌ترین نقطه توقف به کف کندل -> نزدیک حمایت -> CALL
        near_low = min(extrema, key=lambda p: abs(p - current_candle.low))

        key_put = ("near_high", to_bucket(near_high))
        key_call = ("near_low", to_bucket(near_low))

        # لمس سطح نزدیک سقف -> PUT
        if key_put not in self._traded_levels:
            if (prev_tick.price < near_high <= latest_tick.price or
                    prev_tick.price > near_high >= latest_tick.price):
                self._traded_levels.append(key_put)
                return "PUT"

        # لمس سطح نزدیک کف -> CALL
        if key_call not in self._traded_levels:
            if (prev_tick.price < near_low <= latest_tick.price or
                    prev_tick.price > near_low >= latest_tick.price):
                self._traded_levels.append(key_call)
                return "CALL"

        return None
