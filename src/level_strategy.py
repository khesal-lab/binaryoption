"""
استراتژی ثابت (Rule-Based) برای ربات مخصوص جمع‌آوری داده
================================================================

استراتژی: کندل جاری در حال شکل‌گیری معیار است.

داخل این کندل، نقاط توقف (اکسترمم‌های محلی) شناسایی می‌شوند. دو سطح
کلیدی انتخاب می‌شوند: نزدیک‌ترین به سقف (HIGH) و نزدیک‌ترین به کف (LOW).
وقتی قیمت یکی از این سطوح را لمس کرد، جهت معامله به این صورت تعیین می‌شود:

    حالت ۱ — تمام نقاط توقف زیر اپن (ناحیه نزولی):
        هر دو سطح -> PUT (موافق جهت نزول)

    حالت ۲ — تمام نقاط توقف بالای اپن (ناحیه صعودی):
        هر دو سطح -> CALL (موافق جهت صعود)

    حالت ۳ — نقاط توقف هم بالای اپن هم پایین اپن (نوسان دو طرف اپن):
        نزدیک‌ترین به سقف -> PUT  (نزدیک مقاومت)
        نزدیک‌ترین به کف  -> CALL (نزدیک حمایت)

هر سطح فقط یک‌بار در طول عمر کندل جاری فعال می‌شود. با بسته‌شدن کندل
و شروع کندل جدید، سطوح پاک شده و از نو محاسبه می‌شوند.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.browser_session import Tick
from src.feature_engineering import (
    Candle,
    TickHistory,
    classify_region_vs_open,
    find_local_extrema,
    level_strategy_direction_for,
)
import config


@dataclass
class LevelStrategyTracker:
    match_tolerance_ratio: float = config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO

    _traded_levels: list = field(default_factory=list, init=False)

    def on_new_candle(self, closed_candle: Candle) -> None:
        """با شروع کندل جدید، تمام سطوح معامله‌شده پاک می‌شوند."""
        self._traded_levels = []

    def reset(self) -> None:
        """مثل on_new_candle، اما برای مواقعی که کندل قبلی اصلاً معتبر نیست (مثلاً تعویض نماد)."""
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

        # استخراج همهٔ اکسترمم‌های محلی (نقاط توقف) داخل کندل جاری - منطق
        # مشترک با compute_level_strategy_context در feature_engineering.py
        # (تا تصمیم واقعی و فیچر ثبت‌شده هیچ‌وقت از هم جدا نشوند).
        extrema = find_local_extrema(ticks_in_candle)

        if not extrema:
            return None

        tolerance = candle_range * self.match_tolerance_ratio
        open_price = current_candle.open

        def to_bucket(price: float) -> int:
            return round(price / tolerance) if tolerance > 0 else round(price * 1_000_000)

        region = classify_region_vs_open(extrema, open_price)

        # نزدیک‌ترین نقطه توقف به سقف و کف کندل
        near_high = min(extrema, key=lambda p: abs(current_candle.high - p))
        near_low = min(extrema, key=lambda p: abs(p - current_candle.low))

        key_high = ("near_high", to_bucket(near_high))
        key_low = ("near_low", to_bucket(near_low))

        # لمس سطح نزدیک سقف
        if key_high not in self._traded_levels:
            if (prev_tick.price < near_high <= latest_tick.price or
                    prev_tick.price > near_high >= latest_tick.price):
                self._traded_levels.append(key_high)
                return level_strategy_direction_for(region, near_high_stall=True)

        # لمس سطح نزدیک کف
        if key_low not in self._traded_levels:
            if (prev_tick.price < near_low <= latest_tick.price or
                    prev_tick.price > near_low >= latest_tick.price):
                self._traded_levels.append(key_low)
                return level_strategy_direction_for(region, near_high_stall=False)

        return None
