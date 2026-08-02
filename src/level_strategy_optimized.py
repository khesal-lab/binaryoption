"""
نسخهٔ بهینه‌شدهٔ استراتژی ثابت (level_strategy.py) - با یادگیری درون‌کندلی
================================================================================

نسخهٔ اصلی (src/level_strategy.py) کاملاً دست‌نخورده باقی می‌ماند؛ این فایل یک
نسخهٔ جایگزین است که با config.LEVEL_STRATEGY_VARIANT = "optimized" انتخاب
می‌شود (پیش‌فرض همچنان "basic" یعنی همان فایل قبلی).

تفاوت با نسخهٔ اصلی: نسخهٔ اصلی سه حالت دارد -

    ۱. همهٔ نقاط توقف زیر Open (ناحیهٔ نزولی) -> هر دو سطح PUT
    ۲. همهٔ نقاط توقف بالای Open (ناحیهٔ صعودی) -> هر دو سطح CALL
    ۳. نقاط توقف هر دو طرف Open (نوسان دوطرفه) -> نزدیک سقف=PUT، نزدیک کف=CALL

فرض حالت ۳ این است که لبه‌های کندل نقش مقاومت/حمایت دارند و قیمت از آن‌ها
برمی‌گردد. اما گاهی رنج کندل مدام از یک طرف در حال بزرگ‌تر شدن است - یعنی
همان لبه‌ای که رویش فرض «بازگشت» گذاشته بودیم، پیوسته شکسته می‌شود و معامله
پشت سرهم می‌بازد. در این حالت، ادامهٔ معامله در جهتِ همان شکستِ مداوم منطقی‌تر
از فرض بازگشت است.

این نسخه فقط برای حالت ۳ (نوسان دوطرفه): اگر config.LEVEL_STRATEGY_REVERSE_
AFTER_LOSSES معاملهٔ متوالیِ همین حالت در همین کندل باخت باشند، جهتِ باقیِ
معاملاتِ همین حالت در همین کندل معکوس می‌شود (تا کندل تمام شود و دوباره از
صفر حساب شود). حالت‌های ۱ و ۲ بدون هیچ تغییری باقی می‌مانند.

برای این‌که این کلاس بداند نتیجهٔ هر سیگنال (که با تأخیر TRADE_EXPIRY_SECONDS
مشخص می‌شود) برد بوده یا باخت، متد on_result(result) باید بعد از مشخص‌شدن هر
نتیجهٔ واقعی صدا زده شود (نگاه کنید به collect_data.py که این را از طریق
on_result_callback مربوط به DataLogger انجام می‌دهد).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.browser_session import Tick
from src.feature_engineering import Candle, TickHistory, classify_region_vs_open, find_local_extrema
import config

_OPPOSITE = {"CALL": "PUT", "PUT": "CALL"}


@dataclass
class OptimizedLevelStrategyTracker:
    match_tolerance_ratio: float = config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO
    reverse_after_losses: int = config.LEVEL_STRATEGY_REVERSE_AFTER_LOSSES

    _traded_levels: list = field(default_factory=list, init=False)
    # چند باخت متوالیِ حالت «نوسان دوطرفه» در همین کندل رخ داده؟
    _spans_open_loss_streak: int = field(default=0, init=False)
    # آیا در همین کندل، به‌خاطر رسیدن به آستانهٔ باخت متوالی، جهتِ حالت «نوسان
    # دوطرفه» معکوس شده؟ تا پایان همین کندل True می‌ماند (حتی اگر بعدش ببرد).
    _reversed_in_current_candle: bool = field(default=False, init=False)
    # به ازای هر سیگنال صادرشده، این‌که آیا حالت «نوسان دوطرفه» بود یا نه -
    # چون نتایج با تأخیر و به همان ترتیب صدور سیگنال‌ها می‌رسند (FIFO).
    _pending_signal_modes: deque = field(default_factory=deque, init=False)

    def on_new_candle(self, closed_candle: Candle) -> None:
        """با شروع کندل جدید، سطوح معامله‌شده و وضعیت معکوس‌سازیِ درون‌کندلی پاک می‌شوند."""
        self._traded_levels = []
        self._spans_open_loss_streak = 0
        self._reversed_in_current_candle = False

    def reset(self) -> None:
        """مثل on_new_candle، اما برای مواقعی که کندل قبلی اصلاً معتبر نیست (مثلاً تعویض نماد)."""
        self._traded_levels = []
        self._spans_open_loss_streak = 0
        self._reversed_in_current_candle = False
        self._pending_signal_modes.clear()

    def on_result(self, result: int) -> None:
        """
        بعد از مشخص‌شدن نتیجهٔ واقعیِ هر معامله (توسط DataLogger) صدا زده
        می‌شود. result باید ۱ (برد) یا ۰ (باخت) باشد. فقط معاملاتی که در حالت
        «نوسان دوطرفه» صادر شده باشند در شمارش/معکوس‌سازی مشارکت می‌کنند.
        """
        if not self._pending_signal_modes:
            return
        was_spans_open = self._pending_signal_modes.popleft()
        if not was_spans_open:
            return

        if result == 0:
            self._spans_open_loss_streak += 1
            if self._spans_open_loss_streak >= self.reverse_after_losses:
                self._reversed_in_current_candle = True
        else:
            self._spans_open_loss_streak = 0

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
        spans_open = region == 0

        def direction_for(near_high_stall: bool) -> str:
            """
            جهت معامله را بر اساس موقعیت نوسان نسبت به اپن برمی‌گرداند. فقط در
            حالت «نوسان دوطرفه»، اگر آستانهٔ باخت متوالی در همین کندل رد شده
            باشد، جهت معکوس می‌شود.
            """
            if spans_open:
                direction = "PUT" if near_high_stall else "CALL"
                if self._reversed_in_current_candle:
                    direction = _OPPOSITE[direction]
                return direction
            elif region == 1:
                return "CALL"
            else:
                return "PUT"

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
                self._pending_signal_modes.append(spans_open)
                return direction_for(near_high_stall=True)

        # لمس سطح نزدیک کف
        if key_low not in self._traded_levels:
            if (prev_tick.price < near_low <= latest_tick.price or
                    prev_tick.price > near_low >= latest_tick.price):
                self._traded_levels.append(key_low)
                self._pending_signal_modes.append(spans_open)
                return direction_for(near_high_stall=False)

        return None
