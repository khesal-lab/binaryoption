"""
استراتژی ثابت (Rule-Based) برای ربات مخصوص جمع‌آوری داده
================================================================

بر خلاف ربات معامله‌گر مبتنی بر مدل (src/live_predictor.py) که تصمیمش را از
یک مدل آموزش‌دیده می‌گیرد، این ماژول یک استراتژی کاملاً ثابت و از پیش
مشخص را پیاده می‌کند - هدف فقط جمع‌آوری دادهٔ متنوع و برچسب‌خورده (توسط خودِ
پلتفرم) است، نه سودآوری زنده. وین‌ریت این معاملات مهم نیست؛ مهم این است که
دقیقاً در نقاطی که یک تریدر معمولاً روی آن‌ها تصمیم می‌گیرد (تست حمایت/
مقاومت) نمونه با نتیجهٔ واقعی جمع شود.

دو نوع سطح شناسایی و تست می‌شود:

    ۱. کف/سقف کندلِ قبلیِ بسته‌شده: سقف = مقاومت (وقتی قیمت به آن می‌رسد،
       فرض بازگشت به پایین -> PUT)، کف = حمایت (وقتی قیمت به آن می‌رسد،
       فرض بازگشت به بالا -> CALL). هر سطح فقط یک‌بار در طول عمر همان کندل
       تست می‌شود (وقتی کندل جدید شروع شود، دوباره قابل تست است).

    ۲. نقاطی داخل کندلِ در حال شکل‌گیری که قیمت بیش از یک‌بار در آن‌جا متوقف/
       برگشته (یعنی یک اکسترمم محلی که تقریباً هم‌قیمت با یک اکسترمم قبلیِ
       هم‌نوع است) - این‌ها هم مثل مقاومت (اگر قله باشد -> PUT) یا حمایت
       (اگر دره باشد -> CALL) در نظر گرفته می‌شوند؛ هر سطح فقط یک‌بار تست
       می‌شود.
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

    _prev_candle: Optional[Candle] = field(default=None, init=False)
    _prev_high_tested: bool = field(default=False, init=False)
    _prev_low_tested: bool = field(default=False, init=False)
    _traded_intra_levels: list = field(default_factory=list, init=False)

    def on_new_candle(self, closed_candle: Candle) -> None:
        """
        وقتی یک کندل بسته می‌شود صدا زده می‌شود: کندلِ تازه‌بسته‌شده مرجع
        مقاومت/حمایت جدید می‌شود، و همهٔ پرچم‌های «تست‌شده» برای عمر کندل
        تازه صفر می‌شوند (چون هم مرجع کندل قبلی عوض شده، هم کندل جاری تازه
        شروع شده و اکسترمم‌های داخلی‌اش از نو ساخته می‌شوند).
        """
        self._prev_candle = closed_candle
        self._prev_high_tested = False
        self._prev_low_tested = False
        self._traded_intra_levels = []

    def check_signal(
        self,
        prev_tick: Optional[Tick],
        latest_tick: Tick,
        current_candle: Candle,
        tick_history: TickHistory,
    ) -> Optional[str]:
        """
        اگر لحظهٔ فعلی یکی از سطوح (مقاومت/حمایتِ کندل قبلی، یا یک نقطهٔ
        تکرارشدهٔ توقف داخل کندل جاری) را تازه لمس کرده باشد، جهت معاملهٔ
        متناظر ("CALL" یا "PUT") را برمی‌گرداند؛ در غیر این صورت None.
        """
        signal = self._check_prev_candle_levels(prev_tick, latest_tick)
        if signal is not None:
            return signal
        return self._check_intra_candle_stall_levels(current_candle, tick_history)

    def _check_prev_candle_levels(self, prev_tick: Optional[Tick], latest_tick: Tick) -> Optional[str]:
        if self._prev_candle is None or prev_tick is None:
            return None

        resistance = self._prev_candle.high
        support = self._prev_candle.low

        if not self._prev_high_tested and prev_tick.price < resistance <= latest_tick.price:
            self._prev_high_tested = True
            return "PUT"  # تست مقاومت -> فرض بازگشت به پایین

        if not self._prev_low_tested and prev_tick.price > support >= latest_tick.price:
            self._prev_low_tested = True
            return "CALL"  # تست حمایت -> فرض بازگشت به بالا

        return None

    def _check_intra_candle_stall_levels(
        self, current_candle: Candle, tick_history: TickHistory
    ) -> Optional[str]:
        candle_range = current_candle.range
        if candle_range <= 0:
            return None

        ticks_in_candle = [t for t in tick_history.buffer if t.timestamp >= current_candle.start_time]
        if len(ticks_in_candle) < 3:
            return None

        # همهٔ اکسترمم‌های محلی (قله/دره) کندل جاری تا همین لحظه:
        extrema: list[tuple[float, str]] = []
        prev_direction = None
        for i in range(1, len(ticks_in_candle)):
            diff = ticks_in_candle[i].price - ticks_in_candle[i - 1].price
            if diff == 0:
                continue
            direction = 1 if diff > 0 else -1
            if prev_direction is not None and direction != prev_direction:
                kind = "peak" if prev_direction == 1 else "trough"
                extrema.append((ticks_in_candle[i - 1].price, kind))
            prev_direction = direction

        if len(extrema) < 2:
            return None

        tolerance = candle_range * self.match_tolerance_ratio
        latest_price, latest_kind = extrema[-1]

        for earlier_price, earlier_kind in extrema[:-1]:
            if earlier_kind != latest_kind:
                continue
            if abs(latest_price - earlier_price) > tolerance:
                continue
            # این سطح قبلاً هم لمس شده - یعنی یک سطح تکرارشونده (حمایت/مقاومتِ
            # داخلی) است. با یک کلید سطلی (Bucket) از تست دوبارهٔ همان سطح
            # جلوگیری می‌کنیم.
            bucket = round(latest_price / tolerance) if tolerance > 0 else round(latest_price, 8)
            level_key = (latest_kind, bucket)
            if level_key in self._traded_intra_levels:
                continue
            self._traded_intra_levels.append(level_key)
            return "PUT" if latest_kind == "peak" else "CALL"

        return None
