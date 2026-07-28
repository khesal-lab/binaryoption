"""
ساختار بازار: سوئینگ، لگ، حمایت/مقاومت، روند و شکل نسبی چارت
================================================================

این ماژول تمام مفاهیمی را که یک تریدر باتجربه با نگاه به چارت (و نه با نگاه به
مقدار خام قیمت) تشخیص می‌دهد، به‌صورت عددی و **کاملاً نسبی** محاسبه می‌کند:

    - سوئینگ‌های قیمتی (Swing High/Low) با روش فرکتال
    - لگ (Leg) فعلی: جهت، طول زمانی، اندازه، و نسبت آن به لگ‌های قبلی
    - نزدیک‌ترین حمایت/مقاومت اخیر و فاصلهٔ نرمال‌شده با ATR تا آن‌ها
    - آیا های/لوی کندل قبلی یا سوئینگ اخیر شکسته شده (Breakout)
    - درصد بازگشت فیبوناچی لگ فعلی نسبت به لگ قبلی
    - رژیم روند (صعودی/نزولی/رنج) بر اساس توالی سوئینگ‌ها
    - تعداد کندل‌های هم‌رنگ متوالی (Streak)
    - نسبت انبساط/انقباض نوسان (ATR کوتاه به ATR بلند)
    - عدم‌تقارن سایه‌ها نسبت به میانگین اخیر
    - شکل نرمال‌شدهٔ ۲۰ کندل اخیر چارت (برای یادگیری الگوهای بزرگ توسط مدل)

هیچ‌کدام از خروجی‌های این ماژول مقدار خام قیمت نیستند؛ همه یا نسبت‌اند، یا
درصد، یا پرچم صفر/یک، یا برچسب رژیم (طبقه‌بندی).
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

import config
from src.feature_engineering import Candle

SwingKind = Literal["high", "low"]


@dataclass
class SwingPoint:
    index: int          # اندیس کندل در تاریخچهٔ داخلی (برای دیباگ)
    price: float
    timestamp: float
    kind: SwingKind


@dataclass
class Leg:
    start_price: float
    start_time: float
    direction: Literal["up", "down"]

    def range(self, current_price: float) -> float:
        return abs(current_price - self.start_price)

    def signed_range(self, current_price: float) -> float:
        return current_price - self.start_price


def _true_range(candle: Candle, prev_close: Optional[float]) -> float:
    if prev_close is None:
        return candle.high - candle.low
    return max(
        candle.high - candle.low,
        abs(candle.high - prev_close),
        abs(candle.low - prev_close),
    )


class MarketStructureTracker:
    """
    این کلاس به ازای هر کندل تکمیل‌شده (نه هر تیک) به‌روزرسانی می‌شود
    (`ingest_closed_candle`) و در هر لحظه که بخواهیم (مثلاً لحظهٔ ورود به معامله)
    می‌توانیم `get_features(current_price, current_time)` را صدا بزنیم تا
    وضعیت لحظه‌ای «نسبت به ساختار چارت» را دریافت کنیم.
    """

    def __init__(
        self,
        history_size: int = config.CANDLE_HISTORY_SIZE,
        chart_window: int = config.CHART_WINDOW_CANDLES,
        fractal_k: int = config.SWING_FRACTAL_K,
        sr_lookback_swings: int = config.SR_LOOKBACK_SWINGS,
        atr_period_short: int = config.ATR_PERIOD_SHORT,
        atr_period_long: int = config.ATR_PERIOD_LONG,
        leg_ratio_lookback: int = config.LEG_RATIO_LOOKBACK,
        wick_asymmetry_lookback: int = config.WICK_ASYMMETRY_LOOKBACK,
        trend_swing_lookback: int = config.TREND_SWING_LOOKBACK,
    ):
        self.chart_window = chart_window
        self.fractal_k = fractal_k
        self.sr_lookback_swings = sr_lookback_swings
        self.atr_period_short = atr_period_short
        self.atr_period_long = atr_period_long
        self.leg_ratio_lookback = leg_ratio_lookback
        self.wick_asymmetry_lookback = wick_asymmetry_lookback
        self.trend_swing_lookback = trend_swing_lookback

        self.candles: deque[Candle] = deque(maxlen=history_size)
        self.swing_points: deque[SwingPoint] = deque(maxlen=50)
        self._last_confirmed_index: int = -1
        self._candle_counter: int = -1  # اندیس افزایشی کل کندل‌های دیده‌شده

    # -- ورودی: هر بار یک کندل تکمیل شود این متد صدا زده می‌شود ------------------
    def ingest_closed_candle(self, candle: Candle) -> None:
        self.candles.append(candle)
        self._candle_counter += 1
        self._try_confirm_swing()

    def _try_confirm_swing(self) -> None:
        """
        با روش فرکتال (k=1 پیش‌فرض): کندل «وسط» سوئینگ است اگر های/لوی آن از
        هر دو همسایهٔ چپ و راستش بیشتر/کمتر باشد. چون همسایهٔ راست باید از قبل
        وجود داشته باشد، تشخیص سوئینگ همیشه با یک کندل تأخیر تأیید می‌شود —
        این تأخیر ذاتی و غیرقابل‌اجتناب است (سوئینگ را فقط بعد از گذشتنش می‌شود فهمید).
        """
        k = self.fractal_k
        window = 2 * k + 1
        if len(self.candles) < window:
            return

        candles_list = list(self.candles)
        mid_pos = len(candles_list) - 1 - k  # اندیس کندل میانی در همین لیست
        mid_candle_global_index = self._candle_counter - k
        if mid_candle_global_index <= self._last_confirmed_index:
            return  # قبلاً بررسی شده

        mid = candles_list[mid_pos]
        left = candles_list[mid_pos - k: mid_pos]
        right = candles_list[mid_pos + 1: mid_pos + 1 + k]

        is_swing_high = all(mid.high > c.high for c in left) and all(mid.high > c.high for c in right)
        is_swing_low = all(mid.low < c.low for c in left) and all(mid.low < c.low for c in right)

        if is_swing_high:
            self.swing_points.append(SwingPoint(mid_candle_global_index, mid.high, mid.start_time, "high"))
        if is_swing_low:
            self.swing_points.append(SwingPoint(mid_candle_global_index, mid.low, mid.start_time, "low"))

        self._last_confirmed_index = mid_candle_global_index

    # -- محاسبات کمکی --------------------------------------------------------
    def _atr(self, period: int) -> Optional[float]:
        candles_list = list(self.candles)
        if len(candles_list) < 2:
            return None
        trs = []
        for i in range(1, len(candles_list)):
            trs.append(_true_range(candles_list[i], candles_list[i - 1].close))
        window = trs[-period:] if len(trs) >= 1 else trs
        if not window:
            return None
        return sum(window) / len(window)

    def _last_swings_by_kind(self, kind: SwingKind, n: int) -> list[SwingPoint]:
        matches = [p for p in self.swing_points if p.kind == kind]
        return matches[-n:]

    def _current_leg(self, current_price: float) -> Optional[Leg]:
        if not self.swing_points:
            return None
        last_swing = self.swing_points[-1]
        direction = "up" if current_price >= last_swing.price else "down"
        return Leg(start_price=last_swing.price, start_time=last_swing.timestamp, direction=direction)

    def _streak(self) -> int:
        """تعداد کندل‌های هم‌رنگ متوالی؛ عدد مثبت=سبز، منفی=قرمز."""
        candles_list = list(self.candles)
        if not candles_list:
            return 0
        last_color = candles_list[-1].is_bullish
        count = 0
        for c in reversed(candles_list):
            if c.is_bullish == last_color:
                count += 1
            else:
                break
        return count if last_color else -count

    def _wick_asymmetry(self, candle: Candle) -> float:
        total = candle.upper_wick + candle.lower_wick
        if total <= 0:
            return 0.0
        return (candle.upper_wick - candle.lower_wick) / total

    def get_normalized_chart_shape(self) -> list[dict]:
        """
        شکل نرمال‌شدهٔ N کندل اخیر (پیش‌فرض ۲۰ تا) — هر مقدار بین ۰ و ۱ نسبت به
        بازهٔ های/لوی کل پنجره. اگر قیمت جاری از این بازه فراتر برود (Breakout از
        محدودهٔ دیداری چارت)، عدد نرمال‌شده می‌تواند بزرگ‌تر از ۱ یا کوچک‌تر از ۰
        شود — این خودش یک سیگنال معنادار است، پس عمداً Clip نمی‌شود.
        """
        candles_list = list(self.candles)[-self.chart_window:]
        if not candles_list:
            return []
        window_high = max(c.high for c in candles_list)
        window_low = min(c.low for c in candles_list)
        span = window_high - window_low
        if span <= 0:
            return []

        def norm(v: float) -> float:
            return (v - window_low) / span

        return [
            {
                "o": norm(c.open),
                "h": norm(c.high),
                "l": norm(c.low),
                "c": norm(c.close),
                "bullish": int(c.is_bullish),
            }
            for c in candles_list
        ]

    def _swing_range(self) -> Optional[float]:
        """
        محدودهٔ (Range) سوئینگ فعلی: فاصلهٔ بین آخرین سوئینگ های و آخرین سوئینگ لو
        شناسایی‌شده — معیار «بزرگی» ساختار نوسانی جاری که همه‌چیز دیگر (اندازهٔ
        کندل، اندازهٔ لگ) نسبت به آن سنجیده می‌شود.
        """
        last_high = self._last_swings_by_kind("high", 1)
        last_low = self._last_swings_by_kind("low", 1)
        if not last_high or not last_low:
            return None
        rng = abs(last_high[-1].price - last_low[-1].price)
        return rng if rng > 0 else None

    # -- خروجی اصلی: ویژگی‌های ساختاری در لحظه --------------------------------
    def get_features(
        self,
        current_price: float,
        current_time: float,
        current_candle_range: Optional[float] = None,
    ) -> dict:
        features: dict = {}

        # --- آیا اصلاً سوئینگی شناسایی شده؟ (پیش‌نیاز همهٔ ویژگی‌های زیر) ---
        features["has_swing_data"] = int(bool(self.swing_points))
        swing_range = self._swing_range()
        features["swing_range_defined"] = int(swing_range is not None)

        # نسبت محدودهٔ کندل جاری به محدودهٔ سوئینگ: این کندل چه سهمی از کل نوسان دارد
        features["candle_range_to_swing_range_ratio"] = (
            current_candle_range / swing_range
            if (current_candle_range is not None and swing_range)
            else None
        )

        atr_short = self._atr(self.atr_period_short)
        atr_long = self._atr(self.atr_period_long)
        features["volatility_ratio_short_long"] = (
            atr_short / atr_long if (atr_short is not None and atr_long) else None
        )

        prev_candle = self.candles[-1] if self.candles else None

        # --- شکست های/لوی کندل قبلی (Breakout) ---
        if prev_candle:
            features["broke_prev_candle_high"] = int(current_price > prev_candle.high)
            features["broke_prev_candle_low"] = int(current_price < prev_candle.low)
        else:
            features["broke_prev_candle_high"] = None
            features["broke_prev_candle_low"] = None

        # --- حمایت/مقاومت اخیر ---
        recent_highs = self._last_swings_by_kind("high", self.sr_lookback_swings)
        recent_lows = self._last_swings_by_kind("low", self.sr_lookback_swings)
        resistances = [p.price for p in recent_highs if p.price > current_price]
        supports = [p.price for p in recent_lows if p.price < current_price]
        nearest_resistance = min(resistances) if resistances else None
        nearest_support = max(supports) if supports else None

        atr_ref = atr_long or atr_short
        features["dist_to_resistance_atr"] = (
            (nearest_resistance - current_price) / atr_ref if (nearest_resistance is not None and atr_ref) else None
        )
        features["dist_to_support_atr"] = (
            (current_price - nearest_support) / atr_ref if (nearest_support is not None and atr_ref) else None
        )
        features["broke_recent_swing_high"] = int(
            bool(recent_highs) and current_price > recent_highs[-1].price
        )
        features["broke_recent_swing_low"] = int(
            bool(recent_lows) and current_price < recent_lows[-1].price
        )

        # --- لگ فعلی و رژیم روند ---
        current_leg = self._current_leg(current_price)
        if current_leg:
            features["leg_direction"] = 1 if current_leg.direction == "up" else -1
            # طول لگ برحسب «تعداد کندل» به‌جای ثانیهٔ خام، تا مستقل از تایم‌فریم/سرعت اجرا باشد
            features["leg_length_in_candles"] = (
                (current_time - current_leg.start_time) / config.CANDLE_TIMEFRAME_SECONDS
                if config.CANDLE_TIMEFRAME_SECONDS
                else None
            )
            leg_range = current_leg.range(current_price)

            # نسبت محدودهٔ لگ فعلی به محدودهٔ سوئینگ (آیا این لگ نسبت به کل نوسان بزرگ است یا کوچک)
            features["leg_range_to_swing_range_ratio"] = (
                leg_range / swing_range if swing_range else None
            )

            # نسبت طول لگ فعلی به میانگین چند لگ قبلی (آیا لگ فعلی کشیده‌تر از حد معمول است)
            all_swings_sorted = sorted(self.swing_points, key=lambda p: p.timestamp)
            prev_ranges = []
            for i in range(len(all_swings_sorted) - 1, 0, -1):
                prev_ranges.append(abs(all_swings_sorted[i].price - all_swings_sorted[i - 1].price))
                if len(prev_ranges) >= self.leg_ratio_lookback:
                    break
            avg_prev_leg_range = sum(prev_ranges) / len(prev_ranges) if prev_ranges else None
            features["leg_extension_ratio"] = (
                leg_range / avg_prev_leg_range if avg_prev_leg_range else None
            )

            # درصد بازگشت فیبوناچی نسبت به لگ قبلی
            if len(self.swing_points) >= 2:
                prev_leg_range_signed = self.swing_points[-1].price - self.swing_points[-2].price
                if prev_leg_range_signed != 0:
                    features["fib_retracement_of_prev_leg"] = (
                        current_price - self.swing_points[-1].price
                    ) / prev_leg_range_signed
                else:
                    features["fib_retracement_of_prev_leg"] = None
            else:
                features["fib_retracement_of_prev_leg"] = None
        else:
            features["leg_direction"] = None
            features["leg_length_in_candles"] = None
            features["leg_range_to_swing_range_ratio"] = None
            features["leg_extension_ratio"] = None
            features["fib_retracement_of_prev_leg"] = None

        # --- رژیم روند: بر اساس آخرین سوئینگ‌های های/لو ---
        highs = self._last_swings_by_kind("high", self.trend_swing_lookback)
        lows = self._last_swings_by_kind("low", self.trend_swing_lookback)
        if len(highs) == self.trend_swing_lookback and len(lows) == self.trend_swing_lookback:
            higher_highs = highs[-1].price > highs[-2].price
            higher_lows = lows[-1].price > lows[-2].price
            lower_highs = highs[-1].price < highs[-2].price
            lower_lows = lows[-1].price < lows[-2].price
            if higher_highs and higher_lows:
                features["trend_regime"] = 1  # صعودی
            elif lower_highs and lower_lows:
                features["trend_regime"] = -1  # نزولی
            else:
                features["trend_regime"] = 0  # رنج/مخلوط
            features["trend_regime_ready"] = 1
        else:
            features["trend_regime"] = 0
            features["trend_regime_ready"] = 0  # هنوز سوئینگ کافی برای تشخیص روند نداریم

        # --- استریک رنگ کندل‌ها ---
        features["candle_color_streak"] = self._streak()

        # --- عدم‌تقارن سایه نسبت به میانگین اخیر ---
        if prev_candle:
            recent_candles = list(self.candles)[-self.wick_asymmetry_lookback:]
            avg_wick_asym = sum(self._wick_asymmetry(c) for c in recent_candles) / len(recent_candles)
            features["wick_asymmetry_current"] = self._wick_asymmetry(prev_candle)
            features["wick_asymmetry_relative"] = self._wick_asymmetry(prev_candle) - avg_wick_asym
        else:
            features["wick_asymmetry_current"] = None
            features["wick_asymmetry_relative"] = None

        # --- شکل نرمال‌شدهٔ چارت (برای یادگیری الگوی بصری توسط مدل) ---
        features["chart_shape_json"] = json.dumps(self.get_normalized_chart_shape())

        return features
