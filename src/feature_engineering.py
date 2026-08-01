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

import json
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

    def reset(self) -> None:
        """خالی‌کردن کامل بافر - مثلاً وقتی نماد معاملاتی عوض شده و تیک‌های قبلی دیگر معتبر نیستند."""
        self.buffer.clear()

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

    def get_recent_velocities_pct(self) -> list[float]:
        """
        دنبالهٔ سرعت درصدی بین هر دو تیک متوالی داخل کل بافر (تا ۹ مقدار برای
        بافر ۱۰تایی) — نه فقط دو عدد خلاصهٔ سرعت/شتاب، بلکه خودِ توالی، تا مدل
        بتواند الگوی دقیق شتاب‌گیری/کندشدن و جهت را در چند تیک آخر قبل از
        معامله ببیند.
        """
        return self._velocity_pct_series()

    def get_ticks_since_direction_change(self) -> int:
        """
        چند تیک متوالی است که حرکت هم‌جهت بوده (بدون تغییر جهت)؟ عدد بزرگ یعنی
        یک حرکت یک‌طرفهٔ نسبتاً پایدار در جریان است؛ عدد کوچک یعنی نوسان/تردید
        اخیر (چند بار جهت عوض شده).
        """
        ticks = list(self.buffer)
        if len(ticks) < 2:
            return 0
        directions = []
        for i in range(1, len(ticks)):
            diff = ticks[i].price - ticks[i - 1].price
            if diff != 0:
                directions.append(1 if diff > 0 else -1)
        if not directions:
            return 0
        last_direction = directions[-1]
        count = 0
        for d in reversed(directions):
            if d == last_direction:
                count += 1
            else:
                break
        return count

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

    def reset(self) -> None:
        """خالی‌کردن کامل بافر - مثلاً وقتی نماد معاملاتی عوض شده و تیک‌های قبلی دیگر معتبر نیستند."""
        self.buffer.clear()

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

    def _micro_swing_points(self) -> tuple[list[float], list[float]]:
        """
        نقاط اکسترمم محلی (نوسان ریز - تغییر جهت تیک‌به‌تیک) در کل بافر تیک‌ها
        (نه فقط کندل جاری - نوسان ریز مستقل از مرز کندل است)، به‌علاوهٔ آخرین
        تیک به‌عنوان انتهای لگِ در حال شکل‌گیری. خروجی: (prices, timestamps)
        هم‌طول و به ترتیب زمانی.
        """
        ticks = list(self.buffer)
        if len(ticks) < 2:
            return [], []
        prices: list[float] = []
        times: list[float] = []
        prev_direction: Optional[int] = None
        for i in range(1, len(ticks)):
            diff = ticks[i].price - ticks[i - 1].price
            if diff == 0:
                continue
            direction = 1 if diff > 0 else -1
            if prev_direction is not None and direction != prev_direction:
                prices.append(ticks[i - 1].price)
                times.append(ticks[i - 1].timestamp)
            prev_direction = direction
        prices.append(ticks[-1].price)
        times.append(ticks[-1].timestamp)
        return prices, times

    def get_micro_swing_features(self) -> dict:
        """
        ویژگی‌های «نوسان ریز» (Micro-Swing) در سطح تیک: هر لگ فاصلهٔ بین دو
        اکسترمم محلی متوالی است (یا از آخرین اکسترمم تا تیک جاری، برای لگِ در
        حال شکل‌گیری). برای آخرین نوسان و پنجره‌های ۳/۵/۱۵ نوسان اخیر محاسبه
        می‌شود:

            - جهت (فقط آخرین نوسان): ۱=صعودی، -۱=نزولی، ۰=نامشخص
            - سرعت: مجموع اندازهٔ حرکت (مسافت، مطلق و درصدی) تقسیم بر کل
              زمان پنجره (درصد/ثانیه)
            - شتاب: تفاضل سرعت لحظه‌ایِ لگِ آخر و لگِ اولِ پنجره، تقسیم بر کل
              زمان پنجره
            - جابجایی (Displacement): درصد تغییر خالص قیمت از ابتدای پنجره تا
              الان (علامت‌دار)
            - مسافت جابجایی (Distance): مجموع قدرمطلق درصد تغییر هر لگ داخل
              پنجره (همیشه >= |جابجایی|، چون رفت‌وبرگشت‌ها را هم می‌شمارد)

        اگر نوسان کافی در بافر نباشد، مقدار خنثی ۰.۰ برگردانده می‌شود.
        """
        prices, times = self._micro_swing_points()

        legs: list[tuple[float, float]] = []
        for i in range(1, len(prices)):
            if prices[i - 1] == 0:
                continue
            pct = (prices[i] - prices[i - 1]) / prices[i - 1]
            dt = times[i] - times[i - 1]
            legs.append((pct, dt))

        def leg_velocity(pct: float, dt: float) -> float:
            return pct / dt if dt > 0 else 0.0

        features: dict = {}

        # -- آخرین نوسان ریز (۱ لگ) --------------------------------------
        if legs:
            last_pct, last_dt = legs[-1]
            last_velocity = leg_velocity(last_pct, last_dt)
            features["micro_swing_last_direction"] = 1 if last_pct > 0 else (-1 if last_pct < 0 else 0)
            features["micro_swing_last_speed_pct"] = last_velocity
            if len(legs) >= 2:
                prev_pct, prev_dt = legs[-2]
                prev_velocity = leg_velocity(prev_pct, prev_dt)
                total_dt = last_dt + prev_dt
                features["micro_swing_last_acceleration_pct"] = (
                    (last_velocity - prev_velocity) / total_dt if total_dt > 0 else 0.0
                )
            else:
                features["micro_swing_last_acceleration_pct"] = 0.0
        else:
            features["micro_swing_last_direction"] = 0
            features["micro_swing_last_speed_pct"] = 0.0
            features["micro_swing_last_acceleration_pct"] = 0.0

        # -- پنجره‌های ۳/۵/۱۵ نوسان ریز اخیر ------------------------------
        for window in (3, 5, 15):
            prefix = f"micro_swing_last{window}"
            window_legs = legs[-window:]
            n = len(window_legs)
            if n == 0:
                features[f"{prefix}_speed_pct"] = 0.0
                features[f"{prefix}_acceleration_pct"] = 0.0
                features[f"{prefix}_displacement_pct"] = 0.0
                features[f"{prefix}_distance_pct"] = 0.0
                continue

            distance_pct = sum(abs(pct) for pct, _ in window_legs)
            total_dt = sum(dt for _, dt in window_legs)

            # جابجایی خالص: ترکیب دقیق بازده‌های درصدی هر لگ (معادل ریاضی
            # (قیمت پایانی - قیمت ابتدایی)/قیمت ابتدایی، بدون نیاز به قیمت خام)
            displacement_pct = 1.0
            for pct, _ in window_legs:
                displacement_pct *= (1 + pct)
            displacement_pct -= 1.0

            features[f"{prefix}_displacement_pct"] = displacement_pct
            features[f"{prefix}_distance_pct"] = distance_pct
            features[f"{prefix}_speed_pct"] = (distance_pct / total_dt) if total_dt > 0 else 0.0

            if n >= 2:
                first_pct, first_dt = window_legs[0]
                last_pct, last_dt = window_legs[-1]
                first_velocity = leg_velocity(first_pct, first_dt)
                last_velocity = leg_velocity(last_pct, last_dt)
                features[f"{prefix}_acceleration_pct"] = (
                    (last_velocity - first_velocity) / total_dt if total_dt > 0 else 0.0
                )
            else:
                features[f"{prefix}_acceleration_pct"] = 0.0

        return features

    def get_edge_behavior_features(
        self,
        candle_start_time: float,
        candle_high: float,
        candle_low: float,
        edge_zone_ratio: float = 0.15,
    ) -> dict:
        """
        رفتار قیمت نسبت به لبهٔ بالا/پایین کندل جاری:
            - چند بار قیمت وارد «منطقهٔ لبه» (پیش‌فرض ۱۵٪ بالایی/پایینی رنج کندل) شده؟
            - آخرین باری که وارد آن منطقه شد، بعد از آن (نه در طول کل کندل، بلکه
              دقیقاً بعد از همان لحظه) قیمت فراتر رفت (Extend) یا همان‌جا رد
              شد/متوقف ماند (Reject/Stall)؟ برای این تشخیص، فقط تیک‌های بعد از
              لحظهٔ تست را نگاه می‌کنیم — نه های/لوی کل کندل — چون ممکن است
              های/لوی واقعی کندل خیلی زودتر (مثلاً همان کندل قبل از این تست)
              رخ داده باشد و به این تست ربطی نداشته باشد.
            - اندازهٔ جهش فعلی (از آخرین نقطهٔ برگشت تا الان) و جهش قبلی، هر دو
              نسبت به رنج کندل — برای مقایسهٔ «آیا این جهش از جهش قبلی بزرگ‌تر
              شده یا کوچک‌تر» (فاصله و مقدار جهش‌های متوالی).
        """
        result = {
            "upper_edge_test_count": 0,
            "lower_edge_test_count": 0,
            "upper_edge_last_outcome": None,
            "lower_edge_last_outcome": None,
            "last_move_size_ratio": None,
            "prev_move_size_ratio": None,
        }
        candle_range = candle_high - candle_low
        ticks_in_candle = [t for t in self.buffer if t.timestamp >= candle_start_time]
        if len(ticks_in_candle) < 3 or candle_range <= 0:
            return result

        # نقاط اکسترمم محلی (Peak/Trough) به‌همراه اندیسشان در دنبالهٔ تیک‌ها
        extrema_idx: list[int] = []
        extrema_price: list[float] = []
        extrema_kind: list[str] = []
        prev_direction = None
        for i in range(1, len(ticks_in_candle)):
            diff = ticks_in_candle[i].price - ticks_in_candle[i - 1].price
            if diff == 0:
                continue
            direction = 1 if diff > 0 else -1
            if prev_direction is not None and direction != prev_direction:
                extrema_idx.append(i - 1)
                extrema_price.append(ticks_in_candle[i - 1].price)
                extrema_kind.append("peak" if prev_direction == 1 else "trough")
            prev_direction = direction

        upper_zone = candle_low + (1 - edge_zone_ratio) * candle_range
        lower_zone = candle_low + edge_zone_ratio * candle_range

        peak_positions = [
            (idx, price) for idx, price, kind in zip(extrema_idx, extrema_price, extrema_kind)
            if kind == "peak" and price >= upper_zone
        ]
        trough_positions = [
            (idx, price) for idx, price, kind in zip(extrema_idx, extrema_price, extrema_kind)
            if kind == "trough" and price <= lower_zone
        ]

        result["upper_edge_test_count"] = len(peak_positions)
        result["lower_edge_test_count"] = len(trough_positions)

        if peak_positions:
            last_idx, last_price = peak_positions[-1]
            prices_after = [t.price for t in ticks_in_candle[last_idx + 1:]]
            extended = bool(prices_after) and max(prices_after) > last_price
            result["upper_edge_last_outcome"] = int(extended)

        if trough_positions:
            last_idx, last_price = trough_positions[-1]
            prices_after = [t.price for t in ticks_in_candle[last_idx + 1:]]
            extended = bool(prices_after) and min(prices_after) < last_price
            result["lower_edge_last_outcome"] = int(extended)

        if extrema_price:
            current_price = ticks_in_candle[-1].price
            result["last_move_size_ratio"] = abs(current_price - extrema_price[-1]) / candle_range
            if len(extrema_price) >= 2:
                result["prev_move_size_ratio"] = abs(extrema_price[-1] - extrema_price[-2]) / candle_range

        return result


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
    def body_ratio(self) -> float:
        """
        نسبت طول بدنه به کل رنج کندل (بدنه/(های−لو))، همیشه بین ۰ و ۱.
        وقتی کندل اصلاً سایه ندارد (Marubozu)، این مقدار دقیقاً ۱.۰ می‌شود —
        بدون نیاز به هیچ عدد قراردادی، چون صورت و مخرج هر دو مقادیر واقعی و
        محدودند (برخلاف نسبت بدنه/سایه که با سایهٔ صفر بی‌نهایت می‌شد).
        از آن‌جا که body_ratio + upper_wick_ratio + lower_wick_ratio همیشه
        دقیقاً برابر ۱ است، این سه عدد با هم شکل کامل کندل را بدون هیچ تکینگی
        توصیف می‌کنند.
        """
        rng = self.range
        return (self.body / rng) if rng > 0 else 0.0

    def as_dict(self, prefix: str) -> dict:
        """
        فقط ویژگی‌های شکلیِ نسبی کندل — بدون هیچ مقدار خام OHLC. نسبت‌های بدنه/سایه
        به‌طور ذاتی مستقل از سطح قیمت‌اند (چون صورت و مخرج هر دو به همان واحد قیمتند).
        """
        rng = self.range
        return {
            f"{prefix}_is_bullish": int(self.is_bullish),
            f"{prefix}_body_ratio": self.body_ratio,
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

    def reset(self) -> None:
        """
        تاریخچهٔ کندل‌ها و کندل در حال شکل‌گیری را کامل پاک می‌کند - مثلاً وقتی
        نماد معاملاتی عوض شده و کندل‌های قبلی مربوط به یک نماد دیگرند.
        """
        self.history.clear()
        self.current = None
        self._current_bucket = None

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
        # دنبالهٔ کامل سرعت/جهت چند تیک اخیر (نه فقط دو عدد خلاصه) و این‌که چند
        # تیک متوالی است حرکت هم‌جهت بوده (پایداری حرکت لحظه‌ای قبل از معامله):
        "recent_tick_velocities_json": json.dumps(tick_buffer.get_recent_velocities_pct()),
        "ticks_since_direction_change": tick_buffer.get_ticks_since_direction_change(),
    }
    # سرعت/شتاب/جهت/جابجایی/مسافت نوسان‌های ریز اخیر (مستقل از مرز کندل):
    snapshot.update(tick_history.get_micro_swing_features())

    current_candle = candle_aggregator.current
    if current_candle and latest:
        rng = current_candle.range
        snapshot["price_position_in_candle"] = (
            (latest.price - current_candle.low) / rng if rng > 0 else 0.5
        )
        snapshot["distance_from_open_ratio"] = (
            (latest.price - current_candle.open) / rng if rng > 0 else 0.0
        )
        # نیمهٔ بالایی کندل (بالاتر از قیمت بازشدن) یا نیمهٔ پایینی (پایین‌تر از
        # قیمت بازشدن) — مستقل از رنگ کندل، صرفاً موقعیت قیمت لحظه‌ای نسبت به
        # نقطهٔ شروع همین کندلِ در حال شکل‌گیری:
        snapshot["price_above_open"] = int(latest.price >= current_candle.open)
        snapshot.update(current_candle.as_dict("candle_curr"))
        snapshot.update(
            tick_history.get_stall_features(current_candle.start_time, current_candle.high, current_candle.low)
        )
        # رفتار قیمت نسبت به لبهٔ بالا/پایین کندل جاری (تست/رد/عبور) و اندازهٔ جهش‌ها:
        snapshot.update(
            tick_history.get_edge_behavior_features(
                current_candle.start_time, current_candle.high, current_candle.low
            )
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
        snapshot["price_above_open"] = None
        snapshot["stall_count_in_candle"] = None
        snapshot["last_stall_position_in_candle"] = None
        snapshot["upper_edge_test_count"] = None
        snapshot["lower_edge_test_count"] = None
        snapshot["upper_edge_last_outcome"] = None
        snapshot["lower_edge_last_outcome"] = None
        snapshot["last_move_size_ratio"] = None
        snapshot["prev_move_size_ratio"] = None
        snapshot["tick_vs_candle_alignment"] = None

    if candle_aggregator.get_previous_candle():
        snapshot.update(candle_aggregator.get_previous_candle().as_dict("candle_prev1"))

    snapshot.update(compare_recent_candles(candle_aggregator.history, n=3))

    return snapshot
