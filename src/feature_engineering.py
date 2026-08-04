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
import math
import statistics
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
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

    def price_at_or_after(self, timestamp: float) -> Optional[float]:
        """
        نزدیک‌ترین قیمت ثبت‌شده بعد (یا مساوی) یک timestamp مشخص را برمی‌گرداند.
        برخلاف TickBuffer.latest_price_after (که فقط چند تیک آخر - پیش‌فرض ۱۰
        تا - را نگه می‌دارد)، این بافر بلندتر (پیش‌فرض ۲۰۰ تیک) است تا حتی بعد
        از چند ثانیه تأخیرِ اضافه (مثلاً صبر برای پیام نتیجهٔ معامله از پلتفرم)
        هنوز تیک لحظهٔ دقیق موردنظر در آن باقی مانده باشد.
        """
        candidates = [t for t in self.buffer if t.timestamp >= timestamp]
        if not candidates:
            return None
        return min(candidates, key=lambda t: t.timestamp - timestamp).price

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


def normalized_window_shape(
    candles: "deque[Candle]", window_size: int, current_candle: Optional[Candle] = None
) -> list[dict]:
    """
    یک «پنجرهٔ دیداری» ثابت از کندل‌های تکمیل‌شده (به‌علاوهٔ کندلِ در حال
    شکل‌گیری فعلی، اگر داده شود، به‌عنوان لبهٔ راست) را می‌گیرد و محور قیمت را
    خودش با همان پنجره Auto-Scale می‌کند (۰ تا ۱ نسبت به های/لوی همان پنجره) -
    هیچ مرجع بیرونی (قیمت خام، ATR، زمان) استفاده نمی‌شود، پس مستقل از نماد/
    سطح قیمت و تایم‌فریم کندل است؛ همان منطقی که MarketStructureTracker برای
    «شکل چارت» در تایم‌فریم اصلی استفاده می‌کند، این‌جا به‌صورت تابع مستقل
    است تا برای هر تایم‌فریم دیگری (مثلاً کندل‌های ریزتر چندثانیه‌ای) هم قابل
    استفاده باشد. ترتیب خروجی از قدیم به جدید است.
    """
    closed = list(candles)
    window = closed[-(window_size - 1):] if current_candle else closed[-window_size:]
    if current_candle:
        window = window + [current_candle]
    if not window:
        return []

    window_high = max(c.high for c in window)
    window_low = min(c.low for c in window)
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
        for c in window
    ]


def compute_micro_market_regime(
    candles: "deque[Candle]", window: int, current_candle: Optional[Candle] = None
) -> dict:
    """
    وضعیت بازار (گاوی/خرسی/رنج) در تایم‌فریم کندل‌های ریز (پیش‌فرض ۵ ثانیه‌ای)،
    بر اساس Efficiency Ratio کافمن: نسبت «جابجایی خالص قیمت» به «مجموع جابجایی‌های
    مرحله‌ای (close به close) بین کندل‌های همان پنجره». این نسبت نزدیک ۱ یعنی
    حرکت یک‌طرفهٔ تمیز (روند واقعی)، نزدیک ۰ یعنی رفت‌وبرگشت بدون جابجایی خالص
    (رنج) - حتی اگر قیمت در همان بازه پرنوسان هم بوده باشد. برخلاف trend_regime
    در market_structure.py (که به سوئینگ‌های های/لوی کندل‌های ۱-دقیقه‌ای نیاز
    دارد و برای پنجرهٔ کوتاه چند-ده‌ثانیه‌ای داده کافی ندارد)، این روش با همین
    تعداد کم کندل هم قابل‌محاسبه است.
    """
    closed = list(candles)[-window:]
    sequence = closed + ([current_candle] if current_candle else [])

    if len(sequence) < 2:
        return {
            "micro_market_efficiency_ratio": None,
            "micro_market_net_direction": None,
            "micro_market_bullish_ratio": None,
            "micro_market_regime": 0,
            "micro_market_regime_ready": 0,
        }

    net_change = sequence[-1].close - sequence[0].open
    step_changes = [abs(sequence[i].close - sequence[i - 1].close) for i in range(1, len(sequence))]
    total_path = sum(step_changes)
    efficiency_ratio = (abs(net_change) / total_path) if total_path > 0 else 0.0
    net_direction = 1 if net_change > 0 else (-1 if net_change < 0 else 0)
    bullish_ratio = sum(1 for c in sequence if c.is_bullish) / len(sequence)

    if net_direction != 0 and efficiency_ratio >= config.MICRO_TREND_EFFICIENCY_THRESHOLD:
        regime = net_direction  # +۱ = گاوی (Bullish)، -۱ = خرسی (Bearish)
    else:
        regime = 0  # رنج

    return {
        "micro_market_efficiency_ratio": efficiency_ratio,
        "micro_market_net_direction": net_direction,
        "micro_market_bullish_ratio": bullish_ratio,
        "micro_market_regime": regime,
        "micro_market_regime_ready": 1,
    }


# ---------------------------------------------------------------------------
# منطق بدون‌حالت (Stateless) استراتژی سطوح (src/level_strategy.py) - این‌جا
# نگه‌داری می‌شود (نه در خودِ level_strategy.py) تا هم آن ماژول بتواند از این
# تابع استفاده کند و هم build_feature_snapshot بدون ایجاد Import چرخه‌ای؛
# تنها منبع حقیقتِ این منطق همین‌جاست - level_strategy.py آن را دوباره
# پیاده‌سازی نمی‌کند، فقط صدا می‌زند.
# ---------------------------------------------------------------------------
def find_local_extrema(ticks: list) -> list[float]:
    """قیمت نقاط توقف/برگشت محلی (Local Extrema) را از یک دنبالهٔ تیک پیدا می‌کند."""
    extrema: list[float] = []
    prev_dir: Optional[int] = None
    for i in range(1, len(ticks)):
        diff = ticks[i].price - ticks[i - 1].price
        if diff == 0:
            continue
        d = 1 if diff > 0 else -1
        if prev_dir is not None and d != prev_dir:
            extrema.append(ticks[i - 1].price)
        prev_dir = d
    return extrema


def classify_region_vs_open(extrema: list[float], open_price: float) -> int:
    """
    +۱ = تمام اکسترمم‌ها بالای open (ناحیهٔ صعودی، بدون نوسان دوطرفه)
    -۱ = تمام اکسترمم‌ها پایین open (ناحیهٔ نزولی، بدون نوسان دوطرفه)
     ۰ = هم بالا هم پایین open (spans_open - نوسان دوطرفهٔ اپن)
    """
    if all(p > open_price for p in extrema):
        return 1
    if all(p < open_price for p in extrema):
        return -1
    return 0


def level_strategy_direction_for(region: int, near_high_stall: bool) -> str:
    """دقیقاً همان منطق direction_for در src/level_strategy.py."""
    if region == 0:
        return "PUT" if near_high_stall else "CALL"
    elif region == 1:
        return "CALL"
    else:
        return "PUT"


def compute_level_strategy_context(tick_history: "TickHistory", current_candle: "Candle") -> Optional[dict]:
    """
    «نظر» استراتژی سطوح (level_strategy.py) را در همین لحظه، به‌صورت فیچرهای
    کاملاً نسبی (نه قانون سخت‌کدشده) برمی‌گرداند - مستقل از این‌که خودِ آن
    استراتژی همین الان معامله را باز کرده یا نه، و مستقل از نماد/سطح قیمت
    (فقط نسبت‌ها). هدف: در دیتای ترین ثبت شود تا خودِ مدل یاد بگیرد کِی به
    این سیگنالِ قانون‌محور اعتماد کند و کِی نه (مثلاً وقتی روند بزرگ‌تر با آن
    مخالف است) - به‌جای این‌که یک آستانهٔ ثابت و دستی در خودِ استراتژی حک شود.
    اگر دادهٔ کافی نبود (کندل بدون رنج یا کمتر از ۳ تیک)، None برمی‌گرداند.
    """
    candle_range = current_candle.range
    if candle_range <= 0:
        return None

    ticks_in_candle = [t for t in tick_history.buffer if t.timestamp >= current_candle.start_time]
    if len(ticks_in_candle) < 3:
        return None

    extrema = find_local_extrema(ticks_in_candle)
    if not extrema:
        return None

    region = classify_region_vs_open(extrema, current_candle.open)

    near_high = min(extrema, key=lambda p: abs(current_candle.high - p))
    near_low = min(extrema, key=lambda p: abs(p - current_candle.low))

    near_high_direction = 1 if level_strategy_direction_for(region, near_high_stall=True) == "CALL" else -1
    near_low_direction = 1 if level_strategy_direction_for(region, near_high_stall=False) == "CALL" else -1

    return {
        "level_strategy_region": region,
        "level_strategy_near_high_position": (near_high - current_candle.low) / candle_range,
        "level_strategy_near_low_position": (near_low - current_candle.low) / candle_range,
        "level_strategy_near_high_implied_direction": near_high_direction,
        "level_strategy_near_low_implied_direction": near_low_direction,
    }


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
def compute_time_of_day_features(timestamp: Optional[float]) -> dict:
    """
    ساعت روز و روز هفته (به‌وقت UTC، همان ساعتی که تیک‌ها با آن رسیده‌اند) را
    به‌صورت چرخه‌ای (sin/cos) کدگذاری می‌کند - نه یک عدد خام ساعت (۰ تا ۲۳)، چون
    عدد خام یعنی ساعت ۲۳ و ساعت ۰ برای مدل «دورترین» مقادیر ممکن به‌نظر می‌رسند
    درحالی‌که در واقعیت فقط یک ساعت فاصله دارند؛ کدگذاری sin/cos این پیوستگی
    دایره‌ای را حفظ می‌کند. هدف: امکان یادگیری الگوهای وابسته به بازهٔ زمانی
    (مثلاً نوسان/نقدینگی متفاوت شب در برابر صبح، یا روزهای هفته در برابر تعطیلات
    آخر هفته در جفت‌ارزهای OTC).
    """
    if timestamp is None:
        return {
            "hour_of_day_sin": None,
            "hour_of_day_cos": None,
            "day_of_week_sin": None,
            "day_of_week_cos": None,
        }
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour_frac = dt.hour + dt.minute / 60.0
    day_of_week = dt.weekday()  # ۰=دوشنبه ... ۶=یکشنبه
    return {
        "hour_of_day_sin": math.sin(2 * math.pi * hour_frac / 24.0),
        "hour_of_day_cos": math.cos(2 * math.pi * hour_frac / 24.0),
        "day_of_week_sin": math.sin(2 * math.pi * day_of_week / 7.0),
        "day_of_week_cos": math.cos(2 * math.pi * day_of_week / 7.0),
    }


def build_feature_snapshot(
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
    micro_candle_aggregator: Optional[CandleAggregator] = None,
) -> dict:
    """
    تمام ویژگی‌های نسبیِ سطح تیک/کندل را در یک دیکشنری جمع می‌کند. ویژگی‌های
    ساختار بازار (سوئینگ/لگ/حمایت‌مقاومت/روند) جداگانه توسط
    market_structure.MarketStructureTracker.get_features() ساخته و در
    DataLogger با این خروجی ترکیب می‌شوند.
    """
    latest = tick_buffer.latest()
    time_reference = latest.timestamp if latest else (candle_aggregator.current.start_time if candle_aggregator.current else None)
    snapshot: dict = {
        **compute_time_of_day_features(time_reference),
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

    history_list = list(candle_aggregator.history)
    prev1 = history_list[-1] if history_list else None
    prev2 = history_list[-2] if len(history_list) >= 2 else None

    if prev1:
        snapshot.update(prev1.as_dict("candle_prev1"))
    if prev2:
        snapshot.update(prev2.as_dict("candle_prev2"))

    # نسبت اندازهٔ (رنج) کندلِ در حال شکل‌گیری فعلی به دو کندل تکمیل‌شدهٔ قبلی،
    # و هم‌رنگ بودن کندل جاری با هرکدام. compare_recent_candles پایین‌تر فقط
    # کندل‌های تکمیل‌شده را با هم مقایسه می‌کند (prev1 با prev2، prev2 با prev3
    # و ...)؛ این‌جا مقایسهٔ کندل جاری (که هنوز تکمیل نشده) با هرکدام از آن دو
    # کندل قبلی اضافه می‌شود - قبلاً چنین مقایسه‌ای اصلاً وجود نداشت.
    snapshot["candle_curr_vs_prev1_range_ratio"] = (
        (current_candle.range / prev1.range) if current_candle and prev1 and prev1.range > 0 else None
    )
    snapshot["candle_curr_color_match_prev1"] = (
        int(current_candle.is_bullish == prev1.is_bullish) if current_candle and prev1 else None
    )
    snapshot["candle_curr_vs_prev2_range_ratio"] = (
        (current_candle.range / prev2.range) if current_candle and prev2 and prev2.range > 0 else None
    )
    snapshot["candle_curr_color_match_prev2"] = (
        int(current_candle.is_bullish == prev2.is_bullish) if current_candle and prev2 else None
    )

    snapshot.update(compare_recent_candles(history_list, n=3))

    # شکل نرمال‌شدهٔ چند کندل ریزِ چندثانیه‌ای اخیر (مستقل از کندل یک‌دقیقه‌ای
    # بالا) - تا مدل تصویری از رفتار قیمت درست در همان چند ثانیهٔ منتهی به
    # لحظهٔ ورود به معامله هم داشته باشد، دقیقاً مثل chart_shape_json ولی در
    # یک تایم‌فریم ریزتر.
    if micro_candle_aggregator is not None:
        snapshot["micro_chart_shape_json"] = json.dumps(
            normalized_window_shape(
                micro_candle_aggregator.history,
                config.MICRO_CHART_WINDOW_CANDLES,
                micro_candle_aggregator.current,
            )
        )
        # وضعیت بازار (گاوی/خرسی/رنج) در همین تایم‌فریم ریز - جدا از شکل چارت بالا:
        snapshot.update(
            compute_micro_market_regime(
                micro_candle_aggregator.history,
                config.MICRO_CHART_WINDOW_CANDLES,
                micro_candle_aggregator.current,
            )
        )
    else:
        snapshot["micro_market_efficiency_ratio"] = None
        snapshot["micro_market_net_direction"] = None
        snapshot["micro_market_bullish_ratio"] = None
        snapshot["micro_market_regime"] = None
        snapshot["micro_market_regime_ready"] = None

    # «نظر» استراتژی سطوح (level_strategy.py) در همین لحظه - نه به‌عنوان یک
    # قانون سخت‌کدشده، بلکه به‌صورت فیچرهای نسبی تا خودِ مدل یاد بگیرد کِی به
    # این سیگنال اعتماد کند (مثلاً وقتی با روند بزرگ‌تر هم‌جهت است) و کِی نه.
    level_strategy_context = (
        compute_level_strategy_context(tick_history, current_candle) if current_candle else None
    )
    if level_strategy_context is not None:
        snapshot.update(level_strategy_context)
    else:
        snapshot["level_strategy_region"] = None
        snapshot["level_strategy_near_high_position"] = None
        snapshot["level_strategy_near_low_position"] = None
        snapshot["level_strategy_near_high_implied_direction"] = None
        snapshot["level_strategy_near_low_implied_direction"] = None

    return snapshot
