"""
بخش سوم: مدیریت حافظه معاملاتی (State & History Tracking)
============================================================

این ماژول نتیجهٔ معاملات اخیر (Win=1 / Loss=0) را نگه می‌دارد و دو چیز را
محاسبه می‌کند:

    ۱. الگوی ۳ نتیجهٔ آخر، مثلاً [1, 0, 1] — همان چیزی که در روش دستی شما
       به‌عنوان یکی از سیگنال‌های تصمیم‌گیری استفاده می‌شود.
    ۲. وین‌ریت لحظه‌ای (Rolling Winrate) روی یک پنجرهٔ بزرگ‌تر (مثلاً ۲۰ معاملهٔ
       آخر) با استفاده از Pandas.
"""

from __future__ import annotations

from collections import deque

import pandas as pd

import config


class TradeHistory:
    """
    نگه‌دارندهٔ کامل تاریخچهٔ نتایج معاملات در طول اجرای اسکریپت.
    """

    def __init__(
        self,
        recent_window: int = config.RECENT_RESULTS_WINDOW,
        winrate_window: int = config.WINRATE_ROLLING_WINDOW,
    ):
        self.recent_window = recent_window
        self.winrate_window = winrate_window

        # فقط N نتیجهٔ آخر برای الگوی سریع (مثل [1,0,1])
        self._recent_results: deque[int] = deque(maxlen=recent_window)

        # کل تاریخچه برای محاسبات آماری دقیق‌تر با pandas
        self._all_results: list[int] = []

    def add_result(self, result: int) -> None:
        """result باید 1 (Win) یا 0 (Loss) باشد."""
        assert result in (0, 1), "نتیجهٔ معامله باید 0 یا 1 باشد"
        self._recent_results.append(result)
        self._all_results.append(result)

    def reset(self) -> None:
        """تاریخچهٔ معاملات را کاملاً پاک می‌کند (برای شروع دوباره از صفر)."""
        self._recent_results.clear()
        self._all_results.clear()

    def get_recent_pattern(self) -> list[int]:
        """الگوی آخرین معاملات، مثلاً [1, 0, 1]. اگر تعداد کافی نباشد کوتاه‌تر است."""
        return list(self._recent_results)

    def get_rolling_winrate(self) -> float:
        """
        وین‌ریت روی آخرین `winrate_window` معامله (پیش‌فرض ۲۰ تا).
        اگر هنوز به این تعداد معامله نرسیده باشیم، روی کل تاریخچهٔ موجود حساب می‌شود.
        """
        if not self._all_results:
            return 0.0
        series = pd.Series(self._all_results[-self.winrate_window:])
        return float(series.mean())

    def get_overall_winrate(self) -> float:
        """وین‌ریت کلی از ابتدای اجرای اسکریپت."""
        if not self._all_results:
            return 0.0
        return float(pd.Series(self._all_results).mean())

    def total_trades(self) -> int:
        return len(self._all_results)

    def as_feature_dict(self) -> dict:
        """
        خروجی آماده برای اضافه شدن به ردیف دیتاست آموزشی: الگوی اخیر (هر معامله
        در یک ستون جدا) + وین‌ریت‌های لحظه‌ای.
        """
        pattern = self.get_recent_pattern()
        # پدینگ با None اگر هنوز به تعداد کافی معامله نداشته باشیم
        padded = [None] * (self.recent_window - len(pattern)) + pattern
        features = {f"recent_result_{i+1}": val for i, val in enumerate(padded)}
        features["rolling_winrate"] = self.get_rolling_winrate()
        features["overall_winrate"] = self.get_overall_winrate()
        features["total_trades_so_far"] = self.total_trades()
        return features
