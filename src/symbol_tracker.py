"""
تشخیص تغییر نماد (Asset) زنده از روی جریان تیک‌ها
======================================================

اگر کاربر به‌صورت دستی نماد معاملاتی را در خودِ پلتفرم عوض کند، هیچ پیام
صریحی برای این تغییر شنود نمی‌شود - تنها نشانهٔ آن این است که symbol تیک‌های
دریافتی عوض می‌شود (خودِ Tick از browser_session.py همیشه symbol دارد).
این ماژول این تغییر را از روی همان جریان تیک تشخیص می‌دهد تا اسکریپت‌های
فراخوان (main.py/collect_data.py) بافرهای قیمتی/ساختار بازار را که بر مبنای
نماد قبلی ساخته شده‌اند ریست کنند - وگرنه شکل چارت و بقیهٔ فیچرها ترکیبی از
دو نماد مختلف (با دو سطح قیمتی کاملاً بی‌ربط) می‌شوند.

برای جلوگیری از ریست اشتباهی به‌خاطر یک تیکِ پرت/تک (مثلاً یک پیام پس‌زمینه
از نمادی دیگر در واچ‌لیست)، تغییر فقط وقتی قطعی در نظر گرفته می‌شود که چند
تیک متوالی (SYMBOL_SWITCH_CONFIRM_TICKS) همگی نماد جدید را نشان دهند.
"""

from __future__ import annotations

SYMBOL_SWITCH_CONFIRM_TICKS = 2


class SymbolSwitchDetector:
    """
    check(symbol) را به ازای هر تیک صدا بزنید. فقط زمانی True برمی‌گرداند که
    نماد فعال واقعاً (و به‌طور تأییدشده) عوض شده باشد - یعنی همان لحظه‌ای که
    باید بافرهای قیمتی/ساختار بازار ریست شوند.
    """

    def __init__(self, confirm_ticks: int = SYMBOL_SWITCH_CONFIRM_TICKS):
        self.confirm_ticks = confirm_ticks
        self.current_symbol: str | None = None
        self._pending_symbol: str | None = None
        self._pending_count: int = 0

    def check(self, symbol: str) -> bool:
        """
        اولین تیکی که اصلاً دیده می‌شود هرگز «تغییر» محسوب نمی‌شود - فقط
        current_symbol را ست می‌کند. برای تیک‌های بعدی، اگر symbol با
        current_symbol فرق داشت، شمارندهٔ تأیید افزایش می‌یابد؛ فقط وقتی این
        شمارنده به confirm_ticks برسد، تغییر قطعی در نظر گرفته می‌شود
        (current_symbol به‌روز می‌شود) و True برمی‌گردد.
        """
        if self.current_symbol is None:
            self.current_symbol = symbol
            return False

        if symbol == self.current_symbol:
            self._pending_symbol = None
            self._pending_count = 0
            return False

        if symbol == self._pending_symbol:
            self._pending_count += 1
        else:
            self._pending_symbol = symbol
            self._pending_count = 1

        if self._pending_count >= self.confirm_ticks:
            self.current_symbol = symbol
            self._pending_symbol = None
            self._pending_count = 0
            return True

        return False
