"""
کلیک برنامه‌ای (خودکار) روی دکمهٔ واقعی BUY/SELL پلتفرم
==========================================================

وقتی مدل تصمیم می‌گیرد معامله باز کند، این ماژول همان دکمهٔ واقعی BUY/SELL
پلتفرم را در مرورگر پیدا کرده و روی آن `.click()` واقعی صدا می‌زند — دقیقاً
با همان منطق تشخیص متنی که src/trade_click_listener.py برای شنیدن کلیک
استفاده می‌کند. چون این کلیک از دیدگاه صفحه یک کلیک واقعی DOM است، همان
شنوندهٔ کلیکِ از قبل فعال (trade_click_listener) آن را هم‌زمان تشخیص داده و
مسیر ثبت دادهٔ موجود (DataLogger.capture_entry) بدون هیچ کد تکراری صدا زده
می‌شود — یعنی معاملهٔ خودکار هم دقیقاً مثل معاملهٔ دستی ثبت و لیبل‌گذاری
می‌شود.
"""

from __future__ import annotations

from playwright.async_api import Page

import config

_CLICK_SCRIPT = """
(direction) => {
    const BUY_TEXT = %r;
    const SELL_TEXT = %r;
    const MAX_TEXT_LENGTH = 20;
    const text = direction === 'CALL' ? BUY_TEXT : SELL_TEXT;
    const pattern = new RegExp('\\\\b' + text + '\\\\b');

    const all = document.querySelectorAll('button, a, div, span');
    let candidates = [];
    for (const el of all) {
        const t = (el.textContent || '').trim().toUpperCase();
        if (t.length > 0 && t.length <= MAX_TEXT_LENGTH && pattern.test(t)) {
            candidates.push(el);
        }
    }
    if (candidates.length === 0) return false;
    // کوچک‌ترین عنصر منطبق (نزدیک‌ترین به خودِ متن دکمه، نه یک کانتینر بزرگ)
    candidates.sort((a, b) => a.textContent.length - b.textContent.length);
    const preferred = candidates.find(el => el.tagName === 'A' || el.tagName === 'BUTTON') || candidates[0];
    preferred.click();
    return true;
}
""" % (config.BUY_BUTTON_TEXT.upper(), config.SELL_BUTTON_TEXT.upper())


async def click_trade_button(page: Page, direction: str) -> bool:
    """دکمهٔ BUY (اگر direction == 'CALL') یا SELL (اگر 'PUT') را واقعاً کلیک می‌کند."""
    try:
        clicked = await page.evaluate(_CLICK_SCRIPT, direction)
        if not clicked:
            print(f"[TradeExecutor] دکمهٔ {direction} روی صفحه پیدا نشد؛ کلیک خودکار انجام نشد.")
        return bool(clicked)
    except Exception as exc:  # noqa: BLE001 - نباید کل حلقهٔ معاملهٔ خودکار به‌خاطر یک خطای مرورگر متوقف شود
        print(f"[TradeExecutor] خطا هنگام کلیک خودکار: {exc}")
        return False
