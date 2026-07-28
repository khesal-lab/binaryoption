"""
تشخیص کلیک واقعی روی دکمه‌های BUY/SELL خودِ پلتفرم
=======================================================

به‌جای این‌که کاربر در ترمینال حرف بزند (c/p)، این ماژول مستقیماً داخل صفحهٔ
مرورگر یک شنوندهٔ کلیک (Event Listener) روی خودِ دکمه‌های BUY و SELL پلتفرم
تزریق می‌کند. وقتی کاربر همان‌طور که همیشه معامله می‌کند روی این دکمه‌ها کلیک
می‌کند، آن کلیک واقعی شناسایی و به کد پایتون گزارش می‌شود.

روش کار:
    ۱. با page.expose_function یک تابع پایتون در اختیار جاوااسکریپت صفحه
       قرار می‌گیرد (window.__poBotOnTradeClick).
    ۲. با page.add_init_script یک اسکریپت تزریق می‌شود که در هر بارگذاری/
       Reload صفحه از نو اجرا می‌شود (چون Pocket Option یک SPA است و ممکن
       است دکمه‌ها بعداً و به‌صورت پویا رندر شوند، از MutationObserver هم
       استفاده می‌کنیم تا دکمه‌های جدید را هم پیدا کند).
    ۳. تشخیص دکمه صرفاً از روی متن آن (BUY / SELL) است — اگر پلتفرم متن یا
       زبان دکمه‌ها را تغییر داد، مقادیر config.BUY_BUTTON_TEXT /
       config.SELL_BUTTON_TEXT را اصلاح کنید.
"""

from __future__ import annotations

from typing import Callable

from playwright.async_api import Page

import config


def _build_injection_script() -> str:
    buy_text = config.BUY_BUTTON_TEXT.upper()
    sell_text = config.SELL_BUTTON_TEXT.upper()
    return f"""
    (function() {{
        const BUY_TEXT = {buy_text!r};
        const SELL_TEXT = {sell_text!r};

        function normalize(text) {{
            return (text || '').trim().toUpperCase();
        }}

        function attach(el, direction) {{
            if (el.__poBotAttached) return;
            el.__poBotAttached = true;
            el.addEventListener('click', () => {{
                if (window.__poBotOnTradeClick) {{
                    window.__poBotOnTradeClick(direction);
                }}
            }});
        }}

        function scan() {{
            document.querySelectorAll('button, div, span, a').forEach((el) => {{
                // فقط عنصرهای برگ (بدون فرزند عنصری) را بررسی می‌کنیم تا چند بار
                // برای یک دکمهٔ تودرتو ثبت نشود
                if (el.children.length > 0) return;
                const text = normalize(el.textContent);
                if (text === BUY_TEXT) {{
                    attach(el.closest('button') || el, 'CALL');
                }} else if (text === SELL_TEXT) {{
                    attach(el.closest('button') || el, 'PUT');
                }}
            }});
        }}

        // این اسکریپت در ابتدای بارگذاری صفحه (document-start) اجرا می‌شود، یعنی
        // پیش از آن‌که document.body ساخته شود. document.documentElement (خودِ
        // <html>) همیشه از همان ابتدا وجود دارد، پس observer را روی آن می‌بندیم
        // (observe با subtree:true تغییرات داخل body را هم که بعداً ساخته شود پوشش می‌دهد).
        function start() {{
            scan();
            new MutationObserver(scan).observe(document.documentElement, {{ childList: true, subtree: true }});
        }}

        if (document.documentElement) {{
            start();
        }} else {{
            document.addEventListener('DOMContentLoaded', start, {{ once: true }});
        }}
    }})();
    """


async def attach_trade_button_listeners(
    page: Page,
    on_call_click: Callable[[], None],
    on_put_click: Callable[[], None],
) -> None:
    """
    این تابع باید یک‌بار، بعد از لاگین دستی کاربر، صدا زده شود. از آن پس، هر
    کلیک واقعی روی دکمهٔ BUY باعث صدا زده‌شدن on_call_click و هر کلیک روی
    SELL باعث صدا زده‌شدن on_put_click می‌شود — دقیقاً هم‌زمان با لحظه‌ای که
    کاربر آن معامله را واقعاً در پلتفرم باز می‌کند.
    """

    async def _handle_click(direction: str) -> None:
        if direction == "CALL":
            on_call_click()
        else:
            on_put_click()

    await page.expose_function("__poBotOnTradeClick", _handle_click)

    injection = _build_injection_script()
    # برای این‌که بعد از هر Reload/ناوبری دوباره تزریق شود:
    await page.add_init_script(injection)
    # و برای این‌که همین حالا هم روی صفحهٔ از قبل بارگذاری‌شده اجرا شود:
    await page.evaluate(injection)

    print(f"[TradeClickListener] شنود کلیک روی دکمه‌های "
          f"'{config.BUY_BUTTON_TEXT}' و '{config.SELL_BUTTON_TEXT}' فعال شد.")
