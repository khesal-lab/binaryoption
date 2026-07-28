"""
تشخیص کلیک واقعی روی دکمه‌های BUY/SELL خودِ پلتفرم
=======================================================

به‌جای این‌که کاربر در ترمینال حرف بزند (c/p)، این ماژول مستقیماً داخل صفحهٔ
مرورگر یک شنوندهٔ کلیک (Event Listener) تزریق می‌کند. وقتی کاربر همان‌طور که
همیشه معامله می‌کند روی دکمهٔ BUY یا SELL کلیک می‌کند، آن کلیک واقعی شناسایی و
به کد پایتون گزارش می‌شود.

روش کار (Event Delegation): به‌جای اتصال جداگانه به تک‌تک عنصرهای دکمه (که
اگر دکمه یک <div> با آیکون/فاصلهٔ اطراف متن باشد، کلیک روی هر جای دیگر دکمه
غیر از دقیقاً روی حروف را از دست می‌دهد)، یک شنوندهٔ واحد روی کل `document`
در فاز Capture می‌گذاریم. با هر کلیک، از نقطهٔ واقعی کلیک (event.target) چند
سطح به بالا حرکت می‌کنیم و در هر سطح متن آن عنصر را با BUY/SELL مقایسه
می‌کنیم — این‌طوری کلیک روی هر جای بصریِ دکمه (نه فقط دقیقاً روی متن) هم گرفته
می‌شود، و چون شنونده روی خودِ document است، نیازی به دنبال‌کردن رندر شدن
دوبارهٔ دکمه‌ها (مثلاً توسط React) هم نیست.

تشخیص دکمه صرفاً از روی متن آن (BUY / SELL) است — اگر پلتفرم متن یا زبان
دکمه‌ها را تغییر داد، مقادیر config.BUY_BUTTON_TEXT / config.SELL_BUTTON_TEXT
را اصلاح کنید.
"""

from __future__ import annotations

from typing import Callable

from playwright.async_api import Page

import config

# حداکثر تعداد سطحی که از نقطهٔ کلیک به سمت اجداد بالا می‌رویم تا کانتینر
# واقعیِ دکمه را پیدا کنیم (کافی برای چند لایه <span>/<div> تودرتوی معمول).
_MAX_ANCESTOR_LEVELS = 6


def _build_injection_script() -> str:
    buy_text = config.BUY_BUTTON_TEXT.upper()
    sell_text = config.SELL_BUTTON_TEXT.upper()
    return f"""
    (function() {{
        if (window.__poBotClickPatched) return;
        window.__poBotClickPatched = true;

        const BUY_TEXT = {buy_text!r};
        const SELL_TEXT = {sell_text!r};
        const MAX_LEVELS = {_MAX_ANCESTOR_LEVELS};
        // اگر متن کانتینر بیشتر از این تعداد کاراکتر باشد، یعنی به یک کانتینر
        // بزرگ‌تر از خودِ دکمه رسیده‌ایم (مثلاً کل پنل معاملات) و باید دست نگه داریم
        const MAX_TEXT_LENGTH = 20;
        const buyPattern = new RegExp('\\\\b' + BUY_TEXT + '\\\\b');
        const sellPattern = new RegExp('\\\\b' + SELL_TEXT + '\\\\b');

        function normalize(text) {{
            return (text || '').trim().toUpperCase();
        }}

        function findDirection(startEl) {{
            // متن کانتینر ممکن است شامل آیکون یا کاراکترهای دیگر هم باشد (مثلاً
            // «▲BUY»)، پس به‌جای برابری دقیق، وجود کلمهٔ کامل BUY/SELL را با
            // مرز کلمه (\\b) بررسی می‌کنیم تا با کلماتی مثل «BUYER» اشتباه نگیرد.
            let node = startEl;
            for (let i = 0; i < MAX_LEVELS && node; i++, node = node.parentElement) {{
                const text = normalize(node.textContent);
                if (text.length === 0 || text.length > MAX_TEXT_LENGTH) continue;
                if (buyPattern.test(text)) return 'CALL';
                if (sellPattern.test(text)) return 'PUT';
            }}
            return null;
        }}

        // فاز Capture (آرگومان سوم true) تا حتی اگر خودِ پلتفرم بعداً
        // stopPropagation صدا بزند، ما زودتر از آن کلیک را گرفته باشیم.
        document.addEventListener('click', function(event) {{
            const direction = findDirection(event.target);
            if (direction && window.__poBotOnTradeClick) {{
                window.__poBotOnTradeClick(direction);
            }}
        }}, true);
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
