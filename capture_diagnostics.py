"""
اسکریپت تشخیصی: ثبت جامع اطلاعات برای پیدا کردن سیگنال نتیجهٔ معامله
=====================================================================

هدف این اسکریپت، برخلاف main.py، ساخت دیتاست نیست — فقط **جمع‌آوری خام هر نوع
داده‌ای** است که ممکن است نتیجهٔ برد/باخت یک معامله در آن پنهان باشد، تا بشود
با بررسی آن فهمید نتیجه از کدام کانال و با چه فرمتی قابل استخراج است:

    ۱. تمام فریم‌های WebSocket، هم ارسالی و هم دریافتی (با دقت میلی‌ثانیه)
    ۲. تمام درخواست/پاسخ‌های شبکه از نوع XHR/Fetch (که خیلی از بروکرها نتیجهٔ
       تسویهٔ معامله را از این طریق برمی‌گردانند، نه فقط WebSocket)
    ۳. پیام‌های console صفحه (اگر پلتفرم چیزی لاگ کند)

نحوهٔ استفاده:
    python capture_diagnostics.py
    - مثل main.py دستی وارد حساب دمو شوید و Enter بزنید.
    - چند معامله انجام دهید — حتماً ترکیبی از چند برد و چند باخت (که بعداً با
      نگاه‌کردن به این‌که کدام لحظه‌ها برد/باخت بودند، بشود لاگ‌ها را match کرد).
    - در ترمینال 'q' + Enter بزنید تا اسکریپت با ذخیرهٔ همهٔ لاگ‌ها تمام شود.
    - فایل‌های زیر در data/diagnostics/ ساخته می‌شوند؛ همه را بفرستید:
        ws_frames.log       فریم‌های وب‌ساکت (SENT/RECV) با زمان
        network.log         درخواست/پاسخ‌های XHR و Fetch با زمان
        console.log         پیام‌های console صفحه
"""

from __future__ import annotations

import asyncio
import time

from playwright.async_api import Page, Request, Response, WebSocket

import config
from src.browser_session import launch_browser_and_wait_for_login

config.DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

WS_LOG_PATH = config.DIAGNOSTICS_DIR / "ws_frames.log"
NETWORK_LOG_PATH = config.DIAGNOSTICS_DIR / "network.log"
CONSOLE_LOG_PATH = config.DIAGNOSTICS_DIR / "console.log"

_MAX_BODY_CHARS = 3000  # برش بدنهٔ پاسخ‌های بزرگ برای جلوگیری از لاگ‌های حجیم


def _append(path, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as exc:
        print(f"[Diagnostics] خطا در نوشتن {path}: {exc}")


def _attach_websocket_logging(page: Page) -> None:
    """تمام فریم‌های ارسالی و دریافتیِ هر WebSocket صفحه را با زمان ثبت می‌کند."""

    def on_websocket(ws: WebSocket) -> None:
        _append(WS_LOG_PATH, f"[{time.time():.3f}] WS-OPEN {ws.url}")

        def on_sent(payload):
            text = payload if isinstance(payload, str) else payload.decode("utf-8", errors="ignore")
            _append(WS_LOG_PATH, f"[{time.time():.3f}] SENT {text[:_MAX_BODY_CHARS]}")

        def on_received(payload):
            text = payload if isinstance(payload, str) else payload.decode("utf-8", errors="ignore")
            _append(WS_LOG_PATH, f"[{time.time():.3f}] RECV {text[:_MAX_BODY_CHARS]}")

        def on_close():
            _append(WS_LOG_PATH, f"[{time.time():.3f}] WS-CLOSE {ws.url}")

        ws.on("framesent", on_sent)
        ws.on("framereceived", on_received)
        ws.on("close", lambda: on_close())

    page.on("websocket", on_websocket)


def _attach_network_logging(page: Page) -> None:
    """
    فقط درخواست/پاسخ‌های XHR و Fetch را لاگ می‌کند (نه تصاویر/CSS/JS و غیره) تا
    حجم لاگ قابل مدیریت بماند — این‌ها معمول‌ترین راهی هستند که بروکرها نتیجهٔ
    تسویهٔ معامله را (جدا از WebSocket) برمی‌گردانند.
    """

    def on_request(request: Request) -> None:
        if request.resource_type not in ("xhr", "fetch"):
            return
        post_data = request.post_data or ""
        _append(
            NETWORK_LOG_PATH,
            f"[{time.time():.3f}] REQUEST {request.method} {request.url} "
            f"BODY={post_data[:_MAX_BODY_CHARS]}",
        )

    def on_response(response: Response) -> None:
        if response.request.resource_type not in ("xhr", "fetch"):
            return

        async def _log_body():
            try:
                body = await response.text()
            except Exception as exc:  # noqa: BLE001 - می‌خواهیم هر خطایی را بی‌سروصدا رد کنیم
                body = f"<غیرقابل‌خواندن: {exc}>"
            _append(
                NETWORK_LOG_PATH,
                f"[{time.time():.3f}] RESPONSE {response.status} {response.url} "
                f"BODY={body[:_MAX_BODY_CHARS]}",
            )

        asyncio.create_task(_log_body())

    page.on("request", on_request)
    page.on("response", on_response)


def _attach_console_logging(page: Page) -> None:
    def on_console(msg) -> None:
        _append(CONSOLE_LOG_PATH, f"[{time.time():.3f}] [{msg.type}] {msg.text}")

    page.on("console", on_console)


async def main() -> None:
    context, page = await launch_browser_and_wait_for_login()

    _attach_websocket_logging(page)
    _attach_network_logging(page)
    _attach_console_logging(page)

    print("\n" + "=" * 70)
    print("در حال ثبت داده. لطفاً چند معامله (ترکیبی از چند برد و چند باخت) انجام دهید.")
    print(f"لاگ‌ها این‌جا ذخیره می‌شوند: {config.DIAGNOSTICS_DIR}")
    print("برای پایان و ذخیره، در ترمینال 'q' را تایپ و Enter بزنید.")
    print("=" * 70 + "\n")

    loop = asyncio.get_event_loop()
    while True:
        command = await loop.run_in_executor(None, input, "> ")
        if command.strip().lower() == "q":
            break

    await context.close()
    print(f"[Diagnostics] تمام شد. فایل‌های data/diagnostics/*.log را برای بررسی ارسال کنید.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Diagnostics] خروج با Ctrl+C.")
