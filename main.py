"""
اسکریپت اصلی — هماهنگ‌کنندهٔ تمام بخش‌ها
==========================================

این فایل چهار بخش قبلی (browser_session, feature_engineering, state_tracker,
data_logger) را با asyncio.gather به‌صورت هم‌زمان (Concurrent) اجرا می‌کند:

    Task 1: خواندن تیک‌های خام از صف و به‌روزرسانی TickBuffer/CandleAggregator
    Task 2: گوش‌دادن به ورودی کاربر در ترمینال برای ثبت معامله (کلید c/p/q)
    Task 3: نظارت بر سلامت اتصال WebSocket و تلاش برای Reconnect در صورت قطعی

نحوهٔ استفاده:
    python main.py
    - مرورگر باز می‌شود -> به‌صورت دستی وارد حساب دمو شوید -> Enter بزنید.
    - در ترمینال:
        c + Enter  -> ثبت معاملهٔ CALL (خرید/صعودی) در همین لحظه
        p + Enter  -> ثبت معاملهٔ PUT  (فروش/نزولی) در همین لحظه
        q + Enter  -> خروج امن از برنامه
"""

from __future__ import annotations

import asyncio

import config
from src.browser_session import (
    WebSocketListener,
    launch_browser_and_wait_for_login,
    reconnect_page_if_needed,
)
from src.feature_engineering import TickBuffer, TickHistory, CandleAggregator
from src.market_structure import MarketStructureTracker
from src.state_tracker import TradeHistory
from src.data_logger import DataLogger


async def tick_consumer_task(
    tick_queue: asyncio.Queue,
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
    market_structure: MarketStructureTracker,
) -> None:
    """
    Task 1: پیوسته از صف تیک‌های خام می‌خواند، بافرها و کندل‌ساز را به‌روز می‌کند.
    هر بار که یک کندل یک‌دقیقه‌ای بسته شود، همان کندل به MarketStructureTracker
    داده می‌شود تا سوئینگ/لگ/حمایت‌مقاومت/روند را به‌روزرسانی کند.
    این حلقه باید همیشه سریع باشد تا هیچ تأخیری روی داده‌های real-time نیفتد.
    """
    while True:
        tick = await tick_queue.get()
        tick_buffer.add(tick)
        tick_history.add(tick)
        closed_candle = candle_aggregator.add_tick(tick)
        if closed_candle is not None:
            market_structure.ingest_closed_candle(closed_candle)


async def hotkey_listener_task(data_logger: DataLogger, stop_event: asyncio.Event) -> None:
    """
    Task 2: ورودی ترمینال را بدون بلاک‌کردن event loop می‌خواند (با
    run_in_executor) و بر اساس آن معاملهٔ CALL/PUT ثبت می‌کند یا برنامه را
    خاتمه می‌دهد.

    توجه: این روش نیازمند فوکوس روی ترمینال است (نه هات‌کی سراسری/Global).
    اگر نیاز به هات‌کی سراسری (حتی وقتی فوکوس روی مرورگر است) دارید، می‌توانید
    از کتابخانهٔ خارجی `keyboard` استفاده کنید؛ آن کتابخانه در لینوکس به دسترسی
    root نیاز دارد که به همین دلیل این‌جا استفاده نشده است.
    """
    loop = asyncio.get_event_loop()
    print("\nراهنما: 'c' + Enter برای CALL | 'p' + Enter برای PUT | 'q' + Enter برای خروج\n")

    while not stop_event.is_set():
        user_input = await loop.run_in_executor(None, input, "> ")
        command = user_input.strip().lower()

        if command == "c":
            data_logger.capture_entry("CALL")
        elif command == "p":
            data_logger.capture_entry("PUT")
        elif command == "q":
            print("[Main] درخواست خروج دریافت شد...")
            stop_event.set()
        else:
            print("ورودی نامعتبر. از 'c'، 'p' یا 'q' استفاده کنید.")


async def connection_watchdog_task(
    ws_listener: WebSocketListener,
    page,
    stop_event: asyncio.Event,
) -> None:
    """
    Task 3: هر چند ثانیه یک‌بار بررسی می‌کند که آیا WebSocket فعالی وجود دارد.
    اگر برای مدتی طولانی هیچ اتصالی وجود نداشت (مثلاً به‌خاطر قطعی اینترنت)،
    صفحه را Reload می‌کند تا اتصال از نو برقرار شود.
    """
    disconnect_grace_seconds = 15
    while not stop_event.is_set():
        await asyncio.sleep(5)
        if not ws_listener.connected_event.is_set():
            print(f"[Watchdog] هیچ اتصال WebSocket فعالی نیست؛ "
                  f"{disconnect_grace_seconds} ثانیه صبر و سپس تلاش برای اتصال مجدد...")
            try:
                await asyncio.wait_for(
                    ws_listener.connected_event.wait(), timeout=disconnect_grace_seconds
                )
            except asyncio.TimeoutError:
                await reconnect_page_if_needed(page)


async def main() -> None:
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    tick_buffer = TickBuffer()
    tick_history = TickHistory()
    candle_aggregator = CandleAggregator()
    market_structure = MarketStructureTracker()
    trade_history = TradeHistory()

    context, page = await launch_browser_and_wait_for_login()

    ws_listener = WebSocketListener(tick_queue)
    ws_listener.attach(page)

    data_logger = DataLogger(tick_buffer, tick_history, candle_aggregator, market_structure, trade_history)

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(
            tick_consumer_task(tick_queue, tick_buffer, tick_history, candle_aggregator, market_structure)
        ),
        asyncio.create_task(hotkey_listener_task(data_logger, stop_event)),
        asyncio.create_task(connection_watchdog_task(ws_listener, page, stop_event)),
    ]

    try:
        # منتظر می‌مانیم تا کاربر دستور خروج ('q') بدهد
        await stop_event.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        data_logger.close()
        await context.close()
        print(f"[Main] دیتاست در {config.CSV_LOG_PATH} و {config.SQLITE_DB_PATH} ذخیره شد.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Main] خروج با Ctrl+C.")
