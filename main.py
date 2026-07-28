"""
اسکریپت اصلی — هماهنگ‌کنندهٔ تمام بخش‌ها
==========================================

این فایل بخش‌های قبلی (browser_session, feature_engineering, market_structure,
state_tracker, data_logger, trade_click_listener) را با asyncio.gather به‌صورت
هم‌زمان (Concurrent) اجرا می‌کند:

    Task 1: خواندن تیک‌های خام از صف و به‌روزرسانی TickBuffer/CandleAggregator
    Task 2: گوش‌دادن به ورودی ترمینال، فقط برای خروج امن ('q') یا ثبت دستی
            معامله به‌عنوان جایگزین (c/p) — روش اصلی ثبت معامله، کلیک واقعی
            روی دکمه‌های BUY/SELL خودِ پلتفرم است (task_click_listener)
    Task 3: نظارت بر سلامت اتصال WebSocket و تلاش برای Reconnect در صورت قطعی

نحوهٔ استفاده:
    python main.py
    - مرورگر باز می‌شود -> به‌صورت دستی وارد حساب دمو شوید -> Enter بزنید.
    - از این‌جا به بعد، دقیقاً مثل همیشه معامله کنید: هر کلیک واقعی روی دکمهٔ
      BUY یا SELL پلتفرم به‌طور خودکار شناسایی و ثبت می‌شود.
    - در ترمینال هم می‌توانید به‌جای کلیک، 'c' (CALL) یا 'p' (PUT) تایپ کنید
      (روش جایگزین/دستی)، و 'q' + Enter برای خروج امن.
"""

from __future__ import annotations

import asyncio

import config
from src.browser_session import (
    DealResultBuffer,
    WebSocketListener,
    launch_browser_and_wait_for_login,
    reconnect_page_if_needed,
)
from src.feature_engineering import TickBuffer, TickHistory, CandleAggregator
from src.market_structure import MarketStructureTracker
from src.state_tracker import TradeHistory
from src.data_logger import DataLogger
from src.trade_click_listener import attach_trade_button_listeners


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
    run_in_executor). روش اصلی ثبت معامله، کلیک واقعی روی دکمه‌های BUY/SELL
    خودِ پلتفرم است (نگاه کنید به src/trade_click_listener.py)؛ این تابع فقط
    یک راه جایگزین/دستی (c/p) و راه خروج امن (q) فراهم می‌کند.
    """
    loop = asyncio.get_event_loop()
    print("\nراهنما: حالا می‌توانید مثل همیشه روی BUY/SELL در پلتفرم کلیک کنید و خودکار ثبت می‌شود.")
    print("جایگزین دستی: 'c' + Enter برای CALL | 'p' + Enter برای PUT")
    print("'r' + Enter برای پاک‌کردن کامل دیتاست و شروع از صفر | 'q' + Enter برای خروج\n")

    while not stop_event.is_set():
        user_input = await loop.run_in_executor(None, input, "> ")
        command = user_input.strip().lower()

        if command == "c":
            data_logger.capture_entry("CALL")
        elif command == "p":
            data_logger.capture_entry("PUT")
        elif command == "r":
            print("⚠️  این کار کل دیتاست (CSV و SQLite) را برای همیشه پاک می‌کند.")
            confirm = await loop.run_in_executor(
                None, input, "برای تأیید 'yes' را تایپ کنید (هر چیز دیگری = لغو): "
            )
            if confirm.strip().lower() == "yes":
                data_logger.reset_dataset()
            else:
                print("لغو شد؛ داده‌ها دست‌نخورده ماندند.")
        elif command == "q":
            print("[Main] درخواست خروج دریافت شد...")
            stop_event.set()
        else:
            print("ورودی نامعتبر. از 'c'، 'p'، 'r' یا 'q' استفاده کنید.")


async def connection_watchdog_task(
    ws_listener: WebSocketListener,
    page,
    stop_event: asyncio.Event,
) -> None:
    """
    Task 3: هر چند ثانیه یک‌بار بررسی می‌کند که آیا WebSocket فعالی وجود دارد.
    اگر برای مدتی طولانی هیچ اتصالی وجود نداشت (مثلاً به‌خاطر قطعی اینترنت)،
    صفحه را Reload می‌کند تا اتصال از نو برقرار شود.

    نکته: درست بعد از لاگین دستی، برقراری اولین اتصال WebSocket طبیعتاً چند
    ثانیه طول می‌کشد (صفحه هنوز کامل بارگذاری نشده). قبل از این‌که اصلاً شروع
    به نظارت کنیم، به همین اتصال اول یک مهلت جداگانه (initial_grace_seconds)
    می‌دهیم — وگرنه همین تأخیر طبیعی را اشتباهی «قطعی» تشخیص می‌دهد و صفحه را
    بی‌دلیل Reload می‌کند (دقیقاً همان رفتاری که ممکن است دیده باشید).
    """
    initial_grace_seconds = config.INITIAL_CONNECTION_GRACE_SECONDS
    disconnect_grace_seconds = 15

    try:
        await asyncio.wait_for(ws_listener.connected_event.wait(), timeout=initial_grace_seconds)
    except asyncio.TimeoutError:
        pass  # اگر واقعاً بعد از این مدت هم وصل نشد، حلقهٔ زیر خودش دوباره تلاش می‌کند

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

    deal_buffer = DealResultBuffer()

    context, page = await launch_browser_and_wait_for_login()

    ws_listener = WebSocketListener(tick_queue, deal_buffer)
    ws_listener.attach(page)

    data_logger = DataLogger(
        tick_buffer, tick_history, candle_aggregator, market_structure, trade_history, deal_buffer
    )

    # شنود کلیک واقعی روی دکمه‌های BUY/SELL خودِ پلتفرم — روش اصلی ثبت معامله
    await attach_trade_button_listeners(
        page,
        on_call_click=lambda: data_logger.capture_entry("CALL"),
        on_put_click=lambda: data_logger.capture_entry("PUT"),
    )

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
