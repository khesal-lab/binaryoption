"""
اسکریپت اختصاصی جمع‌آوری داده با استراتژی ثابت تست حمایت/مقاومت
====================================================================

بر خلاف main.py (که یا منتظر کلیک دستی شما می‌ماند یا از یک مدل آموزش‌دیده
برای تصمیم‌گیری استفاده می‌کند)، این اسکریپت یک استراتژی کاملاً قانون‌محور و
ثابت اجرا می‌کند - هدف فقط جمع‌آوری سریع و متنوع دادهٔ برچسب‌خورده (با نتیجهٔ
واقعی خودِ پلتفرم) است، نه سودآوری زنده. وین‌ریت این معاملات مهم نیست.

استراتژی (جزئیات در src/level_strategy.py):
    ۱. کف/سقف کندلِ قبلیِ بسته‌شده به‌عنوان حمایت/مقاومت: رسیدن قیمت به سقف
       -> PUT (فرض بازگشت به پایین)، رسیدن به کف -> CALL (فرض بازگشت به بالا).
    ۲. نقاطی داخل کندلِ در حال شکل‌گیری که قیمت بیش از یک‌بار در آن‌جا متوقف/
       برگشته (حمایت/مقاومت داخلی) - همان منطق PUT/CALL.

نحوهٔ استفاده: دقیقاً مثل main.py - دستی لاگین کنید، Enter بزنید، و از آن‌جا
به بعد این اسکریپت خودش (بدون نیاز به کلیک شما) روی سطوح تشخیص‌داده‌شده
معامله باز می‌کند. نتیجه در همان دیتاست معمول (config.CSV_LOG_PATH /
SQLITE_DB_PATH) با meta_trade_source="level_strategy" ثبت می‌شود - کاملاً
قابل تفکیک از معاملات دستی یا معاملات ربات مدل‌محور (main.py با AUTO_TRADE_ENABLED).

نسخهٔ استراتژی: config.LEVEL_STRATEGY_VARIANT مشخص می‌کند کدام‌یک از
src/level_strategy.py ("basic"، پیش‌فرض) یا src/level_strategy_optimized.py
("optimized" - معکوس‌سازی جهت بعد از باخت‌های متوالی در حالت نوسان دوطرفه)
اجرا شود.

هشدار: فقط روی حساب دمو اجرا کنید.
"""

from __future__ import annotations

import asyncio
import time

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
from src.trade_executor import click_trade_button
from src.level_strategy import LevelStrategyTracker
from src.level_strategy_optimized import OptimizedLevelStrategyTracker
from src.symbol_tracker import SymbolSwitchDetector
from src.config_reloader import config_hot_reload_task


async def tick_consumer_and_strategy_task(
    tick_queue: asyncio.Queue,
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
    market_structure: MarketStructureTracker,
    level_tracker: LevelStrategyTracker | OptimizedLevelStrategyTracker,
    page,
    click_source_holder: dict,
    data_logger: DataLogger,
    micro_candle_aggregator: CandleAggregator,
) -> None:
    """
    نسخهٔ گسترش‌یافتهٔ tick_consumer_task اصلی (main.py): علاوه بر
    به‌روزرسانی بافرها/کندل‌ساز/ساختار بازار، بعد از هر تیک استراتژی تست
    سطوح را هم بررسی می‌کند و در صورت سیگنال، دکمهٔ واقعی BUY/SELL را کلیک
    می‌کند. مقایسهٔ تیک فعلی با تیک قبلی برای تشخیص «عبور از سطح» لازم است،
    پس تیک قبلی این‌جا نگه داشته می‌شود.

    اگر کاربر نماد معاملاتی را دستی در پلتفرم عوض کند، SymbolSwitchDetector
    این تغییر را از روی symbol تیک‌های ورودی تشخیص می‌دهد و علاوه بر بافرهای
    قیمتی/ساختار بازار، استراتژی سطوح (level_tracker) هم ریست می‌شود و
    prev_tick پاک می‌شود - وگرنه اولین مقایسهٔ بعد از تعویض نماد، تیک نماد
    جدید را با تیک نماد قدیمی (با سطح قیمتی کاملاً بی‌ربط) مقایسه می‌کند.
    """
    last_click_monotonic = 0.0
    prev_tick = None
    symbol_detector = SymbolSwitchDetector()

    while True:
        tick = await tick_queue.get()

        if symbol_detector.check(tick.symbol):
            print(f"[CollectData] تغییر نماد شناسایی شد (نماد جدید: {tick.symbol}) - "
                  f"بافرهای قیمتی، ساختار بازار و استراتژی سطوح ریست شدند.")
            tick_buffer.reset()
            tick_history.reset()
            candle_aggregator.reset()
            market_structure.reset()
            level_tracker.reset()
            micro_candle_aggregator.reset()
            prev_tick = None

        if tick.symbol != symbol_detector.current_symbol:
            # این تیک مربوط به نماد دیگری است که هم‌زمان (مثلاً به‌خاطر یک هشدار قیمتی
            # یا آیتم در لیست علاقه‌مندی‌ها) روی همان وب‌سوکت جاری شده، نه نماد چارتِ
            # فعال. تا وقتی SymbolSwitchDetector این نماد را به‌عنوان تعویض واقعی تأیید
            # نکرده، نباید وارد کندل/بافر/استراتژی شود - وگرنه هم high/low کندل با
            # قیمت نماد بی‌ربط آلوده می‌شود و هم level_tracker.check_signal ممکن است
            # prev_tick و tick را از دو نماد مختلف مقایسه کند و معاملهٔ واقعی اشتباه بزند.
            continue

        tick_buffer.add(tick)
        tick_history.add(tick)
        closed_candle = candle_aggregator.add_tick(tick)
        if closed_candle is not None:
            market_structure.ingest_closed_candle(closed_candle)
            level_tracker.on_new_candle(closed_candle)
        micro_candle_aggregator.add_tick(tick)

        current_candle = candle_aggregator.current
        if current_candle is not None and prev_tick is not None and not data_logger.is_bot_trading_blocked():
            signal = level_tracker.check_signal(prev_tick, tick, current_candle, tick_history)
            if signal is not None:
                now = time.monotonic()
                if now - last_click_monotonic >= config.LEVEL_STRATEGY_MIN_CLICK_INTERVAL_SECONDS:
                    last_click_monotonic = now
                    click_source_holder["source"] = "level_strategy"
                    asyncio.create_task(click_trade_button(page, signal))

        prev_tick = tick


async def hotkey_listener_task(data_logger: DataLogger, stop_event: asyncio.Event) -> None:
    loop = asyncio.get_event_loop()
    print("\nحالت جمع‌آوری داده با استراتژی تست حمایت/مقاومت فعال است.")
    print("این اسکریپت خودش روی سطوح تشخیص‌داده‌شده معامله باز می‌کند؛ نیازی به کلیک شما نیست.")
    print("'resume' + Enter برای فعال‌کردن دوبارهٔ معاملهٔ خودکار بعد از توقف به‌خاطر پی‌آوت پایین "
          "(بدون نیاز به ری‌استارت - مثلاً بعد از تغییر دستی ارز)")
    print("'r' + Enter برای پاک‌کردن کامل دیتاست و شروع از صفر | 'q' + Enter برای خروج\n")

    while not stop_event.is_set():
        user_input = await loop.run_in_executor(None, input, "> ")
        command = user_input.strip().lower()

        if command == "resume":
            if data_logger.resume_trading():
                print("[CollectData] معاملهٔ خودکار دوباره فعال شد.")
            else:
                print("[CollectData] معاملهٔ خودکار همین الان هم متوقف نبوده است.")
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
            print("[CollectData] درخواست خروج دریافت شد...")
            stop_event.set()
        else:
            print("ورودی نامعتبر. از 'resume'، 'r' یا 'q' استفاده کنید.")


async def connection_watchdog_task(
    ws_listener: WebSocketListener,
    page,
    stop_event: asyncio.Event,
) -> None:
    initial_grace_seconds = config.INITIAL_CONNECTION_GRACE_SECONDS
    disconnect_grace_seconds = 15

    try:
        await asyncio.wait_for(ws_listener.connected_event.wait(), timeout=initial_grace_seconds)
    except asyncio.TimeoutError:
        pass

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
    micro_candle_aggregator = CandleAggregator(timeframe_seconds=config.MICRO_CANDLE_TIMEFRAME_SECONDS)
    market_structure = MarketStructureTracker()
    trade_history = TradeHistory()

    if config.LEVEL_STRATEGY_VARIANT == "optimized":
        level_tracker = OptimizedLevelStrategyTracker()
        print("[CollectData] نسخهٔ optimized استراتژی سطوح فعال است "
              f"(معکوس‌سازی جهت بعد از {config.LEVEL_STRATEGY_REVERSE_AFTER_LOSSES} باخت متوالی "
              "در حالت نوسان دوطرفه).")
    else:
        level_tracker = LevelStrategyTracker()

    deal_buffer = DealResultBuffer()

    context, page = await launch_browser_and_wait_for_login()

    ws_listener = WebSocketListener(tick_queue, deal_buffer)
    ws_listener.attach(page)

    def _on_trade_result(source: str, result: int) -> None:
        # فقط نسخهٔ optimized متد on_result دارد و فقط نتیجهٔ معاملات همین
        # استراتژی (نه معاملات دستی احتمالی) باید بازخوردش داده شود.
        if source == "level_strategy" and hasattr(level_tracker, "on_result"):
            level_tracker.on_result(result)

    data_logger = DataLogger(
        tick_buffer, tick_history, candle_aggregator, market_structure, trade_history, deal_buffer,
        page=page, on_result_callback=_on_trade_result,
        micro_candle_aggregator=micro_candle_aggregator,
    )
    await data_logger.install_page_controls()
    print(f"[CollectData] نظارت پی‌آوت فعال: اگر پی‌آوت واقعیِ یک معامله کمتر از "
          f"{config.MIN_PAYOUT_PERCENT}٪ باشد، معاملهٔ خودکار این اسکریپت متوقف می‌شود.")
    print("[CollectData] یک دکمهٔ «توقف/ازسرگیری معاملهٔ خودکار» هم پایین-چپ صفحهٔ مرورگر اضافه شد.")
    print(f"[CollectData] توقف بعد از باخت متوالی: {config.CONSECUTIVE_LOSSES_FOR_COOLDOWN} باخت پیاپی -> "
          f"{config.CONSECUTIVE_LOSS_COOLDOWN_SECONDS:.0f} ثانیه توقف.")
    print("[CollectData] بارگذاری زندهٔ config.py فعال است - تغییر config.py در حین اجرا (بدون نیاز به "
          "ری‌استارت) روی مقادیری مثل توقف باخت متوالی/پی‌آوت اعمال می‌شود.")

    # وقتی استراتژی دکمه را برنامه‌ای کلیک می‌کند، درست قبل از کلیک این مقدار
    # به "level_strategy" تغییر می‌کند تا شنوندهٔ کلیک زیر بداند این معامله را
    # با کدام برچسب ثبت کند (تفکیک از معاملات دستی احتمالی در همین اجرا).
    click_source_holder = {"source": "manual"}

    def _make_on_click(direction: str):
        def _callback() -> None:
            source = click_source_holder["source"]
            click_source_holder["source"] = "manual"
            data_logger.capture_entry(direction, source=source)
        return _callback

    await attach_trade_button_listeners(
        page,
        on_call_click=_make_on_click("CALL"),
        on_put_click=_make_on_click("PUT"),
    )

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(
            tick_consumer_and_strategy_task(
                tick_queue, tick_buffer, tick_history, candle_aggregator, market_structure,
                level_tracker, page, click_source_holder, data_logger,
                micro_candle_aggregator,
            )
        ),
        asyncio.create_task(hotkey_listener_task(data_logger, stop_event)),
        asyncio.create_task(connection_watchdog_task(ws_listener, page, stop_event)),
        asyncio.create_task(config_hot_reload_task(stop_event)),
    ]

    try:
        await stop_event.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        data_logger.close()
        await context.close()
        print(f"[CollectData] دیتاست در {config.CSV_LOG_PATH} و {config.SQLITE_DB_PATH} ذخیره شد.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[CollectData] خروج با Ctrl+C.")
