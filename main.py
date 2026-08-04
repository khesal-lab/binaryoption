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
import time

import config
from src.browser_session import (
    AssetPayoutTracker,
    DealResultBuffer,
    WebSocketListener,
    launch_browser_and_wait_for_login,
    reconnect_page_if_needed,
)
from src.feature_engineering import (
    TickBuffer,
    TickHistory,
    CandleAggregator,
    build_feature_snapshot,
    compute_level_strategy_context,
)
from src.market_structure import MarketStructureTracker
from src.state_tracker import TradeHistory
from src.data_logger import DataLogger
from src.trade_click_listener import attach_trade_button_listeners
from src.trade_executor import click_trade_button
from src.live_predictor import LivePredictor
from src.symbol_tracker import SymbolSwitchDetector
from src.config_reloader import config_hot_reload_task


async def tick_consumer_task(
    tick_queue: asyncio.Queue,
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
    market_structure: MarketStructureTracker,
    micro_candle_aggregator: CandleAggregator,
) -> None:
    """
    Task 1: پیوسته از صف تیک‌های خام می‌خواند، بافرها و کندل‌ساز را به‌روز می‌کند.
    هر بار که یک کندل یک‌دقیقه‌ای بسته شود، همان کندل به MarketStructureTracker
    داده می‌شود تا سوئینگ/لگ/حمایت‌مقاومت/روند را به‌روزرسانی کند. کندل‌ساز
    ریزِ چندثانیه‌ای (micro_candle_aggregator) هم موازی و از همان جریان تیک
    به‌روز می‌شود - فقط برای «شکل چارت» ریز در build_feature_snapshot استفاده
    می‌شود، نیازی به ساختار بازار جداگانه ندارد.
    این حلقه باید همیشه سریع باشد تا هیچ تأخیری روی داده‌های real-time نیفتد.

    اگر کاربر نماد معاملاتی را دستی در پلتفرم عوض کند، تیک‌های جدید symbol
    متفاوتی خواهند داشت - SymbolSwitchDetector این تغییر را تشخیص می‌دهد و
    تمام بافرهای قیمتی/ساختار بازار (که بر مبنای نماد قبلی ساخته شده‌اند)
    ریست می‌شوند، وگرنه فیچرهای بعدی ترکیبی از دو نماد بی‌ربط می‌شوند.
    """
    symbol_detector = SymbolSwitchDetector()

    while True:
        tick = await tick_queue.get()

        if symbol_detector.check(tick.symbol):
            print(f"[Main] تغییر نماد شناسایی شد (نماد جدید: {tick.symbol}) - "
                  f"بافرهای قیمتی و ساختار بازار ریست شدند.")
            tick_buffer.reset()
            tick_history.reset()
            candle_aggregator.reset()
            market_structure.reset()
            micro_candle_aggregator.reset()

        if tick.symbol != symbol_detector.current_symbol:
            # این تیک مربوط به نماد دیگری است که هم‌زمان (مثلاً به‌خاطر یک هشدار قیمتی
            # یا آیتم در لیست علاقه‌مندی‌ها) روی همان وب‌سوکت جاری شده، نه نماد چارتِ
            # فعال. تا وقتی SymbolSwitchDetector این نماد را به‌عنوان تعویض واقعی تأیید
            # نکرده، نباید وارد کندل/بافر/ساختار بازار شود؛ وگرنه high/low کندل با
            # قیمت یک نماد کاملاً بی‌ربط آلوده می‌شود.
            continue

        tick_buffer.add(tick)
        tick_history.add(tick)
        closed_candle = candle_aggregator.add_tick(tick)
        if closed_candle is not None:
            market_structure.ingest_closed_candle(closed_candle)
        micro_candle_aggregator.add_tick(tick)


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
    print("'resume' + Enter برای فعال‌کردن دوبارهٔ معاملهٔ خودکار بعد از توقف به‌خاطر پی‌آوت پایین "
          "(بدون نیاز به ری‌استارت - مثلاً بعد از تغییر دستی ارز)")
    print("'r' + Enter برای پاک‌کردن کامل دیتاست و شروع از صفر | 'q' + Enter برای خروج\n")

    while not stop_event.is_set():
        user_input = await loop.run_in_executor(None, input, "> ")
        command = user_input.strip().lower()

        if command == "c":
            data_logger.capture_entry("CALL")
        elif command == "p":
            data_logger.capture_entry("PUT")
        elif command == "resume":
            if data_logger.resume_trading():
                print("[Main] معاملهٔ خودکار دوباره فعال شد.")
            else:
                print("[Main] معاملهٔ خودکار همین الان هم متوقف نبوده است.")
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
            print("ورودی نامعتبر. از 'c'، 'p'، 'resume'، 'r' یا 'q' استفاده کنید.")


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


async def auto_trade_task(
    page,
    predictor: LivePredictor,
    tick_buffer: TickBuffer,
    tick_history: TickHistory,
    candle_aggregator: CandleAggregator,
    market_structure: MarketStructureTracker,
    trade_history: TradeHistory,
    click_source_holder: dict,
    pending_confidence_holder: dict,
    data_logger: DataLogger,
    stop_event: asyncio.Event,
    micro_candle_aggregator: CandleAggregator,
) -> None:
    """
    Task اختیاری: فقط وقتی config.AUTO_TRADE_ENABLED فعال و فایل مدل موجود
    باشد اجرا می‌شود. با فاصلهٔ کوتاه (poll_interval) وضعیت لحظه‌ای بازار را
    می‌سازد، از مدل برای هر دو جهت (CALL/PUT) احتمال برد می‌گیرد، و اگر
    بهترین جهت از آستانهٔ اطمینان config.AUTO_TRADE_CONFIDENCE_THRESHOLD عبور
    کرد، خودش روی همان دکمهٔ واقعی BUY/SELL کلیک می‌کند (src/trade_executor).
    برای این‌که ثبت نتیجه با معاملهٔ قبلی قاطی نشود، بعد از هر معاملهٔ خودکار
    حداقل config.AUTO_TRADE_COOLDOWN_SECONDS ثانیه صبر می‌کند.

    نکتهٔ مهم دربارهٔ زمان اجرای پیش‌بینی: تقریباً تمام دادهٔ ترین این مدل از
    معاملات level_strategy.py آمده - یعنی فقط از لحظاتی که قیمت دقیقاً به یک
    اکسترمم نزدیک سقف/کف کندل جاری برخورد کرده. اگر این تابع در هر poll_interval
    (بدون هیچ قیدی) از مدل پیش‌بینی بخواهد، مدل را در موقعیت‌هایی می‌سنجد که
    اصلاً چیزی شبیهشان ندیده (Train/Serve Skew) - همین می‌تواند دلیل اصلی
    وین‌ریت پایین‌تر AUTO_TRADE نسبت به خودِ level_strategy در عمل باشد. به
    همین دلیل، قبل از ساختن اسنپ‌شات کامل، ابتدا با compute_level_strategy_context
    بررسی می‌شود که آیا قیمت لحظه‌ای همین الان هم نزدیک یکی از همان سطوح
    کلیدی (نزدیک سقف/کف کندل) هست یا نه؛ اگر نه، این دور نادیده گرفته می‌شود.
    """
    poll_interval = 0.2
    last_trade_monotonic = 0.0
    diagnostics_interval_seconds = 30.0
    last_diagnostics_print = time.monotonic()
    # چون از این به بعد اکثر دورها به‌خاطر gating جدید (فقط نزدیک سطوح کلیدی
    # level_strategy) رد می‌شوند، بدون این شمارنده‌ها هیچ سرنخی از این نیست که
    # آیا اصلاً چیزی در حال بررسی هست یا مثلاً هنوز تیک کافی نرسیده - هر
    # چند ثانیه یک‌بار خلاصهٔ همین دلایل رد شدن چاپ می‌شود.
    skip_counts = {
        "blocked": 0, "cooldown": 0, "not_ready": 0, "no_candle": 0,
        "no_level_context": 0, "not_near_level": 0, "low_confidence": 0,
    }
    gate_pass_count = 0

    print(f"[AutoTrade] حالت معاملهٔ خودکار فعال شد | جهت‌گیری: "
          f"{config.AUTO_TRADE_DIRECTION_MODE} | آستانهٔ اطمینان: "
          f"{config.AUTO_TRADE_CONFIDENCE_THRESHOLD:.0%} | فاصلهٔ حداقلی بین معاملات: "
          f"{config.AUTO_TRADE_COOLDOWN_SECONDS} ثانیه")

    while not stop_event.is_set():
        await asyncio.sleep(poll_interval)

        now = time.monotonic()
        if now - last_diagnostics_print >= diagnostics_interval_seconds:
            print(f"[AutoTrade] گزارش {diagnostics_interval_seconds:.0f} ثانیهٔ اخیر: "
                  f"{gate_pass_count} بار وارد بررسیِ مدل شد، "
                  f"{skip_counts['not_near_level']} بار نزدیک هیچ سطح کلیدی‌ای نبود، "
                  f"{skip_counts['no_level_context']} بار هنوز اکسترمم/بازگشت قیمتی در کندل جاری شکل نگرفته بود، "
                  f"{skip_counts['no_candle']} بار کندل/تیک آماده نبود، "
                  f"{skip_counts['not_ready']} بار هنوز تیک/وب‌ساکت آماده نبود، "
                  f"{skip_counts['low_confidence']} بار اطمینان مدل کمتر از آستانه بود، "
                  f"{skip_counts['blocked']} بار معاملهٔ خودکار متوقف بود، "
                  f"{skip_counts['cooldown']} بار در فاصلهٔ خنک‌شدن بعد از معاملهٔ قبلی بود.")
            for key in skip_counts:
                skip_counts[key] = 0
            gate_pass_count = 0
            last_diagnostics_print = now

        if data_logger.is_bot_trading_blocked():
            skip_counts["blocked"] += 1
            continue

        if now - last_trade_monotonic < config.AUTO_TRADE_COOLDOWN_SECONDS:
            skip_counts["cooldown"] += 1
            continue

        latest = tick_buffer.latest()
        if latest is None or not tick_buffer.is_ready():
            skip_counts["not_ready"] += 1
            continue

        current_candle = candle_aggregator.current
        if current_candle is None or current_candle.range <= 0:
            skip_counts["no_candle"] += 1
            continue

        level_context = compute_level_strategy_context(tick_history, current_candle)
        if level_context is None:
            skip_counts["no_level_context"] += 1
            continue

        # آیا قیمت لحظه‌ای همین الان نزدیک یکی از همان سطوح کلیدی (نزدیک سقف/کف
        # کندل جاری) است؟ همان تلورانس خودِ level_strategy.py، تا این تشخیص «شبیه
        # لحظهٔ سیگنال‌دهی» با چیزی که مدل واقعاً از رویش ترین دیده هم‌خوان باشد.
        price_position = (latest.price - current_candle.low) / current_candle.range
        near_high_gap = abs(price_position - level_context["level_strategy_near_high_position"])
        near_low_gap = abs(price_position - level_context["level_strategy_near_low_position"])
        if (near_high_gap > config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO
                and near_low_gap > config.LEVEL_STRATEGY_MATCH_TOLERANCE_RATIO):
            skip_counts["not_near_level"] += 1
            continue

        gate_pass_count += 1
        snapshot = build_feature_snapshot(tick_buffer, tick_history, candle_aggregator, micro_candle_aggregator)
        snapshot.update(
            market_structure.get_features(latest.price, latest.timestamp, candle_aggregator.current)
        )
        snapshot.update(trade_history.as_feature_dict())

        if config.AUTO_TRADE_DIRECTION_MODE == "level_strategy":
            # جهت را خودِ مدل انتخاب نمی‌کند؛ همان جهتِ implied سطحی که قیمت
            # الان به آن نزدیک‌تر است (سقف یا کف کندل جاری) گرفته می‌شود و
            # مدل فقط اطمینانِ همان جهت را می‌سنجد (فیلتر، نه انتخاب‌گر).
            if near_high_gap <= near_low_gap:
                direction_value = level_context["level_strategy_near_high_implied_direction"]
            else:
                direction_value = level_context["level_strategy_near_low_implied_direction"]
            direction = "CALL" if direction_value == 1 else "PUT"
            confidence = predictor.predict_confidence_for_direction(snapshot, direction)
        else:
            direction, confidence = predictor.predict_direction(snapshot)

        if direction is None or confidence < config.AUTO_TRADE_CONFIDENCE_THRESHOLD:
            skip_counts["low_confidence"] += 1
            continue

        click_source_holder["source"] = "bot"
        pending_confidence_holder["confidence"] = confidence
        clicked = await click_trade_button(page, direction)
        if clicked:
            last_trade_monotonic = now
            print(f"[AutoTrade] معاملهٔ خودکار {direction} ثبت شد (اطمینان مدل: {confidence:.1%})")
        else:
            click_source_holder["source"] = "manual"
            pending_confidence_holder["confidence"] = None


async def main() -> None:
    tick_queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    tick_buffer = TickBuffer()
    tick_history = TickHistory()
    candle_aggregator = CandleAggregator()
    micro_candle_aggregator = CandleAggregator(timeframe_seconds=config.MICRO_CANDLE_TIMEFRAME_SECONDS)
    market_structure = MarketStructureTracker()
    trade_history = TradeHistory()

    deal_buffer = DealResultBuffer()
    asset_payout_tracker = AssetPayoutTracker()

    context, page = await launch_browser_and_wait_for_login()

    ws_listener = WebSocketListener(tick_queue, deal_buffer, asset_payout_tracker)
    ws_listener.attach(page)

    data_logger = DataLogger(
        tick_buffer, tick_history, candle_aggregator, market_structure, trade_history, deal_buffer, page=page,
        micro_candle_aggregator=micro_candle_aggregator,
        asset_payout_tracker=asset_payout_tracker,
    )
    await data_logger.install_page_controls()
    print(f"[Main] نظارت پی‌آوت فعال: اگر پی‌آوت واقعیِ یک معامله کمتر از "
          f"{config.MIN_PAYOUT_PERCENT}٪ باشد، معاملهٔ خودکار (AUTO_TRADE) متوقف می‌شود.")
    print("[Main] یک دکمهٔ «توقف/ازسرگیری معاملهٔ خودکار» هم پایین-چپ صفحهٔ مرورگر اضافه شد.")
    tiers_text = " | ".join(
        f"{n} باخت پیاپی -> {s:.0f} ثانیه" for n, s in config.CONSECUTIVE_LOSS_COOLDOWN_TIERS
    )
    print(f"[Main] توقف بعد از باخت متوالی (چند سطحی): {tiers_text}")
    print("[Main] بارگذاری زندهٔ config.py فعال است - تغییر config.py در حین اجرا (بدون نیاز به "
          "ری‌استارت) روی مقادیری مثل توقف باخت متوالی/پی‌آوت/آستانهٔ اطمینان اعمال می‌شود.")

    # وقتی معاملهٔ خودکار (auto_trade_task) دکمه را برنامه‌ای کلیک می‌کند، درست
    # قبل از کلیک این مقدار را به "bot" تغییر می‌دهد؛ شنوندهٔ کلیک زیر که هم
    # روی کلیک دستی کاربر و هم روی همین کلیک برنامه‌ای صدا زده می‌شود، این
    # مقدار را می‌خواند تا بداند این معامله را با کدام برچسب (manual/bot) ثبت کند.
    click_source_holder = {"source": "manual"}
    # همان الگوی click_source_holder: چون خودِ AUTO_TRADE مستقیماً capture_entry
    # را صدا نمی‌زند (بلکه دکمهٔ خودِ پلتفرم را کلیک می‌کند و این کلیک، شنوندهٔ
    # زیر را فعال می‌کند)، اطمینان مدل هم باید از همین طریق واسط منتقل شود تا
    # لحظهٔ ثبت معامله در دسترس باشد - برای معاملهٔ دستی همیشه None می‌ماند.
    pending_confidence_holder: dict = {"confidence": None}

    def _make_on_click(direction: str):
        def _callback() -> None:
            source = click_source_holder["source"]
            click_source_holder["source"] = "manual"
            confidence = pending_confidence_holder["confidence"]
            pending_confidence_holder["confidence"] = None
            data_logger.capture_entry(direction, source=source, confidence=confidence)
        return _callback

    # شنود کلیک واقعی روی دکمه‌های BUY/SELL خودِ پلتفرم — روش اصلی ثبت معامله
    await attach_trade_button_listeners(
        page,
        on_call_click=_make_on_click("CALL"),
        on_put_click=_make_on_click("PUT"),
    )

    predictor: LivePredictor | None = None
    if config.AUTO_TRADE_ENABLED:
        if config.LIVE_MODEL_JSON_PATH.exists() and config.LIVE_MODEL_FEATURES_PATH.exists():
            predictor = LivePredictor(config.LIVE_MODEL_JSON_PATH, config.LIVE_MODEL_FEATURES_PATH)
        else:
            print(f"[Main] AUTO_TRADE_ENABLED فعال است ولی فایل مدل پیدا نشد "
                  f"({config.LIVE_MODEL_JSON_PATH})؛ حالت معاملهٔ خودکار غیرفعال می‌ماند.")

    stop_event = asyncio.Event()

    tasks = [
        asyncio.create_task(
            tick_consumer_task(
                tick_queue, tick_buffer, tick_history, candle_aggregator, market_structure,
                micro_candle_aggregator,
            )
        ),
        asyncio.create_task(hotkey_listener_task(data_logger, stop_event)),
        asyncio.create_task(connection_watchdog_task(ws_listener, page, stop_event)),
        asyncio.create_task(config_hot_reload_task(stop_event)),
    ]

    if predictor is not None:
        tasks.append(
            asyncio.create_task(
                auto_trade_task(
                    page, predictor, tick_buffer, tick_history, candle_aggregator,
                    market_structure, trade_history, click_source_holder, pending_confidence_holder,
                    data_logger, stop_event, micro_candle_aggregator,
                )
            )
        )

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
