"""
بخش چهارم: ضبط لحظهٔ معامله و لیبل‌گذاری (Data Logger)
=========================================================

این ماژول قلب سیستم جمع‌آوری دیتاست است:

    ۱. وقتی کاربر یک کلید را می‌فشارد (CALL یا PUT)، تابع capture_entry تمام
       ویژگی‌های **نسبی** همان لحظه (سرعت/شتاب درصدی، ساختار کندل، سوئینگ/لگ،
       حمایت/مقاومت، روند، الگوی معاملات اخیر) را عکس‌برداری (Snapshot) می‌کند.
    ۲. بعد از گذشت TRADE_EXPIRY_SECONDS (پیش‌فرض ۳ ثانیه، دقیقاً مثل معاملات
       واقعی شما)، قیمت لحظهٔ انقضا را از TickBuffer می‌خواند و نتیجه را
       Win=1 / Loss=0 لیبل می‌زند.
    ۳. ردیف کامل (ویژگی‌ها + لیبل) در یک فایل CSV با pandas و هم‌زمان در یک
       دیتابیس SQLite ذخیره می‌شود.

نکتهٔ ستون‌ها: ستون‌هایی که با پیشوند `meta_` شروع می‌شوند (مثل meta_entry_price)
فقط برای ردیابی/دیباگ‌اند و **نباید** به‌عنوان ورودی مدل استفاده شوند — چون مقدار
خام قیمت هستند. تمام ستون‌های دیگر (غیر از meta_*، direction، result) نسبی‌اند
و برای آموزش مدل مناسب‌اند.

نکتهٔ نتیجهٔ معامله: از دو منبع تلاش می‌شود نتیجه گرفته شود:
    ۱. «نتیجهٔ خودِ پلتفرم» (ws_deal): با شنود پیام‌های WebSocket که شبیه
       اعلام نتیجهٔ معامله‌اند (تابع extract_deal_candidates در
       browser_session.py). چون فرمت دقیق این پیام‌ها مستند نیست، این روش
       هیوریستیک است و ممکن است نیاز به تنظیم داشته باشد.
    ۲. «محاسبهٔ داخلی» (tick_fallback): مقایسهٔ قیمت لحظهٔ ورود و خروج از روی
       تیک‌های خودمان — همیشه در دسترس است و اگر روش اول چیزی پیدا نکرد،
       جایگزین آن می‌شود.
ستون result_source مشخص می‌کند نتیجهٔ نهایی از کدام منبع آمده است.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import pandas as pd

import config
from src.browser_session import DealResultBuffer
from src.feature_engineering import TickBuffer, TickHistory, CandleAggregator, build_feature_snapshot
from src.market_structure import MarketStructureTracker
from src.state_tracker import TradeHistory

Direction = Literal["CALL", "PUT"]


@dataclass
class PendingTrade:
    """یک معاملهٔ باز که هنوز منتظر نتیجهٔ آن (بعد از ۳ ثانیه) هستیم."""
    direction: Direction
    entry_price: float
    entry_time: float
    entry_wall_time: float  # زمان محلی (time.time()) لحظهٔ کلیک، برای همبستگی با DealResultBuffer
    entry_symbol: str
    feature_snapshot: dict = field(default_factory=dict)
    source: str = "manual"  # "manual" (کلیک کاربر) یا "bot" (کلیک خودکار مدل)


def _is_refund_deal(lower_map: dict) -> bool:
    """
    آیا این پیام «نتیجهٔ معامله» نشان می‌دهد پلتفرم خودش مبلغ را ریفاند کرده
    (نه برد نه باخت واقعی - مثلاً به‌خاطر مشکل سرور یا تساوی قیمت)؟ خودِ
    Pocket Option این را با فیلدهای refundTime/refundTimestamp غیر-null در
    پیام نتیجهٔ معامله (updateClosedDeals) نشان می‌دهد.
    """
    for key in ("refundtime", "refundtimestamp"):
        if lower_map.get(key) not in (None, 0, "", "null"):
            return True
    return False


def _interpret_deal_candidate(deal: dict) -> Optional[int]:
    """
    تلاش می‌کند از یک دیکشنری کاندیدای «نتیجهٔ معامله» یک لیبل ۰/۱ دربیاورد.
    ترتیب اولویت: اول ریفاند صریح (config.REFUND_COUNTS_AS_LOSS)، بعد کلیدهای
    صریح برد/باخت، سپس کلیدهای عددی سود/زیان. اگر هیچ‌کدام قابل تفسیر نبود،
    None برمی‌گرداند (یعنی از fallback استفاده شود).
    """
    lower_map = {str(k).lower(): v for k, v in deal.items()}

    if _is_refund_deal(lower_map):
        return 0 if config.REFUND_COUNTS_AS_LOSS else 1

    for key in ("iswin", "is_win", "win"):
        if key in lower_map:
            value = lower_map[key]
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value > 0)
            if isinstance(value, str):
                return int(value.strip().lower() in ("1", "true", "win"))

    for key in ("profit", "amount", "payout", "closeprofit", "close_profit"):
        if key in lower_map and isinstance(lower_map[key], (int, float)):
            return int(lower_map[key] > 0)

    if "status" in lower_map and isinstance(lower_map["status"], str):
        status = lower_map["status"].strip().lower()
        if status in ("win", "won"):
            return 1
        if status in ("loss", "lose", "lost"):
            return 0

    return None


def _extract_payout_percent(raw_deal: Optional[dict], result: int) -> Optional[float]:
    """
    درصد پی‌آوت واقعیِ همین معامله را از روی پیام «نتیجهٔ معامله» خودِ پلتفرم
    (نه نمایش صفحه) تخمین می‌زند:
        ۱. اگر خودِ پیام یک کلید صریح "payout" داشته باشد، همان استفاده می‌شود
           (هم فرمت ۰ تا ۱ مثل ۰.۹۲ و هم فرمت ۰ تا ۱۰۰ مثل ۹۲ پشتیبانی می‌شود).
        ۲. وگرنه، فقط در معاملات بُرد، از نسبت profit/amount محاسبه می‌شود
           (چون در باخت این نسبت چیزی دربارهٔ پی‌آوت نمی‌گوید - کل مبلغ از
           دست می‌رود، نه نسبتی از آن).
    اگر هیچ‌کدام در دسترس نبود، None برمی‌گرداند (یعنی این معامله قابل بررسی
    از نظر پی‌آوت نیست).
    """
    if raw_deal is None:
        return None
    lower_map = {str(k).lower(): v for k, v in raw_deal.items()}

    if "payout" in lower_map and isinstance(lower_map["payout"], (int, float)):
        value = float(lower_map["payout"])
        return value * 100 if value <= 1.5 else value

    if result == 1:
        profit = lower_map.get("profit")
        amount = lower_map.get("amount")
        if isinstance(profit, (int, float)) and isinstance(amount, (int, float)) and amount > 0:
            return (profit / amount) * 100

    return None


class DataLogger:
    """
    مسئول ثبت لحظهٔ ورود به معامله، صبر تا انقضا، لیبل‌گذاری نتیجه و
    ذخیرهٔ ساختاریافتهٔ داده در CSV/SQLite.
    """

    def __init__(
        self,
        tick_buffer: TickBuffer,
        tick_history: TickHistory,
        candle_aggregator: CandleAggregator,
        market_structure: MarketStructureTracker,
        trade_history: TradeHistory,
        deal_buffer: Optional[DealResultBuffer] = None,
        page=None,
        on_result_callback: Optional[Callable[[str, int], None]] = None,
        csv_path=config.CSV_LOG_PATH,
        sqlite_path=config.SQLITE_DB_PATH,
        expiry_seconds: float = config.TRADE_EXPIRY_SECONDS,
        deal_result_wait_seconds: float = config.DEAL_RESULT_WAIT_SECONDS,
        consecutive_loss_cooldown_tiers: Optional[list[tuple[int, float]]] = None,
        micro_candle_aggregator: Optional[CandleAggregator] = None,
    ):
        self.tick_buffer = tick_buffer
        self.tick_history = tick_history
        self.candle_aggregator = candle_aggregator
        # کندل‌ساز ریزِ چندثانیه‌ای اختیاری (پیش‌فرض ۵ ثانیه) - فقط برای اضافه‌کردن
        # «شکل چارت» ریز به اسنپ‌شات هر معامله (micro_chart_shape_json)؛ اگر داده
        # نشود، این ویژگی صرفاً از اسنپ‌شات حذف می‌ماند.
        self.micro_candle_aggregator = micro_candle_aggregator
        self.market_structure = market_structure
        self.trade_history = trade_history
        self.deal_buffer = deal_buffer
        # برای نمایش بنر هشدار پی‌آوت پایین مستقیماً روی صفحهٔ مرورگر (اختیاری -
        # اگر داده نشود، فقط در ترمینال هشدار داده می‌شود).
        self.page = page
        # اختیاری: بعد از مشخص‌شدن نتیجهٔ هر معامله با (source, result) صدا زده
        # می‌شود - مثلاً برای این‌که استراتژی سطوح بهینه‌شده (level_strategy_
        # optimized.py) از نتیجهٔ واقعی معاملات خودش مطلع شود.
        self.on_result_callback = on_result_callback
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.expiry_seconds = expiry_seconds
        self.deal_result_wait_seconds = deal_result_wait_seconds
        # اگر مقدار صریحی داده نشده باشد (حالت عادی - main.py/collect_data.py این
        # پارامتر را پاس نمی‌دهند)، سطوح همیشه مستقیم از خودِ ماژول config خوانده
        # می‌شوند (نه یک‌بار اسنپ‌شات‌شده این‌جا) - تا اگر config.py در حین اجرا تغییر
        # کند و config_reloader.py دوباره‌اش را reload کند، همین‌جا هم بدون نیاز به
        # ری‌استارت اسکریپت بلافاصله اثر بگذارد. تست‌ها می‌توانند برای مقدار ثابت/سریع،
        # این پارامتر را صریح پاس بدهند.
        self._consecutive_loss_cooldown_tiers_override = consecutive_loss_cooldown_tiers
        # وین‌ریت جداگانه فقط برای معاملاتی که خودِ ربات (نه کاربر) باز کرده،
        # تا بشود عملکرد لحظه‌ای مدل را مستقل از معاملات دستی دنبال کرد.
        self.bot_trade_history = TradeHistory()
        # وقتی پی‌آوت واقعیِ یک معامله از config.MIN_PAYOUT_PERCENT کمتر باشد،
        # این پرچم True می‌شود. اسکریپت‌های فراخوان (collect_data.py/main.py)
        # قبل از هر کلیک برنامه‌ای این پرچم را بررسی می‌کنند و در صورت True،
        # دیگر معامله‌ای باز نمی‌کنند - معاملهٔ دستی خودِ شما هرگز مسدود نمی‌شود.
        self.trading_paused = False
        # چند معاملهٔ متوالی (که پی‌آوتشان قابل تشخیص بود) زیر آستانه بوده‌اند؟
        # برای جلوگیری از توقف اشتباهی به‌خاطر یک خوانش نادرست تکی (تشخیص
        # heuristic پی‌آوت گاهی ممکن است رکورد معاملهٔ دیگری را بگیرد).
        self._low_payout_streak = 0
        # آخرین درصد پی‌آوتی که باعث توقف شد - برای این‌که بعد از رفرش/Reload
        # صفحه (که DOM و بنر تزریق‌شده را از بین می‌برد) بتوانیم همان بنر را
        # با همان پیام دوباره بسازیم.
        self._last_low_payout_percent: Optional[float] = None
        # چند معاملهٔ برنامه‌ای (خودکار، نه دستی) پشت‌سرهم باخت بوده‌اند؟ فقط
        # نتیجهٔ معاملات بات (source != "manual") در این شمارنده اثر می‌گذارد.
        self._consecutive_loss_streak = 0
        # اگر با توقف کوتاه‌مدت بعد از باخت متوالی مواجه شدیم، معاملهٔ خودکار
        # تا این لحظه (time.monotonic()) متوقف می‌ماند - جدا از trading_paused
        # که توقف پی‌آوت پایین است و نیاز به resume دستی دارد.
        self._loss_cooldown_until_monotonic: float = 0.0

        self._init_sqlite()

    @property
    def consecutive_loss_cooldown_tiers(self) -> list[tuple[int, float]]:
        if self._consecutive_loss_cooldown_tiers_override is not None:
            return self._consecutive_loss_cooldown_tiers_override
        return config.CONSECUTIVE_LOSS_COOLDOWN_TIERS

    def _cooldown_seconds_for_streak(self, streak: int) -> Optional[float]:
        """
        بین همهٔ سطوحی که آستانه‌شان (تعداد باخت متوالی) به streak رسیده،
        بیشترین مدت توقف را برمی‌گرداند - یعنی اگر زنجیرهٔ باخت از چند سطح
        عبور کرده باشد، سخت‌گیرانه‌ترین سطح اعمال می‌شود (نه لزوماً سطحی که
        بالاترین آستانه را دارد، بلکه سطحی که مدت توقفش بیشتر است). اگر هیچ
        سطحی هنوز نرسیده باشد، None برمی‌گرداند.
        """
        applicable = [
            cooldown_seconds
            for threshold, cooldown_seconds in self.consecutive_loss_cooldown_tiers
            if streak >= threshold
        ]
        return max(applicable) if applicable else None

    # -- راه‌اندازی اولیهٔ SQLite ---------------------------------------------
    def _init_sqlite(self) -> None:
        """
        جدول trades را در صورت نبودن می‌سازد. از ستون‌های پویا (Dynamic) با
        JSON استفاده نمی‌کنیم بلکه هر بار که یک دیتافریم جدید نوشته می‌شود،
        ساختار جدول را با pandas.to_sql (if_exists='append') هماهنگ نگه می‌داریم.
        """
        self._conn = sqlite3.connect(self.sqlite_path)

    def close(self) -> None:
        self._conn.close()

    def reset_dataset(self) -> None:
        """
        دیتاست را کاملاً از صفر شروع می‌کند: فایل CSV حذف، جدول SQLite خالی، و
        تاریخچهٔ معاملات (وین‌ریت/الگوی اخیر) پاک می‌شود. برای زمانی که می‌خواهید
        بدون داده‌های قبلی از نو شروع کنید (مثلاً بعد از تغییر ساختار ویژگی‌ها).
        این عمل غیرقابل بازگشت است.
        """
        if self.csv_path.exists():
            self.csv_path.unlink()
        self._conn.execute("DROP TABLE IF EXISTS trades")
        self._conn.commit()
        self.trade_history.reset()
        self.bot_trade_history.reset()
        print(f"[DataLogger] دیتاست پاک شد ({self.csv_path.name} و {self.sqlite_path.name}). "
              f"از این لحظه دوباره از صفر ذخیره می‌شود.")

    # -- ثبت لحظهٔ ورود --------------------------------------------------------
    def capture_entry(self, direction: Direction, source: str = "manual") -> None:
        """
        این تابع در لحظهٔ فشردن کلید توسط کاربر (source="manual") یا کلیک
        خودکار ربات معامله‌گر (source="bot") صدا زده می‌شود. یک اسنپ‌شات کامل
        و کاملاً نسبی از وضعیت فعلی بازار می‌گیرد و یک Task پس‌زمینه برای ارزیابی
        نتیجه بعد از expiry_seconds ثانیه ایجاد می‌کند (بدون بلاک کردن بقیهٔ برنامه).
        """
        latest = self.tick_buffer.latest()
        if latest is None:
            print("[DataLogger] هنوز هیچ تیکی دریافت نشده؛ معامله ثبت نشد.")
            return

        snapshot = build_feature_snapshot(
            self.tick_buffer, self.tick_history, self.candle_aggregator, self.micro_candle_aggregator
        )
        snapshot.update(
            self.market_structure.get_features(latest.price, latest.timestamp, self.candle_aggregator.current)
        )
        snapshot.update(self.trade_history.as_feature_dict())
        snapshot["direction"] = direction
        snapshot["meta_trade_source"] = source

        pending = PendingTrade(
            direction=direction,
            entry_price=latest.price,
            entry_time=latest.timestamp,
            entry_wall_time=time.time(),
            entry_symbol=latest.symbol,
            feature_snapshot=snapshot,
            source=source,
        )

        source_tag = f" [{source}]" if source != "manual" else ""
        print(f"[DataLogger] معامله{source_tag} {direction} ثبت شد. "
              f"در حال انتظار برای نتیجه ({self.expiry_seconds} ثانیه)...")

        asyncio.create_task(self._evaluate_and_log(pending))

    # -- ارزیابی نتیجه و ذخیره --------------------------------------------------
    async def _evaluate_and_log(self, pending: PendingTrade) -> None:
        # صبر واقعی (Wall-clock) به‌اندازهٔ زمان انقضای معامله
        await asyncio.sleep(self.expiry_seconds)

        # کمی زمان اضافه می‌دهیم تا مطمئن شویم تیک بعد از لحظهٔ انقضا رسیده است
        exit_timestamp = pending.entry_time + self.expiry_seconds
        exit_price = await self._wait_for_price_after(exit_timestamp)

        if exit_price is None:
            print("[DataLogger] هشدار: قیمت خروج پیدا نشد؛ این معامله لیبل نمی‌گیرد.")
            return

        tick_fallback_result = self._determine_result(pending.direction, pending.entry_price, exit_price)

        # اولویت با نتیجهٔ خودِ پلتفرم (ws_deal)؛ اگر پیدا نشد یا قابل تفسیر نبود،
        # از محاسبهٔ داخلی (tick_fallback) استفاده می‌کنیم.
        result = tick_fallback_result
        result_source = "tick_fallback"
        raw_deal = await self._find_platform_deal_result(pending)
        if raw_deal is not None:
            interpreted = _interpret_deal_candidate(raw_deal)
            if interpreted is not None:
                result = interpreted
                result_source = "ws_deal"

        self.trade_history.add_result(result)
        if pending.source != "manual":
            self.bot_trade_history.add_result(result)
            if result == 1:
                self._consecutive_loss_streak = 0
            else:
                self._consecutive_loss_streak += 1
                cooldown_seconds = self._cooldown_seconds_for_streak(self._consecutive_loss_streak)
                if cooldown_seconds is not None:
                    self._loss_cooldown_until_monotonic = time.monotonic() + cooldown_seconds
                    print(f"[DataLogger] ⏸ {self._consecutive_loss_streak} باخت متوالی در معاملات "
                          f"برنامه‌ای - معاملهٔ خودکار برای {cooldown_seconds:.0f} "
                          f"ثانیه متوقف می‌شود (معاملهٔ دستی شما آزاد است).")
                    asyncio.create_task(self._show_loss_cooldown_banner(cooldown_seconds))
        if self.on_result_callback is not None:
            self.on_result_callback(pending.source, result)

        payout_percent = _extract_payout_percent(raw_deal, result)

        row = dict(pending.feature_snapshot)
        # ستون‌های meta_* فقط برای ردیابی/دیباگ‌اند؛ مقدار خام قیمت دارند و
        # نباید به‌عنوان ورودی مدل استفاده شوند.
        row["meta_symbol"] = pending.entry_symbol
        row["meta_entry_price"] = pending.entry_price
        row["meta_entry_timestamp"] = pending.entry_time
        row["meta_exit_price"] = exit_price
        row["meta_exit_timestamp"] = exit_timestamp
        row["meta_raw_deal_json"] = json.dumps(raw_deal) if raw_deal is not None else None
        row["meta_payout_percent"] = payout_percent
        # تنها ویژگی نسبی مشتق از ورود/خروج (قابل استفاده در تحلیل، نه لزوماً در مدل):
        row["price_change_pct"] = (
            (exit_price - pending.entry_price) / pending.entry_price if pending.entry_price else None
        )
        row["result_source"] = result_source  # ws_deal یا tick_fallback
        row["tick_fallback_result"] = tick_fallback_result  # برای مقایسه/اعتبارسنجی دو روش
        row["result"] = result  # Win = 1 / Loss = 0  <-- لیبل نهایی برای آموزش مدل

        self._append_row(row)

        if payout_percent is not None:
            if payout_percent < config.MIN_PAYOUT_PERCENT:
                self._low_payout_streak += 1
                if self._low_payout_streak < config.LOW_PAYOUT_CONFIRM_TRADES:
                    print(f"[DataLogger] پی‌آوت این معامله {payout_percent:.1f}٪ بود (کمتر از حد مجاز "
                          f"{config.MIN_PAYOUT_PERCENT}٪) - در انتظار تأیید با معاملهٔ بعدی قبل از توقف "
                          f"({self._low_payout_streak}/{config.LOW_PAYOUT_CONFIRM_TRADES}).")
                elif not self.trading_paused:
                    self.trading_paused = True
                    print(f"[DataLogger] ⚠️ پی‌آوت واقعی {config.LOW_PAYOUT_CONFIRM_TRADES} معاملهٔ متوالی "
                          f"زیر حد مجاز {config.MIN_PAYOUT_PERCENT}٪ بود (آخرین مقدار: {payout_percent:.1f}٪). "
                          f"معاملهٔ برنامه‌ای (خودکار) از این لحظه متوقف شد؛ معاملهٔ دستی شما همچنان آزاد است.")
                    asyncio.create_task(self._show_low_payout_banner(payout_percent))
                    asyncio.create_task(self._sync_toggle_button())
            else:
                # یک معاملهٔ واقعاً با پی‌آوت خوب، هر زنجیرهٔ قبلیِ خوانش‌های کم را باطل می‌کند.
                self._low_payout_streak = 0

        outcome_text = "WIN ✅" if result == 1 else "LOSS ❌"
        lifetime_total = self._lifetime_trade_count()
        target = config.MIN_TRADES_FOR_INITIAL_TRAINING
        progress = f" | مجموع کل معاملات ثبت‌شده: {lifetime_total}"
        if target:
            progress += f" از حدود {target} (نمونهٔ اولیهٔ پیشنهادی) — {min(100, lifetime_total / target * 100):.0f}٪"
        source_tag = f" [{pending.source}]" if pending.source != "manual" else ""
        print(f"[DataLogger] نتیجهٔ معامله{source_tag}: {outcome_text} (منبع: {result_source}) | "
              f"وین‌ریت کلی از شروع اجرا: {self.trade_history.get_overall_winrate():.2%}{progress}")
        if pending.source != "manual":
            print(f"[DataLogger] وین‌ریت کلیِ ربات ({pending.source}) از شروع اجرا: "
                  f"{self.bot_trade_history.get_overall_winrate():.2%} "
                  f"روی {self.bot_trade_history.total_trades()} معاملهٔ خودکار")

    async def _show_low_payout_banner(self, payout_percent: float) -> None:
        """
        یک بنر قرمز ثابت بالای خودِ صفحهٔ مرورگر تزریق می‌کند (نه فقط ترمینال)
        تا افت پی‌آوت حتی وقتی کاربر ترمینال را زیر نظر ندارد هم قابل‌دیدن باشد.
        یک دکمهٔ × برای بستن دستی هم دارد (فقط بنر را مخفی می‌کند - برای فعال‌
        کردن دوباره خودِ معاملهٔ خودکار باید دستور resume را در ترمینال بزنید).
        اگر page در دسترس نباشد (مثلاً هنگام تست)، فقط از این تابع بی‌اثر برمی‌گردد.
        """
        if self.page is None:
            return
        # ذخیره می‌شود تا اگر صفحه رفرش/Reload شد (و همین بنر از بین رفت)،
        # بتوانیم دقیقاً همین بنر را با _on_page_load دوباره بسازیم.
        self._last_low_payout_percent = payout_percent
        message = (
            f"⚠️ پی‌آوت به {payout_percent:.1f}٪ افت کرد (کمتر از حد مجاز "
            f"{config.MIN_PAYOUT_PERCENT}٪) - معاملهٔ خودکار متوقف شد. "
            f"برای فعال‌سازی دوباره، 'resume' را در ترمینال تایپ کنید یا دکمهٔ ازسرگیری را بزنید."
        )
        js = """
        (text) => {
            let el = document.getElementById('__low_payout_banner__');
            if (!el) {
                el = document.createElement('div');
                el.id = '__low_payout_banner__';
                el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;' +
                    'background:#c0392b;color:#fff;font:bold 16px/1.4 sans-serif;' +
                    'text-align:center;padding:10px 44px;direction:rtl;';

                const textSpan = document.createElement('span');
                textSpan.id = '__low_payout_banner_text__';
                el.appendChild(textSpan);

                const closeBtn = document.createElement('button');
                closeBtn.textContent = '×';
                closeBtn.setAttribute('aria-label', 'Close');
                closeBtn.style.cssText = 'position:absolute;left:12px;top:50%;' +
                    'transform:translateY(-50%);background:transparent;border:none;' +
                    'color:#fff;font-size:24px;font-weight:bold;cursor:pointer;' +
                    'line-height:1;padding:0 8px;';
                closeBtn.onclick = () => { el.style.display = 'none'; };
                el.appendChild(closeBtn);

                document.body.appendChild(el);
            }
            el.querySelector('#__low_payout_banner_text__').textContent = text;
            el.style.display = 'block';
        }
        """
        try:
            await self.page.evaluate(js, message)
        except Exception as exc:  # noqa: BLE001 - نمایش بنر نباید ثبت دیتاست را مختل کند
            print(f"[DataLogger] هشدار: نمایش بنر پی‌آوت پایین ناموفق بود: {exc}")

    async def _hide_low_payout_banner(self) -> None:
        """بنر پی‌آوت پایین را مخفی می‌کند (بعد از دستور resume)."""
        if self.page is None:
            return
        js = "() => { const el = document.getElementById('__low_payout_banner__'); if (el) el.style.display = 'none'; }"
        try:
            await self.page.evaluate(js)
        except Exception as exc:  # noqa: BLE001 - نباید ادامهٔ برنامه را مختل کند
            print(f"[DataLogger] هشدار: مخفی‌کردن بنر ناموفق بود: {exc}")

    async def _show_loss_cooldown_banner(self, cooldown_seconds: float) -> None:
        """
        یک بنر نارنجی موقت بالای صفحه نشان می‌دهد که به‌خاطر باخت‌های متوالی
        معاملهٔ خودکار موقتاً متوقف شده. برخلاف بنر پی‌آوت پایین، نیازی به
        دستور resume نیست - این تابع خودش صبر می‌کند تا زمان توقف
        (self._loss_cooldown_until_monotonic) واقعاً تمام شود و بعد بنر را
        خودکار مخفی می‌کند. اگر در همین حین توقف دوباره تمدید شده باشد (باخت
        متوالی دیگری رخ داده، شاید حتی با مدت توقفِ سطح بعدی)، حلقه دوباره صبر
        می‌کند تا زمان جدید هم بگذرد.
        """
        if self.page is not None:
            message = (
                f"⏸ {self._consecutive_loss_streak} باخت متوالی در معاملات برنامه‌ای - معاملهٔ "
                f"خودکار برای {cooldown_seconds:.0f} ثانیه متوقف شد و خودش "
                f"دوباره فعال می‌شود. معاملهٔ دستی شما آزاد است."
            )
            js = """
            (text) => {
                let el = document.getElementById('__loss_cooldown_banner__');
                if (!el) {
                    el = document.createElement('div');
                    el.id = '__loss_cooldown_banner__';
                    el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;' +
                        'background:#d68910;color:#fff;font:bold 16px/1.4 sans-serif;' +
                        'text-align:center;padding:10px 16px;direction:rtl;';

                    const textSpan = document.createElement('span');
                    textSpan.id = '__loss_cooldown_banner_text__';
                    el.appendChild(textSpan);

                    document.body.appendChild(el);
                }
                el.querySelector('#__loss_cooldown_banner_text__').textContent = text;
                el.style.display = 'block';
            }
            """
            try:
                await self.page.evaluate(js, message)
            except Exception as exc:  # noqa: BLE001 - نمایش بنر نباید ثبت دیتاست را مختل کند
                print(f"[DataLogger] هشدار: نمایش بنر توقف کوتاه‌مدت ناموفق بود: {exc}")

        while True:
            remaining = self._loss_cooldown_until_monotonic - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(remaining)

        await self._hide_loss_cooldown_banner()

    async def _hide_loss_cooldown_banner(self) -> None:
        """بنر توقف کوتاه‌مدت را مخفی می‌کند (بعد از اتمام خودکار زمان توقف)."""
        if self.page is None:
            return
        js = "() => { const el = document.getElementById('__loss_cooldown_banner__'); if (el) el.style.display = 'none'; }"
        try:
            await self.page.evaluate(js)
        except Exception as exc:  # noqa: BLE001 - نباید ادامهٔ برنامه را مختل کند
            print(f"[DataLogger] هشدار: مخفی‌کردن بنر توقف کوتاه‌مدت ناموفق بود: {exc}")

    def resume_trading(self) -> bool:
        """
        معاملهٔ برنامه‌ای متوقف‌شده (به‌خاطر پی‌آوت پایین) را دوباره فعال می‌کند،
        بدون نیاز به ری‌استارت کل اسکریپت - مثلاً بعد از این‌که خودتان دستی ارز
        را به نمادی با پی‌آوت بهتر تغییر داده‌اید. اگر از قبل متوقف نبوده،
        False برمی‌گرداند (کاری برای انجام نبود).
        """
        if not self.trading_paused:
            return False
        self.trading_paused = False
        self._low_payout_streak = 0
        asyncio.create_task(self._hide_low_payout_banner())
        asyncio.create_task(self._sync_toggle_button())
        return True

    def is_bot_trading_blocked(self) -> bool:
        """
        True اگر معاملهٔ برنامه‌ای (خودکار) فعلاً باید متوقف بماند - یا به‌خاطر
        توقف پی‌آوت پایین (trading_paused، که نیاز به دستور resume دستی دارد)
        یا به‌خاطر توقف کوتاه‌مدت بعد از باخت‌های متوالی (که خودش طبق سطح
        رسیده‌شده در consecutive_loss_cooldown_tiers تمام می‌شود). فراخوان‌های
        main.py/collect_data.py قبل از هر کلیک برنامه‌ای این تابع را چک
        می‌کنند؛ معاملهٔ دستی خودِ شما هرگز توسط این تابع مسدود نمی‌شود.
        """
        return self.trading_paused or time.monotonic() < self._loss_cooldown_until_monotonic

    # -- دکمهٔ توقف/ازسرگیری روی خودِ صفحهٔ مرورگر -------------------------------
    async def install_page_controls(self) -> None:
        """
        یک دکمهٔ شناور «توقف/ازسرگیری معاملهٔ خودکار» مستقیماً روی صفحهٔ مرورگر
        تزریق می‌کند - برای این‌که نیازی به تایپ دستور در ترمینال نباشد. باید
        دقیقاً یک‌بار، بعد از ساخت DataLogger با page واقعی، await شود (مثلاً
        در main() اسکریپت). اگر page در دسترس نباشد، بی‌اثر برمی‌گردد.

        چون رفرش/Reload صفحه (چه دستی توسط کاربر، چه توسط reconnect_page_if_
        needed) کل DOM را از نو می‌سازد - یعنی دکمه و بنر پی‌آوت از بین
        می‌روند - این‌جا روی رویداد 'domcontentloaded' صفحه هم مشترک می‌شویم تا
        بعد از هر بارگذاری دوباره، همین کنترل‌ها را با وضعیت فعلی بازسازی کنیم.
        expose_function خودش توسط Playwright بین navigation ها حفظ می‌شود، پس
        فقط یک‌بار در همین‌جا (نه در هر reload) صدا زده می‌شود.
        """
        if self.page is None:
            return

        await self.page.expose_function("pocketBotToggleTrading", self._toggle_trading_from_page)
        await self._inject_toggle_button()
        self.page.on("domcontentloaded", lambda: asyncio.create_task(self._on_page_load()))

    async def _on_page_load(self) -> None:
        """
        بعد از هر بارگذاریِ دوبارهٔ صفحه (رفرش دستی یا Reconnect خودکار) صدا
        زده می‌شود: دکمهٔ توقف/ازسرگیری را دوباره می‌سازد و اگر معاملهٔ خودکار
        از قبل متوقف بوده، همان بنر پی‌آوت را هم دوباره نمایش می‌دهد - تا هیچ‌
        کدام از این کنترل‌ها با رفرش صفحه گم نشوند.
        """
        await self._inject_toggle_button()
        await self._sync_toggle_button()
        if self.trading_paused and self._last_low_payout_percent is not None:
            await self._show_low_payout_banner(self._last_low_payout_percent)

    async def _inject_toggle_button(self) -> None:
        """فقط تزریق DOM دکمه (بدون expose_function) - جداگانه تا هم در راه‌اندازی اولیه، هم بعد از هر reload صدا زده شود."""
        if self.page is None:
            return
        js = """
        () => {
            let btn = document.getElementById('__trading_toggle_btn__');
            if (btn) return;
            btn = document.createElement('button');
            btn.id = '__trading_toggle_btn__';
            btn.type = 'button';
            btn.textContent = '⏸ توقف معاملهٔ خودکار';
            btn.style.cssText = 'position:fixed;bottom:12px;left:12px;z-index:2147483647;' +
                'padding:8px 16px;border:none;border-radius:6px;font:bold 14px sans-serif;' +
                'cursor:pointer;color:#fff;background:#34495e;direction:rtl;box-shadow:0 2px 8px rgba(0,0,0,.3);';
            btn.onclick = async () => {
                btn.disabled = true;
                try {
                    const paused = await window.pocketBotToggleTrading();
                    btn.textContent = paused ? '▶ ازسرگیری معاملهٔ خودکار' : '⏸ توقف معاملهٔ خودکار';
                    btn.style.background = paused ? '#27ae60' : '#34495e';
                } finally {
                    btn.disabled = false;
                }
            };
            document.body.appendChild(btn);
        }
        """
        try:
            await self.page.evaluate(js)
        except Exception as exc:  # noqa: BLE001 - نباید راه‌اندازی برنامه را مختل کند
            print(f"[DataLogger] هشدار: افزودن دکمهٔ توقف/ازسرگیری ناموفق بود: {exc}")

    async def _toggle_trading_from_page(self) -> bool:
        """
        از طریق دکمهٔ روی صفحه صدا زده می‌شود (با page.expose_function). وضعیت
        توقف را toggle می‌کند و در صورت ازسرگیری، بنر پی‌آوت را هم مخفی می‌کند.
        مقدار جدید trading_paused را برمی‌گرداند تا خودِ دکمه برچسبش را عوض کند.
        """
        self.trading_paused = not self.trading_paused
        action = "متوقف" if self.trading_paused else "دوباره فعال"
        print(f"[DataLogger] معاملهٔ خودکار از طریق دکمهٔ روی صفحه {action} شد.")
        if self.trading_paused:
            self._low_payout_streak = 0
        else:
            await self._hide_low_payout_banner()
        return self.trading_paused

    async def _sync_toggle_button(self) -> None:
        """برچسب/رنگ دکمهٔ روی صفحه را با self.trading_paused هماهنگ می‌کند (وقتی تغییر از مسیر دیگری - نه خودِ دکمه - اتفاق افتاده)."""
        if self.page is None:
            return
        js = """
        (paused) => {
            const btn = document.getElementById('__trading_toggle_btn__');
            if (!btn) return;
            btn.textContent = paused ? '▶ ازسرگیری معاملهٔ خودکار' : '⏸ توقف معاملهٔ خودکار';
            btn.style.background = paused ? '#27ae60' : '#34495e';
        }
        """
        try:
            await self.page.evaluate(js, self.trading_paused)
        except Exception as exc:  # noqa: BLE001 - نباید ادامهٔ برنامه را مختل کند
            print(f"[DataLogger] هشدار: هماهنگ‌سازی دکمهٔ توقف/ازسرگیری ناموفق بود: {exc}")

    def _lifetime_trade_count(self) -> int:
        """
        تعداد کل معاملات ثبت‌شده در طول عمر دیتاست (نه فقط اجرای جاری) — از
        روی SQLite که همیشه معتبر و انباشته از همهٔ اجراهای قبلی است.
        """
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        if cursor.fetchone() is None:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    async def _find_platform_deal_result(self, pending: PendingTrade) -> Optional[dict]:
        """
        بعد از لحظهٔ انقضا، تا deal_result_wait_seconds ثانیه دنبال یک پیام
        «نتیجهٔ معامله» از خودِ پلتفرم می‌گردد که زمان دریافتش حوالی لحظهٔ
        انقضا باشد. اگر DealResultBuffer تنظیم نشده باشد، فوراً None برمی‌گرداند.
        """
        if self.deal_buffer is None:
            return None

        expected_settle_wall_time = pending.entry_wall_time + self.expiry_seconds
        deadline = time.monotonic() + self.deal_result_wait_seconds
        window_start = expected_settle_wall_time - 1.0

        while time.monotonic() < deadline:
            window_end = time.time() + 0.01
            deal = self.deal_buffer.find_and_consume(window_start, window_end)
            if deal is not None:
                return deal
            await asyncio.sleep(0.1)

        return None

    async def _wait_for_price_after(self, exit_timestamp: float, max_wait: float = 2.0) -> float | None:
        """
        اگر هنوز تیکی با زمان >= exit_timestamp نرسیده باشد (مثلاً به‌خاطر تأخیر
        شبکه)، تا max_wait ثانیهٔ اضافه صبر می‌کند. از asyncio.sleep استفاده
        می‌شود (نه time.sleep) تا event loop بلاک نشود و تیک‌های جدید بتوانند
        در همین حین به‌طور هم‌زمان به بافر اضافه شوند.
        """
        deadline = time.monotonic() + max_wait
        price = self.tick_buffer.latest_price_after(exit_timestamp)
        while price is None and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            price = self.tick_buffer.latest_price_after(exit_timestamp)
        return price

    @staticmethod
    def _determine_result(direction: Direction, entry_price: float, exit_price: float) -> int:
        """
        منطق برد/باخت دقیقاً مثل قوانین Pocket Option:
            CALL برنده است اگر قیمت خروج > قیمت ورود
            PUT برنده است اگر قیمت خروج < قیمت ورود
            تساوی دقیق طبق config.TIE_COUNTS_AS_LOSS مدیریت می‌شود.
        """
        if exit_price == entry_price:
            return 0 if config.TIE_COUNTS_AS_LOSS else 1

        if direction == "CALL":
            return int(exit_price > entry_price)
        else:  # PUT
            return int(exit_price < entry_price)

    # -- ذخیره‌سازی -------------------------------------------------------------
    def _append_row(self, row: dict) -> None:
        """
        یک ردیف را هم به SQLite و هم به CSV اضافه می‌کند.

        SQLite همیشه منبع معتبر (Source of Truth) است، چون pandas.to_sql
        مقادیر را بر اساس **نام** ستون درج می‌کند، نه موقعیت — پس حتی اگر
        ستون‌های جدیدی در طول زمان اضافه شوند (که در این پروژهٔ در حال توسعه
        طبیعی است)، دادهٔ قدیمی هیچ‌وقت با مقدار اشتباه قاطی نمی‌شود.

        CSV اما یک فایل متنی خطی است: اگر یک بار (مثلاً به‌خاطر یک نسخهٔ قدیمی
        کد) ردیفی با تعداد ستون متفاوت از هدر نوشته شده باشد، خودِ فایل CSV
        برای همیشه خراب و غیرقابل‌خواندن می‌ماند (چون pandas نمی‌داند کدام
        مقدار به کدام ستون تعلق دارد). به همین دلیل، به‌جای Append کردن یا حتی
        تلاش برای تعمیر فایل خراب، هر بار **کل CSV را از روی همان SQLite معتبر
        دوباره می‌سازیم** — این‌طوری خرابی CSV اصلاً امکان‌پذیر نیست.
        """
        df = pd.DataFrame([row])

        # --- SQLite: اول این‌جا درج می‌شود (منبع معتبر) ---
        self._sync_sqlite_columns(df.columns)
        df.to_sql("trades", self._conn, if_exists="append", index=False)

        # --- CSV: کامل از روی SQLite بازسازی می‌شود، نه Append ---
        try:
            full = pd.read_sql("SELECT * FROM trades", self._conn)
            full.to_csv(self.csv_path, index=False)
        except Exception as exc:  # noqa: BLE001 - نباید کل ثبت معامله را به‌خاطر خطای نوشتن CSV متوقف کند
            print(f"[DataLogger] هشدار: بازسازی CSV ناموفق بود (دادهٔ SQLite دست‌نخورده است): {exc}")

    def _sync_sqlite_columns(self, columns) -> None:
        """جدول trades را در صورت نبودن ستون‌های جدید، با ALTER TABLE به‌روز می‌کند."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        if cursor.fetchone() is None:
            return  # جدول هنوز ساخته نشده؛ to_sql با if_exists='append' خودش می‌سازد
        existing_columns = {info[1] for info in self._conn.execute("PRAGMA table_info(trades)")}
        for col in columns:
            if col not in existing_columns:
                self._conn.execute(f'ALTER TABLE trades ADD COLUMN "{col}"')
        self._conn.commit()
