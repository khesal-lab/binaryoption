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
from typing import Literal, Optional

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


def _interpret_deal_candidate(deal: dict) -> Optional[int]:
    """
    تلاش می‌کند از یک دیکشنری کاندیدای «نتیجهٔ معامله» یک لیبل ۰/۱ دربیاورد.
    ترتیب اولویت: کلیدهای صریح برد/باخت، سپس کلیدهای عددی سود/زیان.
    اگر هیچ‌کدام قابل تفسیر نبود، None برمی‌گرداند (یعنی از fallback استفاده شود).
    """
    lower_map = {str(k).lower(): v for k, v in deal.items()}

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
        csv_path=config.CSV_LOG_PATH,
        sqlite_path=config.SQLITE_DB_PATH,
        expiry_seconds: float = config.TRADE_EXPIRY_SECONDS,
        deal_result_wait_seconds: float = config.DEAL_RESULT_WAIT_SECONDS,
    ):
        self.tick_buffer = tick_buffer
        self.tick_history = tick_history
        self.candle_aggregator = candle_aggregator
        self.market_structure = market_structure
        self.trade_history = trade_history
        self.deal_buffer = deal_buffer
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.expiry_seconds = expiry_seconds
        self.deal_result_wait_seconds = deal_result_wait_seconds

        self._init_sqlite()

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
        print(f"[DataLogger] دیتاست پاک شد ({self.csv_path.name} و {self.sqlite_path.name}). "
              f"از این لحظه دوباره از صفر ذخیره می‌شود.")

    # -- ثبت لحظهٔ ورود --------------------------------------------------------
    def capture_entry(self, direction: Direction) -> None:
        """
        این تابع در لحظهٔ فشردن کلید توسط کاربر صدا زده می‌شود. یک اسنپ‌شات کامل
        و کاملاً نسبی از وضعیت فعلی بازار می‌گیرد و یک Task پس‌زمینه برای ارزیابی
        نتیجه بعد از expiry_seconds ثانیه ایجاد می‌کند (بدون بلاک کردن بقیهٔ برنامه).
        """
        latest = self.tick_buffer.latest()
        if latest is None:
            print("[DataLogger] هنوز هیچ تیکی دریافت نشده؛ معامله ثبت نشد.")
            return

        snapshot = build_feature_snapshot(self.tick_buffer, self.tick_history, self.candle_aggregator)
        snapshot.update(
            self.market_structure.get_features(latest.price, latest.timestamp, self.candle_aggregator.current)
        )
        snapshot.update(self.trade_history.as_feature_dict())
        snapshot["direction"] = direction

        pending = PendingTrade(
            direction=direction,
            entry_price=latest.price,
            entry_time=latest.timestamp,
            entry_wall_time=time.time(),
            entry_symbol=latest.symbol,
            feature_snapshot=snapshot,
        )

        print(f"[DataLogger] معامله {direction} ثبت شد. "
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

        row = dict(pending.feature_snapshot)
        # ستون‌های meta_* فقط برای ردیابی/دیباگ‌اند؛ مقدار خام قیمت دارند و
        # نباید به‌عنوان ورودی مدل استفاده شوند.
        row["meta_symbol"] = pending.entry_symbol
        row["meta_entry_price"] = pending.entry_price
        row["meta_entry_timestamp"] = pending.entry_time
        row["meta_exit_price"] = exit_price
        row["meta_exit_timestamp"] = exit_timestamp
        row["meta_raw_deal_json"] = json.dumps(raw_deal) if raw_deal is not None else None
        # تنها ویژگی نسبی مشتق از ورود/خروج (قابل استفاده در تحلیل، نه لزوماً در مدل):
        row["price_change_pct"] = (
            (exit_price - pending.entry_price) / pending.entry_price if pending.entry_price else None
        )
        row["result_source"] = result_source  # ws_deal یا tick_fallback
        row["tick_fallback_result"] = tick_fallback_result  # برای مقایسه/اعتبارسنجی دو روش
        row["result"] = result  # Win = 1 / Loss = 0  <-- لیبل نهایی برای آموزش مدل

        self._append_row(row)

        outcome_text = "WIN ✅" if result == 1 else "LOSS ❌"
        lifetime_total = self._lifetime_trade_count()
        target = config.MIN_TRADES_FOR_INITIAL_TRAINING
        progress = f" | مجموع کل معاملات ثبت‌شده: {lifetime_total}"
        if target:
            progress += f" از حدود {target} (نمونهٔ اولیهٔ پیشنهادی) — {min(100, lifetime_total / target * 100):.0f}٪"
        print(f"[DataLogger] نتیجهٔ معامله: {outcome_text} (منبع: {result_source}) | "
              f"وین‌ریت لحظه‌ای: {self.trade_history.get_rolling_winrate():.2%}{progress}")

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
