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
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

import config
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
    entry_symbol: str
    feature_snapshot: dict = field(default_factory=dict)


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
        csv_path=config.CSV_LOG_PATH,
        sqlite_path=config.SQLITE_DB_PATH,
        expiry_seconds: float = config.TRADE_EXPIRY_SECONDS,
    ):
        self.tick_buffer = tick_buffer
        self.tick_history = tick_history
        self.candle_aggregator = candle_aggregator
        self.market_structure = market_structure
        self.trade_history = trade_history
        self.csv_path = csv_path
        self.sqlite_path = sqlite_path
        self.expiry_seconds = expiry_seconds

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

        result = self._determine_result(pending.direction, pending.entry_price, exit_price)

        self.trade_history.add_result(result)

        row = dict(pending.feature_snapshot)
        # ستون‌های meta_* فقط برای ردیابی/دیباگ‌اند؛ مقدار خام قیمت دارند و
        # نباید به‌عنوان ورودی مدل استفاده شوند.
        row["meta_symbol"] = pending.entry_symbol
        row["meta_entry_price"] = pending.entry_price
        row["meta_entry_timestamp"] = pending.entry_time
        row["meta_exit_price"] = exit_price
        row["meta_exit_timestamp"] = exit_timestamp
        # تنها ویژگی نسبی مشتق از ورود/خروج (قابل استفاده در تحلیل، نه لزوماً در مدل):
        row["price_change_pct"] = (
            (exit_price - pending.entry_price) / pending.entry_price if pending.entry_price else None
        )
        row["result"] = result  # Win = 1 / Loss = 0  <-- لیبل نهایی برای آموزش مدل

        self._append_row(row)

        outcome_text = "WIN ✅" if result == 1 else "LOSS ❌"
        print(f"[DataLogger] نتیجهٔ معامله: {outcome_text} | "
              f"وین‌ریت لحظه‌ای: {self.trade_history.get_rolling_winrate():.2%}")

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
        """یک ردیف را هم به CSV و هم به SQLite اضافه می‌کند."""
        df = pd.DataFrame([row])

        # --- CSV ---
        write_header = not self.csv_path.exists()
        df.to_csv(self.csv_path, mode="a", header=write_header, index=False)

        # --- SQLite ---
        df.to_sql("trades", self._conn, if_exists="append", index=False)
