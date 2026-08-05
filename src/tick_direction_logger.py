"""
لاگ خام جهت تیک‌ها - مستقل از سطح قیمت
==========================================

فقط علامت تغییر قیمت هر تیک نسبت به تیک قبلیِ همان نماد را ثبت می‌کند
(+۱ صعودی / -۱ نزولی / ۰ بدون تغییر) - نه خودِ قیمت. هدف این است که بعداً
با analyze_tick_direction_patterns.py بشود بررسی کرد آیا در توالی خام
تغییرات جهت الگوی تکرارشوندهٔ آماری معنادار (فراتر از یک random walk) وجود
دارد یا نه، قبل از این‌که چنین چیزی به‌عنوان فیچر وارد مدل شود.

نوشتن روی SQLite به‌ازای هر تیک کند است (این حلقه باید همیشه real-time
بماند)، پس ردیف‌ها در حافظه بافر می‌شوند و هر flush_every تیک یک‌جا با
executemany نوشته می‌شوند.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import config
from src.browser_session import Tick

TABLE_NAME = "tick_directions"


class TickDirectionLogger:
    def __init__(
        self,
        sqlite_path: Path = config.SQLITE_DB_PATH,
        flush_every: int = config.TICK_DIRECTION_LOG_FLUSH_EVERY,
    ):
        self.flush_every = flush_every
        self._conn = sqlite3.connect(sqlite_path)
        self._init_table()
        # آخرین قیمت شناخته‌شدهٔ هر نماد - چون کلیدش خودِ نماد است، نیازی به
        # ریست دستی هنگام تعویض نماد نیست: تیک نماد جدید همیشه با آخرین قیمتِ
        # همان نماد (نه نماد قبلی) مقایسه می‌شود.
        self._last_price_by_symbol: dict[str, float] = {}
        self._pending: list[tuple[float, str, int]] = []

    def _init_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                symbol TEXT NOT NULL,
                direction INTEGER NOT NULL
            )
            """
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_symbol_ts "
            f"ON {TABLE_NAME}(symbol, timestamp)"
        )
        self._conn.commit()

    def record(self, tick: Tick) -> None:
        last_price = self._last_price_by_symbol.get(tick.symbol)
        if last_price is not None:
            if tick.price > last_price:
                direction = 1
            elif tick.price < last_price:
                direction = -1
            else:
                direction = 0
            self._pending.append((tick.timestamp, tick.symbol, direction))
            if len(self._pending) >= self.flush_every:
                self.flush()
        self._last_price_by_symbol[tick.symbol] = tick.price

    def flush(self) -> None:
        if not self._pending:
            return
        self._conn.executemany(
            f"INSERT INTO {TABLE_NAME} (timestamp, symbol, direction) VALUES (?, ?, ?)",
            self._pending,
        )
        self._conn.commit()
        self._pending.clear()

    def close(self) -> None:
        self.flush()
        self._conn.close()
