"""
بارگذاری دوبارهٔ زندهٔ config.py (Hot Reload)
=================================================

اسکریپت‌های اصلی (main.py/collect_data.py) فقط یک‌بار در ابتدای اجرا
`import config` را صدا می‌زنند. اگر شما فایل config.py را در حین اجرا ویرایش
کنید، به‌خودی‌خود هیچ اثری روی پردازش در حالِ اجرا ندارد - چون پایتون فایل را
دوباره نمی‌خواند مگر این‌که صریحاً درخواست شود.

این ماژول یک Task پس‌زمینه فراهم می‌کند که هر چند ثانیه یک‌بار زمان آخرین
تغییرِ config.py را بررسی می‌کند و اگر عوض شده بود، با importlib.reload()
خودِ ماژول config را در همان جای حافظه‌اش دوباره اجرا می‌کند - یعنی مقادیر
جدید مستقیماً روی همان آبجکت ماژولی که همهٔ فایل‌های دیگر پروژه با
`import config` به آن اشاره می‌کنند نوشته می‌شود.

نکتهٔ مهم: این فقط برای مقادیری واقعاً «زنده» است که کد در لحظهٔ استفاده
مستقیماً `config.NAME` را می‌خواند (نه مقداری که یک‌بار در سازندهٔ یک کلاس
اسنپ‌شات و در یک attribute جدا ذخیره شده - آن‌ها همچنان فقط با ری‌استارت
اسکریپت به‌روز می‌شوند). برای مثال config.CONSECUTIVE_LOSSES_FOR_COOLDOWN و
config.CONSECUTIVE_LOSS_COOLDOWN_SECONDS در DataLogger عمداً به همین شکل
(بدون اسنپ‌شات، همیشه مستقیم از config خوانده می‌شوند - نگاه کنید به
src/data_logger.py) طراحی شده‌اند تا با این مکانیزم واقعاً زنده باشند.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import config


async def config_hot_reload_task(stop_event: asyncio.Event, poll_interval_seconds: float = 2.0) -> None:
    """
    Task پس‌زمینه: هر poll_interval_seconds ثانیه زمان آخرین تغییر config.py
    را بررسی می‌کند. اگر تغییر کرده بود، config را دوباره reload می‌کند و در
    ترمینال اطلاع می‌دهد. اگر در حین reload خطایی رخ دهد (مثلاً یک syntax
    error موقت به‌خاطر ذخیرهٔ ناقص فایل)، فقط پیام خطا چاپ می‌شود و تنظیمات
    قبلی دست‌نخورده باقی می‌مانند - این Task هیچ‌وقت به‌خاطر یک config.py
    خراب کل اسکریپت را متوقف نمی‌کند.
    """
    config_path = Path(config.__file__)
    try:
        last_mtime = config_path.stat().st_mtime
    except OSError:
        last_mtime = None

    while not stop_event.is_set():
        await asyncio.sleep(poll_interval_seconds)

        try:
            mtime = config_path.stat().st_mtime
        except OSError:
            continue

        if last_mtime is not None and mtime == last_mtime:
            continue
        last_mtime = mtime

        try:
            importlib.reload(config)
            print("[Config] تغییرات config.py شناسایی و بارگذاری شد.")
        except Exception as exc:  # noqa: BLE001 - یک config.py خراب نباید کل اسکریپت را متوقف کند
            print(f"[Config] خطا در بارگذاری دوبارهٔ config.py (تنظیمات قبلی دست‌نخورده ماند): {exc}")
