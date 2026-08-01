"""
بخش اول: راه‌اندازی مرورگر و اتصال/شنود WebSocket
====================================================

این ماژول مسئول موارد زیر است:
    ۱. باز کردن Chrome با Playwright به‌صورت غیر Headless تا کاربر بتواند
       به‌صورت دستی وارد حساب دمو (Demo) خودش در Pocket Option شود.
    ۲. شنود (Intercept) تمام فریم‌های WebSocket صفحه و استخراج تیک‌های قیمت
       (Symbol, Price, Timestamp با دقت میلی‌ثانیه) از داخل آن‌ها.
    ۳. مدیریت قطع/وصل شدن WebSocket و تلاش مجدد در صورت خطای شبکه.

نکتهٔ مهم دربارهٔ فرمت پیام‌ها:
    پروتکل داخلی Pocket Option مستند و رسمی نیست و ممکن است در طول زمان تغییر کند.
    به همین دلیل این ماژول تمام فریم‌های خام دریافتی را (در صورت فعال بودن
    DEBUG_LOG_RAW_FRAMES) در فایل data/raw_ws_frames.log ذخیره می‌کند تا در صورتی
    که تابع extract_ticks_from_payload() تیک‌ها را درست تشخیص نداد، بتوانید با
    نگاه‌کردن به فریم‌های واقعی، الگوی جدید را به آن تابع اضافه کنید.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from playwright.async_api import async_playwright, WebSocket, BrowserContext, Page

import config


@dataclass
class Tick:
    """یک تیک قیمت لحظه‌ای."""
    symbol: str
    price: float
    timestamp: float  # یونیکس‌تایم با دقت میلی‌ثانیه (float)

    @property
    def datetime_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)


# الگوی شناسایی پیام‌های socket.io: عددی در ابتدای رشته (مثلاً "42") و بعد JSON
_SOCKET_IO_PREFIX_RE = re.compile(r"^\d+")


def _log_raw_frame(raw: str) -> None:
    """ذخیره فریم خام برای بررسی دستی و اصلاح parser در آینده."""
    if not config.DEBUG_LOG_RAW_FRAMES:
        return
    try:
        with open(config.RAW_WS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(raw[:2000] + "\n")  # برش برای جلوگیری از فایل‌های حجیم
    except OSError:
        pass


def _strip_socketio_prefix(raw: str) -> Optional[object]:
    """
    فریم‌های socket.io معمولاً به شکل «کد عددی» + «JSON» هستند، مثل:
        42["updateStream",[["#AUDCAD",1690000000.123,0.87421]]]
    این تابع کد عددی ابتدایی را حذف کرده و باقیمانده را JSON پارس می‌کند.
    """
    match = _SOCKET_IO_PREFIX_RE.match(raw)
    body = raw[match.end():] if match else raw
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None


def _looks_like_tick_row(item) -> bool:
    """
    تشخیص می‌دهد آیا یک لیست شبیه [symbol, timestamp, price] است یا نه.
    این هیوریستیک (Heuristic) عمومی است چون فرمت دقیق پیام رسمی مستند نشده.
    """
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return False
    symbol, ts, price = item[0], item[1], item[2]
    return (
        isinstance(symbol, str)
        and isinstance(ts, (int, float))
        and isinstance(price, (int, float))
    )


def extract_ticks_from_payload(payload) -> list[Tick]:
    """
    به‌صورت بازگشتی (Recursive) در ساختار JSON پیام دنبال ردیف‌های شبیه
    تیک قیمت می‌گردد و لیستی از آبجکت‌های Tick برمی‌گرداند.

    اگر پیام‌های واقعی Pocket Option در پروژهٔ شما ساختار متفاوتی داشتند،
    کافی است این تابع را با توجه به نمونه‌های ذخیره‌شده در
    data/raw_ws_frames.log اصلاح کنید.
    """
    ticks: list[Tick] = []

    def _walk(node):
        if _looks_like_tick_row(node):
            symbol, ts, price = node[0], node[1], node[2]
            # برخی سرورها timestamp را به میلی‌ثانیه می‌فرستند؛ به ثانیه تبدیل می‌کنیم
            ts_seconds = ts / 1000.0 if ts > 1e12 else float(ts)
            ticks.append(Tick(symbol=symbol, price=float(price), timestamp=ts_seconds))
            return
        if isinstance(node, list):
            for child in node:
                _walk(child)
        elif isinstance(node, dict):
            for child in node.values():
                _walk(child)

    _walk(payload)
    return ticks


# کلیدهایی که در پیام‌های «نتیجهٔ معامله»/«بستن قرارداد» بروکرهای مشابه معمولاً دیده می‌شوند.
# فرمت دقیق Pocket Option مستند نیست؛ این یک حدس منطقی است که باید با نگاه‌کردن به
# data/raw_ws_frames.log بعد از چند معاملهٔ واقعی تأیید/اصلاح شود.
_DEAL_MARKER_KEYS = {
    "profit", "win", "iswin", "is_win", "result", "status",
    "amount", "payout", "closeprofit", "close_profit", "deal",
}


def extract_deal_candidates(payload) -> list[dict]:
    """
    به‌صورت بازگشتی دنبال دیکشنری‌هایی می‌گردد که حداقل یکی از کلیدهای بالا را
    دارند — این‌ها «کاندیدای پیام نتیجهٔ معامله» هستند، نه لزوماً قطعی.
    خروجی این تابع صرفاً برای تلاش برای گرفتن نتیجهٔ معامله از خودِ پلتفرم است؛
    اگر چیزی پیدا نشود یا اشتباه تفسیر شود، سیستم به‌طور خودکار به محاسبهٔ
    داخلی (بر اساس تیک‌های خودمان) برمی‌گردد.
    """
    candidates: list[dict] = []

    def _walk(node):
        if isinstance(node, dict):
            keys_lower = {str(k).lower() for k in node.keys()}
            if keys_lower & _DEAL_MARKER_KEYS:
                candidates.append(node)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return candidates


class DealResultBuffer:
    """
    نگه‌دارندهٔ کاندیداهای اخیر «پیام نتیجهٔ معامله» که از WebSocket شنود شده‌اند،
    به همراه زمان محلی دریافت (wall-clock) هر کدام. DataLogger بعد از هر معامله
    در یک بازهٔ زمانی مشخص (حوالی لحظهٔ انقضا) دنبال نزدیک‌ترین کاندیدا می‌گردد.
    """

    def __init__(self, maxlen: int = 100):
        self.buffer: deque[tuple[float, dict]] = deque(maxlen=maxlen)

    def add(self, raw: dict) -> None:
        self.buffer.append((time.time(), raw))

    def find_and_consume(self, after_time: float, before_time: float) -> Optional[dict]:
        """
        اولین کاندیدایی که زمان دریافتش داخل بازهٔ [after_time, before_time] باشد
        را برمی‌گرداند و از بافر حذف می‌کند (تا برای معاملهٔ دیگری دوباره استفاده نشود).
        """
        for i, (received_at, raw) in enumerate(self.buffer):
            if after_time <= received_at <= before_time:
                del self.buffer[i]
                return raw
        return None


class WebSocketListener:
    """
    این کلاس به تمام WebSocketهای صفحه گوش می‌دهد، تیک‌های استخراج‌شده را
    داخل یک asyncio.Queue می‌ریزد و در صورت قطع اتصال، وضعیت را لاگ می‌کند.
    """

    def __init__(self, tick_queue: asyncio.Queue, deal_buffer: Optional[DealResultBuffer] = None):
        self.tick_queue = tick_queue
        self.deal_buffer = deal_buffer
        self._active_sockets: set[WebSocket] = set()
        self.connected_event = asyncio.Event()

    def attach(self, page: Page) -> None:
        """این متد باید بعد از باز شدن صفحه صدا زده شود تا شنود شروع شود."""
        page.on("websocket", self._on_websocket_created)

    def _on_websocket_created(self, ws: WebSocket) -> None:
        print(f"[WebSocket] اتصال جدید شناسایی شد: {ws.url}")
        self._active_sockets.add(ws)
        self.connected_event.set()

        ws.on("framereceived", lambda payload: self._handle_frame(payload))
        ws.on("close", lambda: self._on_close(ws))
        ws.on("socketerror", lambda err: print(f"[WebSocket] خطا: {err}"))

    def _on_close(self, ws: WebSocket) -> None:
        print(f"[WebSocket] اتصال بسته شد: {ws.url}")
        self._active_sockets.discard(ws)
        if not self._active_sockets:
            self.connected_event.clear()

    def _handle_frame(self, raw_payload) -> None:
        # payload می‌تواند str یا bytes باشد
        raw_str = raw_payload if isinstance(raw_payload, str) else raw_payload.decode("utf-8", errors="ignore")
        _log_raw_frame(raw_str)

        payload = _strip_socketio_prefix(raw_str)
        if payload is None:
            return

        ticks = extract_ticks_from_payload(payload)
        for tick in ticks:
            try:
                self.tick_queue.put_nowait(tick)
            except asyncio.QueueFull:
                # اگر پردازش کندتر از دریافت باشد، قدیمی‌ترین را دور می‌ریزیم
                # تا همیشه تازه‌ترین قیمت در دسترس باشد (مهم برای تصمیم‌گیری real-time)
                try:
                    self.tick_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.tick_queue.put_nowait(tick)

        if self.deal_buffer is not None:
            for deal in extract_deal_candidates(payload):
                self.deal_buffer.add(deal)


async def launch_browser_and_wait_for_login() -> tuple[BrowserContext, Page]:
    """
    مرورگر را با Playwright باز می‌کند، به صفحهٔ حساب دمو Pocket Option می‌رود
    و منتظر می‌ماند تا کاربر به‌صورت دستی لاگین کند.

    از یک پروفایل دائمی (Persistent Context) استفاده می‌شود تا در اجراهای بعدی
    نیازی به لاگین مجدد نباشد.

    اگر config.BROWSER_CHANNEL = "chrome" باشد، به‌جای Chromium باندل‌شدهٔ
    Playwright، از Google Chrome واقعیِ نصب‌شده روی سیستم استفاده می‌شود -
    بعضی پلتفرم‌ها Chromium باندل‌شده را «مرورگر ناشناخته» تشخیص می‌دهند و
    اجازهٔ لاگین نمی‌دهند. علاوه بر این، پرچم --disable-blink-features=
    AutomationControlled هم اضافه می‌شود تا نشانه‌های آشکار خودکارسازی
    (navigator.webdriver و مشابه) کمتر قابل‌تشخیص باشند.
    """
    playwright = await async_playwright().start()

    launch_kwargs = dict(
        user_data_dir=str(config.USER_DATA_DIR),
        headless=config.BROWSER_HEADLESS,
        args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
        no_viewport=True,
    )
    if config.BROWSER_CHANNEL:
        launch_kwargs["channel"] = config.BROWSER_CHANNEL

    context = await playwright.chromium.launch_persistent_context(**launch_kwargs)
    page = context.pages[0] if context.pages else await context.new_page()

    await page.goto(config.POCKET_OPTION_URL, wait_until="domcontentloaded")

    print("\n" + "=" * 70)
    print("لطفاً به‌صورت دستی وارد حساب دمو (Demo) خود در Pocket Option شوید.")
    print("پس از این‌که صفحهٔ معاملاتی و چارت قیمت کاملاً بارگذاری شد،")
    print("در همین ترمینال Enter را بزنید تا اسکریپت ادامه پیدا کند.")
    print("=" * 70 + "\n")

    # منتظر ورودی کاربر در ترمینال می‌مانیم بدون بلاک کردن event loop
    await asyncio.get_event_loop().run_in_executor(None, input, ">>> پس از لاگین، Enter را بزنید: ")

    return context, page


async def reconnect_page_if_needed(page: Page) -> None:
    """
    در صورت قطعی اینترنت یا بسته شدن ناگهانی WebSocket، این تابع صفحه را
    Reload می‌کند تا اتصال WebSocket دوباره برقرار شود.
    """
    try:
        await page.reload(wait_until="domcontentloaded", timeout=30_000)
        print("[Reconnect] صفحه با موفقیت مجدداً بارگذاری شد.")
    except Exception as exc:  # noqa: BLE001 - می‌خواهیم هر نوع خطای شبکه را بگیریم و ادامه دهیم
        print(f"[Reconnect] تلاش برای اتصال مجدد ناموفق بود: {exc}")
