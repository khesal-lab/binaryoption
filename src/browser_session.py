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


# پیام‌های «رویداد باینری» socket.io (وقتی سرور یک Attachment جدا می‌فرستد) به شکل
# «کد عددی» + «-» + JSON هستند، مثل:
#   451-["successcloseOrder",{"_placeholder":true,"num":0}]
# و بدنهٔ واقعی پیام (Attachment) در یک فریم جداگانهٔ بعدی می‌رسد. چون این فریم دوم
# هیچ نامی از رویداد ندارد، بدون ردیابی این نام از فریم اول، غیرممکن است بشود
# successopenOrder (که فقط پیش‌بینی خوش‌بینانهٔ سود در لحظهٔ باز شدن معامله است،
# همیشه مثبت) را از successcloseOrder (نتیجهٔ واقعیِ بسته‌شدن معامله) تشخیص داد.
_SOCKET_IO_BINARY_EVENT_RE = re.compile(r"^\d+-")


def _extract_binary_event_name(raw: str) -> Optional[str]:
    """
    اگر raw یک فریمِ «اعلام رویداد باینری» باشد، نام رویداد را برمی‌گرداند؛
    وگرنه None (یعنی این فریم یا یک پیام معمولی است، یا خودِ بدنهٔ یک Attachment).
    """
    match = _SOCKET_IO_BINARY_EVENT_RE.match(raw)
    if not match:
        return None
    try:
        parsed = json.loads(raw[match.end():])
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
        return parsed[0]
    return None


# کف قابل‌قبول برای timestamp (تقریباً سال ۲۰۰۱ به بعد، به ثانیه). پیام‌های دیگری مثل
# successsignals/stats شکل [symbol, seconds, count] دارند (مثل ["AUDCHF_otc", 180, 293])
# که بدون این حد پایین، دقیقاً شبیه یک ردیف تیک واقعی تشخیص داده می‌شدند و به‌عنوان
# قیمت و timestamp جعلی وارد صف تیک‌ها می‌شدند.
_MIN_PLAUSIBLE_TICK_TIMESTAMP = 1_000_000_000


def _looks_like_tick_row(item) -> bool:
    """
    تشخیص می‌دهد آیا یک لیست شبیه [symbol, timestamp, price] است یا نه.
    این هیوریستیک (Heuristic) عمومی است چون فرمت دقیق پیام رسمی مستند نشده.
    """
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return False
    symbol, ts, price = item[0], item[1], item[2]
    if not (
        isinstance(symbol, str)
        and isinstance(ts, (int, float))
        and isinstance(price, (int, float))
    ):
        return False
    ts_seconds = ts / 1000.0 if ts > 1e12 else float(ts)
    return ts_seconds >= _MIN_PLAUSIBLE_TICK_TIMESTAMP


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


# پیام updateAssets لیست کامل دارایی‌های قابل‌معامله و پی‌آوت لحظه‌ایِ هرکدام را
# می‌فرستد. هر ردیف چیزی شبیه این است (نمونهٔ واقعی مشاهده‌شده در
# data/raw_ws_frames.log):
#   [5,"#AAPL","Apple","stock",2,50,60,30,3,0,170,0,[],1785256500,false,[...],-1,60,1785256500]
# اندیس ۱ = نماد (با پیشوند #)، اندیس ۳ = نوع دارایی (stock/currency/...)،
# اندیس ۵ = درصد پی‌آوت لحظه‌ای. مثل بقیهٔ این ماژول، فرمت رسمی مستند نیست؛
# این هم یک هیوریستیک است (نه استخراج دقیق بر اساس مستندات پروتکل).
def _looks_like_asset_payout_row(item) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 6:
        return False
    symbol, _display_name, asset_type, payout = item[1], item[2], item[3], item[5]
    return (
        isinstance(symbol, str)
        and symbol.startswith("#")
        and isinstance(asset_type, str)
        and isinstance(payout, (int, float))
        and 0 <= payout <= 100
    )


def extract_asset_payouts_from_payload(payload) -> list[dict]:
    """
    به‌صورت بازگشتی دنبال ردیف‌های شبیه دارایی/پی‌آوت (پیام updateAssets)
    می‌گردد. خروجی: لیستی از {"symbol", "asset_type", "payout_percent"}.
    """
    results: list[dict] = []

    def _walk(node):
        if _looks_like_asset_payout_row(node):
            results.append({
                "symbol": node[1].lstrip("#"),
                "asset_type": node[3],
                "payout_percent": float(node[5]),
            })
            return
        if isinstance(node, list):
            for child in node:
                _walk(child)
        elif isinstance(node, dict):
            for child in node.values():
                _walk(child)

    _walk(payload)
    return results


class AssetPayoutTracker:
    """
    آخرین درصد پی‌آوت شناخته‌شده برای هر نماد را نگه می‌دارد (از روی پیام‌های
    updateAssets که پلتفرم به‌صورت دوره‌ای/هنگام تغییر می‌فرستد). هدف: در لحظهٔ
    ورود به یک معامله بشود فهمید این نماد در حال حاضر نسبت به سایر دارایی‌های
    هم‌نوع (هم asset_type) در کجای «لیست پی‌آوت» قرار دارد - به‌عنوان یک سیگنال
    نسبیِ رژیم بازار، نه یک شناسهٔ ثابت نماد.
    """

    def __init__(self):
        self._payouts: dict[str, dict] = {}

    def update_from_payload(self, payload) -> None:
        for entry in extract_asset_payouts_from_payload(payload):
            self._payouts[entry["symbol"]] = entry

    def get_payout(self, symbol: str) -> Optional[float]:
        entry = self._payouts.get(symbol)
        return entry["payout_percent"] if entry else None

    def get_rank_percentile(self, symbol: str) -> Optional[dict]:
        """
        رتبه (۱=بالاترین پی‌آوت) و صدک (۰=بالاترین، ۱=پایین‌ترین) این نماد در
        مقایسه با سایر نمادهای هم‌نوع (همان asset_type) که تا این لحظه دیده
        شده‌اند. اگر خودِ نماد یا حداقل یک همتای دیگر هنوز دیده نشده باشد
        (مقایسه‌ای بی‌معنی با فقط یک عضو)، None برمی‌گرداند.
        """
        entry = self._payouts.get(symbol)
        if entry is None:
            return None
        peers = [
            e["payout_percent"] for e in self._payouts.values()
            if e.get("asset_type") == entry.get("asset_type")
        ]
        if len(peers) < 2:
            return None
        peers_sorted = sorted(peers, reverse=True)
        rank = peers_sorted.index(entry["payout_percent"]) + 1
        return {
            "payout_rank": rank,
            "payout_total_peers": len(peers),
            "payout_percentile": (rank - 1) / (len(peers) - 1),
        }


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

    def __init__(
        self,
        tick_queue: asyncio.Queue,
        deal_buffer: Optional[DealResultBuffer] = None,
        asset_payout_tracker: Optional[AssetPayoutTracker] = None,
    ):
        self.tick_queue = tick_queue
        self.deal_buffer = deal_buffer
        self.asset_payout_tracker = asset_payout_tracker
        self._active_sockets: set[WebSocket] = set()
        self.connected_event = asyncio.Event()
        # نام رویداد باینری اعلام‌شده در فریمِ قبلی (اگر بود) - چون بدنهٔ واقعیِ
        # آن رویداد در فریم بعدی، جدا و بدون نام می‌رسد. به محض مصرف در همان
        # فریمِ بعدی، دوباره None می‌شود (نگاه کنید به _extract_binary_event_name).
        self._pending_binary_event_name: Optional[str] = None

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

        # اگر این فریم فقط «اعلام یک رویداد باینری» باشد (بدنهٔ واقعی در فریم
        # بعدی می‌رسد)، نام رویداد را برای فریم بعدی نگه می‌داریم و همین‌جا تمام.
        binary_event_name = _extract_binary_event_name(raw_str)
        if binary_event_name is not None:
            self._pending_binary_event_name = binary_event_name
            return

        # این فریم، بدنهٔ واقعیِ همان رویداد باینریِ قبلی است (اگر بود) - یک‌بار
        # مصرف می‌شود تا به فریم‌های نامرتبط بعدی نشت نکند.
        associated_event_name = self._pending_binary_event_name
        self._pending_binary_event_name = None

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

        # نکتهٔ حیاتی: پیام «باز شدن معامله» (successopenOrder) هم دقیقاً همان
        # کلیدهای «نتیجهٔ معامله» را دارد (profit/amount) - ولی مقدار profit آن
        # همیشه مثبت است (پیش‌بینیِ خوش‌بینانهٔ سود در صورت برد، نه نتیجهٔ واقعی).
        # اگر این پیام اشتباهاً به‌عنوان نتیجهٔ یک معامله تفسیر شود، هر معامله
        # (چه واقعاً برنده چه بازنده) به‌عنوان WIN لیبل می‌خورد. برای جلوگیری از
        # این آلودگی، فقط رویدادهایی که در نامشان "close" هست (یا نامشان اصلاً
        # شناخته‌نشده - برای سازگاری با فرمت‌های ساده‌ای که این مشکل را ندارند)
        # اجازهٔ ورود به deal_buffer را دارند؛ رویدادهای شناخته‌شدهٔ غیر-close
        # (مثل successopenOrder) صریحاً رد می‌شوند و به جای آن‌ها محاسبهٔ داخلی
        # (tick_fallback) جایگزین می‌شود - که همیشه در دسترس و درست است.
        is_known_non_result_event = (
            associated_event_name is not None and "close" not in associated_event_name.lower()
        )
        if self.deal_buffer is not None and not is_known_non_result_event:
            for deal in extract_deal_candidates(payload):
                self.deal_buffer.add(deal)

        if self.asset_payout_tracker is not None:
            self.asset_payout_tracker.update_from_payload(payload)


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
