"""
اسکریپت یک‌باره برای ساخت پروفایل اولیهٔ Playwright از روی پروفایل واقعیِ
Chrome روزمرهٔ شما
================================================================================

اگر پلتفرم هنگام لاگین از داخل اسکریپت‌های این پروژه پیام «مرورگر شناخته‌شده
نیست» می‌دهد، ولی با همان اکانت در Chrome روزمرهٔ خودتان (که از قبل لاگین
کرده‌اید) مشکلی ندارید، این اسکریپت یک کپی از همان پروفایل واقعیِ Chrome را
در config.USER_DATA_DIR قرار می‌دهد - یعنی از این به بعد main.py/collect_data.py
(که حالا با config.BROWSER_CHANNEL = "chrome" از خودِ Chrome واقعیِ نصب‌شده
استفاده می‌کنند، نه Chromium باندل‌شدهٔ Playwright) یک نشستِ از قبل
تأییدشده را می‌بینند، نه یک مرورگر تازه و ناشناس.

⚠️ قبل از اجرای این اسکریپت، Chrome را کاملاً ببندید (هیچ پنجره/تب بازی
نداشته باشد) تا تمام دادهٔ نشست روی دیسک نوشته شده باشد.

این اسکریپت فقط از پروفایل واقعی «می‌خواند» و هیچ تغییری در آن نمی‌دهد؛
فقط یک کپی در مسیر دیگری (USER_DATA_DIR) می‌سازد. اگر USER_DATA_DIR از قبل
چیزی داشته باشد (مثلاً نشست دموی قبلی)، کامل رونویسی می‌شود - قبل از این کار
از شما تأیید گرفته می‌شود.

استفاده:
    python setup_chrome_profile.py                  # حدس خودکار مسیر پروفایل
    python setup_chrome_profile.py /path/to/profile  # مسیر دستی (مثلاً اگر
                                                      # از پروفایل غیر پیش‌فرض
                                                      # Chrome استفاده می‌کنید)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import config

# فایل‌های قفل مخصوص همان پردازهٔ در حال اجرای Chrome - کپی‌کردنشان می‌تواند
# باعث شود مرورگر بعدی فکر کند پروفایل «در حال استفادهٔ یک نمونهٔ دیگر» است.
_EXCLUDED_NAMES = {"SingletonLock", "SingletonCookie", "SingletonSocket"}


def _default_chrome_profile_dir() -> Path:
    """حدس مسیر پیش‌فرض پروفایل Chrome روی لینوکس (اوبونتو/دبیان)."""
    home = Path.home()
    candidates = [
        home / ".config" / "google-chrome",
        home / ".config" / "google-chrome-stable",
        home / ".config" / "chromium",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ignore_locks(_directory: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _EXCLUDED_NAMES}


def main() -> None:
    source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else _default_chrome_profile_dir()

    if not source.exists():
        print(f"❌ مسیر پروفایل پیدا نشد: {source}")
        print("مسیر واقعی پروفایل Chrome خودتان را به‌عنوان آرگومان بدهید، مثلاً:")
        print("    python setup_chrome_profile.py ~/.config/google-chrome")
        sys.exit(1)

    destination = config.USER_DATA_DIR

    print(f"مبدأ (پروفایل واقعی Chrome):      {source}")
    print(f"مقصد (پروفایل اختصاصی این ربات): {destination}")
    print()
    print("⚠️  مطمئن شوید Chrome کاملاً بسته است (هیچ پنجره/تبی باز نمانده) - "
          "وگرنه ممکن است دادهٔ نشست کامل روی دیسک نوشته نشده باشد.")
    if destination.exists() and any(destination.iterdir()):
        print(f"⚠️  مسیر مقصد ({destination}) از قبل داده دارد و کامل رونویسی خواهد شد "
              "(مثلاً نشست دموی قبلی این ربات).")
    confirm = input("\nبرای ادامه 'yes' را تایپ کنید (هر چیز دیگری = لغو): ").strip().lower()
    if confirm != "yes":
        print("لغو شد؛ هیچ تغییری اعمال نشد.")
        return

    if destination.exists():
        shutil.rmtree(destination)

    print("در حال کپی‌کردن (بسته به حجم کش Chrome ممکن است کمی طول بکشد)...")
    shutil.copytree(source, destination, ignore=_ignore_locks, dirs_exist_ok=True)

    print(f"\n✅ تمام شد. پروفایل در {destination} آماده است.")
    print("حالا main.py یا collect_data.py را اجرا کنید - اگر نشست Chrome هنوز "
          "معتبر باشد، دیگر نیازی به لاگین دوباره نیست. اگر پلتفرم باز هم درخواست "
          "لاگین کرد، این‌بار باید بدون خطای «مرورگر ناشناخته» انجام شود.")


if __name__ == "__main__":
    main()
