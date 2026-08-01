"""
تبدیل اسنپ‌شات خام ویژگی‌ها به فرمت عددی مناسب برای مدل یادگیری ماشین
========================================================================

این ماژول تنها جایی است که ساختار «ورودی مدل» تعریف می‌شود، و دقیقاً همین
تابع هم در اسکریپت آموزش (train/train_model.py) و هم در پیش‌بینی زنده
(src/live_predictor.py) صدا زده می‌شود — تا هیچ‌وقت تفاوتی بین ستون‌های
دیده‌شده در آموزش و ستون‌های ساخته‌شده در لحظهٔ معامله به‌وجود نیاید.

چهار کار انجام می‌شود:
    ۱. Flatten کردن ستون‌های JSON (شکل چارت و توالی سرعت تیک) به چند ستون
       عددی مجزا با طول ثابت (Padding با مقدار خنثی اگر داده کافی نبود).
    ۲. تبدیل جهت معامله (direction: "CALL"/"PUT") به یک ویژگی عددی
       (direction_call: ۱ یا ۰) — چون در معاملهٔ زنده، جهت خودش یکی از
       ورودی‌های تصمیم است (مدل هر دو جهت را امتحان می‌کند)، نه یک برچسب.
    ۳. ساخت چند ویژگی «هم‌جهت با تصمیم» (Direction-Interaction): برای این‌که
       مدل مجبور نباشد خودش از میان ده‌ها فیچر بی‌ربط، تعامل بین direction_call
       و فیچرهای جهت‌دار (مومنتوم، روند، رنگ کندل، فاصله تا حمایت/مقاومت) را
       کشف کند — که با تعداد دادهٔ محدود عملاً رخ نمی‌دهد (دیده شد که
       XGBoost حتی یک‌بار هم روی direction_call به‌تنهایی Split نمی‌زند)،
       این تعامل‌ها را صریح می‌سازیم: مثلاً «آیا مومنتوم هم‌جهت با معاملهٔ
       انتخابی است» یا «چقدر فاصله در جهت مطلوب معامله وجود دارد».
    ۴. حذف قطعی ستون‌هایی که یا متادیتای خام قیمت‌اند، یا (مهم‌تر) فقط بعد
       از پایان معامله محاسبه می‌شوند — یعنی در لحظهٔ ورود اصلاً وجود ندارند
       (مثل price_change_pct) و نگه‌داشتنشان در دادهٔ آموزشی «نشت اطلاعات از
       آینده» (Data Leakage) است: دقتی که با آن‌ها روی داده تست به دست
       می‌آید کاذب است، چون هنگام معاملهٔ واقعی چنین ستونی در دسترس نیست.
"""

from __future__ import annotations

import json
from typing import Any

import config

# ستون‌هایی که هرگز نباید به مدل داده شوند: یا متادیتای خام/دیباگ‌اند، یا
# فقط بعد از پایان معامله (با دانستن قیمت خروج) قابل محاسبه‌اند.
_EXCLUDED_COLUMNS = {
    "meta_symbol",
    "meta_entry_price",
    "meta_entry_timestamp",
    "meta_exit_price",
    "meta_exit_timestamp",
    "meta_raw_deal_json",
    "meta_trade_source",
    "result_source",
    "tick_fallback_result",
    "price_change_pct",  # فقط بعد از پایان معامله معلوم می‌شود - نشت اطلاعات
    "result",
    # شمارندهٔ تجمعی/وین‌ریت کلی تا این لحظه: چون فقط با گذر زمان/تعداد معامله
    # افزایش/تغییر می‌کند، در تقسیم تصادفی (K-Fold/train_test_split) می‌تواند
    # به‌جای یک الگوی بازار واقعی، «بازهٔ زمانی/رژیم دادهٔ این ردیف» را به مدل
    # نشان دهد - سیگنالی که در معاملهٔ زنده (که همیشه در انتهای دنباله است)
    # تعمیم پیدا نمی‌کند.
    "total_trades_so_far",
    "overall_winrate",
}

_CHART_SHAPE_FIELDS = ("o", "h", "l", "c", "bullish")

# منبع واحد حقیقت برای فیچرهای «جهت‌دار» (خودِ direction_call + هر فیچری که
# _add_direction_interaction_features می‌سازد). هر جای دیگری از کد (مثل
# گزارش train_model.py) که نیاز به شناسایی این ستون‌ها دارد، باید از همین
# مجموعه استفاده کند - نه یک فیلتر رشته‌ای جداگانه که ممکن است با تغییر نام
# یکی از این فیچرها هماهنگ نماند (دقیقاً همان باگی که قبلاً باعث شد
# streak_aligned_with_direction اشتباهاً «بدون سیگنال جهت» گزارش شود).
DIRECTION_INTERACTION_COLUMNS = frozenset({
    "direction_call",
    "momentum_in_direction",
    "momentum_smoothed_in_direction",
    "acceleration_in_direction",
    "trend_aligned_with_direction",
    "streak_aligned_with_direction",
    "candle_color_aligned_with_direction",
    "distance_to_target_atr",
    "distance_to_obstacle_atr",
})

# فیچرهای «نوسان ریز» (Micro-Swing): یک خانوادهٔ به‌هم‌مرتبط از فیچرها (نه
# مستقل از هم)، و در ترین‌های مختلف همیشه چند عضو (نه لزوماً همیشه یکی)
# از این خانواده در فیچرهای برتر ظاهر شده‌اند. چون تعدادشان کم است (۱۵ ستون)،
# کل خانواده - صرف‌نظر از رتبهٔ تک‌تک اعضا در یک ترین خاص - در مدل «فقط
# فیچرهای برتر» نگه داشته می‌شود، تا نوسان رتبه بین اجراها یکی از آن‌ها را
# اتفاقی حذف نکند.
MICRO_SWING_COLUMNS = frozenset({
    "micro_swing_last_direction",
    "micro_swing_last_speed_pct",
    "micro_swing_last_acceleration_pct",
    "micro_swing_last3_speed_pct",
    "micro_swing_last3_acceleration_pct",
    "micro_swing_last3_displacement_pct",
    "micro_swing_last3_distance_pct",
    "micro_swing_last5_speed_pct",
    "micro_swing_last5_acceleration_pct",
    "micro_swing_last5_displacement_pct",
    "micro_swing_last5_distance_pct",
    "micro_swing_last15_speed_pct",
    "micro_swing_last15_acceleration_pct",
    "micro_swing_last15_displacement_pct",
    "micro_swing_last15_distance_pct",
})


def chart_shape_long_current_candle_columns() -> frozenset[str]:
    """
    ستون‌های تخت‌شدهٔ آخرین کندل (کندلِ جاری) در پنجرهٔ بلندِ شکل چارت - یعنی
    موقعیت کندل فعلی نسبت به یک بازهٔ وسیع‌تر (نه فقط پنجرهٔ کوتاه ۱۰ کندلی).
    در آخرین ترین، دقیقاً همین ستون‌ها (به‌خصوص های و اوپن) رتبهٔ ۲ و ۷ اهمیت
    را در کل مدل داشتند؛ برای این‌که نوسان رتبه در ترین‌های بعدی این سیگنالِ
    اثبات‌شده را اتفاقی از مدل «فقط فیچرهای برتر» حذف نکند، همیشه نگه داشته
    می‌شوند.
    """
    last_idx = config.CHART_WINDOW_CANDLES_LONG - 1
    return frozenset(f"chart_shape_long_{last_idx}_{field}" for field in _CHART_SHAPE_FIELDS)


def _flatten_chart_shape_generic(row: dict, json_key: str, prefix: str, window: int) -> None:
    raw = row.pop(json_key, None)
    try:
        candles = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        candles = []
    if not isinstance(candles, list):
        candles = []

    pad_needed = window - len(candles)
    for i in range(window):
        idx = i - pad_needed  # کندل‌های کم‌تر از پنجره از چپ Pad می‌شوند
        candle = candles[idx] if 0 <= idx < len(candles) else None
        for field in _CHART_SHAPE_FIELDS:
            neutral = 0.5 if field != "bullish" else 0
            value = candle.get(field, neutral) if candle else neutral
            row[f"{prefix}_{i}_{field}"] = value


def _flatten_chart_shape(row: dict) -> None:
    _flatten_chart_shape_generic(row, "chart_shape_json", "chart_shape", config.CHART_WINDOW_CANDLES)


def _flatten_chart_shape_long(row: dict) -> None:
    """پنجرهٔ دوم و بزرگ‌تر (پیش‌فرض ۲۵ کندل) - همان روش Flatten، پیشوند جدا (chart_shape_long_*) تا با پنجرهٔ کوچک قاطی نشود."""
    _flatten_chart_shape_generic(
        row, "chart_shape_long_json", "chart_shape_long", config.CHART_WINDOW_CANDLES_LONG
    )


def _flatten_tick_velocities(row: dict) -> None:
    raw = row.pop("recent_tick_velocities_json", None)
    length = max(config.TICK_BUFFER_SIZE - 1, 0)
    try:
        values = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        values = []
    if not isinstance(values, list):
        values = []

    pad_needed = length - len(values)
    for i in range(length):
        idx = i - pad_needed
        row[f"tick_velocity_{i}"] = values[idx] if 0 <= idx < len(values) else 0.0


def _encode_direction(row: dict) -> None:
    direction = row.pop("direction", None)
    row["direction_call"] = 1 if direction == "CALL" else 0


def _add_direction_interaction_features(row: dict) -> None:
    """
    فیچرهای «هم‌جهت با تصمیم»: به‌جای این‌که مدل خودش از میان ده‌ها فیچر
    بی‌ربط تعامل بین جهت معامله و فیچرهای جهت‌دار را کشف کند (که با چند صد
    ردیف داده عملاً اتفاق نمی‌افتد)، این تعامل را صریح می‌سازیم. sign=+۱ برای
    CALL و -۱ برای PUT است؛ ضرب کردن یک فیچر جهت‌دار در sign یعنی «این فیچر
    هم‌جهت با معاملهٔ انتخابی است یا مخالف آن».
    """
    sign = 1.0 if row.get("direction_call") == 1 else -1.0

    def _num(key: str, default: float = 0.0) -> float:
        value = row.get(key)
        return default if value is None else float(value)

    row["momentum_in_direction"] = sign * _num("velocity_pct")
    row["momentum_smoothed_in_direction"] = sign * _num("velocity_pct_smoothed")
    row["acceleration_in_direction"] = sign * _num("acceleration_pct")
    row["trend_aligned_with_direction"] = sign * _num("trend_regime")
    row["streak_aligned_with_direction"] = sign * _num("candle_color_streak")
    row["candle_color_aligned_with_direction"] = sign * (2 * _num("candle_curr_is_bullish", 0.5) - 1)

    # برای CALL، «هدف» صعود قیمت است (مقاومت مانع، حمایت پشتیبان)؛ برای PUT
    # برعکس. این دو فیچر می‌گویند «در جهتی که رفته‌ام چقدر راه باز است» و
    # «نزدیک‌ترین دیوار مخالف من چقدر فاصله دارد».
    dist_to_resistance = row.get("dist_to_resistance_atr")
    dist_to_support = row.get("dist_to_support_atr")
    if sign > 0:
        row["distance_to_target_atr"] = dist_to_resistance
        row["distance_to_obstacle_atr"] = dist_to_support
    else:
        row["distance_to_target_atr"] = dist_to_support
        row["distance_to_obstacle_atr"] = dist_to_resistance


def flatten_snapshot_for_model(snapshot: dict) -> dict[str, float]:
    """
    یک اسنپ‌شات خام (همان دیکشنری که DataLogger.capture_entry می‌سازد، یا یک
    ردیف خوانده‌شده از SQLite) را به دیکشنری کاملاً عددی برای مدل تبدیل می‌کند.
    """
    row = dict(snapshot)

    _encode_direction(row)
    _add_direction_interaction_features(row)
    _flatten_chart_shape(row)
    _flatten_chart_shape_long(row)
    _flatten_tick_velocities(row)

    for col in _EXCLUDED_COLUMNS:
        row.pop(col, None)

    numeric_row: dict[str, float] = {}
    for key, value in row.items():
        if value is None:
            numeric_row[key] = 0.0
        elif isinstance(value, bool):
            numeric_row[key] = float(value)
        elif isinstance(value, (int, float)):
            numeric_row[key] = float(value)
        else:
            try:
                numeric_row[key] = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass  # ستون متنی غیرقابل‌تبدیل - نادیده گرفته می‌شود
    return numeric_row
