"""
تحلیل زنجیره‌های برد/باخت پشت‌سرهم استراتژی سطوح (level_strategy)
===================================================================

هدف: پیدا کردن تفاوت بین موقعیت‌هایی که استراتژی سطوح (collect_data.py) پشت‌سرهم
برنده بوده و موقعیت‌هایی که پشت‌سرهم بازنده بوده - مخصوصاً در شرایطی که قیمت با
نوسان بین کندل‌ها ولی در نهایت در یک مسیر اصلی (روند) حرکت می‌کند.

فرضیهٔ اصلی که این اسکریپت آزمایش می‌کند:
    level_strategy.py فقط به اکسترمم‌های محلی *داخل همان کندل جاری* نسبت به
    open آن کندل نگاه می‌کند (تابع check_signal در src/level_strategy.py) و
    هیچ آگاهی‌ای از روند/استریک چندکندلی بزرگ‌تر ندارد. وقتی نوسان داخل کندل
    هر دو طرف open را لمس کند («spans_open»)، استراتژی همیشه یک سیگنال
    بازگشتی (Mean-Reversion) می‌زند: نزدیک سقف -> PUT، نزدیک کف -> CALL - حتی
    اگر روند بزرگ‌تر (چند کندل قبل) قویاً در یک جهت باشد و همان لمس سقف/کف در
    واقع فقط یک مکث کوچک قبل از ادامهٔ روند بوده باشد، نه بازگشت واقعی.

    اگر این فرضیه درست باشد، باید ببینیم: معاملاتی که جهتشان با روند بزرگ‌تر
    هم‌جهت بوده (trend_aligned_with_direction > 0) به‌طور محسوسی وین‌ریت
    بالاتری از معاملاتی دارند که خلاف روند بزرگ‌تر بوده‌اند
    (trend_aligned_with_direction < 0) - همین سه فیچر که از قبل در پروژه
    محاسبه و ذخیره می‌شوند (src/ml_features.py:_add_direction_interaction_features):

        trend_aligned_with_direction        sign(direction) * trend_regime
        streak_aligned_with_direction       sign(direction) * candle_color_streak
        candle_color_aligned_with_direction sign(direction) * (کندل جاری صعودی؟)

    sign = +1 برای CALL و -1 برای PUT. یعنی مثبت = هم‌جهت با روند/استریک/رنگ
    کندل جاری، منفی = خلاف آن‌ها.

این اسکریپت مستقیماً از همان دیتابیس SQLite که main.py/collect_data.py
می‌سازند (config.SQLITE_DB_PATH) می‌خواند - نیازی به هیچ دادهٔ اضافی نیست و
هیچ تغییری هم در دیتابیس نمی‌دهد (فقط خواندنی).

اجرا:
    python analyze_level_strategy.py
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

import config
from src.ml_features import flatten_snapshot_for_model

MIN_STREAK_LEN = 2  # حداقل طول زنجیره برای این‌که «پشت‌سرهم» حساب شود
N_RANDOMNESS_SIMULATIONS = 2000  # تعداد شبیه‌سازی برای آزمایش تصادفی‌بودن زنجیره‌ها

# فیچرهای پیوستهٔ دیگر (غیر از هم‌جهتی با روند) که ممکن است بین زنجیره‌های
# برد/باخت فرق داشته باشند - از قبل در پروژه محاسبه و ذخیره می‌شوند.
CONTINUOUS_CANDIDATE_COLUMNS = [
    ("volatility_ratio_short_long", "نسبت نوسان کوتاه‌مدت به بلندمدت (انبساط/انقباض)"),
    ("wick_asymmetry_current", "عدم‌تقارن سایهٔ کندل قبلی"),
]

INTERACTION_COLUMNS = [
    "trend_aligned_with_direction",
    "streak_aligned_with_direction",
    "candle_color_aligned_with_direction",
    "level_strategy_region_aligned_with_direction",
    "level_strategy_near_high_signal_aligned_with_direction",
    "level_strategy_near_low_signal_aligned_with_direction",
]


def _load_level_strategy_trades() -> pd.DataFrame:
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM trades", conn)
    finally:
        conn.close()

    df = df[df["meta_trade_source"] == "level_strategy"].copy()
    df = df.sort_values("meta_entry_timestamp").reset_index(drop=True)
    return df


def _add_interaction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """برای هر ردیف، همان فیچرهای هم‌جهت‌بودن را با flatten_snapshot_for_model
    (دقیقاً همان مسیری که train_model.py استفاده می‌کند) محاسبه می‌کند. اگر
    ردیف‌های قدیمی‌تر ستون خام لازم (مثلاً level_strategy_region) را نداشته
    باشند، flatten_snapshot_for_model خودش با مقدار خنثی (۰) پر می‌کند."""
    extracted = {c: [] for c in INTERACTION_COLUMNS}
    for row in df.to_dict("records"):
        flat = flatten_snapshot_for_model(row)
        for c in INTERACTION_COLUMNS:
            extracted[c].append(flat.get(c))
    for c in INTERACTION_COLUMNS:
        df[c] = extracted[c]
    return df


def _mark_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """به هر ردیف streak_id (شمارهٔ زنجیرهٔ نتایج یکسان متوالی) و
    streak_len (طول کل همان زنجیره) اضافه می‌کند."""
    streak_id = (df["result"] != df["result"].shift()).cumsum()
    df["streak_id"] = streak_id
    df["streak_len"] = df.groupby("streak_id")["streak_id"].transform("size")
    return df


def _report_split(df: pd.DataFrame, column: str, label: str) -> None:
    print(f"\n--- وین‌ریت بر اساس {label} ({column}) ---")
    aligned = df[df[column] > 0]
    opposed = df[df[column] < 0]
    neutral = df[df[column] == 0]
    for name, subset in (("هم‌جهت (>0)", aligned), ("خلاف‌جهت (<0)", opposed), ("خنثی (=0)", neutral)):
        if len(subset) == 0:
            print(f"  {name}: هیچ معامله‌ای نبود")
            continue
        winrate = subset["result"].mean()
        print(f"  {name}: {len(subset)} معامله، وین‌ریت {winrate:.1%}")


def _max_run_length(values: np.ndarray, target: bool) -> int:
    max_run = 0
    current = 0
    for v in values:
        if v == target:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _report_streak_randomness(df: pd.DataFrame, n_sims: int = N_RANDOMNESS_SIMULATIONS) -> None:
    """
    قبل از دنبال‌کردن هر فیچری برای توضیح زنجیره‌های برد/باخت، اول باید دید
    آیا اصلاً این زنجیره‌ها از چیزی «غیرعادی» خبر می‌دهند، یا فقط نوسان طبیعیِ
    یک فرآیند تصادفی با همین وین‌ریت هستند - چون با وین‌ریت ۵۹٪ روی ۲۰۰۰+
    معامله، زنجیره‌های ۵-۶تایی برد یا باخت کاملاً طبیعی و منتظره‌اند، حتی اگر
    هیچ الگوی واقعی‌ای در کار نباشد (توهم الگو/Clustering Illusion).
    """
    results = df["result"].to_numpy().astype(bool)
    n = len(results)
    p = results.mean()
    observed_max_loss = _max_run_length(results, False)
    observed_max_win = _max_run_length(results, True)

    rng = np.random.default_rng(42)
    sim_max_loss = np.empty(n_sims, dtype=int)
    sim_max_win = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        sim = rng.random(n) < p
        sim_max_loss[i] = _max_run_length(sim, False)
        sim_max_win[i] = _max_run_length(sim, True)

    loss_percentile = (sim_max_loss < observed_max_loss).mean()
    win_percentile = (sim_max_win < observed_max_win).mean()

    print(f"\n--- آزمایش تصادفی‌بودن زنجیره‌ها (شبیه‌سازی {n_sims} بار با وین‌ریت {p:.1%} روی {n} معامله) ---")
    print(f"طولانی‌ترین زنجیرهٔ باخت واقعی: {observed_max_loss}  |  در شبیه‌سازی‌های تصادفی معمولاً "
          f"{'کوتاه‌تر یا برابر' if loss_percentile < 0.95 else 'کوتاه‌تر'} از این بود "
          f"(صدک {loss_percentile:.0%}) -> "
          f"{'در محدودهٔ طبیعی نوسان تصادفی است' if loss_percentile < 0.95 else 'غیرعادی است؛ احتمالاً یک الگوی واقعی وجود دارد'}")
    print(f"طولانی‌ترین زنجیرهٔ برد واقعی: {observed_max_win}  |  صدک {win_percentile:.0%} -> "
          f"{'در محدودهٔ طبیعی نوسان تصادفی است' if win_percentile < 0.95 else 'غیرعادی است؛ احتمالاً یک الگوی واقعی وجود دارد'}")


def _report_by_symbol(df: pd.DataFrame) -> None:
    print("\n--- وین‌ریت به تفکیک نماد (meta_symbol) ---")
    if "meta_symbol" not in df.columns:
        print("  ستون meta_symbol موجود نیست.")
        return
    grouped = df.groupby("meta_symbol")["result"].agg(["mean", "count"]).sort_values("count", ascending=False)
    for symbol, row in grouped.iterrows():
        print(f"  {symbol}: {int(row['count'])} معامله، وین‌ریت {row['mean']:.1%}")


def _report_continuous_split(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df.columns or df[column].isna().all():
        print(f"\n--- {label} ({column}): داده‌ای موجود نیست ---")
        return
    valid = df[df[column].notna()]
    median = valid[column].median()
    above = valid[valid[column] > median]
    below = valid[valid[column] <= median]
    print(f"\n--- وین‌ریت بر اساس {label} ({column})، میانه={median:.3f} ---")
    print(f"  بالاتر از میانه: {len(above)} معامله، وین‌ریت {above['result'].mean():.1%}")
    print(f"  پایین یا مساوی میانه: {len(below)} معامله، وین‌ریت {below['result'].mean():.1%}")


def main() -> None:
    df = _load_level_strategy_trades()
    if df.empty:
        print("هیچ معاملهٔ level_strategy در دیتابیس پیدا نشد.")
        return

    print(f"تعداد کل معاملات level_strategy: {len(df)}")
    print(f"وین‌ریت کلی: {df['result'].mean():.1%}")

    df = _add_interaction_columns(df)
    df = _mark_streaks(df)

    # --- اول: آیا اصلاً این زنجیره‌ها غیرعادی‌اند، یا نوسان طبیعیِ یک فرآیند
    # تصادفی با همین وین‌ریت‌اند؟ اگر غیرعادی نبودند، دنبال «فیچر توضیح‌دهنده»
    # گشتن ممکن است اصلاً بی‌فایده باشد (توهم الگو).
    _report_streak_randomness(df)

    ready_mask = df["trend_regime_ready"] == 1 if "trend_regime_ready" in df.columns else pd.Series(True, index=df.index)
    print(f"\nتعداد معاملاتی که هنگام ورود، روند (trend_regime) قابل‌تشخیص بوده: "
          f"{int(ready_mask.sum())} از {len(df)} ({ready_mask.mean():.1%})")

    # --- فرضیهٔ اول: وین‌ریت هم‌جهت با روند در برابر خلاف روند ---
    _report_split(df[ready_mask], "trend_aligned_with_direction", "هم‌جهتی با روند بزرگ‌تر (trend_regime)")
    _report_split(df, "streak_aligned_with_direction", "هم‌جهتی با استریک رنگ کندل‌ها")
    _report_split(df, "candle_color_aligned_with_direction", "هم‌جهتی با رنگ کندل جاری")

    # --- فرضیهٔ دوم: خودِ «نظر» لحظه‌ای level_strategy (منطقهٔ اکسترمم‌ها نسبت
    # به open) - این ستون‌ها فقط برای معاملاتی که بعد از اضافه‌شدن این فیچر
    # جمع‌آوری شده باشند مقدار واقعی دارند؛ برای دادهٔ قدیمی‌تر خنثی (۰) است.
    _report_split(df, "level_strategy_region_aligned_with_direction", "هم‌جهتی با منطقهٔ اکسترمم‌های کندل جاری")
    _report_split(df, "level_strategy_near_high_signal_aligned_with_direction", "هم‌جهتی با سیگنال نزدیک سقف")
    _report_split(df, "level_strategy_near_low_signal_aligned_with_direction", "هم‌جهتی با سیگنال نزدیک کف")

    # --- فرضیه‌های دیگر: نماد و نوسان/عدم‌تقارن سایه ---
    _report_by_symbol(df)
    for col, label in CONTINUOUS_CANDIDATE_COLUMNS:
        _report_continuous_split(df, col, label)

    # --- مقایسهٔ زنجیره‌های برد پشت‌سرهم در برابر باخت پشت‌سرهم ---
    streaks = df[df["streak_len"] >= MIN_STREAK_LEN]
    win_streak_trades = streaks[streaks["result"] == 1]
    loss_streak_trades = streaks[streaks["result"] == 0]
    print(f"\n--- مقایسهٔ معاملات داخل زنجیره‌های برد/باخت پشت‌سرهم (طول >= {MIN_STREAK_LEN}) ---")
    print(f"تعداد معامله داخل زنجیرهٔ برد: {len(win_streak_trades)}")
    print(f"تعداد معامله داخل زنجیرهٔ باخت: {len(loss_streak_trades)}")
    compare_cols = INTERACTION_COLUMNS + [c for c, _ in CONTINUOUS_CANDIDATE_COLUMNS if c in df.columns]
    for col in compare_cols:
        w_mean = win_streak_trades[col].mean()
        l_mean = loss_streak_trades[col].mean()
        print(f"  {col}: میانگین در زنجیرهٔ برد = {w_mean:+.3f}  |  میانگین در زنجیرهٔ باخت = {l_mean:+.3f}")

    print("\nراهنمای خواندن نتیجه: در هر بخش بالا، اگر وین‌ریت دو گروه به‌وضوح از هم "
          "فاصله داشت (مثلاً بیش از ۱۰-۱۵ واحد درصد)، یا اگر بخش «تصادفی‌بودن زنجیره‌ها» "
          "غیرعادی بودن را نشان داد، همان فیچر مسیر خوبی برای بهبود استراتژی است. اگر همهٔ "
          "این تفاوت‌ها ناچیز بودند و زنجیره‌ها هم در محدودهٔ طبیعی تصادف بودند، یعنی این "
          "زنجیره‌های برد/باخت به‌احتمال زیاد فقط نوسان طبیعیِ یک استراتژی با وین‌ریت ثابت‌اند، "
          "نه نشانهٔ یک موقعیت قابل‌تشخیص - و باید فیچرهای دیگری (یا حتی زمان/سشن معاملاتی) را "
          "بررسی کرد.")


if __name__ == "__main__":
    main()
