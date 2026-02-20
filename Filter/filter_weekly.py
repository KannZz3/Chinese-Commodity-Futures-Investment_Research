import numpy as np
import pandas as pd

__all__ = [
    "build_weekly_filters",
    "locate_test_start_week",
    "attach_weekly_filters_to_daily",
]

# ============================================================
# 1) Build weekly filters / 构建周线过滤因子
# ============================================================

def build_weekly_filters(
    weekly_df: pd.DataFrame,
    price_col: str = "close",
    hold_col: str = "hold",
    vol_window: int = 4,
    trend_fast: int = 13,
    trend_slow: int = 26,
    crowd_lookback: int = 4,
    crowd_hold_thresh: float = 0.2,
    crowd_price_thresh: float = 0.05,
    use_prev_week: bool = True
) -> pd.DataFrame:
    """
    Construct weekly filter factors on the *full* weekly_df.
    在完整 weekly_df 上构造周频过滤因子。

    Parameters / 参数
    ----------------
    weekly_df : pd.DataFrame
        Weekly data with a DatetimeIndex. Must contain price_col and hold_col.
        周频数据，索引需为 DatetimeIndex，必须包含 close/hold
    price_col : str
        Weekly close price column name / 周收盘价列名
    hold_col : str
        Weekly open interest column name / 周持仓量(OI)列名
    vol_window : int
        Rolling window for weekly return volatility / 周收益波动窗口（周数）
    trend_fast, trend_slow : int
        Weekly MA windows for trend / 周均线窗口（快/慢）
    crowd_lookback : int
        Lookback (weeks) for crowding metrics / 拥挤度回看周数
    crowd_hold_thresh, crowd_price_thresh : float
        Thresholds for "crowded long" / 多头拥挤阈值（持仓&价格累计涨幅）
    use_prev_week : bool
        If True, output at week t stores stats from t-1, so bfill mapping onto daily
        uses last week's info (no look-ahead).
        若 True：周 t 行存放 t-1 的统计量，用于下一周日频

    Returns / 返回
    -------------
    weekly_factors : pd.DataFrame
        Index aligned with weekly_df, columns include:
            wk_ret, wk_vol, wk_vol_regime,
            wk_ma_fast, wk_ma_slow, wk_trend_raw, wk_trend_sign,
            wk_hold_chg, wk_hold_chg_L, wk_price_chg_L,
            wk_crowded_long
        与 weekly_df 同索引的周因子表（便于后续 join）。
    """
    # ---- basic checks / 基础校验 ----
    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        raise ValueError("build_weekly_filters: weekly_df.index must be a DatetimeIndex.")
    if price_col not in weekly_df.columns:
        raise ValueError(f"build_weekly_filters: weekly_df is missing price column '{price_col}'.")
    if hold_col not in weekly_df.columns:
        raise ValueError(f"build_weekly_filters: weekly_df is missing hold column '{hold_col}'.")

    w = weekly_df.sort_index().copy()

    # ========== 1) Weekly return & weekly volatility (RAW, as-of week t) ==========
    # 周收益 & 周收益波动（先算当周 t 的 raw 值）
    w_ret_raw = w[price_col].pct_change()
    _minp_vol = max(1, vol_window // 2)  # safety / 防止 min_periods=0
    w_vol_raw = w_ret_raw.rolling(vol_window, min_periods=_minp_vol).std()

    # NOTE: keep your original global quantile thresholds (no change)
    # 注意：此处仍采用全样本分位数阈值（保持你现有逻辑不变）
    q_low = w_vol_raw.quantile(0.3)
    q_high = w_vol_raw.quantile(0.7)

    def classify_vol(v: float):
        """0=low vol, 1=mid, 2=high / 波动分档：低/中/高"""
        if np.isnan(v):
            return np.nan
        if v <= q_low:
            return 0
        elif v >= q_high:
            return 2
        else:
            return 1

    w_vol_regime_raw = w_vol_raw.apply(classify_vol)

    # ========== 2) Weekly trend via MA_fast / MA_slow (RAW, as-of week t) ==========
    # 周趋势（快慢均线差）
    _minp_fast = max(1, trend_fast // 2)
    _minp_slow = max(1, trend_slow // 2)
    ma_fast_raw = w[price_col].rolling(trend_fast, min_periods=_minp_fast).mean()
    ma_slow_raw = w[price_col].rolling(trend_slow, min_periods=_minp_slow).mean()
    trend_raw_raw = ma_fast_raw - ma_slow_raw

    trend_sign_raw = np.sign(trend_raw_raw)
    eps = 1e-8
    trend_sign_raw = trend_sign_raw.where(trend_raw_raw.abs() > eps, 0.0)  # deadzone / 小幅差值归零

    # ========== 3) Position crowding metrics (RAW, as-of week t) ==========
    # 拥挤度：持仓累计上升 + 价格累计上升 => crowded_long
    hold = w[hold_col].astype(float)
    hold_chg_raw = hold.diff()  # 单周绝对变化 / 1-week absolute change
    hold_chg_L_raw = hold - hold.shift(crowd_lookback)  # L周绝对变化 / L-week abs change

    price = w[price_col].astype(float)
    price_chg_L_raw = price / price.shift(crowd_lookback) - 1.0  # L周累计涨跌幅 / L-week return

    hold_base = hold.shift(crowd_lookback)
    with np.errstate(divide="ignore", invalid="ignore"):
        hold_pct_chg_L_raw = np.where(
            hold_base > 0,
            hold_chg_L_raw / hold_base,
            np.nan
        )

    crowded_long_raw = (
        (hold_pct_chg_L_raw > crowd_hold_thresh) &
        (price_chg_L_raw > crowd_price_thresh)
    ).astype(float)

    # ========== 4) Convert to "as-of previous week" template==========
    # 若 use_prev_week=True：在周索引 t 处存 t-1 的信息，供下一周日频使用（bfill 对齐）
    if use_prev_week:
        w_ret = w_ret_raw.shift(1)
        w_vol = w_vol_raw.shift(1)
        w_vol_regime = w_vol_regime_raw.shift(1)

        ma_fast = ma_fast_raw.shift(1)
        ma_slow = ma_slow_raw.shift(1)
        trend_raw = trend_raw_raw.shift(1)
        trend_sign = trend_sign_raw.shift(1)

        hold_chg = hold_chg_raw.shift(1)
        hold_chg_L = hold_chg_L_raw.shift(1)
        price_chg_L = price_chg_L_raw.shift(1)
        crowded_long = crowded_long_raw.shift(1)
    else:
        w_ret = w_ret_raw
        w_vol = w_vol_raw
        w_vol_regime = w_vol_regime_raw

        ma_fast = ma_fast_raw
        ma_slow = ma_slow_raw
        trend_raw = trend_raw_raw
        trend_sign = trend_sign_raw

        hold_chg = hold_chg_raw
        hold_chg_L = hold_chg_L_raw
        price_chg_L = price_chg_L_raw
        crowded_long = crowded_long_raw

    # ========== 5) Assemble factor DataFrame ==========
    weekly_factors = pd.DataFrame(index=w.index)
    weekly_factors["wk_ret"] = w_ret
    weekly_factors["wk_vol"] = w_vol
    weekly_factors["wk_vol_regime"] = w_vol_regime
    weekly_factors["wk_ma_fast"] = ma_fast
    weekly_factors["wk_ma_slow"] = ma_slow
    weekly_factors["wk_trend_raw"] = trend_raw
    weekly_factors["wk_trend_sign"] = trend_sign
    weekly_factors["wk_hold_chg"] = hold_chg
    weekly_factors["wk_hold_chg_L"] = hold_chg_L
    weekly_factors["wk_price_chg_L"] = price_chg_L
    weekly_factors["wk_crowded_long"] = crowded_long

    return weekly_factors


# ============================================================
# 2) Locate test start week / 定位测试集起始周
# ============================================================

def locate_test_start_week(
    test_df: pd.DataFrame,
    weekly_df: pd.DataFrame
):
    """
    Locate the corresponding weekly bar for the first day in the test daily sample.
    找到测试集首个交易日对应的周线 bar

    Returns / 返回
    -------------
    first_test_date : pd.Timestamp
        Earliest date in test_df / 测试集最早日期
    start_week_date : pd.Timestamp
        First weekly date in weekly_df that is >= first_test_date
        weekly_df 中第一个 >= first_test_date 的周索引日期
    """
    if not isinstance(test_df.index, pd.DatetimeIndex):
        raise ValueError("locate_test_start_week: test_df.index must be a DatetimeIndex.")
    if not isinstance(weekly_df.index, pd.DatetimeIndex):
        raise ValueError("locate_test_start_week: weekly_df.index must be a DatetimeIndex.")

    first_test_date = test_df.index.min()

    # Find the weekly bar that "contains" this date (e.g., W-FRI index) / 找到包含该日的周bar
    weekly_after = weekly_df[weekly_df.index >= first_test_date]
    if weekly_after.empty:
        raise ValueError(
            f"locate_test_start_week: weekly_df has no weekly data with index >= first_test_date ({first_test_date})."
        )

    start_week_date = weekly_after.index[0]
    return first_test_date, start_week_date


# ============================================================
# 3) Attach weekly filters to daily / 周因子映射到日频
# ============================================================

def attach_weekly_filters_to_daily(daily_df: pd.DataFrame, weekly_filters: pd.DataFrame) -> pd.DataFrame:
    """
    Attach weekly filter signals to a daily DataFrame.
    将周频过滤因子对齐并合并到日频数据上

    Parameters / 参数
    ----------------
    daily_df : pd.DataFrame
        Index = daily trading dates / 日频交易日索引
    weekly_filters : pd.DataFrame
        Index = weekly dates (e.g. W-FRI), columns = wk_* factors.
        周索引（例如每周五），列为 wk_* 因子

    Logic / 对齐逻辑
    ----------------
    Reindex weekly filters onto daily index with method="bfill":
        For a daily date d, pick the first weekly date W such that W >= d,
        and use row at W.

    Under use_prev_week=True:
        The row at weekly date W stores previous week's info (t-1),
        so all days in week W use last week's statistics (no look-ahead).

    Returns / 返回
    -------------
    merged : pd.DataFrame
        daily_df joined with wk_* columns / 合并后的日频数据（含 wk_* 列）
    """
    wf_daily = weekly_filters.reindex(daily_df.index, method="bfill")
    merged = daily_df.join(wf_daily)
    return merged
