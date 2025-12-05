# Features/features_construction.py
# ============================================
#  纯价格行为因子构建
# ============================================

import numpy as np
import pandas as pd

__all__ = ["build_daily_features"]


def build_daily_features(
    df: pd.DataFrame,
    mom_windows=(2, 4, 8, 16),
    vol_windows=(4, 8, 16),
    ma_windows=(4, 8, 16, 32),
    vol_ma_windows=(5, 10, 20),
    hold_ma_windows=(5, 10, 20),
) -> pd.DataFrame:
    """
    基于日频 OHLCV + hold 构建 CTA 因子。

    参数
    ----
    df : pd.DataFrame
        index 为 datetime，columns 至少包含:
        [open, high, low, close, volume, hold]

    返回
    ----
    data : pd.DataFrame
        在 df 基础上增加所有因子的 DataFrame
    """
    data = df.copy()

    # =========================
    # 3. 基础特征：单根收益 & K 线形态
    # =========================
    data["ret_1"] = data["close"].pct_change()

    # K 线形态相关：实体、范围、上下影线
    data["body"] = data["close"] - data["open"]
    data["range"] = data["high"] - data["low"]
    data["upper_shadow"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]

    # 占比（归一化）
    data["body_pct"] = data["body"] / data["range"].replace(0, np.nan)
    data["upper_shadow_pct"] = data["upper_shadow"] / data["range"].replace(0, np.nan)
    data["lower_shadow_pct"] = data["lower_shadow"] / data["range"].replace(0, np.nan)

    data["is_bull"] = (data["close"] > data["open"]).astype(int)
    data["is_bear"] = (data["close"] < data["open"]).astype(int)

    # =========================
    # 4. 动量 & 波动率 & 均线因子
    # =========================

    # -------- 1. 动量（Momentum）--------
    for n in mom_windows:
        col = f"mom_{n}"
        data[col] = data["close"].pct_change(n)

        # slope：动量变化速度（百分比变化，更强）
        data[f"{col}_slope"] = data[col].pct_change()

        # 多周期 slope（趋势加速度 proxy）
        data[f"{col}_slope_3"] = data[col].pct_change(3)
        data[f"{col}_slope_5"] = data[col].pct_change(5)

        # 二阶导数（趋势加速度）
        data[f"{col}_acc"] = data[col].diff().diff()

    # -------- 2. 波动率（Volatility）--------
    for n in vol_windows:
        col = f"vol_{n}"
        data[col] = data["ret_1"].rolling(n).std()

        # slope：波动率变化 => 反转/趋势加速的重要信号
        data[f"{col}_slope"] = data[col].pct_change()

        # 多周期 slope
        data[f"{col}_slope_3"] = data[col].pct_change(3)
        data[f"{col}_slope_5"] = data[col].pct_change(5)

        # 二阶导数：波动率加速度（趋势强度衡量）
        data[f"{col}_acc"] = data[col].diff().diff()

    # -------- 3. 均线（MA）--------
    for n in ma_windows:
        ma_col = f"ma_{n}"
        data[ma_col] = data["close"].rolling(n).mean()
        data[f"close_ma_{n}"] = data["close"] / data[ma_col] - 1

        # slope：均线的百分比变化（比 diff 更稳定）
        data[f"{ma_col}_slope"] = data[ma_col].pct_change()

        # 多周期 slope（趋势强度）
        data[f"{ma_col}_slope_3"] = data[ma_col].pct_change(3)
        data[f"{ma_col}_slope_5"] = data[ma_col].pct_change(5)

        # 二阶导数（趋势加速度）
        data[f"{ma_col}_acc"] = data[ma_col].diff().diff()

        # 角度 slope（强 signal）
        data[f"ma_{n}_angle"] = np.arctan(
            data[ma_col].diff() / (data[ma_col].shift(1).abs() + 1e-9)
        )

    # =========================
    # 5. 成交量 & 持仓量因子
    # =========================

    # 1. Volume returns
    data["vol_ret_1"] = data["volume"].pct_change()
    data["vol_ret_5"] = data["volume"].pct_change(5)
    data["vol_ret_10"] = data["volume"].pct_change(10)

    # 2. Volume MA & bias
    for n in vol_ma_windows:
        ma_col = f"vol_ma_{n}"
        data[ma_col] = data["volume"].rolling(n).mean()
        data[f"vol_ma_bias_{n}"] = data["volume"] / data[ma_col] - 1

    # 3. Volume volatility
    for n in vol_ma_windows:
        data[f"vol_vol_{n}"] = data["volume"].pct_change().rolling(n).std()

    # 4. Volume-Price divergence
    data["vol_price_div_1"] = data["volume"].pct_change() - data["close"].pct_change()
    data["vol_price_div_5"] = data["volume"].pct_change(5) - data["close"].pct_change(5)

    # =========================
    # Hold (Open Interest) Features
    # =========================

    # 1. Hold returns
    data["hold_ret_1"] = data["hold"].pct_change()
    data["hold_ret_5"] = data["hold"].pct_change(5)
    data["hold_ret_10"] = data["hold"].pct_change(10)

    # 2. Hold MA & bias
    for n in hold_ma_windows:
        ma_col = f"hold_ma_{n}"
        data[ma_col] = data["hold"].rolling(n).mean()
        data[f"hold_ma_bias_{n}"] = data["hold"] / data[ma_col] - 1

    # 3. Hold volatility
    for n in hold_ma_windows:
        data[f"hold_vol_{n}"] = data["hold"].pct_change().rolling(n).std()

    # 4. Price–OI divergence
    data["price_oi_div_1"] = data["close"].pct_change() - data["hold"].pct_change()
    data["price_oi_div_5"] = data["close"].pct_change(5) - data["hold"].pct_change(5)

    # 5. Volume / Hold ratio
    data["volume_hold_ratio"] = data["volume"] / (data["hold"] + 1e-6)

    # 6. Volume–Price correlation
    data["vp_ratio"] = (
        data["volume"].pct_change().rolling(5).corr(data["close"].pct_change())
    )

    # =========================
    # 6. 量价仓组合因子
    # =========================
    data["price_vol_confirm"] = np.sign(data["ret_1"]) * data["vol_ret_1"]
    data["price_oi_trend"] = np.sign(data["ret_1"]) * data["hold_ret_1"]
    data["absorption_ratio"] = data["volume"] / (data["body"].abs() + 1e-3)
    data["effort_result_body_abs"] = data["volume"] * data["body"].abs()
    data["pv_corr_5"] = data["ret_1"].rolling(5).corr(data["vol_ret_1"])
    data["po_corr_5"] = data["ret_1"].rolling(5).corr(data["hold_ret_1"])

    return data
