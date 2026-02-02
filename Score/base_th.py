import numpy as np
import pandas as pd

__all__ = ["choose_base_th_from_train"]

# ============================================
# 1) Choose a reasonable base_th
# 1) 选择合理的 base_th（波动自适应阈值）
# ============================================

def choose_base_th_from_train(
    train_df,
    close_col="close",
    base_th_ref=0.01,     # base when daily vol≈ref_vol / 基准阈值（当日波动≈ref_vol）
    vol_lookback=50,      # rolling vol window / 波动率滚动窗口
    ref_vol=0.01,         # reference vol / 参考波动率（默认1%）
    floor_mult=0.5,       # lower bound multiplier / 下限倍数
    ceil_mult=3.0,        # upper bound multiplier / 上限倍数
    winsor_pct=1.0,       # winsorize ret tails (%) / 日收益双尾winsor百分位
    min_vol=0.002,        # vol lower clamp / 波动率下界
    max_vol=0.1,          # vol upper clamp / 波动率上界
    verbose=True
):
    """
    Vol-adaptive base_th from training set.
    用训练集“典型日波动率”自适应选择 base_th。

    Steps / 步骤
    1) ret = pct_change(close) (optional winsorize tails)
       用 close 算日收益 ret（可选双尾裁剪）
    2) vol_median = median( rolling_std(ret, vol_lookback) ), then clamp to [min_vol, max_vol]
       取 rolling 波动率中位数，并夹逼到合理区间
    3) base_th_data = base_th_ref * (vol_median / ref_vol)
       按波动率线性缩放
    4) base_th = clip(base_th_data, floor_mult*base_th_ref, ceil_mult*base_th_ref)
       再加 floor/ceil 保护
    """

    # ============================
    # 1) Daily returns / 日收益
    # ============================
    if close_col not in train_df.columns:
        raise KeyError(f"[choose_base_th_from_train] '{close_col}' not found in train_df columns.")

    close = train_df[close_col].astype(float)
    ret = close.pct_change().dropna()

    # Not enough samples -> fallback
    # 样本不足 -> 直接返回基准值
    if len(ret) < vol_lookback:
        if verbose:
            print("[choose_base_th_from_train] Too few data points, returning base_th_ref directly.")
        return float(base_th_ref)

    # ============================
    # 1.5) Winsorize / 双尾裁剪
    # ============================
    if winsor_pct is not None and winsor_pct > 0:
        low_q, high_q = np.percentile(ret, [winsor_pct, 100 - winsor_pct])
        ret = ret.clip(low_q, high_q)

    # ============================
    # 2) Vol median / 波动率中位数
    # ============================
    rolling_vol = ret.rolling(vol_lookback).std()
    vol_median_raw = rolling_vol.median()

    if (vol_median_raw is None) or (not np.isfinite(vol_median_raw)) or (vol_median_raw <= 0):
        if verbose:
            print("[choose_base_th_from_train] vol_median invalid, returning base_th_ref directly.")
        return float(base_th_ref)

    vol_median = float(vol_median_raw)
    vol_median_before_clip = vol_median

    # Clamp vol into reasonable range
    # 波动率夹逼到合理区间
    if min_vol is not None:
        vol_median = max(vol_median, float(min_vol))
    if max_vol is not None:
        vol_median = min(vol_median, float(max_vol))

    # ============================
    # 3) Scale base_th / 按波动缩放
    # ============================
    base_th_data = base_th_ref * (vol_median / ref_vol)

    # ============================
    # 4) Floor/Ceil / 上下界保护
    # ============================
    base_th_floor = floor_mult * base_th_ref
    base_th_ceil  = ceil_mult  * base_th_ref

    base_th_clipped = max(base_th_data, base_th_floor)
    base_th_clipped = min(base_th_clipped, base_th_ceil)

    if verbose:
        print(
            "[choose_base_th_from_train] "
            f"vol_median_raw={vol_median_before_clip:.4f}, "
            f"vol_median_used={vol_median:.4f}, "
            f"base_th_data={base_th_data:.4f}, "
            f"floor={base_th_floor:.4f}, ceil={base_th_ceil:.4f} "
            f"→ base_th={base_th_clipped:.4f}"
        )

    return float(base_th_clipped)


