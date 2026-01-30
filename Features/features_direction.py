# Features/features_direction.py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress

__all__ = ["infer_factor_sign", "build_sign_dict"]

# =========================================================
# Usage / 调用示例
# ---------------------------------------------------------
# from Features.features_direction import infer_factor_sign, build_sign_dict
#
# # 1) Single factor / 单因子方向
# sign, detail = infer_factor_sign(df, factor="mom_5", future_col="future_ret_H")
#
# # 2) Batch build sign dict / 批量生成方向字典
# sign_dict, detail_dict = build_sign_dict(
#     factor_list=feature_cols,
#     train_df=train_df,
#     future_col="future_ret_H",
#     n_group=5,
#     bootstrap_n=300,
#     min_confidence=0.7,
# )
# =========================================================


def infer_factor_sign(
    df: pd.DataFrame,
    factor: str,
    future_col: str = "future_ret_H",
    n_group: int = 5,
    min_samples: int = 300,
    bootstrap_n: int = 200,
    min_confidence: float = 0.7,
):
    """
    Infer factor direction / 因子方向推断

    Goal / 目标
    ----
    Infer whether a factor should be used as + (long), - (short), or dropped (0),
    based on its relationship with future returns.
    推断因子与未来收益的方向：+1（正向）、-1（反向）、0（不稳定/丢弃）。

    Components / 组成模块
    ----
    1) Quantile bucketing (qcut): top vs bottom return spread
       分位分组：最高分组 vs 最低分组收益差
    2) IC sign: Pearson correlation direction
       IC 方向：皮尔森相关方向
    3) Monotonicity: Spearman on grouped means
       单调性：对分组均值做 Spearman
    3b) Regression slope + t-stat (constraint)
       回归斜率与 t 值：作为约束/一致性增强
    4) Block bootstrap stability on qcut sign
       分块 bootstrap：方向稳定性（概率）
    5) Weighted ensemble + confidence filter
       加权集成 + 置信度过滤

    Returns / 返回
    ----
    final_sign : int
        +1 / -1 / 0
    detail : dict
        Intermediate diagnostics / 中间诊断信息
    """

    # ======================
    # 0) Prepare data / 数据准备
    # ======================
    sub = df[[factor, future_col]].dropna().copy()
    if len(sub) < min_samples:
        return 0, {"error": "too_few_samples", "n": int(len(sub)), "min_samples": int(min_samples)}

    x = sub[factor].to_numpy()
    y = sub[future_col].to_numpy()

    # ======================
    # 1) Qcut bucketing / 分位分组方向
    # ======================
    try:
        sub["group"] = pd.qcut(x, n_group, labels=False, duplicates="drop")
        g = sub.groupby("group")[future_col].mean()
        # Direction: top bucket mean - bottom bucket mean
        # 方向：最高分组均值 - 最低分组均值
        qcut_sign = np.sign(g.iloc[-1] - g.iloc[0])
    except Exception:
        qcut_sign = 0
        g = None

    # ======================
    # 2) IC sign / IC 方向（Pearson）
    # ======================
    IC = np.corrcoef(x, y)[0, 1]
    if np.isnan(IC):
        ic_sign = 0
    else:
        ic_sign = np.sign(IC)

    # ======================
    # 3) Monotonicity / 单调性（Spearman）
    #    Using group means from qcut buckets
    # ======================
    try:
        if g is None:
            mono_sign = 0
            spearman_r = np.nan
        else:
            group_returns = g.values
            group_index = np.arange(len(group_returns))
            spearman_r, _ = spearmanr(group_index, group_returns)
            mono_sign = np.sign(spearman_r) if not np.isnan(spearman_r) else 0
    except Exception:
        mono_sign = 0
        spearman_r = np.nan

    # ========================================
    # 3b) Regression slope + t-stat / 回归斜率与 t 值
    #     (Constraint + mild reinforcement)
    # ========================================
    reg_sign = 0
    reg_tstat = np.nan
    reg_slope = np.nan

    try:
        # Centering does not change slope sign, but improves numerical stability.
        # 去均值不改变斜率符号，但更稳健。
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        slope, intercept, r_value, p_value, std_err = linregress(x_centered, y_centered)
        reg_slope = slope

        if (std_err is None) or (std_err == 0) or np.isnan(std_err) or np.isnan(slope):
            reg_sign = 0
            reg_tstat = np.nan
        else:
            reg_sign = int(np.sign(slope))
            reg_tstat = slope / std_err
    except Exception:
        reg_sign = 0
        reg_tstat = np.nan
        reg_slope = np.nan

    # ======================
    # 4) Bootstrap stability / Bootstrap 稳定性（分块）
    # ======================
    signs = []
    n = len(sub)

    # Block bootstrap: each block ≈ 5% of sample, at least 5 obs
    # 分块 bootstrap：每块约为样本 5%，最少 5 条
    block_size = max(5, n // 20)
    n_blocks = int(np.ceil(n / block_size))

    for _ in range(bootstrap_n):
        idx_list = []
        for _ in range(n_blocks):
            start_max = n - block_size
            if start_max < 0:
                start_max = 0
            start = np.random.randint(0, start_max + 1)
            idx_block = range(start, min(start + block_size, n))
            idx_list.extend(idx_block)

        idx = np.array(idx_list[:n])
        xb = x[idx]
        yb = y[idx]

        # Bootstrap Qcut sign (same as main)
        # bootstrap 分位分组方向（与主逻辑一致）
        try:
            dfb = pd.DataFrame({"f": xb, "r": yb}).dropna()
            dfb["g"] = pd.qcut(dfb["f"], n_group, labels=False, duplicates="drop")
            gb = dfb.groupby("g")["r"].mean()
            bsign = np.sign(gb.iloc[-1] - gb.iloc[0])
        except Exception:
            bsign = 0

        signs.append(bsign)

    signs = np.array(signs)
    pos_prob = np.mean(signs == 1)
    neg_prob = np.mean(signs == -1)

    # Bootstrap direction / Bootstrap 方向
    if pos_prob > neg_prob:
        bootstrap_sign = 1
    elif neg_prob > pos_prob:
        bootstrap_sign = -1
    else:
        bootstrap_sign = 0

    bootstrap_confidence = max(pos_prob, neg_prob)

    # ======================
    # 5) Final decision / 最终方向（加权集成 + 约束）
    # ======================
    # Weights / 权重（保持你的设定）
    w_qcut = 0.35
    w_ic = 0.35
    w_mono = 0.2
    w_boot = 0.1

    raw_score = (
        w_qcut * qcut_sign
        + w_ic * ic_sign
        + w_mono * mono_sign
        + w_boot * bootstrap_sign
    )

    # Regression constraint / 回归约束（保持你的设定）
    reg_t_min = 2.0   # |t| >= 2: significant / 显著阈值
    w_reg = 0.15      # mild boost / 小幅增强权重

    if not np.isnan(reg_tstat) and abs(reg_tstat) >= reg_t_min and reg_sign != 0:
        raw_sign_before_reg = np.sign(raw_score)

        if raw_sign_before_reg != 0:
            if raw_sign_before_reg != reg_sign:
                # Significant regression but opposite direction -> veto
                # 回归显著但方向相反 -> 直接否决
                raw_score = 0.0
            else:
                # Consistent -> mild reinforcement
                # 一致 -> 小幅增强
                raw_score = raw_score + w_reg * reg_sign
        # If raw_score == 0: keep it 0, regression alone doesn't decide
        # 若 raw_score==0：保持 0，不让回归单独决定方向

    final_sign = int(np.sign(raw_score))

    # Bootstrap confidence filter / Bootstrap 置信度过滤
    if bootstrap_confidence < min_confidence:
        final_sign = 0

    # ======================
    # Pack results / 输出诊断信息
    # ======================
    detail = {
        "qcut_sign": int(qcut_sign),
        "ic": IC,
        "ic_sign": int(ic_sign),
        "spearman": spearman_r,
        "mono_sign": int(mono_sign),
        "bootstrap_sign": int(bootstrap_sign),
        "bootstrap_confidence": bootstrap_confidence,
        "final_sign_rawscore": raw_score,
        "final_sign": int(final_sign),

        # Regression diagnostics / 回归诊断
        "reg_slope": float(reg_slope) if not np.isnan(reg_slope) else np.nan,
        "reg_tstat": float(reg_tstat) if not np.isnan(reg_tstat) else np.nan,
        "reg_sign": int(reg_sign),

        # Block bootstrap config / 分块 bootstrap 配置
        "bootstrap_block_size": int(block_size),
        "bootstrap_n": int(bootstrap_n),
    }

    return final_sign, detail


def build_sign_dict(
    factor_list,
    train_df: pd.DataFrame,
    future_col: str = "future_ret_H",
    n_group: int = 5,
    bootstrap_n: int = 300,
    min_confidence: float = 0.7,
):
    """
    Build factor -> sign mapping / 批量生成因子方向字典

    This runs `infer_factor_sign` for each factor on the training set, and returns:
    对训练集中的每个因子运行 `infer_factor_sign`，输出：

    Returns / 返回
    ----
    sign_dict : dict
        {factor_name: +1 / -1 / 0}
    detail_dict : dict
        {factor_name: full diagnostics returned by infer_factor_sign}
        {factor_name: infer_factor_sign 的完整诊断信息}
    """

    # ======================
    # Loop over factors / 遍历因子
    # ======================
    sign_dict = {}
    detail_dict = {}

    for f in factor_list:
        sign, detail = infer_factor_sign(
            train_df,
            factor=f,
            future_col=future_col,
            n_group=n_group,
            bootstrap_n=bootstrap_n,
            min_confidence=min_confidence,
        )

        # Ensure stable int type / 统一成 int（+1/-1/0）
        sign = int(sign)
        sign_dict[f] = sign
        detail_dict[f] = detail

    return sign_dict, detail_dict


