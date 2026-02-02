# Score/Score_fun.py

import numpy as np
import pandas as pd

__all__ = ["add_score"]

# ============================
# Factor scoring
# 因子得分计算
# ============================

def add_score(
    df,
    up_factors,
    down_factors,
    up_sign_dict,
    down_sign_dict,
    lookback=50,
    winsor_limit=5,
    use_robust=True,
    remove_corr=True,
    corr_threshold=0.7,
    factor_group_map=None,      # ⚠ 当前版本中不再使用，所有因子等权 / unused: equal-weight all factors
    conflict_check=True,
    conflict_threshold=0.1      # 单因子“强贡献”阈值（按贡献值）/ strong threshold on weighted contribution
):
    """
    新版 add_score（无因子分组版本） / add_score (no factor-group, equal-weight)

    输入 / Inputs
    -----------
    up_factors      : 多头侧因子名列表（UP 模型 TopN） / UP-side factor list (TopN)
    down_factors    : 空头侧因子名列表（DOWN 模型 TopN） / DOWN-side factor list (TopN)
    up_sign_dict    : 因子在 UP 方向上的 sign（+1/-1/0） / {factor -> sign} for UP
    down_sign_dict  : 因子在 DOWN 方向上的 sign（+1/-1/0） / {factor -> sign} for DOWN

    输出 / Outputs (added columns)
    ----------------------------
    score_long  : 多头得分（3日平滑） / long score (3-day smoothed)
    score_short : 空头得分（3日平滑） / short score (3-day smoothed)
    score       : score_long - score_short / net score

    Notes / 说明
    ------------
    conflict_threshold 的比较对象是“单因子加权贡献”：
        contrib_all_w = (Z * sign) * (1/n_factors)
    因此阈值尺度应与 n_factors 相关（否则可能过大导致永不触发）。
    """

    df = df.copy()

    # ------------------------------------------------------------
    # helper: early-return with NaN scores
    # 辅助：统一 NaN 早退出口
    # ------------------------------------------------------------
    def _return_nan_scores(_df):
        _df["score"] = np.nan
        _df["score_long"] = np.nan
        _df["score_short"] = np.nan
        return _df

    # ============================================================
    # 0) Clean factor lists & sign-conflict filtering
    # 0) 清洗列表 & 方向冲突过滤
    # ============================================================
    # Deduplicate while keeping order
    # 去重且保序
    up_factors = list(dict.fromkeys(up_factors))
    down_factors = list(dict.fromkeys(down_factors))

    # Drop factors that appear in BOTH lists but with OPPOSITE non-zero signs.
    # 若同一因子同时出现在 up/down 且方向相反（且均非0），则两边同时剔除。
    overlap = set(up_factors) & set(down_factors)
    conflict_drop = set()

    for f in overlap:
        su = np.sign(up_sign_dict.get(f, 0.0))
        sd = np.sign(down_sign_dict.get(f, 0.0))
        if su != 0 and sd != 0 and su != sd:
            print(f"[警告] 因子 {f} 在 up_sign_dict / down_sign_dict 中方向相反，将从多头/空头列表中移除。")
            conflict_drop.add(f)

    if conflict_drop:
        up_factors = [f for f in up_factors if f not in conflict_drop]
        down_factors = [f for f in down_factors if f not in conflict_drop]

    # Union factor list (order preserved)
    # 合并后的总因子列表（保序；允许同因子同时出现在 up/down，只要方向一致）
    all_factors = list(dict.fromkeys(up_factors + down_factors))

    if len(all_factors) == 0:
        return _return_nan_scores(df)

    # ============================================================
    # 1) Optional: remove collinear factors (greedy correlation filter)
    # 1) 可选：剔除共线因子（贪心相关性过滤，顺序敏感）
    # ============================================================
    if remove_corr and len(all_factors) > 1:
        existing = [f for f in all_factors if f in df.columns]
        if len(existing) > 1:
            # Absolute correlation matrix
            # 绝对相关系数矩阵
            corr = df[existing].corr().abs()

            # Greedy keep: keep first, then keep f only if corr(f, kept) < threshold for all kept
            # 贪心保留：先留第一个；之后仅当与已留集合内所有因子相关 < 阈值时才保留
            keep = []
            for f in existing:
                if not keep:
                    keep.append(f)
                    continue
                if all(corr[f][k] < corr_threshold for k in keep):
                    keep.append(f)

            kept = set(keep)
            all_factors = [f for f in all_factors if f in kept]
            up_factors = [f for f in up_factors if f in kept]
            down_factors = [f for f in down_factors if f in kept]

    if len(all_factors) == 0:
        return _return_nan_scores(df)

    # ============================================================
    # 2) Rolling Z-score per factor (robust optional) + winsorize
    # 2) 逐因子 rolling Z-score（可选 robust）+ winsorize 截断
    # ============================================================
    valid_factors = []

    for f in all_factors:
        if f not in df.columns:
            print(f"[跳过因子] {f}: 不在 df 列中")
            continue

        series = df[f].astype(float)

        if use_robust:
            # Robust Z: (x - median) / (MAD * 1.4826)
            # Robust Z：用 rolling median + rolling MAD
            rolling_med = series.rolling(lookback).median()
            rolling_mad = (series - rolling_med).abs().rolling(lookback).median()
            z = (series - rolling_med) / (rolling_mad * 1.4826 + 1e-6)
        else:
            # Standard Z: (x - mean) / std
            # 标准 Z：用 rolling mean + rolling std
            rolling_mean = series.rolling(lookback).mean()
            rolling_std = series.rolling(lookback).std()
            z = (series - rolling_mean) / (rolling_std + 1e-6)

        # Skip invalid z (all NaN or constant)
        # 若 z 全 NaN 或无波动（常数）则跳过
        if z.isna().all() or z.std(skipna=True) == 0:
            print(f"[跳过因子] {f}: Z 值无效（可能全 NaN 或常数）")
            continue

        # Winsorize to control extremes
        # winsor 截断：限制极端值影响
        z = z.clip(-winsor_limit, winsor_limit)

        # Store z column for later matrix build
        # 存储 z 列，供后续构造 Z 矩阵
        df[f"_z_{f}"] = z.astype(float)
        valid_factors.append(f)

    if len(valid_factors) == 0:
        return _return_nan_scores(df)

    # Keep only factors with valid z
    # 只保留 z 有效因子中的 up/down
    up_valid = [f for f in up_factors if f in valid_factors]
    down_valid = [f for f in down_factors if f in valid_factors]

    # Union of valid factors (order preserved)
    # 有效因子并集（保序）
    valid_union = list(dict.fromkeys(up_valid + down_valid))
    if len(valid_union) == 0:
        return _return_nan_scores(df)

    # ============================================================
    # 3) Build Z matrix + equal weights + signs
    # 3) 构造 Z 矩阵 + 等权权重 + 三套方向 sign
    # ============================================================
    # 3.1 Z matrix (columns = factor names)
    # 3.1 Z 矩阵（列名为因子名）
    z_cols = [f"_z_{f}" for f in valid_union]
    Z = df[z_cols].copy()
    Z.columns = valid_union

    # 3.2 Equal weight across ALL valid_union factors
    # 3.2 valid_union 内所有因子等权
    n_factors = len(valid_union)
    if n_factors == 0:
        return _return_nan_scores(df)

    equal_weight = 1.0 / n_factors
    w_series = pd.Series({f: equal_weight for f in valid_union}, dtype=float)

    # 3.3 Signs
    # 3.3 三套 sign：
    #   (1) up_signs   : for long score only / 仅用于多头得分
    #   (2) down_signs : for short score only / 仅用于空头得分
    #   (3) union_signs: for conflict check   / 用于冲突检测（合并方向）
    up_signs = pd.Series(
        {f: float(np.sign(up_sign_dict.get(f, 0.0))) for f in up_valid},
        dtype=float
    )
    down_signs = pd.Series(
        {f: float(np.sign(down_sign_dict.get(f, 0.0))) for f in down_valid},
        dtype=float
    )

    union_signs_dict = {}
    for f in valid_union:
        su = np.sign(up_sign_dict.get(f, 0.0))
        sd = np.sign(down_sign_dict.get(f, 0.0))

        if f in up_valid and f not in down_valid:
            s = su
        elif f in down_valid and f not in up_valid:
            s = sd
        else:
            # Appears in both; opposite sign already removed in step 0
            # 同时出现在 up/down；相反方向已在第0步剔除
            s = su if su != 0 else sd

        union_signs_dict[f] = float(s)

    union_signs = pd.Series(union_signs_dict, dtype=float)

    # ============================================================
    # 4) Compute raw long / short scores (both >= 0)
    # 4) 计算多头/空头原始得分（均为非负）
    # ============================================================

    # 4.1 Long score: only positive contributions from UP side
    # 4.1 多头得分：只累计 UP 侧“正贡献”
    if len(up_valid) > 0:
        Z_up = Z[up_valid]
        # contrib_up: align Z to UP direction (positive = bullish under UP definition)
        # contrib_up：将 Z 按 UP 的方向对齐（正=偏多，负=偏空）
        contrib_up = Z_up.mul(up_signs, axis=1)

        # Apply equal weights
        # 乘以等权权重
        contrib_up_w = contrib_up.mul(w_series[up_valid], axis=1)

        # Keep only bullish part (>=0)
        # 只取“牛市贡献”（>=0）
        bull_up = contrib_up_w.clip(lower=0.0)
        score_long_raw = bull_up.sum(axis=1)
    else:
        score_long_raw = pd.Series(0.0, index=df.index)

    # 4.2 Short score: only bearish contributions from DOWN side (as positive magnitude)
    # 4.2 空头得分：只累计 DOWN 侧“负贡献”的绝对值
    if len(down_valid) > 0:
        Z_down = Z[down_valid]
        # contrib_down: align Z to DOWN direction (positive = bullish under DOWN definition)
        # contrib_down：将 Z 按 DOWN 的方向对齐（正=偏多，负=偏空）
        contrib_down = Z_down.mul(down_signs, axis=1)

        # Apply equal weights
        # 乘以等权权重
        contrib_down_w = contrib_down.mul(w_series[down_valid], axis=1)

        # Keep only bearish part: (-contrib) clipped at 0
        # 只取“熊市贡献”：(-贡献) 且裁剪到 >=0
        bear_down = (-contrib_down_w).clip(lower=0.0)
        score_short_raw = bear_down.sum(axis=1)
    else:
        score_short_raw = pd.Series(0.0, index=df.index)

    # Net signal (not directly saved; kept for readability)
    # 净信号（不直接保存，仅用于逻辑可读性）
    raw_score = score_long_raw - score_short_raw

    # ============================================================
    # 5) Conflict suppression using union_signs (optional)
    # 5) 冲突抑制：用合并方向 union_signs（可选）
    # ============================================================
    if conflict_check:
        # 5.1 Total contribution matrix on merged direction
        # 5.1 在合并方向下的“总贡献矩阵”
        # contrib_all: Z aligned to union direction (positive=pro-bull, negative=pro-bear)
        # contrib_all：Z 按 union 方向对齐（正=偏多，负=偏空）
        contrib_all = Z.mul(union_signs, axis=1)

        # Apply equal weights -> weighted contribution per factor
        # 乘以等权 -> 单因子“加权贡献”
        contrib_all_w = contrib_all.mul(w_series, axis=1)

        # 5.2 Count strong positive / strong negative factors per timestamp
        # 5.2 每个时点统计“强多/强空”的因子数量
        # strong_pos: count of factors with contrib > +threshold
        # strong_neg: count of factors with contrib < -threshold
        strong_pos = (contrib_all_w >  conflict_threshold).sum(axis=1)
        strong_neg = (contrib_all_w < -conflict_threshold).sum(axis=1)
        total_strong = strong_pos + strong_neg

        # 5.3 Determine dominant side by raw long vs raw short
        # 5.3 用原始 long/short 判断当日主导方向
        bull_mask = score_long_raw > score_short_raw
        bear_mask = score_short_raw > score_long_raw
        tie_mask  = score_long_raw == score_short_raw

        # 5.4 Minority limit rule: allow at most 1/3 strong signals on the opposite side
        # 5.4 少数派规则：反方向强信号超过强信号总数的 1/3 则判冲突
        minority_limit = total_strong / 3.0

        # 5.5 Conflict conditions
        # 5.5 冲突判定条件
        # - Bull-dominant day but too many strong negatives -> conflict
        # - Bear-dominant day but too many strong positives -> conflict
        # - Tie day but any strong signals exist -> conflict (strong bull/bear offset)
        bull_conflict = bull_mask & (strong_neg > minority_limit)
        bear_conflict = bear_mask & (strong_pos > minority_limit)
        tie_conflict  = tie_mask  & (total_strong > 0)

        conflict_mask = bull_conflict | bear_conflict | tie_conflict

        # 5.6 Neutralize on conflict timestamps
        # 5.6 冲突时点中性化：分数置零
        raw_score[conflict_mask]       = 0.0
        score_long_raw[conflict_mask]  = 0.0
        score_short_raw[conflict_mask] = 0.0

    # ============================================================
    # 6) Light smoothing (CTA-style): 3-day rolling mean
    # 6) 轻度平滑（CTA常用）：3日均线
    # ============================================================
    df["score_long"]  = score_long_raw.rolling(3).mean()
    df["score_short"] = score_short_raw.rolling(3).mean()
    df["score"]       = df["score_long"] - df["score_short"]

    return df

