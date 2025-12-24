import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import kurtosis

__all__ = ["check_missing_values", "fill_missing_values", "FeaturePreprocessor"]


# ============================
# 1. 缺失值检验
# ============================
def check_missing_values(data: pd.DataFrame, valid_by_finite: bool = False) -> pd.DataFrame:
    """
    遍历所有列（因子），统计 NaN 和 inf（+inf/-inf）的分布，并把“缺失/异常”拆成：
      1) 总 NaN / 总 Inf
      2) 有效区间内 NaN / Inf
      3) 有效区间外（leading + trailing）NaN / Inf

    有效区间的定义：
      - valid_by_finite=False（默认）：沿用原逻辑，用“非 NaN”的首/尾作为有效区间边界。
        也就是说：inf 不会影响有效区间边界，只影响区间内/外的 Inf 计数。
      - valid_by_finite=True：用“finite（既非 NaN 也非 inf）”的首/尾作为有效区间边界。
        这对你目前存在 inf 的场景更严格、更合理。

    返回：
      每个因子一行的统计表 DataFrame。
    """
    missing_summary = [] 

    for col in data.columns:
        s = data[col] 

        # ========= 1) 构建三种 mask（布尔数组） =========
        is_nan = s.isna() 
        is_inf = np.isinf(s.to_numpy(dtype=float, copy=False)) 
        is_nonfinite = is_nan.to_numpy() | is_inf

        # ========= 2) 统计全列的 NaN / Inf 数量 =========
        total_nan = int(is_nan.sum())
        total_inf = int(is_inf.sum())

        # ========= 3) 决定“有效区间”的首尾边界 =========
        if valid_by_finite:
            # finite_mask=True 表示该位置是有效数（既不是 NaN，也不是 inf）
            finite_mask = ~is_nonfinite

            if finite_mask.any():
                finite_arr = finite_mask.to_numpy()

                # np.argmax 找到第一个 True 的位置（第一个有效数）
                first_pos = int(np.argmax(finite_arr))
                # np.where 找到所有 True 的位置，取最后一个就是最后一个有效数
                last_pos = int(np.where(finite_arr)[0][-1])

                first_valid_index = s.index[first_pos]
                last_valid_index = s.index[last_pos]
            else:
                # 全列都 non-finite（全 NaN/inf）：没有有效区间
                first_valid_index = None
                last_valid_index = None

        else:
            # 默认逻辑：只按“非 NaN”的首尾决定有效区间（inf 会被当成有效值）
            first_valid_index = s.first_valid_index()
            last_valid_index = s.last_valid_index()

        # ========= 4) 统计有效区间内/外的 NaN / Inf =========
        if (first_valid_index is not None) and (last_valid_index is not None):
            # 取有效区间（包含首尾）
            in_range = s.loc[first_valid_index:last_valid_index]

            # 区间内 NaN 数
            in_nan = int(in_range.isna().sum())
            # 区间内 Inf 数
            in_inf = int(np.isinf(in_range.to_numpy(dtype=float, copy=False)).sum())

            # 区间外（leading + trailing）= 总量 - 区间内
            out_nan = total_nan - in_nan
            out_inf = total_inf - in_inf
        else:
            # 没有有效区间：区间内都算 0，区间外就是总量
            in_nan = in_inf = 0
            out_nan, out_inf = total_nan, total_inf

        # ========= 5) 汇总本列统计结果 =========
        missing_summary.append(
            {
                "Factor": col,

                # 全列统计
                "Total NaN": total_nan,
                "Total Inf": total_inf,

                # 有效区间内统计
                "NaN in Valid Range": in_nan,
                "Inf in Valid Range": in_inf,

                # 有效区间外统计（leading + trailing）
                "NaN outside Valid Range": out_nan,
                "Inf outside Valid Range": out_inf,

                # 有效区间边界（方便你定位“哪天开始有效/哪天结束有效”）
                "First Valid Index": first_valid_index,
                "Last Valid Index": last_valid_index,
            }
        )

    # list[dict] -> DataFrame
    missing_summary_df = pd.DataFrame(missing_summary)

    # ========= 6) 设置显示选项（保证输出表不会被截断） =========
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    return missing_summary_df


# ===============================
# 2. 缺失值处理
# ===============================
def fill_missing_values(data: pd.DataFrame):
    data = data.copy()

    # 获取每个因子的有效区间
    def get_valid_range(col):
        first_valid_index = data[col].first_valid_index()
        last_valid_index = data[col].last_valid_index()
        if pd.notna(first_valid_index) and pd.notna(last_valid_index):
            return data.loc[first_valid_index:last_valid_index, col]
        else:
            return data[col]  # 如果没有有效范围，返回整个列

    # ========== 1. 动量 / slope / ret类 ==========
    for c in data.columns:
        if c.startswith("mom_") or "slope" in c or c == "ret_1":
            valid_range = get_valid_range(c)
            if valid_range.size > 0:  # 只在有效范围内填补
                print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
                data[c] = data[c].fillna(method='ffill')  # 前值填补
                print(f"After filling {c}, NaN count: {data[c].isna().sum()}")
               
    # ========== 2. MA 均线 ==========
    for c in data.columns:
        if c.startswith("ma_") and "slope" not in c:
            valid_range = get_valid_range(c)
            if valid_range.size > 0:  # 只在有效范围内填补
                print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
                data[c] = data[c].fillna(data["close"])
                print(f"After filling {c}, NaN count: {data[c].isna().sum()}")

    # ========== 4. 检验后, 特殊处理部分 ==========
    # 处理 `mom_2_slope_3` 和 `mom_4_slope_3` 的缺失值，使用前值填补
    for c in ["mom_2_slope_3", "mom_4_slope_3"]:
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill")  
            print(f"After filling {c}, NaN count: {data[c].isna().sum()}")

    # 处理比率因子的缺失值：使用前值填补
    for c in ["body_pct", "upper_shadow_pct", "lower_shadow_pct"]:
        valid_range = get_valid_range(c)
        if valid_range.size > 0:
            print(f"Before filling {c}, NaN count: {data[c].isna().sum()}")
            data[c] = data[c].fillna(method="ffill") 
            print(f"After filling {c}, NaN count: {data[c].isna().sum()}")
    first_pos = []
    for c in data.columns:
        m = data[c].notna().to_numpy()
        if m.any():                      # 该列至少有一个有效值
            first_pos.append(int(m.argmax()))  # 首个有效值位置 = leading NaN 行数
    max_leading = max(first_pos) if first_pos else 0

    return data.iloc[max_leading:].copy()


# =========================
# 3. 因子预处理
# =========================

