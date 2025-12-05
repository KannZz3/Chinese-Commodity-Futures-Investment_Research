from typing import Tuple
import pandas as pd

__all__ = ["add_labels", "make_train_dataset"]


def add_labels(
    data: pd.DataFrame,
    H: int,
    TH: float,
    future_col: str = "future_ret_H",
    up_col: str = "y_up",
    down_col: str = "y_down",
) -> pd.DataFrame:
    """
    根据未来 H 根 K 线的 close-to-close 收益构造 up / down 标签。

    参数
    ----
    data : pd.DataFrame
        index 按时间升序，至少包含列 'close'
    H : int
        预测窗口长度，例如 H=3 表示用 t→t+3 收益打标签
    TH : float
        阈值，|future_ret| >= TH 视为大涨/大跌
    future_col : str
        存储未来收益的列名（默认 'future_ret_H'）
    up_col : str
        大涨标签列名（默认 'y_up'）
    down_col : str
        大跌标签列名（默认 'y_down'）

    返回
    ----
    df : pd.DataFrame
        在原数据基础上增加 future_ret_H / y_up / y_down 的 DataFrame
    """
    df = data.copy()

    # 1) future close-to-close return
    df[future_col] = df["close"].shift(-H) / df["close"] - 1.0

    # 2) up / down 标签
    df[up_col] = (df[future_col] >= TH).astype(int)
    df[down_col] = (df[future_col] <= -TH).astype(int)

    return df


def make_train_dataset(
    data: pd.DataFrame,
    future_col: str = "future_ret_H",
) -> pd.DataFrame:
    """
    删除未来不可见样本（未来收益为 NaN 的行）。

    参数
    ----
    data : pd.DataFrame
        已经包含 future_col 的数据
    future_col : str
        未来收益列名（默认 'future_ret_H'）

    返回
    ----
    data_train : pd.DataFrame
        过滤掉尾部无法打标签样本后的训练集
    """
    mask = data[future_col].notna()
    return data.loc[mask].copy()
