import numpy as np
import pandas as pd

__all__ = ["backtest_money"]


# ============================================================
# 1) Money-based backtest / 资金曲线回测（保证金 + 杠杆）
# ============================================================
def backtest_money(
    df,
    score_col="score",
    T=None,                 # Must be provided externally / 必须由外部传入（训练集确定）
    base_th=0.01,           # choose_base_th_from_train / 基础阈值（训练集选择）
    max_hold_days=10,       # Maximum holding period / 最大持仓天数
    init_capital=500000,    # Initial capital / 初始资金
    pos_pct=0.3,            # Margin fraction per trade / 单笔保证金占比
    leverage=10,            # Directional leverage / 方向杠杆
    shift_signal=True,      # Execute signal on next bar / 信号后移一根K（避免同bar）

    # ====== Weekly filter parameters / 周过滤参数（只影响开仓 gating） ======
    use_weekly_filter=True,
    wk_trend_col="wk_trend_sign",
    wk_vol_regime_col="wk_vol_regime",
    wk_crowded_long_col="wk_crowded_long",
    block_vol_regime=2,
    allow_trend_zero_open=False,

    # ====== Performance statistics parameters / 统计参数（Sharpe 等） ======
    rf_annual=0.012,        # Annual risk-free rate / 年化无风险
    annual_trading_days=252 # Trading days for annualization / 年化交易日
):
    """
    Money-based backtest (margin-style sizing).
    资金口径回测（保证金占用 + 杠杆名义头寸）。

    Notes / 说明：
    - Uses close-to-close returns: ret = close.pct_change().
      使用收盘到收盘收益。
    - Weekly filters only gate opening; no forced exit while holding.
      周过滤仅限制开仓，不强制平仓。
    """

    df = df.copy()

    # === 0) Underlying daily return / 标的日收益 ===
    df["ret"] = df["close"].pct_change().fillna(0.0)

    if T is None:
        raise ValueError(
            "[cta_backtest_money] Parameter T is not specified. "
            "Please first compute T on the training set using choose_T_from_train (or similar), "
            "and then pass it into this function."
        )

    # === 1) Raw directional signal / 原始方向信号 ===
    raw_score = df[score_col]

    sig = pd.Series(0.0, index=df.index, dtype=float)
    sig[raw_score >  T] =  1.0
    sig[raw_score < -T] = -1.0

    if shift_signal:
        # Signal takes effect on next bar / 信号下一根K生效
        sig = sig.shift(1).fillna(0.0)

    df["signal_raw"] = sig

    # === 1.5) Weekly filters (open gating only) / 周过滤（只影响开仓） ===
    weekly_ok = pd.Series(True, index=df.index, dtype=bool)

    if use_weekly_filter:
        # Trend filter / 趋势过滤：方向需与周趋势一致
        if wk_trend_col in df.columns:
            trend = df[wk_trend_col]

            if not allow_trend_zero_open:
                weekly_ok &= (trend != 0).fillna(False)

            same_dir_or_zero = ((trend * sig) >= 0) | (sig == 0)
            weekly_ok &= same_dir_or_zero.fillna(False)

        # Vol regime filter / 波动状态过滤：高波动屏蔽开仓
        if wk_vol_regime_col in df.columns:
            vol_reg = df[wk_vol_regime_col]
            high_vol_mask = (vol_reg == block_vol_regime)
            weekly_ok &= ~(high_vol_mask & (sig != 0))

        # Crowding filter / 拥挤度过滤：拥挤时屏蔽做多开仓
        if wk_crowded_long_col in df.columns:
            crowded = df[wk_crowded_long_col].fillna(0.0)
            block_long = (crowded >= 0.5) & (sig > 0)
            weekly_ok &= ~block_long

    sig_eff = sig.copy()
    sig_eff[~weekly_ok] = 0.0
    df["signal_eff"] = sig_eff
    df["weekly_ok"] = weekly_ok.astype(float)

    # === 2) Initialize state / 初始化状态 ===
    capital = init_capital
    pos_dir = 0.0
    hold_value = 0.0
    cum_ret = 0.0
    hold_days = 0
    reached_tp1 = False

    capital_list = []
    pos_exposure_list = []
    hold_value_list = []

    trade_returns = []   # per-trade cum_ret / 逐笔收益（保证金口径）

    # === 3) Main loop / 主循环 ===
    for i in range(len(df)):
        daily_ret_raw = df["ret"].iloc[i]
        signal_today = df["signal_eff"].iloc[i]

        # A) Open when flat / 空仓开仓
        if pos_dir == 0.0 and signal_today != 0.0:
            pos_dir = signal_today
            hold_value = capital * pos_pct
            cum_ret = 0.0
            hold_days = 0
            reached_tp1 = False

        # B) Daily P&L / 每日盈亏（真实资金）
        effective_notional = pos_dir * leverage * hold_value
        daily_profit = effective_notional * daily_ret_raw
        capital += daily_profit

        # C) Update trade stats if holding / 持仓则更新逐笔统计
        if pos_dir != 0.0:
            cum_ret += pos_dir * leverage * daily_ret_raw
            hold_days += 1

            if cum_ret >= base_th:
                reached_tp1 = True

            # D) Exit rules / 平仓规则
            exit_flag = False
            if cum_ret >= 2 * base_th:                      # take-profit / 止盈
                exit_flag = True
            elif cum_ret <= -0.9 * base_th:                 # stop-loss / 止损
                exit_flag = True
            elif reached_tp1 and cum_ret <= 0.65 * base_th:  # pullback TP / 回撤止盈
                exit_flag = True
            elif hold_days >= max_hold_days:                # time stop / 时间止损
                exit_flag = True

            if exit_flag:
                trade_returns.append(cum_ret)

                pos_dir = 0.0
                hold_value = 0.0
                cum_ret = 0.0
                hold_days = 0
                reached_tp1 = False

        # E) Record daily state / 记录每日状态
        capital_list.append(capital)
        pos_exposure_list.append(pos_dir * leverage)
        hold_value_list.append(hold_value)

    # === 4) Write back curves / 回写曲线 ===
    df["capital"] = capital_list
    df["pos"] = pos_exposure_list
    df["hold_value"] = hold_value_list
    df["equity"] = df["capital"] / float(init_capital)

    # === 5) Stats (attach to attrs) / 统计（写入 attrs） ===
    equity_series = pd.Series(capital_list, index=df.index)
    port_ret = equity_series.pct_change().fillna(0.0)

    rf_daily = (1.0 + rf_annual) ** (1.0 / annual_trading_days) - 1.0
    excess_ret = port_ret - rf_daily

    if excess_ret.std() > 0:
        sharpe_annual = (excess_ret.mean() / excess_ret.std()) * np.sqrt(annual_trading_days)
    else:
        sharpe_annual = np.nan

    trade_returns = np.asarray(trade_returns, dtype=float)
    n_trades = int(trade_returns.size)
    n_win = int((trade_returns > 0).sum())
    n_loss = int((trade_returns < 0).sum())
    win_rate = float(n_win / n_trades) if n_trades > 0 else np.nan

    if n_loss > 0:
        worst_losses = np.sort(trade_returns[trade_returns < 0])
        worst5_loss_ret = worst_losses[:5].tolist()
    else:
        worst5_loss_ret = []

    bt_stats = {
        "n_trades": n_trades,
        "n_win": n_win,
        "n_loss": n_loss,
        "win_rate": win_rate,
        "worst5_loss_ret": worst5_loss_ret,
        "sharpe_annual": sharpe_annual,
        "rf_annual": rf_annual,
        "annual_trading_days": annual_trading_days,
    }

    df.attrs["bt_stats"] = bt_stats

    return df
