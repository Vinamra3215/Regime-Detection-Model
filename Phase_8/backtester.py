
import numpy as np
import pandas as pd
import logging

from config import (
    INITIAL_CAPITAL, EVAL_START, EVAL_END,
    MAX_POSITIONS, STOP_LOSS_PCT, MIN_HOLDING_DAYS,
    ZERODHA_STT_PCT, ZERODHA_EXCHANGE_TXN_PCT,
    ZERODHA_GST_PCT, ZERODHA_SEBI_PCT,
    ZERODHA_STAMP_DUTY_PCT, SLIPPAGE_PCT,
    PHASE7_SIGNALS, PHASE1_LABEL_DIR,
    NIFTY_50_TICKERS,
)

log = logging.getLogger(__name__)


def compute_transaction_cost(trade_value, is_buy=True):
    brokerage = 0

    stt = trade_value * ZERODHA_STT_PCT if not is_buy else 0

    exchange_txn = trade_value * ZERODHA_EXCHANGE_TXN_PCT

    gst = (brokerage + exchange_txn) * ZERODHA_GST_PCT

    sebi = trade_value * ZERODHA_SEBI_PCT

    stamp = trade_value * ZERODHA_STAMP_DUTY_PCT if is_buy else 0

    slippage = trade_value * SLIPPAGE_PCT

    total = brokerage + stt + exchange_txn + gst + sebi + stamp + slippage
    return total


class BacktestPosition:
    def __init__(self, ticker, entry_price, shares, entry_date, signal_strength):
        self.ticker = ticker
        self.entry_price = entry_price
        self.shares = shares
        self.entry_date = entry_date
        self.signal_strength = signal_strength
        self.holding_days = 0
        self.peak_price = entry_price
        self.entry_cost = compute_transaction_cost(entry_price * shares, is_buy=True)

    def update(self, current_price):
        self.holding_days += 1
        self.peak_price = max(self.peak_price, current_price)

    def unrealized_pnl(self, current_price):
        return (current_price - self.entry_price) * self.shares

    def unrealized_return(self, current_price):
        return (current_price - self.entry_price) / self.entry_price

    def should_stop_loss(self, current_price):
        ret = self.unrealized_return(current_price)
        return ret < -STOP_LOSS_PCT and self.holding_days >= MIN_HOLDING_DAYS

    def can_exit(self):
        return self.holding_days >= MIN_HOLDING_DAYS

    def close(self, exit_price):
        exit_cost = compute_transaction_cost(exit_price * self.shares, is_buy=False)
        gross_pnl = (exit_price - self.entry_price) * self.shares
        net_pnl = gross_pnl - self.entry_cost - exit_cost
        total_costs = self.entry_cost + exit_cost
        return net_pnl, total_costs


def run_backtest(daily_signals_df):
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)

    ticker_prices = {}
    for ticker in NIFTY_50_TICKERS:
        path = PHASE1_LABEL_DIR / f"{ticker}_labelled.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            if "Close" in df.columns:
                ticker_prices[ticker] = df[["Close"]].loc[eval_start:eval_end]
            elif "log_return_1d" in df.columns:
                df_eval = df.loc[eval_start:eval_end]
                if len(df_eval) > 0:
                    closes = [100.0]
                    for ret in df_eval["log_return_1d"].values[1:]:
                        closes.append(closes[-1] * np.exp(ret))
                    df_eval = df_eval.copy()
                    df_eval["Close"] = closes[:len(df_eval)]
                    ticker_prices[ticker] = df_eval[["Close"]]

    all_dates = sorted(daily_signals_df["date"].unique())
    all_dates = [d for d in all_dates if eval_start <= pd.Timestamp(d) <= eval_end]

    log.info(f"Backtesting {len(all_dates)} trading days, {len(ticker_prices)} tickers with prices")

    capital = INITIAL_CAPITAL
    positions = {}
    daily_records = []
    trade_log = []
    total_costs = 0.0

    for date in all_dates:
        date_ts = pd.Timestamp(date)
        day_signals = daily_signals_df[daily_signals_df["date"] == date]

        positions_value = 0.0
        day_costs = 0.0
        day_realized_pnl = 0.0
        stops = 0

        tickers_to_close = []
        for ticker, pos in positions.items():
            if ticker in ticker_prices and date_ts in ticker_prices[ticker].index:
                price = ticker_prices[ticker].loc[date_ts, "Close"]
                pos.update(price)
                positions_value += price * pos.shares

                if pos.should_stop_loss(price):
                    tickers_to_close.append((ticker, "STOP_LOSS", price))
                    stops += 1

        for ticker, pos in positions.items():
            if ticker in [t[0] for t in tickers_to_close]:
                continue
            ticker_signal = day_signals[day_signals["ticker"] == ticker]
            if len(ticker_signal) > 0:
                signal = ticker_signal.iloc[0]["signal"]
                if signal == "FLAT" and pos.can_exit():
                    if ticker in ticker_prices and date_ts in ticker_prices[ticker].index:
                        price = ticker_prices[ticker].loc[date_ts, "Close"]
                        tickers_to_close.append((ticker, "SIGNAL_CHANGE", price))

        for ticker, reason, price in tickers_to_close:
            if ticker in positions:
                pos = positions[ticker]
                pnl, costs = pos.close(price)
                day_realized_pnl += pnl
                day_costs += costs
                capital += price * pos.shares + pnl
                trade_log.append({
                    "ticker": ticker,
                    "entry_date": str(pos.entry_date),
                    "exit_date": str(date),
                    "entry_price": pos.entry_price,
                    "exit_price": price,
                    "shares": pos.shares,
                    "gross_pnl": (price - pos.entry_price) * pos.shares,
                    "transaction_costs": costs + pos.entry_cost,
                    "net_pnl": pnl,
                    "return_pct": pos.unrealized_return(price) * 100,
                    "holding_days": pos.holding_days,
                    "exit_reason": reason,
                })
                del positions[ticker]

        if len(positions) < MAX_POSITIONS:
            available_capital = capital - sum(
                p.entry_price * p.shares for p in positions.values()
            )
            slots = MAX_POSITIONS - len(positions)

            long_signals = day_signals[
                day_signals["signal"].str.contains("LONG", na=False)
            ].copy()
            long_signals = long_signals[
                ~long_signals["ticker"].isin(positions.keys())
            ]
            long_signals = long_signals.sort_values("position_size", ascending=False)

            for _, row in long_signals.head(slots).iterrows():
                ticker = row["ticker"]
                if ticker not in ticker_prices:
                    continue
                if date_ts not in ticker_prices[ticker].index:
                    continue

                price = ticker_prices[ticker].loc[date_ts, "Close"]
                per_stock = available_capital / max(slots, 1)
                shares = int(per_stock / price)

                if shares > 0 and per_stock > 0:
                    cost = compute_transaction_cost(price * shares, is_buy=True)
                    capital -= price * shares + cost
                    day_costs += cost
                    positions[ticker] = BacktestPosition(
                        ticker, price, shares, date,
                        row.get("position_size", 0.5)
                    )

        total_costs += day_costs

        pos_val = sum(
            ticker_prices[t].loc[date_ts, "Close"] * p.shares
            for t, p in positions.items()
            if t in ticker_prices and date_ts in ticker_prices[t].index
        )
        portfolio_value = capital + pos_val

        daily_records.append({
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": capital,
            "positions_value": pos_val,
            "n_positions": len(positions),
            "day_realized_pnl": day_realized_pnl,
            "day_costs": day_costs,
            "cumulative_costs": total_costs,
            "stops_triggered": stops,
        })

    for ticker, pos in list(positions.items()):
        last_date = pd.Timestamp(all_dates[-1])
        if ticker in ticker_prices and last_date in ticker_prices[ticker].index:
            price = ticker_prices[ticker].loc[last_date, "Close"]
            pnl, costs = pos.close(price)
            trade_log.append({
                "ticker": ticker,
                "entry_date": str(pos.entry_date),
                "exit_date": str(all_dates[-1]),
                "entry_price": pos.entry_price,
                "exit_price": price,
                "shares": pos.shares,
                "gross_pnl": (price - pos.entry_price) * pos.shares,
                "transaction_costs": costs + pos.entry_cost,
                "net_pnl": pnl,
                "return_pct": pos.unrealized_return(price) * 100,
                "holding_days": pos.holding_days,
                "exit_reason": "END_OF_PERIOD",
            })

    portfolio_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)

    log.info(f"Backtest complete: {len(trade_df)} trades, Rs {total_costs:,.0f} total costs")
    return portfolio_df, trade_df, total_costs


def compute_buy_and_hold(daily_signals_df):
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = pd.Timestamp(EVAL_END)

    per_stock = INITIAL_CAPITAL / len(NIFTY_50_TICKERS)
    total_value = INITIAL_CAPITAL
    all_dates = sorted(daily_signals_df["date"].unique())
    all_dates = [d for d in all_dates if eval_start <= pd.Timestamp(d) <= eval_end]

    records = []
    for date in all_dates:
        day_data = daily_signals_df[daily_signals_df["date"] == date]
        day_return = day_data["actual_return"].mean() if len(day_data) > 0 else 0
        total_value *= (1 + day_return)
        records.append({"date": date, "buyhold_value": total_value})

    return pd.DataFrame(records)

