
import numpy as np
import pandas as pd
import logging

from config import (
    STOP_LOSS_PCT, TRAILING_STOP_PCT,
    MAX_PORTFOLIO_RISK, MIN_HOLDING_DAYS,
    INITIAL_CAPITAL, STRONG_SIGNAL_SIZE, WEAK_SIGNAL_SIZE,
)

log = logging.getLogger(__name__)

class Position:
    def __init__(self, ticker, direction, size, entry_date):
        self.ticker = ticker
        self.direction = direction
        self.size = size
        self.entry_date = entry_date
        self.holding_days = 0
        self.cumulative_return = 0.0
        self.peak_return = 0.0
        self.stopped_out = False

    def update(self, daily_return):
        self.holding_days += 1
        position_return = self.direction * self.size * daily_return
        self.cumulative_return += position_return
        self.peak_return = max(self.peak_return, self.cumulative_return)
        return position_return

    def check_stop_loss(self):
        if self.cumulative_return < -STOP_LOSS_PCT:
            self.stopped_out = True
            return True
        if self.peak_return > 0.01:
            drawdown = self.peak_return - self.cumulative_return
            if drawdown > TRAILING_STOP_PCT:
                self.stopped_out = True
                return True
        return False

    def can_exit(self):
        return self.holding_days >= MIN_HOLDING_DAYS

def simulate_portfolio(daily_signals_dict):
    all_dates = set()
    for ticker, df in daily_signals_dict.items():
        all_dates.update(df["date"].values)
    all_dates = sorted(all_dates)

    if not all_dates:
        log.warning("No dates found for simulation")
        return pd.DataFrame()

    n_stocks = len(daily_signals_dict)
    per_stock_capital = INITIAL_CAPITAL / max(n_stocks, 1)

    positions = {}
    capital = INITIAL_CAPITAL
    daily_records = []
    trade_log = []

    for date in all_dates:
        day_pnl = 0.0
        active_positions = 0
        long_exposure = 0.0
        short_exposure = 0.0
        stops_triggered = 0

        for ticker, df in daily_signals_dict.items():
            day_data = df[df["date"] == date]
            if len(day_data) == 0:
                continue
            row = day_data.iloc[0]

            actual_return = row["actual_return"]
            signal = row["signal"]
            position_size = row["position_size"]

            if ticker in positions:
                pos = positions[ticker]
                pos_return = pos.update(actual_return)
                day_pnl += pos_return * per_stock_capital

                if pos.check_stop_loss() and pos.can_exit():
                    stops_triggered += 1
                    trade_log.append({
                        "ticker": ticker,
                        "entry_date": pos.entry_date,
                        "exit_date": date,
                        "direction": "LONG" if pos.direction > 0 else "SHORT",
                        "return": pos.cumulative_return,
                        "holding_days": pos.holding_days,
                        "exit_reason": "STOP_LOSS",
                    })
                    del positions[ticker]
                    continue

                current_dir = 1 if signal.endswith("LONG") else (-1 if signal.endswith("SHORT") else 0)
                if pos.can_exit():
                    if signal == "FLAT" or current_dir != pos.direction:
                        trade_log.append({
                            "ticker": ticker,
                            "entry_date": pos.entry_date,
                            "exit_date": date,
                            "direction": "LONG" if pos.direction > 0 else "SHORT",
                            "return": pos.cumulative_return,
                            "holding_days": pos.holding_days,
                            "exit_reason": "SIGNAL_CHANGE",
                        })
                        del positions[ticker]

            if ticker not in positions and signal != "FLAT":
                direction = 1 if signal.endswith("LONG") else -1
                size = STRONG_SIGNAL_SIZE if signal.startswith("STRONG") else WEAK_SIGNAL_SIZE

                positions[ticker] = Position(ticker, direction, size, date)

            if ticker in positions:
                active_positions += 1
                pos = positions[ticker]
                if pos.direction > 0:
                    long_exposure += pos.size
                else:
                    short_exposure += pos.size

        capital += day_pnl
        daily_records.append({
            "date": date,
            "daily_pnl": day_pnl,
            "cumulative_pnl": capital - INITIAL_CAPITAL,
            "portfolio_value": capital,
            "daily_return": day_pnl / max(capital - day_pnl, 1),
            "active_positions": active_positions,
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "stops_triggered": stops_triggered,
        })

    portfolio_df = pd.DataFrame(daily_records)
    trade_df = pd.DataFrame(trade_log)

    if len(trade_df) > 0:
        log.info(f"  Simulation: {len(trade_df)} trades, "
                 f"{(trade_df['exit_reason'] == 'STOP_LOSS').sum()} stop-outs")
        winning = trade_df[trade_df["return"] > 0]
        log.info(f"  Win rate: {len(winning)/len(trade_df)*100:.1f}% "
                 f"({len(winning)}/{len(trade_df)})")

    return portfolio_df, trade_df
