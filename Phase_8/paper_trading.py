
import pandas as pd
import logging
from datetime import datetime

from config import (
    KITE_API_KEY, KITE_API_SECRET, PAPER_TRADE_MODE,
    INITIAL_CAPITAL, RESULTS_DIR,
)

log = logging.getLogger(__name__)


class PaperTrader:

    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.positions = {}
        self.order_history = []
        self.is_paper = PAPER_TRADE_MODE
        self.kite = None

        if not self.is_paper and KITE_API_KEY:
            try:
                from kiteconnect import KiteConnect
                self.kite = KiteConnect(api_key=KITE_API_KEY)
                log.info("Connected to Zerodha Kite API (LIVE mode)")
            except ImportError:
                log.warning("kiteconnect not installed. Falling back to paper mode.")
                self.is_paper = True
        else:
            log.info("Running in PAPER TRADE mode")

    def place_order(self, ticker, quantity, side, price=None):
        order = {
            "timestamp": datetime.now().isoformat(),
            "ticker": ticker,
            "quantity": quantity,
            "side": side,
            "price": price,
            "status": "EXECUTED",
            "mode": "PAPER" if self.is_paper else "LIVE",
        }

        if self.is_paper:
            if side == "BUY":
                self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            elif side == "SELL":
                self.positions[ticker] = self.positions.get(ticker, 0) - quantity
                if self.positions[ticker] <= 0:
                    del self.positions[ticker]
        else:
            if self.kite:
                try:
                    order_type = "MARKET" if price is None else "LIMIT"
                    txn = "BUY" if side == "BUY" else "SELL"
                    order_id = self.kite.place_order(
                        variety="regular",
                        exchange="NSE",
                        tradingsymbol=ticker.replace(".NS", ""),
                        transaction_type=txn,
                        quantity=quantity,
                        product="CNC",
                        order_type=order_type,
                        price=price,
                    )
                    order["order_id"] = order_id
                    order["status"] = "PLACED"
                except Exception as e:
                    order["status"] = "FAILED"
                    order["error"] = str(e)

        self.order_history.append(order)
        return order

    def get_positions(self):
        if self.is_paper:
            return self.positions.copy()
        elif self.kite:
            return self.kite.positions()
        return {}

    def get_order_history(self):
        return pd.DataFrame(self.order_history)

    def generate_paper_trade_report(self, signals_df):
        recommendations = []

        for _, row in signals_df.iterrows():
            signal = row.get("signal", "FLAT")
            ticker = row.get("ticker", "")
            conf = row.get("confidence", 0)

            if signal == "FLAT":
                if ticker in self.positions:
                    recommendations.append({
                        "action": "SELL",
                        "ticker": ticker,
                        "reason": f"Signal changed to FLAT (conf={conf:.2f})",
                        "urgency": "MEDIUM",
                    })
            elif signal in ("STRONG_LONG", "WEAK_LONG"):
                if ticker not in self.positions:
                    recommendations.append({
                        "action": "BUY",
                        "ticker": ticker,
                        "signal": signal,
                        "confidence": conf,
                        "reason": f"Bull regime detected ({signal}, conf={conf:.2f})",
                        "urgency": "HIGH" if signal == "STRONG_LONG" else "MEDIUM",
                    })

        report = pd.DataFrame(recommendations)
        if len(report) > 0:
            report = report.sort_values("urgency", ascending=True)
            report.to_csv(RESULTS_DIR / "paper_trade_recommendations.csv", index=False)

        log.info(f"Paper trade report: {len(recommendations)} recommendations")
        return report

