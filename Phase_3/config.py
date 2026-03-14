
import os
from pathlib import Path
import torch

BASE_DIR          = Path(__file__).resolve().parent
PROJECT_DIR       = BASE_DIR.parent
PHASE1_LABEL_DIR  = PROJECT_DIR / "results" / "phase_1" / "data" / "labelled"
PHASE2_RESULTS_DIR = PROJECT_DIR / "results" / "phase_2_results"

RESULTS_DIR            = PROJECT_DIR / "results" / "phase_3"
RAW_NEWS_DIR           = RESULTS_DIR / "raw_news"
SCORED_NEWS_DIR        = RESULTS_DIR / "scored_news"
SENTIMENT_FEATURES_DIR = RESULTS_DIR / "sentiment_features"
MARKET_DATA_DIR        = RESULTS_DIR / "market_data"
PLOTS_DIR              = RESULTS_DIR / "plots"
LOG_DIR                = RESULTS_DIR / "logs"

for d in [RESULTS_DIR, RAW_NEWS_DIR, SCORED_NEWS_DIR,
          SENTIMENT_FEATURES_DIR, MARKET_DATA_DIR, PLOTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "BAJAJFINSV.NS", "NTPC.NS", "POWERGRID.NS", "TECHM.NS", "ONGC.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "M&M.NS", "HINDALCO.NS",
    "COALINDIA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "APOLLOHOSP.NS",
    "ADANIPORTS.NS", "ADANIENT.NS", "GRASIM.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "SHRIRAMFIN.NS", "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS",
    "BRITANNIA.NS", "ITC.NS", "BAJAJ-AUTO.NS", "BEL.NS", "TRENT.NS",
]

TICKER_TO_COMPANY = {
    "RELIANCE.NS": ["Reliance Industries", "Reliance", "RIL"],
    "TCS.NS": ["Tata Consultancy", "TCS"],
    "HDFCBANK.NS": ["HDFC Bank", "HDFCBANK"],
    "INFY.NS": ["Infosys", "INFY"],
    "ICICIBANK.NS": ["ICICI Bank", "ICICI"],
    "HINDUNILVR.NS": ["Hindustan Unilever", "HUL"],
    "SBIN.NS": ["State Bank of India", "SBI", "State Bank"],
    "BHARTIARTL.NS": ["Bharti Airtel", "Airtel"],
    "BAJFINANCE.NS": ["Bajaj Finance"],
    "KOTAKBANK.NS": ["Kotak Mahindra Bank", "Kotak Bank", "Kotak"],
    "LT.NS": ["Larsen & Toubro", "Larsen Toubro", "L&T"],
    "AXISBANK.NS": ["Axis Bank"],
    "ASIANPAINT.NS": ["Asian Paints"],
    "MARUTI.NS": ["Maruti Suzuki", "Maruti"],
    "HCLTECH.NS": ["HCL Technologies", "HCL Tech"],
    "SUNPHARMA.NS": ["Sun Pharma", "Sun Pharmaceutical"],
    "TITAN.NS": ["Titan Company", "Titan"],
    "WIPRO.NS": ["Wipro"],
    "ULTRACEMCO.NS": ["UltraTech Cement", "UltraTech"],
    "NESTLEIND.NS": ["Nestle India", "Nestle"],
    "BAJAJFINSV.NS": ["Bajaj Finserv"],
    "NTPC.NS": ["NTPC"],
    "POWERGRID.NS": ["Power Grid", "PowerGrid"],
    "TECHM.NS": ["Tech Mahindra"],
    "ONGC.NS": ["ONGC", "Oil and Natural Gas"],
    "TATAMOTORS.NS": ["Tata Motors"],
    "TATASTEEL.NS": ["Tata Steel"],
    "JSWSTEEL.NS": ["JSW Steel", "JSW"],
    "M&M.NS": ["Mahindra & Mahindra", "Mahindra", "M&M"],
    "HINDALCO.NS": ["Hindalco"],
    "COALINDIA.NS": ["Coal India"],
    "DRREDDY.NS": ["Dr Reddy", "Dr. Reddy"],
    "DIVISLAB.NS": ["Divi's Laboratories", "Divis Lab"],
    "CIPLA.NS": ["Cipla"],
    "APOLLOHOSP.NS": ["Apollo Hospitals", "Apollo"],
    "ADANIPORTS.NS": ["Adani Ports"],
    "ADANIENT.NS": ["Adani Enterprises", "Adani"],
    "GRASIM.NS": ["Grasim Industries", "Grasim"],
    "HDFCLIFE.NS": ["HDFC Life"],
    "SBILIFE.NS": ["SBI Life"],
    "SHRIRAMFIN.NS": ["Shriram Finance", "Shriram"],
    "BPCL.NS": ["BPCL", "Bharat Petroleum"],
    "EICHERMOT.NS": ["Eicher Motors", "Eicher"],
    "HEROMOTOCO.NS": ["Hero MotoCorp", "Hero Moto"],
    "INDUSINDBK.NS": ["IndusInd Bank", "IndusInd"],
    "BRITANNIA.NS": ["Britannia"],
    "ITC.NS": ["ITC"],
    "BAJAJ-AUTO.NS": ["Bajaj Auto"],
    "BEL.NS": ["Bharat Electronics", "BEL"],
    "TRENT.NS": ["Trent"],
}

TICKER_TO_SECTOR = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "BPCL.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "Finance", "BAJAJFINSV.NS": "Finance", "SHRIRAMFIN.NS": "Finance",
    "HDFCLIFE.NS": "Insurance", "SBILIFE.NS": "Insurance",
    "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "ITC.NS": "FMCG",
    "BHARTIARTL.NS": "Telecom",
    "LT.NS": "Infra", "ADANIPORTS.NS": "Infra", "ADANIENT.NS": "Infra",
    "GRASIM.NS": "Infra", "ULTRACEMCO.NS": "Infra",
    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer", "TRENT.NS": "Consumer",
    "MARUTI.NS": "Auto", "TATAMOTORS.NS": "Auto", "EICHERMOT.NS": "Auto",
    "HEROMOTOCO.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "M&M.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "DIVISLAB.NS": "Pharma",
    "CIPLA.NS": "Pharma", "APOLLOHOSP.NS": "Pharma",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "COALINDIA.NS": "Mining",
    "NTPC.NS": "Power", "POWERGRID.NS": "Power", "BEL.NS": "Defence",
}

RSS_FEEDS = {
    "moneycontrol_markets": "https://www.moneycontrol.com/rss/marketreports.xml",
    "moneycontrol_business": "https://www.moneycontrol.com/rss/business.xml",
    "moneycontrol_stocks": "https://www.moneycontrol.com/rss/lateststnews.xml",
    "et_markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "et_stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "et_companies": "https://economictimes.indiatimes.com/news/company/rssfeeds/2143429.cms",
    "livemint_market": "https://www.livemint.com/rss/market",
    "livemint_companies": "https://www.livemint.com/rss/companies",
}

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock+NSE&hl=en-IN&gl=IN&ceid=IN:en"
NEWS_REQUEST_TIMEOUT = 15
NEWS_DELAY_BETWEEN   = 1.0
MAX_NEWS_PER_TICKER  = 50

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_BATCH_SIZE = 32
FINBERT_MAX_LENGTH = 512

DATA_START    = "2019-01-01"
DATA_END      = "2025-12-31"
VIX_TICKER    = "^INDIAVIX"
NIFTY_INDEX   = "^NSEI"

SENTIMENT_ROLLING_WINDOWS = [5, 10, 20]

SENTIMENT_FEATURE_COLUMNS = [
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_positive_ratio",
    "news_negative_ratio",
    "news_count",
    "vix_normalized",
    "vix_change_1d",
    "vix_change_5d",
    "vix_percentile",
    "market_return_5d",
    "market_return_20d",
    "market_breadth",
    "sector_sentiment",
    "composite_sentiment",
    "sentiment_momentum_5d",
    "sentiment_momentum_20d",
]

REGIME_TO_IDX = {"Bear": 0, "Sideways": 1, "Bull": 2}
IDX_TO_REGIME = {v: k for k, v in REGIME_TO_IDX.items()}
NUM_CLASSES   = 3

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
