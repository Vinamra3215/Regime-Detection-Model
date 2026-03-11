
import numpy as np
import pandas as pd
import torch
import logging
import sys
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, FINBERT_MODEL_NAME, FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH, RAW_NEWS_DIR, SCORED_NEWS_DIR,
)

log = logging.getLogger(__name__)


def load_finbert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    log.info(f"Loading FinBERT from {FINBERT_MODEL_NAME}...")
    log.info(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    log.info(f"FinBERT loaded successfully on {DEVICE}")
    return tokenizer, model


@torch.no_grad()
def score_texts(texts: list[str], tokenizer, model,
                batch_size: int = FINBERT_BATCH_SIZE) -> list[dict]:
    results = []
    label_map = {0: "positive", 1: "negative", 2: "neutral"}

    for i in tqdm(range(0, len(texts), batch_size), desc="FinBERT scoring"):
        batch_texts = texts[i : i + batch_size]

        batch_texts = [t[:2000] if len(t) > 2000 else t for t in batch_texts]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        for j in range(len(batch_texts)):
            pos, neg, neu = float(probs[j][0]), float(probs[j][1]), float(probs[j][2])
            label_idx = int(np.argmax(probs[j]))
            results.append({
                "positive": pos,
                "negative": neg,
                "neutral": neu,
                "sentiment_label": label_map[label_idx],
                "compound": pos - neg,
            })

    return results


def score_news_dataframe(news_df: pd.DataFrame) -> pd.DataFrame:
    if news_df.empty:
        log.warning("Empty news DataFrame, nothing to score.")
        return news_df

    tokenizer, model = load_finbert()

    texts = news_df["text"].fillna("").tolist()

    valid_mask = [len(t.strip()) > 5 for t in texts]
    valid_texts = [t for t, v in zip(texts, valid_mask) if v]

    if not valid_texts:
        log.warning("No valid texts to score.")
        return news_df

    log.info(f"Scoring {len(valid_texts)} texts with FinBERT...")
    scores = score_texts(valid_texts, tokenizer, model)

    score_df = pd.DataFrame(scores)
    valid_indices = [i for i, v in enumerate(valid_mask) if v]

    for col in ["positive", "negative", "neutral", "sentiment_label", "compound"]:
        news_df[col] = np.nan
        if col == "sentiment_label":
            news_df[col] = ""

    for idx, score_idx in enumerate(valid_indices):
        for col in ["positive", "negative", "neutral", "sentiment_label", "compound"]:
            news_df.iloc[score_idx, news_df.columns.get_loc(col)] = scores[idx][col]

    out_path = SCORED_NEWS_DIR / "scored_news.csv"
    news_df.to_csv(out_path, index=False)
    log.info(f"Saved scored news: {out_path} ({len(news_df)} rows)")

    log.info("\n--- Sentiment Distribution ---")
    if "sentiment_label" in news_df.columns:
        counts = news_df["sentiment_label"].value_counts()
        for label, count in counts.items():
            if label:
                log.info(f"  {label}: {count} ({count/len(news_df)*100:.1f}%)")
    log.info(f"  Mean compound: {news_df['compound'].mean():.4f}")
    log.info(f"  Std compound:  {news_df['compound'].std():.4f}")

    return news_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    news_path = RAW_NEWS_DIR / "all_news.csv"
    if not news_path.exists():
        log.error(f"No raw news found at {news_path}. Run news_collector.py first.")
        sys.exit(1)

    df = pd.read_csv(news_path)
    log.info(f"Loaded {len(df)} raw articles")

    scored_df = score_news_dataframe(df)
    print(f"\nScored {len(scored_df)} articles")
    print(scored_df[["ticker", "title", "compound", "sentiment_label"]].head(10))

