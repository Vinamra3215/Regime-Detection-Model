
import argparse
import sys
import time
from pathlib import Path

from config import NIFTY_50_TICKERS, EPOCHS


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 — Sentiment-Enriched Regime Model Pipeline")
    parser.add_argument("--step", choices=["train", "evaluate", "predict", "all"],
                        default="all")
    parser.add_argument("--tickers", nargs="+", default=NIFTY_50_TICKERS)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    t_start = time.time()

    if args.step in ("train", "all"):
        print("\n" + "#" * 70)
        print("  STEP 1/3: TRAINING")
        print("#" * 70)
        from train import train
        best_acc, best_epoch = train(
            tickers=args.tickers,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
        )

    if args.step in ("evaluate", "all"):
        print("\n" + "#" * 70)
        print("  STEP 2/3: EVALUATION")
        print("#" * 70)
        from evaluate import main as eval_main
        try:
            eval_main()
        except SystemExit as e:
            if e.code != 0:
                print("  Evaluation completed (NO-GO, continuing to predictions)")

    if args.step in ("predict", "all"):
        print("\n" + "#" * 70)
        print("  STEP 3/3: PREDICTIONS")
        print("#" * 70)
        from predict import predict_all
        predict_all(args.tickers)

    elapsed = time.time() - t_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n  Phase 4 pipeline completed in {minutes}m {seconds}s")


if __name__ == "__main__":
    main()

