import argparse

from src.train import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ar = ap.parse_args()
    main(ar.fast)
