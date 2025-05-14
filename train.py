import argparse, yaml
from utils.common import load_yaml
from engine.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    Trainer(cfg).fit()

if __name__ == "__main__":
    main()
