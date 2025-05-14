import argparse
from utils.common import load_yaml
from engine.tester import test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    test(cfg, args.weights)

if __name__ == "__main__":
    main()
