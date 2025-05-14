import argparse
from utils.common import load_yaml
from engine.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_only", action='store_true', default=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(
        torch.load(cfg['output_dir']+'/best.pt', map_location='cpu'))
    trainer._run_loader(trainer.val_loader, train=False)

if __name__ == "__main__":
    main()
