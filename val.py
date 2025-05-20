import argparse, torch
from utils.common import load_yaml
from engine.trainer import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    trainer = Trainer(cfg)
    trainer.model.load_state_dict(torch.load(cfg['output_dir']+'/best.pt', map_location='cpu'))
    val_loss = trainer._standard_epoch(train=False)
    print("Validation loss:", val_loss)

if __name__=="__main__":
    main()
