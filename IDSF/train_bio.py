import shutil
from trainer.bio_trainer import PhobertBIOTrainer
from argparse import ArgumentParser
from utils.utils import seed_everything
from configs.bio_config import get_config
import traceback

parser = ArgumentParser(description="phobert training cls qa")
parser.add_argument("--resume", action="store_true", default=False, help="resume training?")
args = parser.parse_args()

if __name__ == "__main__":
    import os

    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if args.resume:
        cfg = get_config(train=False)
    else:
        cfg = get_config(train=True)

    seed_everything(cfg.seed)
    trainer = PhobertBIOTrainer(config=cfg)
    try:
        trainer.train(args.resume)
    except:
        shutil.rmtree(cfg.snapshot_dir)
        traceback.print_exc()
