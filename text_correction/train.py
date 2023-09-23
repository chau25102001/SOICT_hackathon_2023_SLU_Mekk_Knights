import pprint
from trainer.trainer import BartPhoCorrectionTrainer
from argparse import ArgumentParser
from utils.utils import seed_everything
from configs.config import get_config
import traceback

parser = ArgumentParser(description="bartpho training cls qa")
parser.add_argument("--resume", action="store_true", default=False, help="resume training?")
parser.add_argument("--train_batch_size", type=int, default=64, help="batch size")
parser.add_argument("--val_batch_size", type=int, default=64, help="batch size")
args = parser.parse_args()

if __name__ == "__main__":
    import os

    os.environ['TOKENIZERS_PARALLELISM'] = "false"

    if args.resume:
        cfg = get_config(train=False)
    else:
        cfg = get_config(train=True)
    seed_everything(cfg.seed)
    if args.train_batch_size is not None and isinstance(args.train_batch_size, int):
        cfg.train_batch_size = args.train_batch_size
        cfg.total_batch_size = args.train_batch_size
    if args.val_batch_size is not None and isinstance(args.val_batch_size, int):
        cfg.val_batch_size = args.val_batch_size
    trainer = BartPhoCorrectionTrainer(config=cfg)
    try:
        trainer.train()
    except:
        # shutil.rmtree(cfg.snapshot_dir)
        trainer.save_checkpoint("checkpoint_last.pt")
        traceback.print_exc()
