import os
import time
import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import utils
from options import get_train_parser
from system import BARTSeq2seq

logger = logging.getLogger(__name__)


def train():
    parser = get_train_parser()
    args = parser.parse_args()
    utils.set_seed(args)

    if os.path.exists(args.ckpt_dir) and os.listdir(args.ckpt_dir):
        latest_ckpt_path = utils.get_latest_ckpt_path(args.ckpt_dir)
        logger.info('Loading checkpoint {}'.format(latest_ckpt_path))
        new_epoch_id = os.path.basename(latest_ckpt_path).split('_')[0][-2:]
        new_epoch_id = int(new_epoch_id)
        system = BARTSeq2seq.load_from_checkpoint(latest_ckpt_path)
        system.current_epoch = new_epoch_id + 1
        logger.info('Resuming training from epoch {}'.format(system.current_epoch))

    else:
        system = BARTSeq2seq(args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.ckpt_dir, "{epoch:02d}_{val_loss:.2f}"),
        monitor="val_loss", mode="min",
        save_top_k=args.save_topk_ckpt,
        period=3,
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpus,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=checkpoint_callback,
    )

    if args.fp16:
        train_params['precision'] = 16
        train_params['use_amp'] = args.fp16
        train_params['amp_level'] = args.fp16_opt_level

    train_params['logger'] = TensorBoardLogger(f'tboard/{args.domain}',
                                               name=f"{args.tensorboard_dir}_{time.strftime('%Y%m%d_%H%M%S')}")
    if args.n_gpus > 1:
        train_params['distributed_backend'] = 'dp'

    trainer = pl.Trainer(**train_params)
    trainer.current_epoch = system.current_epoch
    trainer.fit(system)

if __name__ == "__main__":
    train()