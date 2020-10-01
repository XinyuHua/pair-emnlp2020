import torch
import time
import collections

from modules import checkpoint_utils, progress_bar
from task import TextPlanningTask

import utils
from options import get_train_parser
from modules.trainer import Trainer
from modules.adam import FairseqAdam
from modules.inverse_square_root_scheduler import InverseSquareRootSchedule


def main():
    parser = get_train_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    task = TextPlanningTask.setup_task(args)
    task.load_dataset(args.valid_set)

    model = task.build_model(args).cuda()
    criterion = task.build_criterion(args).cuda()

    print('| model {}, criterion {}'.format(args.model_name, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    if args.tensorboard_logdir:
        tensorboard_logdir = args.tensorboard_logdir

        if tensorboard_logdir[-1] == '/':
            tensorboard_logdir = tensorboard_logdir[:-1]

        args.tensorboard_logdir = f"{tensorboard_logdir}_{time.strftime('%Y%m%d_%H%M%S')}"
        print('Tensorboard path {}'.format(args.tensorboard_logdir))

    args.ckpt_dir = f'../checkpoints/planning/{args.domain}/{args.exp_name}/'

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = FairseqAdam(args, params)

    lr_scheduler = InverseSquareRootSchedule(args, optimizer)
    lr_scheduler.step_update(0)

    # Build trainer
    trainer = Trainer(args, task, model, criterion, optimizer, lr_scheduler)

    print('| max tokens per GPU = {} and max samples per GPU = {}'.format(
        args.max_tokens,
        args.max_samples,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch
    valid_losses = [None]

    while epoch_itr.epoch < max_epoch:
        # train for one epoch
        train_epoch(args, trainer, epoch_itr)

        if epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr)

        # only use first validation loss to update the learning rate
        trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])



def train_epoch(args, trainer, epoch_itr):
    """Train the model for one epoch."""

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=True,
        shuffle=True,
    )

    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
    )

    for i, sample in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        trainer.train_step(sample)

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        progress.log(stats, tag=args.train_set, step=stats['num_updates'])


    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    progress.print(stats, tag=args.train_set, step=stats['num_updates'])

    # reset training meters
    for k in ['train_nll_loss', 'gnorm']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    nll_loss = trainer.get_meter('train_nll_loss')
    stats['nll_loss'] = nll_loss
    stats['total_loss'] = trainer.get_meter('train_total_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()

    stats['offset_loss'] = trainer.get_meter('train_offset_loss')
    stats['lr'] = trainer.get_lr()
    return stats


def validate(args, trainer, task, epoch_itr):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.valid_set),
        max_tokens=args.max_tokens_valid,
        max_samples=args.max_samples_valid,
        max_positions=min(task.max_positions, trainer.get_model().max_positions),
        seed=args.seed,
        ).next_epoch_itr(shuffle=False)

    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='valid on {}'.format(args.valid_set),
    )

    # reset validation loss meters
    meter = trainer.get_meter('valid_nll_loss')
    if meter is not None:
        meter.reset()

    for sample in progress:
        _, to_print, offset_print = trainer.valid_step(sample)
        if not args.quiet:
            print(to_print)
            print(offset_print)

    # log validation stats
    stats = get_valid_stats(trainer)
    progress.print(stats, tag=args.valid_set, step=trainer.get_num_updates())
    valid_losses.append(stats[args.best_checkpoint_metric].avg)
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['total_loss'] = trainer.get_meter('valid_total_loss')
    nll_loss = trainer.get_meter('valid_nll_loss')
    ppl = utils.get_perplexity(nll_loss.avg)

    stats['nll_loss'] = nll_loss
    stats['ppl'] = ppl
    stats['offset_loss'] = trainer.get_meter('valid_offset_loss')
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        stats['best_loss'] = min(
            checkpoint_utils.save_checkpoint.best, stats['nll_loss'].avg)
    return stats



if __name__=='__main__':
    main()
