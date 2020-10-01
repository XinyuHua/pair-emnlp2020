#
from collections import OrderedDict
import collections
import logging
import os
import re
import traceback
import shutil

import torch
from torch.serialization import default_restore_location


def save_checkpoint(args, trainer, epoch_itr, val_loss):

    def is_better(a, b):
        return a > b if args.maximize_best_checkpoint_metric else a < b

    args.ckpt_dir = f'../checkpoints/planning/{args.domain}/{args.exp_name}/'
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)


    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds['checkpoint{}.pt'.format(epoch)] = (
        end_of_epoch and not args.no_epoch_checkpoints and
        epoch % args.save_interval == 0
    )
    checkpoint_conds['checkpoint_{}_{}.pt'.format(epoch, updates)] = (
        not end_of_epoch and args.save_interval_updates > 0 and
        updates % args.save_interval_updates == 0
    )
    checkpoint_conds['checkpoint_best.pt'] = (
        val_loss is not None and
        (not hasattr(save_checkpoint, 'best') or is_better(val_loss, save_checkpoint.best))
    )
    checkpoint_conds['checkpoint_last.pt'] = not args.no_last_checkpoints

    prev_best = getattr(save_checkpoint, 'best', val_loss)
    if val_loss is not None:
        save_checkpoint.best = val_loss if is_better(val_loss, prev_best) else prev_best
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    if hasattr(save_checkpoint, 'best'):
        extra_state.update({'best': save_checkpoint.best})

    checkpoints = [os.path.join(args.ckpt_dir, fn) for fn, cond in checkpoint_conds.items() if cond]
    if len(checkpoints) > 0:
        trainer.save_checkpoint(checkpoints[0], extra_state)
        for cp in checkpoints[1:]:
            shutil.copyfile(checkpoints[0], cp)

        print('| saved checkpoint {} (epoch {} @ {} updates)'.format(
            checkpoints[0], epoch, updates))

    if not end_of_epoch and args.keep_interval_updates > 0:
        # remove old checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.ckpt_dir, pattern=r'checkpoint_\d+_(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_interval_updates:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)

    if args.keep_last_epochs > 0:
        # remove old epoch checkpoints; checkpoints are sorted in descending order
        checkpoints = checkpoint_paths(
            args.ckpt_dir, pattern=r'checkpoint(\d+)\.pt',
        )
        for old_chk in checkpoints[args.keep_last_epochs:]:
            if os.path.lexists(old_chk):
                os.remove(old_chk)


def load_checkpoint(args, trainer):
    """Load a checkpoint and restore the training iterator."""
    # only one worker should attempt to create the required dir

    if os.path.isabs(args.restore_file):
        checkpoint_path = args.restore_file
    else:
        checkpoint_path = os.path.join(args.ckpt_dir, args.restore_file)

    extra_state = trainer.load_checkpoint(
        checkpoint_path,
        args.reset_optimizer,
        args.reset_lr_scheduler,
        {},
        reset_meters=args.reset_meters,
    )

    if (
        extra_state is not None
        and 'best' in extra_state
        and not args.reset_optimizer
        and not args.reset_meters
    ):
        save_checkpoint.best = extra_state['best']

    if extra_state is not None and not args.reset_dataloader:
        # restore iterator from checkpoint
        itr_state = extra_state['train_iterator']
        epoch_itr = trainer.get_train_iterator(epoch=itr_state['epoch'])
        epoch_itr.load_state_dict(itr_state)
    else:
        epoch_itr = trainer.get_train_iterator(epoch=0)

    trainer.lr_step(epoch_itr.epoch)

    return extra_state, epoch_itr


def load_checkpoint_to_cpu(path, arg_overrides=None):
    """Loads a checkpoint to CPU (with upgrading for backward compatibility)."""
    state = torch.load(
        path, map_location=lambda s, l: default_restore_location(s, 'cpu'),
    )
    args = state['args']
    if arg_overrides is not None:
        for arg_name, arg_val in arg_overrides.items():
            setattr(args, arg_name, arg_val)
    return state


def checkpoint_paths(path, pattern=r'checkpoint(\d+)\.pt'):
    """Retrieves all checkpoints found in `path` directory.

    Checkpoints are identified by matching filename to the specified pattern. If
    the pattern contains groups, the result will be sorted by the first group in
    descending order.
    """
    pt_regexp = re.compile(pattern)
    files = os.listdir(path)

    entries = []
    for i, f in enumerate(files):
        m = pt_regexp.fullmatch(f)
        if m is not None:
            idx = int(m.group(1)) if len(m.groups()) > 0 else i
            entries.append((idx, m.group(0)))
    return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict


def save_state(
    filename, args, model_state_dict, criterion, optimizer, lr_scheduler,
    num_updates, optim_history=None, extra_state=None,
):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    state_dict = {
        'args': args,
        'model': model_state_dict if model_state_dict else {},
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
                'lr_scheduler_state': lr_scheduler.state_dict(),
                'num_updates': num_updates,
            }
        ],
        'extra_state': extra_state,
    }
    if not args.no_save_optimizer_state:
        state_dict['last_optimizer_state'] = convert_state_dict_type(optimizer.state_dict())
    torch_persistent_save(state_dict, filename)
