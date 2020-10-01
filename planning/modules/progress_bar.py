# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

from collections import OrderedDict
import json
from numbers import Number
import os
import re
import sys
import datetime
import time

from tqdm import tqdm

from modules import AverageMeter

g_tbmf_wrapper = None

def build_progress_bar(args, iterator, epoch=None, prefix=None):
    bar = tqdm_progress_bar(iterator, epoch, prefix)
    if args.tensorboard_logdir:
        bar = tensorboard_log_wrapper(bar, args.tensorboard_logdir, args)
    return bar


def format_stat(stat):
    if isinstance(stat, Number):
        stat = '{:g}'.format(stat)
    elif isinstance(stat, AverageMeter):
        stat = '{:.3f}'.format(stat.avg)
    return stat


class progress_bar(object):
    """Abstract class for progress bars."""
    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.offset = getattr(iterable, 'offset', 0)
        self.epoch = epoch
        self.prefix = ''
        if epoch is not None:
            self.prefix += '| epoch {:03d}'.format(epoch)
        if prefix is not None:
            self.prefix += ' | {}'.format(prefix)

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def _str_commas(self, stats):
        return ', '.join(key + '=' + stats[key].strip()
                         for key in stats.keys())

    def _str_pipes(self, stats):
        return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


class tqdm_progress_bar(progress_bar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))


class tensorboard_log_wrapper(progress_bar):
    """Log to tensorboard."""

    def __init__(self, wrapped_bar, tensorboard_logdir, args):
        self.wrapped_bar = wrapped_bar
        self.tensorboard_logdir = tensorboard_logdir
        self.args = args

        try:
            from tensorboardX import SummaryWriter
            self.SummaryWriter = SummaryWriter
            self._writers = {}
        except ImportError:
            print("tensorboard or required dependencies not found, "
                  "please see README for using tensorboard. (e.g. pip install tensorboardX)")
            self.SummaryWriter = None

    def _writer(self, key):
        if self.SummaryWriter is None:
            return None
        if key not in self._writers:
            self._writers[key] = self.SummaryWriter(
                os.path.join(self.tensorboard_logdir, key),
            )
            self._writers[key].add_text('args', str(vars(self.args)))
            self._writers[key].add_text('sys.argv', " ".join(sys.argv))
        return self._writers[key]

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def __exit__(self, *exc):
        for writer in getattr(self, '_writers', {}).values():
            writer.close()
        return False

    def _log_to_tensorboard(self, stats, tag='', step=None):
        writer = self._writer(tag)
        if writer is None:
            return
        if step is None:
            step = stats['num_updates']
        for key in stats.keys() - {'num_updates'}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(key, stats[key], step)
