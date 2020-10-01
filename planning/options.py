import argparse

def _add_common_args(parser):
    parser.add_argument('--data-path', type=str, default="../data/planning/")
    parser.add_argument('--domain', type=str, required=True,
                        choices=['arggen', 'opinion', 'news'])
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--model-name', default='bert-base-cased')
    parser.add_argument('--add-keyphrase-offsets', action='store_true',
                        help='If set to true, add a special embedding to '
                             'represent the word offset of each keyphrase'
                             'in the output. This will be randomly initialized'
                             'and trained from scratch.')
    parser.add_argument('--predict-keyphrase-offset', action='store_true',
                        help='If set to true, predict the word offset for each'
                             'target keyphrase from the last BERT encoder. This'
                             'can be true only if task is `arggen_planning_task`')


    parser.add_argument('--max-prompt-length', default=42, type=int)
    parser.add_argument('--max-kp-length', default=108, type=int)
    parser.add_argument('--max-kp-set-length', default=135, type=int)
    parser.add_argument('--max-tgt-length', default=246, type=int)

    parser.add_argument('--max-positions', default=512, type=int)
    parser.add_argument('--max-tokens', default=50240, type=int)
    parser.add_argument('--max-samples', default=32, type=int)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')

    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--tensorboard-logdir', default=None)
    return


def get_train_parser():
    parser = argparse.ArgumentParser()

    _add_common_args(parser)

    parser.add_argument('--train-set', default='train', type=str)
    parser.add_argument('--valid-set', default='dev', type=str)

    parser.add_argument('--smoothed-offset-ratio', default=0.5, type=float,
                        help='Decay rate (1 no decay, 0.1 decay 90% per step)'
                             'for smoothed cross-entropy on word offset.')
    parser.add_argument('--smoothed-offset-window', default=0, type=int)
    parser.add_argument('--keyphrase-offset-loss-alpha', default=0.1,
                        type=float, help='Coefficient to balance NLL training'
                                         'and keyphrase offset prediction.')

    parser.add_argument('--max-tokens-valid', default=1024, type=int)
    parser.add_argument('--max-samples-valid', default=16, type=int)

    parser.add_argument('--max-epoch', type=int, required=True)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--warmup-updates', type=int, default=5000)
    parser.add_argument('--clip-norm', default=25, type=int)
    parser.add_argument('--validate-interval', default=1)
    parser.add_argument('--best-checkpoint-metric', type=str, default='total_loss')

    parser.add_argument('--no-epoch-checkpoints', action='store_true')
    parser.add_argument('--no-save-optimizer-state', action='store_true')
    parser.add_argument('--no-last-checkpoints', action='store_true')
    parser.add_argument('--maximize-best-checkpoint-metric', action='store_true')
    parser.add_argument('--keep-interval-updates', default=-1, type=int)

    parser.add_argument('--keep-last-epochs', default=-1, type=int)
    parser.add_argument('--reset-optimizer', action='store_true')
    parser.add_argument('--reset-meters', action='store_true')
    parser.add_argument('--reset-lr-scheduler', action='store_true')
    parser.add_argument('--reset-dataloader', action='store_true')
    parser.add_argument('--restore-file', default='checkpoint_last.pt')

    return parser


def get_decode_parser():
    parser = argparse.ArgumentParser()
    _add_common_args(parser)
    parser.add_argument('--test-set', default='test',
                        choices=['test', 'dev', 'train', 'test-toy', 'dev-toy', 'left', 'train-toy', 'dev-toy2'])

    parser.add_argument('--force-control', action='store_true')
    parser.add_argument('--force-kp-copy', action='store_true',
                        help='If set to True, the KP generation pipeline has'
                             'to copy from the source keyphrase.')
    parser.add_argument('--kp-gen-max-time', default=-1, type=int,
                        help='maximum allowed time for each keyphrase.')


    parser.add_argument('--max-gen-tgt-length', default=-1, type=int)
    parser.add_argument('--use-ref-length', action='store_true',
                        help='If set to true, generate until reach the reference'
                             'tgt length.')

    parser.add_argument('--max-tgt-generation', default=246, type=int)

    parser.add_argument('--use-gold-length', action='store_true')
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--do-sampling', action='store_true')
    parser.add_argument('--sampling-topk', default=-1, type=int)
    parser.add_argument('--sampling-topp', default=1.0, type=float)
    parser.add_argument('--repetition-penalty', default=1.0, type=float,
                        help='Repetition penalty, which is from 1.0 (no penalty) to +Inf.')
    parser.add_argument('--disable-ngram-repeat', default=-1, type=int,
                        help='Disable ngram repeat.')

    return parser
