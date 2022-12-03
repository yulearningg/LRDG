import os
import shutil

import numpy as np
import torch.nn as nn

from __main__ import args


def set_devs():
    """Set cpu or a list of gpus"""
    if args.gpus is None or args.gpus == '':
        return None
    elif ',' in args.gpus:
        return [int(i) for i in args.gpus.split(',') if i != '']
    return [int(args.gpus)]


def delete_existing(path):
    """Delete directory if it exists

    Used for automatically rewrites existing log directories

    Arguments:
        path (string): path of the directory
    """
    if os.path.exists(path):
        shutil.rmtree(path)


class AverageMeter(object):
    """Compute and store the average and current value"""

    def __init__(self, size=None):
        self.size = size
        self.reset()

    def reset(self):
        self.count = 0
        if self.size is None:
            self.val = 0
            self.avg = 0
            self.sum = 0
        else:
            assert isinstance(self.size, int)
            self.val = np.zeros((self.size,), dtype=np.float32)
            self.avg = np.zeros((self.size,), dtype=np.float32)
            self.sum = np.zeros((self.size,), dtype=np.float32)

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_state_dict(net, state_dict):
    """Load parameters of pytorch saved model"""
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
