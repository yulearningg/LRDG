import math


def adjust_learning_rate_clr(optimizer, init_lr, epoch, stepsize=200, lr_min=0.00001, lr_max=0.001, gamma=0.9998):
    """Cyclical Learning Rate (CLR)"""
    # The max learning rate is set as the initial learning rate
    lr_max = init_lr
    cycle = math.floor(epoch/(2*stepsize)+1)
    lr_tmp = abs(epoch/stepsize - 2*cycle + 1)
    cur_lr = lr_min + (lr_max - lr_min) * max(0, (1-lr_tmp))
    # Decay the learning rate based on schedule
    # cur_lr = lr_min + (lr_max - lr_min) * max(0, (1-lr_tmp)) * gamma ** epoch
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr
