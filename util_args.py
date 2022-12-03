args = None


def add_args(parser):
    """Return a parser added with args required for training

    Arguments:
        parser : argparse.ArgumentParser
    """
    train = parser.add_argument_group('model training')
    train.add_argument('--record-name', type=str, default='test',
                       help='Record name for saving')
    train.add_argument('--domain', type=str, default='pacs',
                       help='Dataset name')
    train.add_argument('--src', type=str, default='art,photo,cartoon',
                       help='Source domains')
    train.add_argument('--trg', type=str, default='sketch',
                       help='Target domain')
    train.add_argument('--datadir', type=str, default='./input',
                       help='Path to dataset')
    train.add_argument('--logdir', type=str, default='./log',
                       help='Log directory')
    train.add_argument('--savedir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    train.add_argument('--saver', action='store_true',
                       help='Whether save model or not')
    train.add_argument('--resume', type=str, default=None,
                       help='Resume from the checkpoint')
    train.add_argument('--num-epochs', type=int, default=1,
                       help='Number of total training epochs')
    train.add_argument('--batch-size', type=int, default=5,
                       help='mini-batch size (1 = pure stochastic)')
    train.add_argument('--train-trans', type=str, default='train',
                       help='Image transformation for training dataset')
    train.add_argument('--test-trans', type=str, default='test',
                       help='Image transformation for test dataset')
    train.add_argument('--network', type=str, default='resnet18',
                       help='Backbone neural network')
    train.add_argument('--num-classes', type=int, default=7,
                       help='Number of classes')
    train.add_argument('--lambda2', type=float, default=1.,
                       help="Weight for uncertainty loss")
    train.add_argument('--lambda3', type=float, default=1.,
                       help="Weight for image reconstruction loss")
    train.add_argument('--lr', type=float, default=0.1,
                       help='Initial learning rate')
    train.add_argument('--mom', type=float, default=0.9,
                       help='Momentum')
    train.add_argument('--wd', type=float, default=1e-4,
                       help='Weight decay')
    train.add_argument('--num-workers', type=int, default=1,
                       help='Number of cpu workers')
    train.add_argument('--seed', type=int, default=0,
                       help='Manually set RNG seed')
    train.add_argument('--gpus', type=str, default=None,
                       help='The number of the GPUs')
    return train
