
import argparse
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description="Train the SurfaceNet model")
    
    # Dataset options
    parser.add_argument(
        '--dataset',
        type=str,
        default="/home/woody/vlgm/vlgm103v/dataset/Data_Deschaintre18/",
        help='Path to the dataset XML file'
    )
    parser.add_argument('--resize', type=int, default=256, help='Resize dimensions for input images')
    parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='Number of worker threads for DataLoader: -1 for <batch size> threads, 0 for main thread, >0 for background threads'
    )
    
    # Experiment options
    parser.add_argument('-t', '--tag', default='default', help='Experiment tag name')
    parser.add_argument('--logdir', default='exps', type=Path, help='Logging directory for experiments')
    parser.add_argument('-tb', '--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--log-every', type=int, default=20, help='Log every n steps')
    parser.add_argument('--grads-hist', action='store_true', help='Log gradients histogram')
    parser.add_argument('--save-every', type=int, default=1, help='Save checkpoint every n epochs') #100
    
    # Training options
    parser.add_argument('--train-real', action='store_true', help='Enable training on real images')
    parser.add_argument('--train-adversarial', action='store_true', help='Enable adversarial training')
    parser.add_argument('--adv-start', type=int, default= 100, help='Start adversarial training after n steps') #300000
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for training') #64
    parser.add_argument('--optim', default='Adam', help='Optimizer type (Adam, SGD, etc.)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    '''parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming training')'''

    parser.add_argument(
    '--resume',
    type=str,
    default='/home/hpc/vlgm/vlgm103v/genmatpro/idea/exps/default/checkpoints/checkpoint_0',
    help='Path to checkpoint for resuming training'
    )

    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs') #20000
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    return args
