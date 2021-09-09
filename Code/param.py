import argparse
import shlex

# Model argparse analysis 
def train_args(command):
    parser = argparse.ArgumentParser(description='Training')
    # device
    parser.add_argument('--device', default=None, type=str, help='used device')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', 
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-mi_b', '--mini_batch', default=64, type=int,
                        metavar='N', help='load mini batch size (default: 64)')
    # basic set
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # pretrain
    parser.add_argument('--pretrain', default=None,
                        type=str, choices=['MS_Celeb_1M','resume'],
                        help='default: MS_Celeb_1M')   
    # dataset 
    parser.add_argument('--dataset', choices=['FER','AFF','RAF-DB','A+R', 'SFEW'], type=str,
                            help='the name of dataset')
    parser.add_argument('--subset', type=str,
                            help='the sub-type of dataset')
    # learning rate and batch size
    parser.add_argument('--lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_rate', '-lrr', default=0, type=float, help='lr change rate')
    parser.add_argument('--lr_start', '-lrs', default=0, type=int,
                         help='first epoch between lr changes')
    parser.add_argument('--lr_interval', '-lri', default=0, type=int,
                         help='interval of epoch between lr changes')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                            metavar='N', help='train batch size (default: 128)')
    
    # model
    parser.add_argument('--model', type=str,  help='the type of model, include global, gllc, trans')
    parser.add_argument('-n_b', '--blocks', default=2, type=int,
                        help='the number of CNN blocks (default:2)')
    parser.add_argument('--pool_type', '-pt', default='avg',
                        choices=['avg','vit'],
                        type=str, help='type of the last pooling layer')
    # encoder
    parser.add_argument('-n_h', '--num_heads', default=8, type=int,
                        help='the number of heads (default:8)')
    parser.add_argument('--bias', '-bi', default=False, action='store_true', help='attention bias')
    parser.add_argument('--dropout', '-dp', default=0.0 , type=float, help= 'dropout rate')
    # name
    parser.add_argument('--name', default=None , type=str, help= 'template name')
    args = parser.parse_args(shlex.split(command))
    return args

# Run argparse analysis 
def run_args():
    parser = argparse.ArgumentParser(description='Run Trian File')
    parser.add_argument('-s', '--scheme', type=str)
    parser.add_argument('-d', '--device', type=str)
    parser.add_argument('-m', '--mode', type=str, default='train')
    args = parser.parse_args()
    return args