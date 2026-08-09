
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay', type=float, default=0.9, help='learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')

parser.add_argument('--epoch', type=int, default=50, help='# of epochs')
parser.add_argument('--imSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--iter_epoch_ratio', type=float, default=0.4, help='# of ratio of total images as an epoch')
parser.add_argument('--num_class', type=int, default=2, help='# of classes')
parser.add_argument('--checkpoint_path', type=str, default='2fenlei', help='where checkpoint saved')
parser.add_argument('--data_path', type=str, default='../data/classification/', help='where dataset saved.')
parser.add_argument('--test_data_path', type=str, default='../data/classification/', help='where test dataset saved.')

# parser.add_argument('--load_from_checkpoint', type=str, default='/home/pengxiao/virtualenvs/nmi-wsi-diagnosis/checkpoints/classification/trained_model/checkpoint_4_84.67.h5', help='where checkpoint loaded')
parser.add_argument('--load_from_checkpoint', type=str, default='/data/pengxiao/nmi-wsi-diagnosis/checkpoints/classification/trained_model/2fenlei_1_0.92.pth', help='where checkpoint loaded')
parser.add_argument('--optim', type=str, default='rmsp', help='which optimizer')
parser.add_argument('--lr_decay_epoch', type=int, default=2, help='how many epoch to decay learning rate')
parser.add_argument('--eval', action='store_true', help='doing evaluation only')
parser.add_argument('--use_imagenet_weight', action='store_true', help='doing evaluation only')
parser.add_argument('--use_cls_weight', type=int, default=5, help='weight the classifier')
parser.add_argument('--lock_bn', action='store_true', help='if lock batch normalization')

opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# hardcode here
dataset_mean = [0.5,0.5,0.5]
dataset_std = [0.5,0.5,0.5]

checkpoint_root = '../checkpoints/classification/'
opt.checkpoint_path = os.path.join(checkpoint_root, opt.checkpoint_path)
if opt.checkpoint_path != '' and not os.path.isdir(opt.checkpoint_path):
    os.mkdir(opt.checkpoint_path)
