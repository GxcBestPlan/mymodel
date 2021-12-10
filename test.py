from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from mymodel import Net as mymodel
from srcnn import Net as srcnn
from edsr import Net as edsr
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpn_iterative import Net as DBPNITER
from data import get_test_set
from functools import reduce
from utils import calc_psnr
# from scipy.misc import imsave
import scipy.io as sio
import time
import cv2


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='./Dataset')

parser.add_argument('--test_dataset', type=str, default='Set5_LR_x8')
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--model', default='models/DBPNLL_x8.pth', help='sr pretrained base model')
parser.add_argument('--patch_size', type=int, default=1, help='Size of cropped HR image')
parser.add_argument('--data_augmentation', type=bool, default=True)


opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
test_set = get_test_set(opt.input_dir, opt.test_dataset, opt.upscale_factor, opt.patch_size, opt.data_augmentation)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=True)

print('===> Building model')
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) ###D-DBPN
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) ###D-DBPN
elif opt.model_type == 'mymodel':
    model = mymodel(num_channels=3, base_filter=64,  feat=256, num_stages=7, scale_factor=opt.upscale_factor)
elif opt.model_type == 'srcnn':
    model = srcnn()
elif opt.model_type == 'edsr':
    model = edsr()
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) ###D-DBPN

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])


def test():
    avg_psnr = 0
    model.eval()
    # test_epoch = 0
    for batch in testing_data_loader:
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        with torch.no_grad():
            if opt.model_type == 'srcnn':
                prediction = model(bicubic)
            else:
                prediction = model(input)

        if opt.model_type != 'srcnn' and opt.residual:
            prediction = prediction + bicubic

        psnr = calc_psnr(prediction, target)
        # print(len(testing_data_loader))
        print(psnr)
        # print(prediction)
        # print(target)
        avg_psnr += psnr
        # test_epoch += 1
        # print("===> Testing({}/{}): pnsr: {:.4f}".format(test_epoch, len(testing_data_loader), psnr))

    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


if __name__ == '__main__':
    test()
