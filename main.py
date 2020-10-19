from __future__ import print_function
import skimage
import scipy
import os
import torch
import torch.utils.data as Data
import numpy as np
import SimpleITK as sitk
from nibabel import Nifti1Image
from scipy import ndimage
from SFT_rcan import SFT_Net_torch
import torch.backends.cudnn as cudnn
import argparse
import datetime
import math
import glob
import shutil
import matplotlib.pyplot as plt
import cv2
from torch.utils import data
from dataset import *


parser = argparse.ArgumentParser(description='Example')
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='training batch size') ###default=36
parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.02')####try 0.05;0.005;0.0005;0.00005;
5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true',default=True,  help='using GPU or not. default=True')
parser.add_argument('--seed', type=int, default=2, help='random seed to use. Default=1111')


parser.add_argument('--n_resblocks', type=int, default=20,help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,help='number of feature maps')
parser.add_argument('--rgb_range', type=int, default=255,help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,help='number of color channels to use, render 6 channle, without 3 channels')
parser.add_argument('--scale', type=str, default=4,help='super resolution scale')
parser.add_argument('--n_resgroups', type=int, default=10,help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,help='number of feature maps reduction')
parser.add_argument('--crop_size', type=int, default=142,help='crop size')
parser.add_argument('--input_size', type=int, default=128,help='input_image_size')

parser.add_argument('--train_data', type=str, default='../face_dataset/celeba_dataset/training/',help='training dataset directory')
parser.add_argument('--test_data', type=str, default='../face_dataset/celeba_dataset/test/',help='test dataset directory')

opt = parser.parse_args()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opt.lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def psnr_loss(pred, gt):
    imdff = pred - gt
    rmse = (np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 10 * math.log10(1 / rmse)

###########train loader

train_list=load_file_list(opt.train_data)

train_render_list=load_file_list(opt.train_data,render=True)

train_dataset = MyTupleDataset(train_list,train_render_list,crop_size=opt.crop_size, input_height=opt.input_size, up_scale=opt.scale,is_mirror=True)

train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=opt.BATCH_SIZE,      # mini batch size
    shuffle=True,
    num_workers=2)
################################333

###test loader
test_list=load_file_list(opt.test_data)
test_render_list=load_file_list(opt.test_data,render=True)

test_dataset = MyTupleDataset(test_list,test_render_list,crop_size=opt.crop_size,input_height=opt.input_size,up_scale=opt.scale,is_mirror=False)

test_loader = Data.DataLoader(
    dataset=test_dataset,      # torch TensorDataset format
    batch_size=opt.BATCH_SIZE,      # mini batch size
    shuffle=False,
    num_workers=2)

# ###########################
BATCH_SIZE=opt.BATCH_SIZE
cuda = opt.cuda

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
print('===> Building model')

####VDSR parameter:         n_resblocks = args.n_resblocks   n_feats = args.n_feats  args.rgb_range 255)  args.n_colors=3
SFT_model = SFT_Net_torch(opt)
print('#  parameters:', sum(param.numel() for param in SFT_model.parameters()))
print('# trainable parameters:', sum(p.numel() for p in SFT_model.parameters() if p.requires_grad))
# NetS.apply(weights_init)
#print(SFT_model)

if cuda:
    SFT_model = SFT_model.cuda()

optimizer = torch.optim.Adam(SFT_model.parameters(), lr=opt.lr)

#criterion = torch.nn.MSELoss(reduce=True, size_average=True)

criterion = torch.nn.MSELoss(reduction='elementwise_mean')

# percent=sum_batch/BATCH_SIZE
mse_epoch=list()
psnr_epoch=list()
for epoch in range(opt.epoch):   #
    print('epoch',epoch)
    dice_acc = list()
    i=0
    lr = adjust_learning_rate(optimizer, epoch)
    print('learning rate',lr)
    for step, (batch_x, batch_x1, batch_y) in enumerate(train_loader):
        batch_x=batch_x.cuda()
        batch_x1 = batch_x1.cuda()
        batch_y=batch_y.cuda()
        list_x=[batch_x,batch_x1]
        y_pred = SFT_model(list_x)
        loss = criterion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice_acc.append(loss)
    print('MCE number',len(dice_acc))
    print('epoch-----------',epoch)
    print('dice score loss',(torch.mean(torch.stack(dice_acc))))
    mse_epoch.append((torch.mean(torch.stack(dice_acc))).item())

###evlauate step  (training step 11:03-)
    if epoch % 1==0:
        torch.save(SFT_model.state_dict(), 'SFT_epoch'+str(epoch)+'.pth')

    with torch.no_grad():
        #batch_size_val=30
        if epoch % 1 == 0:
            SFT_model.eval()
            val_psnr_list=list()
            for step, (batch_x,batch_x1,batch_y) in enumerate(test_loader):
                val_data=batch_x.cuda()
                val_mask_batch=batch_y.cuda()
                val_data_1 = batch_x1.cuda()
                val_input = [val_data, val_data_1]
                pred_val = SFT_model(val_input)

                pred_val=(pred_val).cpu().numpy()
                val_mask_batch=val_mask_batch.cpu().numpy()

                val_loss=psnr_loss(pred_val,val_mask_batch)
                val_psnr_list.append(val_loss)
            print('val psnr loss', np.mean(val_psnr_list))
            psnr_epoch.append(np.mean(val_psnr_list))
            np.save('psnr_epoch.npy',psnr_epoch)
            np.save('mse_epoch.npy', mse_epoch)