# resolve the package root path
import os
parent = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.dirname(parent)

os.environ['CUDA_VISIBLE_DEVICES']='3'

import sys
sys.path.append(package_root)
from model import StereoSiamNet
from dataset import PairedGraySet

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim


def train_ssn(model_path, model_name, data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #net_ = StereoSiamNet()
    net_ = torch.load('ssn_bremen-cross_decoding-150.pth')
    net_ = net_.to(device)
    #print(net_.dump_info())
    loss_fn = nn.SmoothL1Loss(reduction='mean')
    learning_rate = 1e-3
    #print(net_.parameters)
    opt_ = torch.optim.Adam(net_.parameters(), lr=learning_rate)
    num_epoch = 2000
    batch_size = 8 #8
    print_freq = 100 #100
    save_freq = 50 # 50
    # load dataset
    data_loader = PairedGraySet('../Datasets/SSN/Bremen/', num_epoch, batch_size)
    net_.train()
    counter_ = 0
    for i_epo, i_itr, im1, im2 in data_loader:
        if im1 is None or im2 is None:
            break
        counter_ += 1
        x1 = torch.Tensor(im1).to(device)
        x2 = torch.Tensor(im2).to(device)
        x1r, x2r, _, _ = net_(x1, x2)
        loss_ = loss_fn(x1, x1r) + loss_fn(x2, x2r)
        opt_.zero_grad()
        loss_.backward()
        opt_.step()
        if counter_ % print_freq == 0:
            print('Epoch: %03d Iter: %5d Loss %8.5f' % (i_epo, i_itr, loss_.item()))
        if i_epo > 0 and (i_epo % save_freq == 0) and (i_itr==0):
            torch.save(net_, '%s-%03d.pth' % (model_name, i_epo))
            print('model saved.')
    torch.save(net_, '%s-%03d.pth' % (model_name, num_epoch))
    print('model saved.')
    print('training done.')



if __name__ == '__main__':
    model_dir = '../Models/SSN/Bremen/'
    model_name = 'ssn_bremen-cross_decoding'
    data_dir = '../Datasets/SSN/Bremen/'
    train_ssn(model_dir, model_name, data_dir)
    print('training done!')
    print('trained model saved as: ' + os.path.join(model_dir + model_name))