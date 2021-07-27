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
import cv2


def test_ssn(model_path, data_path, save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #net_ = StereoSiamNet()
    if torch.cuda.is_available():
        net_ = torch.load(model_path)
    else:
        net_ = torch.load(model_path, map_location=torch.device('cpu'))
    net_ = net_.to(device)
    net_.eval()
    #print(net_.dump_info())
    # load dataset
    counter_ = 0
    data_loader = PairedGraySet(data_path, 1, 1, shuffle=False)
    loss_o = [] # loss for original input
    loss_t = [] # loss for texture features
    loss_d1 = [] # loss for decoded outputs#1
    loss_d2 = [] # loss for decoded outputs#2
    for i_epo, i_itr, im1, im2 in data_loader:
        if im1 is None or im2 is None:
            break
        counter_ += 1
        x1 = torch.Tensor(im1).to(device)
        x2 = torch.Tensor(im2).to(device)
        
        x1r, x2r, t1, t2 = net_(x1, x2)
        
        x1r = x1r.detach().cpu().numpy()
        x2r = x2r.detach().cpu().numpy()
        
        t1 = t1.detach().cpu().numpy()*80
        t2 = t2.detach().cpu().numpy()*80
        
        # evaluate the model on texture and decoder of L1 loss, L2 loss
        loss_o.append(np.mean(np.abs(im1 - im2))/np.abs(im1.mean() + im2.mean()))
        loss_t.append(np.mean(np.abs(t1 - t2))/np.abs(t1.mean() + t2.mean()))
        loss_d1.append(np.mean(np.abs(im1 - x1r))/np.abs(im1.mean() + x1r.mean()))
        loss_d2.append(np.mean(np.abs(im2 - x2r))/np.abs(im2.mean() + x2r.mean()))
        
        x1r = np.maximum(0, np.minimum(255, x1r))
        x2r = np.maximum(0, np.minimum(255, x2r))
        
        t1 = np.uint8(np.maximum(0, np.minimum(255, t1))[0,0])
        t2 = np.uint8(np.maximum(0, np.minimum(255, t2))[0,0])

        # save texture map
        cv2.imwrite(save_path + '/%d_t1.png' % counter_, t1)
        cv2.imwrite(save_path + '/%d_t2.png' % counter_, t2)
        
        # save reconstructed 
        x1_np = np.uint8(im1[0,0])
        x2_np = np.uint8(im2[0,0])
        x1r_np = np.uint8(x1r[0,0])
        x2r_np = np.uint8(x2r[0,0])

        cv2.imwrite(save_path + '/%d_x1o.png'%counter_, x1_np)
        cv2.imwrite(save_path + '/%d_x2o.png'%counter_, x2_np)
        cv2.imwrite(save_path + '/%d_x1r.png'%counter_, x1r_np)
        cv2.imwrite(save_path + '/%d_x2r.png'%counter_, x2r_np)
        
        '''
        # save the change of pixels
        x1_ratio = np.float32(x1r_np) / np.float32(np.maximum(x2_np, 1))
        x2_ratio = np.float32(x2r_np) / np.float32(np.maximum(x1_np, 1))
        x1_ratio = np.uint8(x1_ratio * 100 / np.mean(x1_ratio))
        x2_ratio = np.uint8(x2_ratio * 100 / np.mean(x2_ratio))
        cv2.imwrite('test_result/%d_x1_ratio.png'%counter_, x1_ratio)
        cv2.imwrite('test_result/%d_x2_ratio.png'%counter_, x2_ratio)
        
        # save the bias
        x1_bias = np.uint8(np.abs(x1r_np - x2_np)) * 8
        x2_bias = np.uint8(np.abs(x2r_np - x1_np)) * 8
        cv2.imwrite('test_result/%d_x1_bias.png'%counter_, x1_bias)
        cv2.imwrite('test_result/%d_x2_bias.png'%counter_, x2_bias)
        '''
    # print the final statistics
    print("loss_o: %8.5f +/- %8.5f" % (np.array(loss_o).mean(), np.array(loss_o).std()))
    print("loss_t: %8.5f +/- %8.5f" % (np.array(loss_t).mean(), np.array(loss_t).std()))
    print("loss_d1: %8.5f +/- %8.5f" % (np.array(loss_d1).mean(), np.array(loss_d1).std()))
    print("loss_d2: %8.5f +/- %8.5f" % (np.array(loss_d2).mean(), np.array(loss_d2).std()))

    print('test done!')



if __name__ == '__main__':
    model_path = 'ssn_bremen-cross_decoding-1950.pth'
    data_dir = '../Datasets/SSN/test_bremen_wh02-121_in/'
    save_dir = '../Datasets/SSN/test_bremen_wh02-121_out/20210621/Bremen-WH02-121'
    test_ssn(model_path, data_dir, save_dir)