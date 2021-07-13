import sys
sys.path.append('..')
from model import StereoSiamNet

import os.path
import numpy as np
import torchvision.transforms as T

def train_ssl(model_path, model_name, data_path):
    net_ = StereoSiamNet()
    pass


if __name__ == '__main__':
    model_dir = '../Models/SSN/Bremen/'
    model_name = 'ssn_bremen-cross_decoding.pth'
    data_dir = '../Datasets/SSN/Bremen/'
    train_ssl(model_dir, model_name, data_dir)
    print('training done!')
    print('trained model saved as: ' + os.path.join(model_dir + model_name))