import numpy as np
import torch
import torchvision.datasets as D
import torchvision.transforms as T
import glob
import cv2
from collections import Iterator

max_image_size = (640, 480)

class PairedGraySet(Iterator):
    def __init__(self, root_path, num_epoc, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.num_epoc = num_epoc
        self.shuffle = shuffle
        self.epoc = 0
        self.iter = 0
        self.im1 = None
        self.im2 = None
        # load images
        files_main = glob.glob(root_path + '/*/*/*_siam_main.png')
        files_main.sort()
        files_aux = glob.glob(root_path + '/*/*/*_siam_aux.png')
        files_aux.sort()
        assert len(files_aux) == len(files_main)
        assert len(files_main) > batch_size
        # try to load all images into memory
        n_pairs = len(files_main)
        self.images_main = [None]*n_pairs
        self.images_aux = [None]*n_pairs
        for i in range(len(files_main)):
            main_ = cv2.imread(files_main[i], -1) # gray image
            aux_ = cv2.imread(files_aux[i], -1) # gray image
            assert len(main_.shape)==2
            assert len(aux_.shape)==2
            assert main_.shape[0] == aux_.shape[0]
            assert main_.shape[1] == aux_.shape[1]
            self.images_main[i] = cv2.resize(main_, max_image_size, cv2.INTER_LINEAR)
            self.images_aux[i] = cv2.resize(aux_, max_image_size, cv2.INTER_LINEAR)

        # read images from root path, and determine how many
        # batches to run till an epoc is running out.
        self.num_samples = n_pairs
        self.max_iter = self.num_samples // self.batch_size

        # the first index sequence
        if self.shuffle:
            self.ind_seq = np.random.permutation(np.arange(start=0, stop=self.num_samples))
        else:
            self.ind_seq = np.arange(start=0, stop=self.num_samples)
        # preallocation for speed
        h, w = self.images_main[0].shape
        self.im1 = np.zeros([self.batch_size, 1, h, w], np.float32)
        self.im2 = np.zeros([self.batch_size, 1, h, w], np.float32)

    def __next__(self):
        cur_epoc = self.epoc
        cur_iter = self.iter
        if cur_epoc >= self.num_epoc:
            print('training is over, return None!')
            return cur_epoc, cur_iter, None, None
        else:
            # make batch of image pairs 
            beg_ = int(cur_iter*self.batch_size)
            end_ = beg_ + self.batch_size
            for pair_id in range(beg_, end_):
                self.im1[pair_id-beg_, 0, :,:] = self.images_main[self.ind_seq[pair_id]][:,:]
                self.im2[pair_id-beg_, 0, :,:] = self.images_aux[self.ind_seq[pair_id]][:,:]
        # update state to the next
        self.iter += 1
        if self.iter == self.max_iter:
            self.epoc += 1
            self.iter = 0
            # shuffle the index sequence
            if self.shuffle:
                self.ind_seq = np.random.permutation(np.arange(start=0, stop=self.num_samples))
        return cur_epoc, cur_iter, self.im1, self.im2