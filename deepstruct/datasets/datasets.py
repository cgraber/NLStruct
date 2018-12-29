import skimage.io
import numpy as np
import scipy.io as sio
import torch
import torch.cuda
from deepstruct.models import modelconf
from torch.utils.data import Dataset
import sys
import random

from torch.autograd import Variable
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)



class Batch(object):
    def __init__(self, lambdas, max_globals, observations, data, data_masks, msgs, loss_augmentations, lambda_windows = None, max_global_windows = None, other_obs = None):
        self.size = len(observations)
        self.lambdas = lambdas
        self.max_globals = max_globals
        self.observations = observations
        self.data = data
        self.data_masks = data_masks
        self.msgs = msgs
        self.loss_augmentations = loss_augmentations
        self.lambda_windows = lambda_windows
        self.max_global_windows = max_global_windows
        self.other_obs = other_obs

    def __len__(self):
        return self.size

    def __getitem__(self, key):
        return [self.data[key]]

def collate_batch(batch_info):
    data = [info[0] for info in batch_info]
    '''
    if modelconf.USE_GPU:
        obs = [info[1].cuda(async=True) for info in batch_info]
    else:
    '''
    obs = torch.stack([info[1] for info in batch_info])
    if modelconf.USE_GPU:
        obs = obs.cuda(async=True)
    masks = [torch.stack([info[2][i] for info in batch_info]) for i in range(len(batch_info[0][2]))]
    if modelconf.USE_GPU:
        masks = [mask.cuda(async=True) for mask in masks]
    lambdas = [info[3] for info in batch_info]
    max_globals = [info[4] for info in batch_info]
    messages = [info[5] for info in batch_info]
    loss_augmentations = [info[6] for info in batch_info]
    lambda_windows = [info[7] for info in batch_info]
    max_global_windows = [info[8] for info in batch_info]
    if len(batch_info[0]) >= 10:
        other_obs = torch.stack([info[9] for info in batch_info])
        if modelconf.USE_GPU:
            other_obs = other_obs.cuda(async=True)
    else:
        other_obs = None
    return Batch(lambdas, max_globals, obs, data, masks, messages, loss_augmentations, lambda_windows, max_global_windows, other_obs)

class BaseDataset(Dataset):
    def __init__(self, data_len, masks_path):
        super(BaseDataset, self).__init__()
        self.data_len = data_len
        if masks_path == None:
            self.data_masks = None # Will be filled out by model
        else:
            print("LOADING DATA MASKS")
            self.data_masks = torch.load(masks_path)
            print("DONE")

    def init(self, num_graphs, num_messages):
        self.messages = [np.zeros(num_messages, dtype=float) for _ in range(self.data_len)]
        #self.lambdas = [Variable(modelconf.tensor_mod.FloatTensor(num_graphs).fill_(0), requires_grad=False)
        #                for _ in range(self.data_len)]
        self.lambdas = [Variable(modelconf.tensor_mod.FloatTensor(num_graphs).fill_(1), requires_grad=False)
                        for _ in range(self.data_len)]
        self.max_globals = [Variable(modelconf.tensor_mod.FloatTensor(num_graphs).fill_(1), requires_grad=True)
                            for _ in range(self.data_len)]
        self.loss_augmentations = [None for _ in range(self.data_len)]
        self.lambda_windows = [[lambd] for lambd in self.lambdas]
        self.max_global_windows = [[mg] for mg in self.max_globals]
        self.ord = list(range(len(self.messages)))

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.data_masks == None:
            return self.lambdas[self.ord[idx]], self.max_globals[self.ord[idx]], self.messages[self.ord[idx]], self.loss_augmentations[self.ord[idx]], self.lambda_windows[self.ord[idx]], self.max_global_windows[self.ord[idx]]
        else:
            return self.data_masks[self.ord[idx]], self.lambdas[self.ord[idx]], self.max_globals[self.ord[idx]], self.messages[self.ord[idx]], self.loss_augmentations[self.ord[idx]], self.lambda_windows[self.ord[idx]], self.max_global_windows[self.ord[idx]]

    def save_checkpoint(self, file_path):
        info = [self.lambdas, self.max_globals, self.messages]
        torch.save(info, file_path)

    def load_checkpoint(self, file_path):
        self.lambdas, self.max_globals, self.messages = torch.load(file_path)

    def shuffle(self):
        random.shuffle(self.ord)

