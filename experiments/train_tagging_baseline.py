import torch
import math
import random
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.optim.lr_scheduler
import torch.cuda
import argparse, os
from torchvision import transforms
import torchvision.models
from PIL import Image
import time

from deepstruct.models import *
from deepstruct.datasets import *
import deepstruct.models.modelconf

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

TRAIN = 0
VAL = 1
TEST = 2

FULL = 0
R1 = 1

NUM_TRAIN = 10000
NUM_TEST = 10000
NUM_VAL = 5000

order_full = {'structures.txt':0,
        'animals.txt':1,
        'transport.txt':2,
'food.txt':3,
'portrait.txt':4,
'sky.txt':5,
'female.txt':6,
'male.txt':7,
'flower.txt':8,
'people.txt':9,
'river.txt':10,
'sunset.txt':11,
'baby.txt':12,
'plant_life.txt':13,
'indoor.txt':14,
'car.txt':15,
'bird.txt':16,
'dog.txt':17,
'tree.txt':18,
'sea.txt':19,
'night.txt':20,
'lake.txt':21,
'water.txt':22,
'clouds.txt':23}

order_r1 = {'baby_r1.txt':0,
            'bird_r1.txt':1,
            'car_r1.txt':2,
            'clouds_r1.txt':3,
            'dog_r1.txt':4,
            'female_r1.txt':5,
            'flower_r1.txt':6,
            'male_r1.txt':7,
            'night_r1.txt':8,
            'people_r1.txt':9,
            'portrait_r1.txt':10,
            'river_r1.txt':11,
            'sea_r1.txt':12,
            'tree_r1.txt':13,
}


class FlickrTaggingDataset(BaseDataset):
    def __init__(self, dataset, images_folder, save_img_file, annotations_folder, save_label_file, mode, load=False, masks_path=None):
        if dataset == FULL:
            order = order_full
        elif dataset == R1:
            order = order_r1
        else:
            raise Exception('DATASET MUST BE EITHER FULL OR R1')
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        if load:
            print("LOADING PRECOMPUTED IMAGES")
            self.images = torch.load(save_img_file)
            self.labels = torch.load(save_label_file)
        else:
            print("LOADING IMAGES")
            if dataset == FULL:
                self.annotations = [None]*24
            else:
                self.annotations = [None]*14
            for annotation_file in os.listdir(annotations_folder):
                if dataset == FULL and '_r1' in annotation_file:
                    continue
                elif dataset == R1 and '_r1' not in annotation_file:
                    continue
                elif 'README' in annotation_file:
                    continue
                vals = set()
                fin = open(os.path.join(annotations_folder, annotation_file), 'r')
                for line in fin:
                    vals.add(int(line.strip())-1)
                self.annotations[order[annotation_file]] = vals
            self.img_folder = images_folder
            self.img_files = [img_file for img_file in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, img_file)) and 'jpg' in img_file]
            print("NUM IMG FILES: ",len(self.img_files))
            self.img_files.sort(key=lambda name: int(name[2:name.find('.jpg')]))

            if mode == TRAIN:
                self.img_files = self.img_files[:NUM_TRAIN]
            elif mode == TEST:
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN+NUM_TEST]
            else:
                self.img_files = self.img_files[NUM_TRAIN+NUM_TEST:]
            self.images = [None]*len(self.img_files)
            self.labels = []
            for img_file in self.img_files:
                path = os.path.join(self.img_folder, img_file)
                with open(path, 'rb') as f:
                    with Image.open(f) as raw_img:
                        img = self.transform(raw_img.convert('RGB'))
                img_no = int(img_file[2:img_file.find('.jpg')]) - 1
                if mode == TRAIN:
                    img_ind = img_no
                elif mode == TEST:
                    img_ind = img_no - NUM_TRAIN
                else:
                    img_ind = img_no - NUM_TRAIN - NUM_TEST
                label = [0]*len(self.annotations)
                for i,annotation in enumerate(self.annotations):
                    if img_no in annotation:
                        label[i] = 1
                self.images[img_ind] = img
                self.labels.append(label)
            if save_img_file is not None:
                torch.save(self.images, save_img_file)
            if save_label_file is not None:
                torch.save(self.labels, save_label_file)
        super(FlickrTaggingDataset, self).__init__(len(self.images), masks_path)

    def __getitem__(self, idx):
        stuff = super(FlickrTaggingDataset, self).__getitem__(idx)
        return (self.labels[idx], self.images[idx]) + stuff

class FlickrPotentialModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(FlickrPotentialModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.img_size = 224
        self.pair_inds = args_dict['pair_inds']
        if len(node_regions) > 0:
            self.unary_model = torchvision.models.alexnet(pretrained=True)
            
            # There may be a better way to replace the FC layer, but this
            # is the best I could figure out
            tmp = list(self.unary_model.classifier)
            tmp[-1] = nn.Linear(4096, 2*len(node_regions))
            self.unary_model.classifier = nn.Sequential(*tmp)
        if len(pair_regions) > 0:
            #V2:
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).uniform_(-0.1, 0.1))
            #V3:
            #self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(1.0))

    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(observations).float()

    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            result[:, :len(self.node_regions)*self.num_vals] = self.unary_model(self.observations)
        if len(self.pair_regions) > 0:
            result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,1)
        return result


def save_features(model, dataset, batch_size, features_path):
    unary_model = model.graph.potential_model.unary_model
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_batch)
    results = []
    for batch in dataloader:
        imgs = Variable(batch.observations).float()
        results.append(unary_model(imgs).data.cpu())
    results = torch.cat(results)
    torch.save(results, features_path)


def calculate_hamming_loss(dataset, found):
    loss = 0.0
    correct = [datum[0] for datum in dataset]
    for true, guess in zip(correct, found):
        for val1, val2 in zip(true, guess):
            loss += abs(val1 - val2)
    return loss/len(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pretrain alexnet on tagging data')
    parser.add_argument('dataset_type', choices=['full', 'r1'])
    parser.add_argument('working_dir')
    parser.add_argument('--img_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--train_img_file')
    parser.add_argument('--test_img_file')
    parser.add_argument('--val_img_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--test_label_file')
    parser.add_argument('--val_label_file')
    parser.add_argument('--train_features_path')
    parser.add_argument('--test_features_path')
    parser.add_argument('--val_features_path')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--val_interval', type=int, default = 10)
    parser.add_argument('--use_loss_aug', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    if args.gpu:
        modelconf.use_gpu()

    def scaled_hamming_diff(true_val, other_val):
        return abs(true_val - other_val)
    val_scheduler = lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.5)
    
    if args.img_dir == None and args.train_img_file == None:
        print("ERROR: Must specify either an image directory or image file")
        sys.exit(1)
    if args.label_dir == None and args.train_label_file == None:
        print("ERROR: Must specify either a label directory or label file")
        sys.exit(1)
    if args.img_dir == None:
        load = True
    else:
        load = False
    print("LOAD: ",load)
    if args.dataset_type == 'full':
        dataset_type = FULL
        print("RUNNING WITH FULL LABELS")
    else:
        dataset_type = R1
        print("RUNNING WITH R1 LABELS")
    train_data = FlickrTaggingDataset(dataset_type, args.img_dir, args.train_img_file, args.label_dir, args.train_label_file, TRAIN, load=load)
    test_data = FlickrTaggingDataset(dataset_type, args.img_dir, args.test_img_file, args.label_dir, args.test_label_file, TEST, load=load)
    val_data = FlickrTaggingDataset(dataset_type, args.img_dir, args.val_img_file, args.label_dir, args.val_label_file, VAL, load=load)
    train_masks_path = None
    test_masks_path = None

    train_params = {
        'batch_size':10000, 
        'num_epochs':50,
        'l_rate':1e-2, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':0.0, 
        'mp_itrs':100,
        'use_loss_augmented_inference':False,
        'inf_loss':scaled_hamming_diff,
        'val_scheduler':val_scheduler, 
        'checkpoint_dir':args.working_dir,
        'task_loss':calculate_hamming_loss,
        'test_data':val_data,
        'train_masks_path':train_masks_path,
        'test_masks_path':test_masks_path,
        'val_interval':args.val_interval,
    }


    if args.num_epochs is not None:
        train_params['num_epochs'] = args.num_epochs
    if args.l_rate is not None:
        train_params['l_rate'] = args.l_rate
    if args.batch_size is not None:
        train_params['batch_size'] = args.batch_size
    if args.no_l_rate_decay:
        train_params['training_scheduler'] = None
    if args.weight_decay is not None:
        train_params['weight_decay'] = args.weight_decay
    if args.verbose is not None:
        train_params['verbose'] = args.verbose
    if args.use_loss_aug:
        train_params['use_loss_augmented_inference'] = True

    if dataset_type == FULL:
        nodes = list(range(24))
    else:
        nodes = list(range(14))
    pairs = []
    pair_inds = {}
    ind = 0
    for node1 in nodes:
        for node2 in nodes[node1+1:]:
            pairs.append((node1, node2))
            pair_inds[(node1, node2)] = ind
            ind += 1

    args_dict = {'pair_inds':pair_inds}

    full_graph = Graph(nodes, pairs, 2, FlickrPotentialModel, args_dict, False)
    model = PairwiseModel([full_graph], len(nodes), 2, train_params)
    print(train_params) 

    start = time.time()
    obj,  train_results, return_vals = model.train(train_data, None, train_params)
    end = time.time()
    train_time = (end-start)
    print("STARTING TEST")
    start = time.time()
    test_results = model.test(test_data, train_params)
    end = time.time()
    test_time = end-start
    train_test_results = model.test(train_data, train_params)
    train_loss = calculate_hamming_loss(train_data, train_test_results)
    test_loss = calculate_hamming_loss(test_data, test_results)
    print("TRAIN LOSS: ",train_loss)
    print("TEST LOSS: ",test_loss)
    print("TRAIN TIME: ",train_time)
    print("TEST TIME: ",test_time)

    save_features(model, train_data, args.batch_size, args.train_features_path)
    save_features(model, val_data, args.batch_size, args.val_features_path)
    save_features(model, test_data, args.batch_size, args.test_features_path)

