import torch
import skimage.io
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

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)

TRAIN = 0
VAL = 1
TEST = 2

class WordsDataset(Dataset):
    def __init__(self, data_dir, mode):
        super(WordsDataset, self).__init__()
        if mode == TRAIN:
            path = os.path.join(data_dir, 'train/')
            #data_len = 10000
            data_len = 1000
        elif mode == VAL:
            path = os.path.join(data_dir, 'val/')
            #data_len = 2000
            data_len = 200
        elif mode == TEST:
            path = os.path.join(data_dir, 'test/')
            #data_len = 2000
            data_len = 200
        self.data_len = data_len
        self.observations = []
        self.labels = []
        for i in range(data_len):
            tmp_path = os.path.join(path, str(i))
            label_path = os.path.join(tmp_path, 'label.txt')
            with open(label_path, 'r') as fin:
                self.labels.append(torch.from_numpy(np.array([int(label.strip()) for label in fin.readlines()])))
            datum = []
            for j in range(5):
                img_path = os.path.join(tmp_path, '%d.png'%j)
                img = torch.from_numpy(skimage.io.imread(img_path, as_gray=True).flatten()).float()
                img.div_(255)
                datum.append(img)
            self.observations.append(torch.stack(datum))

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]

    def __len__(self):
        return self.data_len

def words_collate(batch_info):
    imgs = torch.cat([info[0] for info in batch_info])
    labels = torch.cat([info[1] for info in batch_info])
    return imgs, labels

def prepare_model():
    hidden_size = 128
    model = nn.Sequential(
        nn.Linear(28*28, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 26),
    )
    return model

def compute_obj(model, batch):
    imgs = batch[0]
    labels = Variable(batch[1])
    inp = model(Variable(imgs))
    return torch.nn.MultiMarginLoss()(inp, labels)

def test_on_batch(model, batch):
    imgs = batch[0]
    labels = batch[1]
    results = model(Variable(imgs))
    vals, guesses = results.max(1, keepdim=True)
    guesses = guesses.view(-1, 5)
    labels = labels.view(-1, 5)
    result = (guesses.data == labels).float()
    char_acc = result.sum()
    word_acc = (result.sum(1) == 5).sum()
    return char_acc, word_acc


def test(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=words_collate)
    model.eval()
    char_acc, word_acc = 0,0
    num_data = 0.0
    for batch in dataloader:
        new_char_acc, new_word_acc = test_on_batch(model, batch)
        char_acc += new_char_acc
        word_acc += new_word_acc
        num_data += len(batch[0])
    return char_acc/num_data, word_acc*5/num_data

def save_classifier(model, classifier_path):
    torch.save(model, classifier_path) 

def train(model, train_data, params):
    l_rate = params.get('l_rate', 1e-4)
    checkpoint_dir = params.get('checkpoint_dir', 'tmp/')
    batch_size = params.get('batch_size', 10)
    training_scheduler = params.get('training_scheduler', lambda opt: 
            torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda epoch:1.0/math.sqrt(epoch) if epoch > 0 else 1.0))
    num_epochs = params.get('num_epochs', 20)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=words_collate)

    model_optimizer = torch.optim.SGD(model.parameters(), lr = params['l_rate'])
    if training_scheduler is not None:
        training_scheduler = training_scheduler(model_optimizer)
    end = start = 0 
    epoch = 0
    train_results = []
    while epoch < num_epochs:
        epoch += 1
        print("EPOCH", epoch, (end-start))
        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
        model.train()
        for batch_ind,batch in enumerate(train_data_loader):
            obj = compute_obj(model, batch)
            print("\tBATCH %d OF %d: %f"%(batch_ind+1, len(train_data_loader), obj.data[0]))
            obj.backward()
            model_optimizer.step()

        end = time.time()
        if epoch%5 == 0:
            train_results.append(test(model, train_data, batch_size))
            print("TRAIN RESULTS: ",train_results[-1])
    return train_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pretrain alexnet on tagging data')
    parser.add_argument('data_directory')
    parser.add_argument('working_dir')
    parser.add_argument('--classifier_path')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--batch_size', type=int)

    train_params = {}
    args = parser.parse_args()
    train_data = WordsDataset(args.data_directory, TRAIN)
    val_data = WordsDataset(args.data_directory, TEST)

    if args.num_epochs is not None:
        train_params['num_epochs'] = args.num_epochs
    if args.l_rate is not None:
        train_params['l_rate'] = args.l_rate
    if args.batch_size is not None:
        train_params['batch_size'] = args.batch_size
    if args.no_l_rate_decay:
        train_params['training_scheduler'] = None

    model = prepare_model()
    test_result = test(model, val_data, args.batch_size)
    train(model, train_data,train_params)
    print("STARTING TEST")
    test_result = test(model, val_data, args.batch_size)
    print("VALIDATION RESULT: ",test_result)

    save_classifier(model, args.classifier_path)


    dataloader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=words_collate)
    for batch in dataloader:
        imgs = batch[0]
        inp = model(Variable(imgs))
        print(inp[0, :])
        break
