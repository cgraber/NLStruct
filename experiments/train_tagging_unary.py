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


class FlickrTaggingDataset(Dataset):
    def __init__(self, dataset, images_folder, save_img_file, annotations_folder, save_label_file, mode, load=False):
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
            self.img_files = [img_file for img_file in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, img_file))]
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
    def __getitem__(self, idx):
        return self.images[idx], torch.FloatTensor(self.labels[idx])

    def __len__(self):
        return len(self.images)

def prepare_alexnet(dataset_type):
    model = torchvision.models.alexnet(pretrained=True)
    tmp = list(model.classifier)
    if dataset_type == FULL:
        tmp[-1] = nn.Linear(4096, 24)
    else:
        tmp[-1] = nn.Linear(4096, 14)
    model.classifier = nn.Sequential(*tmp)
    model = model.cuda()
    return model

def strip_classifier(model):
    tmp = list(model.classifier)
    tmp = tmp[:-1]
    model.classifier = nn.Sequential(*tmp)

def compute_obj(model, batch):
    imgs = batch[0].cuda(async=True)
    labels = Variable(batch[1].cuda(async=True))
    inp = model(Variable(imgs))
    return torch.nn.BCEWithLogitsLoss()(inp, labels)

def test_on_batch(model, batch):
    imgs = batch[0].cuda(async=True)
    labels = batch[1].cuda(async=True)
    results = nn.Sigmoid()(model(Variable(imgs))).round()
    return (results.data-labels).abs().sum()

def test(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    model.eval()
    results = 0
    num_data = 0.0
    for batch in dataloader:
        results += test_on_batch(model, batch)
        num_data += len(batch[0])
    return results/num_data

def save_classifier(model, classifier_path):
    classifier = list(model.classifier)[-1].cpu()
    torch.save(classifier, classifier_path) 

def save_features(model, dataset, batch_size, features_path):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    results = []
    for batch in dataloader:
        imgs = Variable(batch[0].cuda(async=True))
        results.append(model(imgs).data.cpu())
    results = torch.cat(results)
    torch.save(results, features_path)

def train(model, train_data, params):
    l_rate = params.get('l_rate', 1e-4)
    weight_decay = params.get('weight_decay', 0)
    checkpoint_dir = params.get('checkpoint_dir', 'tmp/')
    batch_size = params.get('batch_size', 10)
    validation_data = params.get('validation_data', None)
    verbose = params.get('verbose', False)
    training_scheduler = params.get('training_scheduler', lambda opt: 
            torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda epoch:1.0/math.sqrt(epoch) if epoch > 0 else 1.0))
    num_epochs = params.get('num_epochs', 20)
    task_loss = params.get('task_loss', None)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)

    model_optimizer = torch.optim.SGD(model.parameters(), lr = params['l_rate'], weight_decay = weight_decay)
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
        if epoch%20 == 0 or verbose:
            train_score = test(model, train_data, batch_size)
            if validation_data is not None:
                val_score = test(model, validation_data, batch_size)
                train_results.append((train_score, val_score))
                print("TRAIN RESULTS: ",train_score)
                print("VALIDATION RESULTS: ",val_score)
            else:
                train_results.append(train_score)
                print("TRAIN RESULTS: ",train_results[-1])
    return train_results

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
    parser.add_argument('--classifier_path')
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--verbose', action='store_true')

    train_params = {}
    args = parser.parse_args()
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
    train_params['validation_data'] = val_data

    model = prepare_alexnet(dataset_type)
    test_result = test(model, val_data, args.batch_size)
    train(model, train_data,train_params)
    print("STARTING TEST")
    test_result = test(model, val_data, args.batch_size)
    print("VALIDATION RESULT: ",test_result)

    save_classifier(model, args.classifier_path)

    strip_classifier(model)
    save_features(model, train_data, args.batch_size, args.train_features_path)
    save_features(model, val_data, args.batch_size, args.val_features_path)
    save_features(model, test_data, args.batch_size, args.test_features_path)

