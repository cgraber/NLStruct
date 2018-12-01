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

NUM_TRAIN = 196
NUM_TEST = 66
NUM_VAL = 66

class HorsesSegDataset(Dataset):
    def __init__(self, images_folder, label_folder, mode, load=False):
        self.img_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.label_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            #transforms.ToTensor(),
        ])

        if load:
            print("LOADING PRECOMPUTED IMAGES")
            self.images = torch.load(save_img_file)
            self.labels = torch.load(save_label_file)
        else:
            print("LOADING IMAGES")
            self.img_folder = images_folder
            self.img_files = [img_file for img_file in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, img_file)) and 'preprocessed' not in img_file]
            self.img_files.sort(key=lambda name: int(name[5:name.find('.jpg')]))

            self.label_folder = label_folder
            self.label_files = [label_file for label_file in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, label_file)) and 'preprocessed' not in label_file]
            self.label_files.sort(key=lambda name: int(name[5:name.find('.jpg')]))


            if mode == TRAIN:
                self.img_files = self.img_files[:NUM_TRAIN]
                self.label_files = self.label_files[:NUM_TRAIN]
            elif mode == VAL:
                self.img_files = self.img_files[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
                self.label_files = self.label_files[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
            else:
                self.img_files = self.img_files[NUM_TRAIN+NUM_VAL:]
                self.label_files = self.label_files[NUM_TRAIN+NUM_VAL:]
            self.images = []
            for img_file in self.img_files:
                path = os.path.join(self.img_folder, img_file)
                with open(path, 'rb') as f:
                    with Image.open(f) as raw_img:
                        img = self.img_transform(raw_img.convert('RGB'))
                self.images.append(img)

            self.labels = []
            for label_file in self.label_files:
                path = os.path.join(self.label_folder, label_file)
                with open(path, 'rb') as f:
                    with Image.open(f) as raw_img:
                        label = np.array(self.label_transform(raw_img.convert('1'))).astype('int').flatten()
                self.labels.append(label)
            #fname = os.path.join(self.img_folder, 'preprocessed_imgs')
            #torch.save(self.images, fname)
            #fname = os.path.join(self.label_folder, 'preprocessed_labels')
            #torch.save(self.labels, fname)

    def __getitem__(self, idx):
        return self.images[idx], torch.LongTensor(self.labels[idx])

    def __len__(self):
        return len(self.images)
 


def prepare_model():
    alexnet = torchvision.models.alexnet(pretrained=True)
    end_feats = [alexnet.features[i] for i in range(6,len(alexnet.features))]
    features = nn.Sequential(
        alexnet.features[0],
        alexnet.features[1],
        #Skip max pool
        alexnet.features[3],
        alexnet.features[4],
        nn.MaxPool2d(kernel_size=3, stride=1),
        *end_feats
    )

    classifier = nn.Sequential(
        nn.Dropout(),
        nn.Conv2d(256, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Conv2d(4096, 4096, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(4096, 2, 1)
    )
    #self.upscore = nn.ConvTranspose2d(2, 2, 14, stride=10, bias=False)
    upscore = nn.ConvTranspose2d(2, 2, 14, stride=2, bias=False)
    model = nn.Sequential(features, classifier, upscore)
    model = model.cuda()

    return model

def compute_obj(model, batch):
    imgs = batch[0].cuda(async=True)
    labels = Variable(batch[1].view(-1).cuda(async=True))
    inp = model(Variable(imgs))
    inp = inp.permute(0,2,3,1).contiguous().view(-1, 2)

    return torch.nn.CrossEntropyLoss()(inp, labels)

def test_on_batch(model, batch):
    imgs = batch[0].cuda(async=True)
    true = batch[1].view(-1).numpy()
    val, ind = torch.max(model(Variable(imgs)).cpu().permute(0,2,3,1).contiguous().view(-1,2), 1)
    
    return (ind.data.numpy() == true).sum()

def test(model, dataset, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    model.eval()
    results = 0
    num_data = 0.0
    for batch in dataloader:
        results += test_on_batch(model, batch)
        num_data += batch[1].numpy().size
    return results/num_data

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
    momentum = params.get('momentum', 0)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)

    model_optimizer = torch.optim.SGD(model.parameters(), lr = params['l_rate'], weight_decay = weight_decay, momentum=momentum)
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

def save_features(model, dataset, batch_size, features_path):
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    results = []
    for batch in dataloader:
        imgs = Variable(batch[0].cuda(async=True))
        inp = model(imgs).permute(0,2,3,1).contiguous().view(len(batch[0]), -1)
        results.append(inp.data.cpu())
    results = torch.cat(results)
    torch.save(results, features_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Pretrain alexnet on tagging data')
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
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--load_data', action='store_true')

    train_params = {}
    args = parser.parse_args()

    train_data = HorsesSegDataset(args.img_dir, args.label_dir,  TRAIN, load=args.load_data)
    val_data = HorsesSegDataset(args.img_dir, args.label_dir, VAL, load=args.load_data)
    test_data = HorsesSegDataset(args.img_dir, args.label_dir, TEST, load=args.load_data)

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
    train_params['momentum'] = args.momentum

    model = prepare_model()
    test_result = test(model, val_data, args.batch_size)
    train(model, train_data,train_params)
    print("STARTING TEST")
    test_result = test(model, val_data, args.batch_size)
    print("VALIDATION RESULT: ",test_result)

    #save_classifier(model, args.classifier_path)

    save_features(model, train_data, args.batch_size, args.train_features_path)
    save_features(model, val_data, args.batch_size, args.val_features_path)
    save_features(model, test_data, args.batch_size, args.test_features_path)

