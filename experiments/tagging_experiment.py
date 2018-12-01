from deepstruct.models import *
from deepstruct.datasets import *
import argparse, os
from torchvision import transforms
import torchvision.models
from PIL import Image

import deepstruct.models.modelconf

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
        super(FlickrTaggingDataset, self).__init__(len(self.images), masks_path)
 

    def __getitem__(self, idx):
        stuff = super(FlickrTaggingDataset, self).__getitem__(idx)
        return (self.labels[idx], self.images[idx]) + stuff

class FlickrTaggingDataset_Features(BaseDataset):
    def __init__(self, dataset, feature_file, annotations_folder, save_label_file, mode, images_folder=None, load=False, masks_path=None):
        if dataset == FULL:
            order = order_full
        elif dataset == R1:
            order = order_r1
        else:
            raise Exception('DATASET MUST BE EITHER FULL OR R1')
        self.features = torch.load(feature_file)
        if load:
            print("LOADING PRECOMPUTED LABELS")
            self.labels = torch.load(save_label_file)
            print("DONE")
        else:
            print("LOADING LABELS")
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
                self.labels.append(label)
            print("DONE")
            if save_label_file is not None:
                torch.save(self.labels, save_label_file)
        super(FlickrTaggingDataset_Features, self).__init__(len(self.features), masks_path)

    def __getitem__(self, idx):
        stuff = super(FlickrTaggingDataset_Features, self).__getitem__(idx)
        return (self.labels[idx], self.features[idx, :]) + stuff

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
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(0.0))
        if 'finetune' in args_dict and args_dict['finetune'] == True:
            self.finetune = True
        else:
            self.finetune = False

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

class FlickrBaselineModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(FlickrBaselineModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        if len(node_regions) > 0:
            self.unary_model = nn.Linear(4096, len(node_regions))
            #TMP
            #self.unary_transformer = nn.Linear(len(node_regions), 2*len(node_regions))
            
        if len(pair_regions) > 0:
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions), 2).uniform_(-1.0, 1.0))
        self.pot_div_factor = args_dict['pot_div_factor']
        self.shift_pots = args_dict.get('shift_pots', False)
        if self.shift_pots:
            print("SHIFTING POTS")
        else:
            print("NOT SHIFTING POTS")

    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(observations).float()
    
    def load_classifier(self, path):
        self.unary_model = torch.load(path)

    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            pots = Variable(modelconf.tensor_mod.FloatTensor(len(self.node_regions)*self.num_observations, 2).fill_(0.0))
            tmp = self.unary_model(self.observations)
            if self.shift_pots:
                pots[:,0] = tmp.view(-1, 1)*-0.5
                pots[:,1] = tmp.view(-1, 1)*0.5
            else:
                #pots[:,0] = tmp.view(-1, 1)*-1/10.0
                pots[:,0] = tmp.view(-1, 1)*-1
                #pots[:,1] = tmp.view(-1, 1)/100.0
            result[:, :len(self.node_regions)*self.num_vals] = pots.view(self.num_observations, -1)
        if len(self.pair_regions) > 0:
            #result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations, 1)
            tmp = Variable(modelconf.tensor_mod.FloatTensor(len(self.pair_regions), 4))
            tmp[:, 0] = self.pair_model[:, 0]
            tmp[:, 1] = self.pair_model[:, 1]
            tmp[:, 2] = self.pair_model[:, 1]
            tmp[:, 3] = self.pair_model[:, 0]
            result[:, len(self.node_regions)*self.num_vals:] = tmp.view(1, -1).repeat(self.num_observations, 1)
        result /= self.pot_div_factor
        return result

class FlickrFixedModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(FlickrFixedModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
            
        if len(pair_regions) > 0:
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(1.0))
        self.pot_div_factor = args_dict['pot_div_factor']

    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(observations).float()
    
    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            result[:, :len(self.node_regions)*self.num_vals] = self.observations
        if len(self.pair_regions) > 0:
            result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations, 1)
        result /= self.pot_div_factor
        return result

            
def build_initialized_mlp_global_model_v1(num_graphs, params):
    if 'global_activation' in params:
        if params['global_activation'] == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params['global_activation'] == 'relu':
            activation = lambda: nn.ReLU()
        elif params['global_activation'] == 'hardtanh':
            activation = lambda: nn.Hardtanh()
        else:
            raise Exception("Activation type not valid: ",params['global_activation'])
    else:
        activation = lambda: nn.Sigmoid()
    layer1 = nn.Linear(num_graphs, params['global_hidden_size'])
    layer1.weight.data.fill_(0.0)
    for i in range(num_graphs):
        layer1.weight.data[i, i] = 1.0
    layer1.bias.data.fill_(0.0)

    layer2 = nn.Linear(params['global_hidden_size'], 1)
    layer2.weight.data.fill_(0.0)
    for i in range(num_graphs):
        layer2.weight.data[0, i] = 1.0
    layer2.bias.data.fill_(0.0)
    
    global_model = nn.Sequential(
                layer1,
                activation(),
                layer2
            )
    return global_model

def build_initialized_mlp_global_model_v2(num_graphs, params):
    if 'global_activation' in params:
        if params['global_activation'] == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params['global_activation'] == 'relu':
            activation = lambda: nn.ReLU()
        elif params['global_activation'] == 'hardtanh':
            activation = lambda: nn.Hardtanh()
        elif params['global_activation'] == 'tanh':
            activation = lambda: nn.Tanh()
        elif params['global_activation'] == 'leaky_relu':
            activation = lambda: nn.LeakyReLU(negative_slope=0.25)
        else:
            raise Exception("Activation type not valid: ",params['global_activation'])
    else:
        activation = lambda: nn.Sigmoid()
    num_hidden_layers = params.get('num_global_layers', 1)
    max_val = params.get('global_init_val', None)
    layer1 = nn.Linear(num_graphs, params['global_hidden_size'])
    if max_val is not None:
        layer1.weight.data.uniform_(-1*max_val, max_val)
        layer1.bias.data.fill_(max_val)

    layers = [layer1, activation()]

    for i in range(num_hidden_layers - 1):
        new_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
        if max_val is not None:
            new_layer.weight.data.uniform_(-1*max_val, max_val)
            new_layer.bias.data.fill_(max_val)
        layers.append(new_layer)
        layers.append(activation())

    top_layer = nn.Linear(params['global_hidden_size'], 1)
    if max_val is not None:
        top_layer.weight.data.uniform_(-1*max_val, max_val)
        top_layer.bias.data.fill_(0.0)
    layers.append(top_layer)
    
    global_model = nn.Sequential(*layers)
    return global_model

class GlobalModel_GT(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_GT, self).__init__()
        if 'global_activation' in params:
            if params['global_activation'] == 'sigmoid':
                activation = lambda: nn.Sigmoid()
            elif params['global_activation'] == 'relu':
                activation = lambda: nn.ReLU()
            elif params['global_activation'] == 'hardtanh':
                activation = lambda: nn.Hardtanh()
            elif params['global_activation'] == 'tanh':
                activation = lambda: nn.Tanh()
            elif params['global_activation'] == 'prelu':
                activation = lambda: nn.PReLU()
            elif params['global_activation'] == 'leaky_relu':
                activation = lambda: nn.LeakyReLU(negative_slope=0.25)
            elif params['global_activation'] == 'abs':
                activation = lambda: nn.LeakyReLU(negative_slope=-1.)
            elif params['global_activation'] == 'linear':
                activation = lambda: nn.LeakyReLU(negative_slope=1.)
            else:
                raise Exception("Activation type not valid: ",params['global_activation'])
        else:
            activation = lambda: nn.Sigmoid()
        num_hidden_layers = params.get('num_global_layers', 1)
        max_val = params.get('global_init_val', None)
        layers = []
        if params.get('first_ln', False):
            layers.append(nn.LayerNorm(2*num_graphs, elementwise_affine=False))
        layers.append(nn.Linear(2*num_graphs, params['global_hidden_size']))
        if max_val is not None:
            layers[-1].weight.data.uniform_(-1*max_val, max_val)
            layers[-1].bias.data.fill_(max_val)

        if params.get('global_batchnorm', None):
            layers.append(nn.BatchNorm1d(params['global_hidden_size']))
        elif params.get('global_layernorm', None):
            layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
        layers.append(activation())
        if params.get('use_dropout', False):
            layers.append(nn.Dropout())

        for i in range(num_hidden_layers - 1):
            new_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
            if max_val is not None:
                new_layer.weight.data.uniform_(-1*max_val, max_val)
                new_layer.bias.data.fill_(max_val)
            layers.append(new_layer)
            if params.get('global_batchnorm', None):
                layers.append(nn.batchnorm1d(params['global_hidden_size']))
            elif params.get('global_layernorm', None):
                layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
            layers.append(activation())
            if params.get('use_dropout', False):
                layers.append(nn.Dropout())
        self.use_global_top = params.get('use_global_top', False)
        if self.use_global_top:
            layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
        else:
            top_layer = nn.Linear(params['global_hidden_size'], 1)
            if max_val is not None:
                top_layer.weight.data.uniform_(-1*max_val, max_val)
                top_layer.bias.data.fill_(0.0)
            layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, pots, gt):
        inp = torch.cat([pots, gt], dim=1)
        result = self.global_model(inp)
        if self.use_global_top:
            result = result.sum(dim=1).unsqueeze(1)
        return result



def plot_task_losses(dir_name, model_name, return_vals):
    train_task_losses = return_vals['train_task_losses']
    test_task_losses = return_vals['test_task_losses']
    test_task_losses = list(zip(*test_task_losses))

    plt.figure()
    plt.clf()
    plt.plot(list(range(len(train_task_losses))), train_task_losses, label='Train Hamming Loss')
    plt.plot(test_task_losses[0], test_task_losses[1], label='Test Hamming Loss')
    plt.xlabel('Epoch')
    plt.legend()
    path = os.path.join(dir_name, 'task_losses_%s.pdf'%(model_name))
    plt.savefig(path)
    path = os.path.join(dir_name, 'task_losses_%s.png'%(model_name))
    plt.savefig(path)
    path = os.path.join(dir_name, 'train_task_losses_%s.csv'%(model_name))
    save_data(path, list(range(len(train_task_losses))), train_task_losses)
    path = os.path.join(dir_name, 'test_task_losses_%s.csv'%(model_name))
    save_data(path, test_task_losses[0], test_task_losses[1])

def hamming_diff(true_val, other_val):
    return abs(true_val - other_val)/10.0

def calculate_hamming_loss(dataset, found):
    loss = 0.0
    correct = [datum[0] for datum in dataset]
    for true, guess in zip(correct, found):
        for val1, val2 in zip(true, guess):
            loss += abs(val1 - val2)
    return loss/len(dataset)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run experiment for Flickr tagging dataset')
    parser.add_argument('dataset', choices=['full', 'r1'])
    parser.add_argument('model', choices=['pairwise', 'pairwise_linear', 'global_linear', 'global_quad', 'global_mlp'])
    parser.add_argument('working_dir')
    parser.add_argument('--img_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--train_feat_file')
    parser.add_argument('--val_feat_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--val_label_file')
    parser.add_argument('-p', '--pretrain')
    parser.add_argument('--load_classifier')
    parser.add_argument('--load_pots', action='store_true')
    parser.add_argument('--global_lr', type=float)
    parser.add_argument('--graph_lr', type=float)
    parser.add_argument('--train_interleaved_itrs', type=int, default=100)
    parser.add_argument('--test_interleaved_itrs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['linear', 'abs', 'leaky_relu', 'prelu', 'sigmoid', 'relu', 'hardtanh', 'tanh'])
    parser.add_argument('--global_bn', action='store_true')
    parser.add_argument('--global_ln', action='store_true')
    parser.add_argument('--train_max_globals_l_rate', type=float, default=1e-1)
    parser.add_argument('--train_lambda_l_rate', type=float, default=1e-1)
    parser.add_argument('--test_max_globals_l_rate', type=float, default=1e-1)
    parser.add_argument('--test_lambda_l_rate', type=float, default=1e-1)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--pair_only', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--load_masks', action='store_true')
    parser.add_argument('--train_masks_path')
    parser.add_argument('--val_masks_path')
    parser.add_argument('--mlp_init', choices=['v1', 'v2', 'v3', 'v4'])
    parser.add_argument('--use_loss_aug', action='store_true')
    parser.add_argument('--load_loss_aug', action='store_true')
    parser.add_argument('--loss_aug_save_path')
    parser.add_argument('--load_max_globals', action='store_true')
    parser.add_argument('--max_globals_save_path')
    parser.add_argument('--wide_top', action='store_true')
    parser.add_argument('--interleave', action='store_true')
    parser.add_argument('--use_pd', action='store_true', default=False)
    parser.add_argument('--pot_div_factor', type=float, default=1.0)
    parser.add_argument('--loss_aug_div_factor', type=float, default=1.0)
    parser.add_argument('--val_interval', type=int, default = 5)
    parser.add_argument('--shift_pots', action='store_true')
    parser.add_argument('--global_init_val', type=float)
    parser.add_argument('--l_rate_div', action='store_true')
    parser.add_argument('--reinit', action='store_true')
    parser.add_argument('--num_global_layers', type=int, default=1)
    parser.add_argument('--mp_itrs', type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--use_adam', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--tie_pairs', action='store_true', default=False)
    parser.add_argument('--unary_hidden_layers', type=int, default=2)
    parser.add_argument('--unary_hidden_size', type=int, default=318)
    parser.add_argument('--train_avg_thresh', type=int, default=-1)
    parser.add_argument('--test_avg_thresh', type=int, default=-1)
    parser.add_argument('--pair_diags', action='store_true', default=False)
    parser.add_argument('--pair_one', action='store_true', default=False)
    parser.add_argument('--use_residual', action='store_true')
    parser.add_argument('--mp_eps', type=float, default=0.)
    parser.add_argument('--test_mp_interval', type=int, default=-1)
    parser.add_argument('--train_mp_interval', type=int, default=-1)
    parser.add_argument('--use_feats', action='store_true')
    parser.add_argument('--use_global_beliefs', action='store_true')
    parser.add_argument('--top_sigmoid', action='store_true')
    parser.add_argument('--first_bn', action='store_true')
    parser.add_argument('--first_ln', action='store_true')
    parser.add_argument('--last_ln', action='store_true')
    parser.add_argument('--use_global_top', action='store_true')
    parser.add_argument('--diff_update', action='store_true')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--pd_theta', type=float, default=1.)
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--use_val_scheduler', type=float, default=None)



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
    if args.gpu == True:
        modelconf.use_gpu()
    if args.load_masks:
        train_masks_path = args.train_masks_path
        test_masks_path = args.val_masks_path
    else:
        train_masks_path = None
        test_masks_path = None

    def scaled_hamming_diff(true_val, other_val):
        return abs(true_val - other_val)/args.pot_div_factor

    if args.dataset == 'full':
        dataset_type = FULL
    else:
        dataset_type = R1

    train_data = FlickrTaggingDataset_Features(dataset_type, args.train_feat_file, args.label_dir, args.train_label_file, TRAIN, load=load, masks_path=train_masks_path, images_folder=args.img_dir)
    val_data = None
    #I know its confusing that I'm calling the validation dataset test - sorry
    test_data = FlickrTaggingDataset_Features(dataset_type, args.val_feat_file, args.label_dir, args.val_label_file, VAL, load=load, masks_path=test_masks_path, images_folder=args.img_dir)
    
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

    args_dict = {'pair_inds':pair_inds, 'pot_div_factor':args.pot_div_factor}
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.shift_pots:
        args_dict['shift_pots'] = True
    params = {}
    val_scheduler = lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.5)
    pw_params = {
        'batch_size':10000, 
        'num_epochs':50,
        'l_rate':1e-2, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':args.mp_eps, 
        'mp_itrs':args.mp_itrs,
        'use_loss_augmented_inference':False,
        'inf_loss':scaled_hamming_diff,
        'checkpoint_dir':args.working_dir,
        'task_loss':calculate_hamming_loss,
        'test_data':test_data,
        'train_masks_path':args.train_masks_path,
        'test_masks_path':args.val_masks_path,
        'val_interval':args.val_interval,
        'global_init_val':args.global_init_val,
        'num_global_layers':args.num_global_layers,
        'momentum':args.momentum,
        'use_adam':args.use_adam,
        'weight_decay':args.weight_decay,
        'train_interleaved_itrs':args.train_interleaved_itrs,
        'test_interleaved_itrs':args.test_interleaved_itrs,
        'use_pd':args.use_pd,
        'global_lr':args.global_lr,
        'graph_lr':args.graph_lr,
        'keep_graph_order':True,
        'train_avg_thresh':args.train_avg_thresh,
        'test_avg_thresh':args.test_avg_thresh,
        'global_layernorm':args.global_ln,
        'train_lambda_l_rate':args.train_lambda_l_rate,
        'train_max_globals_l_rate':args.train_max_globals_l_rate,
        'test_lambda_l_rate':args.test_lambda_l_rate,
        'test_max_globals_l_rate':args.test_max_globals_l_rate,
        'use_residual':args.use_residual,
        'train_mp_interval':args.train_mp_interval,
        'test_mp_interval':args.test_mp_interval,
        'top_sigmoid':args.top_sigmoid,
        'first_bn':args.first_bn,
        'first_ln':args.first_ln,
        'last_ln':args.last_ln,
        'use_global_top':args.use_global_top,
        'diff_update':args.diff_update,
        'use_dropout':args.use_dropout,
        'shuffle_data':args.shuffle_data,
        'pd_theta':args.pd_theta,
    }

    if args.wide_top:
        pw_params['wide_top'] = True
    if args.load_max_globals:
        pw_params['load_max_globals'] = True
    if args.max_globals_save_path:
        pw_params['max_globals_save_path'] = args.max_globals_save_path
    if args.use_loss_aug:
        pw_params['use_loss_augmented_inference'] = True
    if args.no_l_rate_decay:
        pw_params['training_scheduler'] = None
    if args.l_rate_div:
        pw_params['training_scheduler'] = lambda opt: torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    if args.num_epochs is not None:
        pw_params['num_epochs'] = args.num_epochs
    if args.l_rate is not None:
        pw_params['l_rate'] = args.l_rate
    if args.batch_size is not None:
        pw_params['batch_size'] = args.batch_size
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.reinit is not None:
        pw_params['reinit'] = args.reinit
    global_params = pw_params.copy()
    global_params['train_interleaved_itrs'] = 500
    global_params['test_interleaved_itrs'] = 500
    global_params['window_size'] = 100

    if args.model == 'pairwise' or args.model == 'pairwise_linear':
        if args.model == 'pairwise_linear':
            args_dict['linear_top'] = True
            if pw_params['l_rate'] != 0.0:
                args_dict['finetune'] = True
            else:
                args_dict['finetune'] = False
                pw_params['l_rate'] = global_params['global_lr']
        full_graph = Graph(nodes, pairs, 2, FlickrBaselineModel, args_dict, False)
        if args.load_classifier is not None:
            full_graph.potential_model.load_classifier(args.load_classifier)
        pw_model = PairwiseModel([full_graph], len(nodes), 2, params)
        print(pw_params) 
        start = time.time()
        obj,  train_results, return_vals = pw_model.train(train_data, val_data, pw_params)
        return_vals['diff_vals'] = None
        end = time.time()
        train_time = end-start
        start = time.time()
        test_results = pw_model.test(test_data, pw_params)
        end = time.time()
        test_time = end-start
        train_test_results = pw_model.test(train_data, pw_params)
        for datum, result in zip(test_data, test_results):
            print("\tCORRECT: ",datum[0])
            print("\tFOUND:   ", result)
        train_loss = calculate_hamming_loss(train_data, train_test_results)
        test_loss = calculate_hamming_loss(test_data, test_results)
        print("TRAIN LOSS: ",train_loss)
        print("TEST LOSS: ",test_loss)
        print("TRAIN TIME: ",train_time)
        print("TEST TIME: ",test_time)
        plot_task_losses(args.working_dir, 'baseline', return_vals)
        graph_results(args.working_dir, 'baseline', return_vals)

    else: 
        if args.model == 'global_linear':    
            global_params['global_model'] = build_linear_global_model
            global_params['interleaved_itrs'] = 1000
        elif args.model == 'global_quad':
            global_params['global_model'] = QuadModel
        elif args.model == 'global_mlp':
            if args.use_gt:
                global_params['global_model'] = GlobalModel_GT
                global_params['global_inputs'] = ['data_masks']
            elif args.mlp_init is None:
                global_params['global_model'] = build_mlp_global_model
            elif args.mlp_init == 'v1':
                global_params['global_model'] = build_initialized_mlp_global_model_v1
            elif args.mlp_init == 'v2':
                global_params['global_model'] = build_initialized_mlp_global_model_v2
            if args.global_hidden_size is None:
                global_params['global_hidden_size'] = 10
            else:
                global_params['global_hidden_size'] = args.global_hidden_size
            if args.global_activation is not None:
                global_params['global_activation'] = args.global_activation
            global_params['interleaved_itrs'] = 1000
        if args.train_interleaved_itrs is not None:
            global_params['train_interleaved_itrs'] = args.train_interleaved_itrs
        if args.test_interleaved_itrs is not None:
            global_params['test_interleaved_itrs'] = args.test_interleaved_itrs
        print("PARAMS: ",global_params)
        graphs = []
        if args.load_pots:
            nodes_graph = Graph(nodes, [], 2, FlickrFixedModel, args_dict, False)
        else:
            nodes_graph = Graph(nodes, [], 2, FlickrBaselineModel, args_dict, False)
        graphs.append(nodes_graph)
        if args.pretrain is not None:
            if args.load_pots:
                full_graph = Graph(nodes, pairs, 2, FlickrPotentialModel, args_dict, False)
                pw_model = PairwiseModel([full_graph], len(nodes), 2, params)
                pw_model.load(args.pretrain)
            else:
                full_graph = Graph(nodes, pairs, 2, FlickrBaselineModel, args_dict, False)
                pw_model = PairwiseModel([full_graph], len(nodes), 2, params)
                pw_model.load(args.pretrain)
                graphs[0].potential_model.unary_model = pw_model.graphs[0].potential_model.unary_model
        '''
        for pair in pairs:
            pair_graph = Graph([], [pair], 26, WordsPotentialModel, args_dict)
            graphs.append(pair_graph)
        '''
        for ind,pair in enumerate(pairs):
            if args.load_pots:
                graphs.append(Graph([], [pair], 2, FlickrFixedModel, args_dict, False))
            else:
                graphs.append(Graph([], [pair], 2, FlickrBaselineModel, args_dict, False))
            if args.pretrain is not None:
                if args.load_pots:
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind*4:(ind+1)*4].data)
                else:
                    #graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind*4:(ind+1)*4].data)
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind:ind+1,:].data)

        '''
        for pair_graph in graphs[len(nodes)+1:]:
            pair_graph.potential_model.pair_model = graphs[len(nodes)].potential_model.pair_model
        '''
        if args.model == 'global_linear' or args.model == 'global_mlp':
            if args.interleave:
                global_model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 2, global_params)
            else:
                global_model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, global_params)
        start = time.time()
        obj, train_results, return_vals = global_model.train(train_data, None, global_params)
        end = time.time()
        train_time = end-start
        start = time.time()
        test_results = global_model.test(test_data, global_params)
        end = time.time()

        train_test_results = global_model.test(train_data, global_params)
        test_time = end-start
        for datum, result in zip(test_data, test_results):
            print("\tCORRECT: ",datum[0])
            print("\tFOUND:   ", result)
        train_loss = calculate_hamming_loss(train_data, train_test_results)
        test_loss = calculate_hamming_loss(test_data, test_results)
        print("TRAIN LOSS: ",train_loss)
        print("TEST LOSS: ", test_loss)

        #pw_model.print_params()
        print("TRAIN TIME: ",train_time)
        print("TEST TIME: ",test_time)
        plot_task_losses(args.working_dir, args.model, return_vals)
        graph_results(args.working_dir, args.model, return_vals)
        



