from deepstruct.datasets import *
from deepstruct.models import *
import argparse, os
from torchvision import transforms
import torchvision.models
from PIL import Image
import deepstruct.models.modelconf

TRAIN = 0
VAL = 1
TEST = 2


NUM_TRAIN = 196
NUM_TEST = 66
NUM_VAL = 66

class HorsesSegDataset(BaseDataset):
    def __init__(self, images_folder, label_folder, mode, load=False, masks_path=None):
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

        self.label_transform_orig = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
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
                        label = np.array(self.label_transform(raw_img.convert('1'))).astype('uint8').flatten()
                self.labels.append(label)
            if mode == TRAIN:
                extra = "train"
            elif mode == VAL:
                extra = "val"
            elif mode == TEST:
                extra = "test"
            fname = os.path.join(self.img_folder, 'preprocessed_imgs_%s'%extra)
            torch.save(self.images, fname)
            fname = os.path.join(self.label_folder, 'preprocessed_labels_%s'%extra)
            torch.save(self.labels, fname)
        super(HorsesSegDataset, self).__init__(len(self.images), masks_path)
 

    def __getitem__(self, idx):
        stuff = super(HorsesSegDataset, self).__getitem__(idx)
        return (self.labels[idx], self.images[idx]) + stuff

class HorsesSegDataset_Features(BaseDataset):
    def __init__(self, images_folder, feature_file, label_folder, mode, load=False, masks_path=None):
        self.features = torch.load(feature_file)

        self.img_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.label_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
        ])
        if load:
            print("LOADING PRECOMPUTED LABELS")
            if mode == TRAIN:
                extra = "train"
            elif mode == VAL:
                extra = "val"
            elif mode == TEST:
                extra = "test"
            save_label_file = os.path.join(label_folder, 'preprocessed_labels_%s'%extra)
            save_image_file = os.path.join(images_folder, 'preprocessed_imgs_%s'%extra)
            self.labels = torch.load(save_label_file)

            self.images = torch.load(save_image_file)
        else:
            print("LOADING LABELS")

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
                        label = np.array(self.label_transform(raw_img.convert('1'))).astype('uint8').flatten()
                self.labels.append(label)

            if mode == TRAIN:
                extra = "train"
            elif mode == VAL:
                extra = "val"
            elif mode == TEST:
                extra = "test"
            fname = os.path.join(self.img_folder, 'preprocessed_imgs_%s'%extra)
            torch.save(self.images, fname)
            fname = os.path.join(self.label_folder, 'preprocessed_labels_%s'%extra)
            torch.save(self.labels, fname)
        print("NUM FEATS: ",len(self.features))
        print("NUM LABELS: ",len(self.labels))
        print("NUM IMAGES: ",len(self.images))
        super(HorsesSegDataset_Features, self).__init__(len(self.features), masks_path)
 

    def __getitem__(self, idx):
        stuff = super(HorsesSegDataset_Features, self).__getitem__(idx)
        return (self.labels[idx], self.features[idx]) + stuff + (self.images[idx],)


class HorsesFeatureModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(HorsesFeatureModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.img_size = 224
        self.pair_inds = args_dict['pair_inds']
        self.tie_pairs = args_dict.get('tie_pairs', False)
        if len(pair_regions) > 0:
            if args_dict.get('tie_pairs', False):
                self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(num_vals*num_vals).uniform_(-1.0, 1.0))
            else:
                self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).uniform_(-1.0, 1.0))
        self.pot_div_factor = args_dict.get('pot_div_factor', 1.0)

    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(observations).float()

    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            result[:, :len(self.node_regions)*self.num_vals] = self.observations
        if len(self.pair_regions) > 0:
            if self.tie_pairs:
                result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,len(self.pair_regions))
            else:
                result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,1)
        result /= self.pot_div_factor
        return result

class HorsesPotentialModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(HorsesPotentialModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.img_size = 224
        self.pair_inds = args_dict['pair_inds']
        if len(node_regions) > 0:
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
            self.upscore = nn.ConvTranspose2d(2, 2, 14, stride=2, bias=False)
            self.unary_model = nn.Sequential(features, classifier, self.upscore)
        if len(pair_regions) > 0:
            #V2:
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).uniform_(-0.1, 0.1))
            #V3:
            #self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(1.0))

    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(torch.stack(observations)).float()

    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            tmp = self.unary_model(self.observations)
            result[:, :len(self.node_regions)*self.num_vals] = tmp.permute(0,2,3,1).contiguous().view(self.num_observations, -1)
        if len(self.pair_regions) > 0:
            result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,1)
        return result

class HorsesCNNTop(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        if params.get('global_batchnorm_all', False):
            self.top = torch.nn.Sequential(
                nn.Conv2d(10, 16, 8, stride=4),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
        else:
            self.top = torch.nn.Sequential(
                nn.Conv2d(10, 16, 8, stride=4),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.ReLU(inplace=True),
            )
        if params.get('global_batchnorm', None):
            self.top2 = torch.nn.Sequential(
                nn.Linear(1152, 256),
                nn.BatchNorm1d(256),
                activation(),
                nn.Linear(256, 1, bias=False),
            )
        elif params.get('global_layernorm', None):
            self.top2 = torch.nn.Sequential(
                nn.Linear(1152, 256),
                nn.LayerNorm(256, elementwise_affine=False),
                activation(),
                nn.Linear(256, 1, bias=False),
            )
        else:
            self.top2 = torch.nn.Sequential(
                nn.Linear(1152, 256),
                activation(),
                nn.Linear(256, 1, bias=False),
            )

    def forward(self, inp):
        unaries = inp[:, :8192].contiguous()
        row_pairs = inp[:, 8192:(8192+16128)].contiguous()
        col_pairs = inp[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(inp), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(inp), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(inp), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))
        final_inp = torch.cat([new_unaries, new_rows, new_cols], dim=1)
        tmp = self.top(final_inp)
        tmp = tmp.view(len(inp), -1)
        result = self.top2(tmp)
        return result

class HorsesCNNTop_GT(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop_GT, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        layers = []
        if params.get('first_bn', False):
            layers.append(nn.BatchNorm2d(23))
        layers.extend([
            nn.Conv2d(23, 32, 8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True),
        ])
        self.top = torch.nn.Sequential(*layers)
        if params.get('global_batchnorm', None):
            self.top2 = torch.nn.Sequential(
                nn.Linear(2304, 512),
                nn.BatchNorm1d(512),
                activation(),
                nn.Linear(512, 1, bias=False),
            )
        elif params.get('global_layernorm', None):
            self.top2 = torch.nn.Sequential(
                nn.Linear(2304, 512),
                nn.LayerNorm(512, elementwise_affine=False),
                activation(),
                nn.Linear(512, 1, bias=False),
            )
        else:
            self.top2 = torch.nn.Sequential(
                nn.Linear(2304, 512),
                activation(),
                nn.Linear(512, 1, bias=False),
            )

    def forward(self, pots, obs, gt):
        unaries = pots[:, :8192].contiguous()
        row_pairs = pots[:, 8192:(8192+16128)].contiguous()
        col_pairs = pots[:, (8192+16128):].contiguous()

        unaries_gt = gt[:, :8192].contiguous()
        row_pairs_gt = gt[:, 8192:(8192+16128)].contiguous()
        col_pairs_gt = gt[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))

        new_u_gt = unaries_gt.view(len(gt), 64, 64, 2).permute(0,3,1,2)
        new_row_gt = row_pairs_gt.view(len(gt), 64, 63, 4).permute(0,3,1,2)
        new_row_gt = nn.functional.pad(new_row_gt, (0,1,0,0,0,0,0,0))
        new_col_gt = col_pairs_gt.view(len(gt), 63, 64, 4).permute(0,3,1,2)
        new_col_gt = nn.functional.pad(new_col_gt, (0,0,0,1,0,0,0,0))
        final_inp = torch.cat([new_unaries, new_rows, new_cols, obs, new_u_gt, new_row_gt, new_col_gt], dim=1)
        tmp = self.top(final_inp)
        tmp = tmp.view(len(pots), -1)
        result = self.top2(tmp)
        return result

class HorsesCNNTop_GTRes(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop_GTRes, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0 and params.get('first_bn', False):
                layers.append(nn.BatchNorm2d(13))
            layers.append(nn.Conv2d(23, 23, 3, padding=1))
            if params.get('global_batchnorm', False):
                layers.append(nn.BatchNorm2d(23))
            layers.append(activation())
            layers.append(nn.Conv2d(23, 23, 3, padding=1))
            block = nn.Sequential(*layers)
            self.blocks.append(block)
            self.add_module('block%d'%ind, block)
        self.ignore_global_top = params.get('ignore_global_top', False)
        if not self.ignore_global_top:
            if params.get('global_top_activation', None) == 'tanh':
                top_activation = lambda: nn.Tanh()
            else:
                top_activation = lambda: nn.Sigmoid()
            if params.get('global_top_full', False):
                self.top = nn.Sequential(
                        top_activation(),
                        nn.Conv2d(23, 23, 1),
                )
            else:
                self.top = nn.Sequential(
                        top_activation(),
                        nn.Conv2d(23, 1, 1),
                )


    def forward(self, pots, obs, gt):

        unaries = pots[:, :8192].contiguous()
        row_pairs = pots[:, 8192:(8192+16128)].contiguous()
        col_pairs = pots[:, (8192+16128):].contiguous()

        unaries_gt = gt[:, :8192].contiguous()
        row_pairs_gt = gt[:, 8192:(8192+16128)].contiguous()
        col_pairs_gt = gt[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))

        new_u_gt = unaries_gt.view(len(gt), 64, 64, 2).permute(0,3,1,2)
        new_row_gt = row_pairs_gt.view(len(gt), 64, 63, 4).permute(0,3,1,2)
        new_row_gt = nn.functional.pad(new_row_gt, (0,1,0,0,0,0,0,0))
        new_col_gt = col_pairs_gt.view(len(gt), 63, 64, 4).permute(0,3,1,2)
        new_col_gt = nn.functional.pad(new_col_gt, (0,0,0,1,0,0,0,0))
        final_inp = torch.cat([new_unaries, new_rows, new_cols, obs, new_u_gt, new_row_gt, new_col_gt], dim=1)
        for block in self.blocks:
            final_inp = block(final_inp) + final_inp
        if not self.ignore_global_top:
            result = self.top(final_inp)
            return result.view(len(pots), -1).sum(dim=1).unsqueeze(1)
        else:
            return nn.functional.sigmoid(final_inp).view(len(pots), -1).sum(dim=1).unsqueeze(1)



class HorsesCNNTop_Res(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop_Res, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params.get('global_activation', None) == 'tanh':
            activation = lambda: nn.Tanh()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0 and params.get('first_bn', False):
                layers.append(nn.BatchNorm2d(13))
            layers.append(nn.Conv2d(13, 13, 3, padding=1))
            if params.get('global_batchnorm', False):
                layers.append(nn.BatchNorm2d(13))
            layers.append(activation())
            layers.append(nn.Conv2d(13, 13, 3, padding=1))
            block = nn.Sequential(*layers)
            self.blocks.append(block)
            self.add_module('block%d'%ind, block)
        self.ignore_global_top = params.get('ignore_global_top', False)
        if not self.ignore_global_top:
            if params.get('global_top_activation', None) == 'tanh':
                top_activation = lambda: nn.Tanh()
            else:
                top_activation = lambda: nn.Sigmoid()
            if params.get('global_top_full', False):
                self.top = nn.Sequential(
                        top_activation(),
                        nn.Conv2d(13, 13, 1),
                )
            else:
                self.top = nn.Sequential(
                        top_activation(),
                        nn.Conv2d(13, 1, 1),
                )

    def forward(self, pots, obs):

        unaries = pots[:, :8192].contiguous()
        row_pairs = pots[:, 8192:(8192+16128)].contiguous()
        col_pairs = pots[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))

        final_inp = torch.cat([new_unaries, new_rows, new_cols, obs], dim=1)
        for block in self.blocks:
            final_inp = block(final_inp) + final_inp
        if not self.ignore_global_top:
            result = self.top(final_inp)
            return result.view(len(pots), -1).sum(dim=1).unsqueeze(1)
        else:
            return nn.functional.sigmoid(final_inp).view(len(pots), -1).sum(dim=1).unsqueeze(1)

class HorsesCNNTop_Res_Beliefs(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop_Res_Beliefs, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params.get('global_activation', None) == 'tanh':
            activation = lambda: nn.Tanh()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0 and params.get('first_bn', False):
                layers.append(nn.BatchNorm2d(23))
            layers.append(nn.Conv2d(23, 23, 3, padding=1))
            if params.get('global_batchnorm', False):
                layers.append(nn.BatchNorm2d(23))
            layers.append(activation())
            layers.append(nn.Conv2d(23, 23, 3, padding=1))
            block = nn.Sequential(*layers)
            self.blocks.append(block)
            self.add_module('block%d'%ind, block)
        self.ignore_global_top = params.get('ignore_global_top', False)
        if not self.ignore_global_top:
            if params.get('global_top_activation', None) == 'tanh':
                top_activation = lambda: nn.Tanh()
            else:
                top_activation = lambda: nn.Sigmoid()
            if params.get('global_top_full', False):
                self.top = nn.Sequential(
                        top_activation(),
                        nn.Conv2d(23, 23, 1),
                )
            else:
                if params.get('global_batchnorm', False):
                    self.top = nn.Sequential(
                            nn.BatchNorm2d(23),
                            top_activation(),
                            nn.Conv2d(23, 1, 1),
                    )
                else:
                    self.top = nn.Sequential(
                            top_activation(),
                            nn.Conv2d(23, 1, 1),
                    )

    def forward(self, pots, beliefs, obs):

        unaries = pots[:, :8192].contiguous()
        row_pairs = pots[:, 8192:(8192+16128)].contiguous()
        col_pairs = pots[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))

        unaries_b = beliefs[:, :8192].contiguous()
        row_pairs_b = beliefs[:, 8192:(8192+16128)].contiguous()
        col_pairs_b = beliefs[:, (8192+16128):].contiguous()

        new_unaries_b = unaries_b.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows_b = row_pairs_b.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows_b = nn.functional.pad(new_rows_b, (0,1,0,0,0,0,0,0))
        new_cols_b = col_pairs_b.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols_b = nn.functional.pad(new_cols_b, (0,0,0,1,0,0,0,0))



        final_inp = torch.cat([new_unaries, new_rows, new_cols, new_unaries_b, new_rows_b, new_cols_b, obs], dim=1)
        for block in self.blocks:
            final_inp = block(final_inp) + final_inp
        if not self.ignore_global_top:
            result = self.top(final_inp)
            return result.view(len(pots), -1).sum(dim=1).unsqueeze(1)
        else:
            return nn.functional.sigmoid(final_inp).view(len(pots), -1).sum(dim=1).unsqueeze(1)


class HorsesCNNTop_Res_NoObs(nn.Module):
    def __init__(self, width, params):
        super(HorsesCNNTop_Res_NoObs, self).__init__()
        if params.get('global_activation', None) == 'relu6':
            activation = lambda: nn.ReLU6(inplace=True)
        elif params.get('global_activation', None) == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        else:
            activation = lambda: nn.ReLU(inplace=True)
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0 and params.get('first_bn', False):
                layers.append(nn.BatchNorm2d(13))
            layers.extend([
                nn.Conv2d(10, 10, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(10, 10, 3, padding=1),
            ])
            block = nn.Sequential(*layers)
            self.blocks.append(block)
            self.add_module('block%d'%ind, block)
        self.ignore_global_top = params.get('ignore_global_top', False)
        if not self.ignore_global_top:
            self.top = nn.Sequential(
                    nn.Sigmoid(),
                    nn.Conv2d(10, 1, 1),
            )

    def forward(self, pots):

        unaries = pots[:, :8192].contiguous()
        row_pairs = pots[:, 8192:(8192+16128)].contiguous()
        col_pairs = pots[:, (8192+16128):].contiguous()

        new_unaries = unaries.view(len(pots), 64, 64, 2).permute(0,3,1,2)
        new_rows = row_pairs.view(len(pots), 64, 63, 4).permute(0,3,1,2)
        new_rows = nn.functional.pad(new_rows, (0,1,0,0,0,0,0,0))
        new_cols = col_pairs.view(len(pots), 63, 64, 4).permute(0,3,1,2)
        new_cols = nn.functional.pad(new_cols, (0,0,0,1,0,0,0,0))

        final_inp = torch.cat([new_unaries, new_rows, new_cols], dim=1)
        for block in self.blocks:
            final_inp = block(final_inp) + final_inp
        if not self.ignore_global_top:
            result = self.top(final_inp)
            return result.view(len(pots), -1).sum(dim=1).unsqueeze(1)
        else:
            return nn.functional.sigmoid(final_inp).view(len(pots), -1).sum(dim=1).unsqueeze(1)





def calculate_accuracy(dataset, found):
    correct = [datum[0] for datum in dataset]
    preds = np.zeros((2,2))
    for true, guess in zip(correct, found):
        for true_p, guess_p in zip(true, guess):
            preds[true_p, guess_p] += 1
    accuracy = (preds[0,0] + preds[1,1])/preds.sum()
    mean_iu = 0.5*(preds[0,0]/(preds[0,0]+preds[0,1] + preds[1, 0]) + preds[1,1]/(preds[1,0]+preds[1,1]+preds[0,1]))
    return accuracy, mean_iu

def save_features(model, dataset, batch_size, features_path):
    unary_model = model.graph.potential_model.unary_model
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_batch)
    results = []
    for batch in dataloader:
        imgs = Variable(torch.stack(batch.observations)).float()
        tmp = unary_model(imgs).data.cpu()
        results.append(tmp.permute(0,2,3,1).contiguous().view(len(batch), -1))
    results = torch.cat(results)
    torch.save(results, features_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run experiment for semantic segmentation')
    parser.add_argument('model', choices=['unary', 'pairwise', 'global_cnn'])
    parser.add_argument('working_dir')
    parser.add_argument('--img_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--load_data', action='store_true', default=False)
    parser.add_argument('--train_feats_file')
    parser.add_argument('--val_feats_file')
    parser.add_argument('--test_feats_file')
    parser.add_argument('-p', '--pretrain')
    parser.add_argument('--global_lr', type=float)
    parser.add_argument('--graph_lr', type=float)
    parser.add_argument('--train_interleaved_itrs', type=int, default=100)
    parser.add_argument('--test_interleaved_itrs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'relu6', 'tanh'])
    parser.add_argument('--global_bn', action='store_true')
    parser.add_argument('--global_ln', action='store_true')
    parser.add_argument('--train_max_globals_l_rate', type=float)
    parser.add_argument('--train_lambda_l_rate', type=float)
    parser.add_argument('--test_max_globals_l_rate', type=float)
    parser.add_argument('--test_lambda_l_rate', type=float)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--load_masks', action='store_true')
    parser.add_argument('--train_masks_path')
    parser.add_argument('--val_masks_path')
    parser.add_argument('--mlp_init', choices=['v1', 'v2'])
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
    parser.add_argument('--val_interval', type=int, default = 10)
    parser.add_argument('--shift_pots', action='store_true')
    parser.add_argument('--global_init_val', type=float, default=0.001)
    parser.add_argument('--l_rate_div', action='store_true')
    parser.add_argument('--reinit', action='store_true')
    parser.add_argument('--num_global_layers', type=int, default=1)
    parser.add_argument('--mp_itrs', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--use_adam', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--tie_pairs', action='store_true', default=False)
    parser.add_argument('--train_avg_thresh', type=int, default=-1)
    parser.add_argument('--test_avg_thresh', type=int, default=-1)
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--use_res', action='store_true')
    parser.add_argument('--first_bn', action='store_true')
    parser.add_argument('--ignore_global_top', action='store_true')
    parser.add_argument('--global_top_full', action='store_true')
    parser.add_argument('--test_mp_interval', type=int, default=-1)
    parser.add_argument('--train_mp_interval', type=int, default=-1)
    parser.add_argument('--reinit_interval', type=int, default=-1)
    parser.add_argument('--global_top_activation', choices=['tanh', 'sigmoid'])
    parser.add_argument('--load_global', action='store_true')
    parser.add_argument('--use_global_beliefs', action='store_true')
    parser.add_argument('--diff_update', action='store_true')
    parser.add_argument('--random_init', type=float, default=None)

    args = parser.parse_args()
    if args.img_dir == None and args.train_img_file == None:
        print("ERROR: Must specify either an image directory or image file")
        sys.exit(1)
    if args.label_dir == None and args.train_label_file == None:
        print("ERROR: Must specify either a label directory or label file")
        sys.exit(1)
    if args.gpu:
        modelconf.use_gpu()
    if args.load_masks:
        train_masks_path = args.train_masks_path
        test_masks_path = args.val_masks_path
    else:
        train_masks_path = None
        test_masks_path = None

    def scaled_identity_diff(true_val, other_val):
        return float(true_val != other_val)/args.loss_aug_div_factor

    save_img_file = os.path.join(args.img_dir, 'preprocessed_imgs')
    save_label_file = os.path.join(args.label_dir, 'preprocessed_imgs')
    if args.model == 'unary':
        train_data = HorsesSegDataset(args.img_dir, args.label_dir,  TRAIN, load=args.load_data, masks_path=train_masks_path)
        val_data = HorsesSegDataset(args.img_dir, args.label_dir, VAL, load=args.load_data, masks_path=train_masks_path)
    else:
        train_data = HorsesSegDataset_Features(args.img_dir, args.train_feats_file, args.label_dir,  TRAIN, load=args.load_data, masks_path=train_masks_path)
        val_data = HorsesSegDataset_Features(args.img_dir, args.val_feats_file, args.label_dir, VAL, load=args.load_data, masks_path=train_masks_path)
    
    num_rows = num_cols = 64
    nodes = list(range(num_rows*num_cols))
    pairs = []
    pair_inds = {}
    ind = 0
    #Constructing pairs in this order to facilitate our top model
    for row in range(num_rows):
        for col in range(num_cols):
            node1 = num_cols*row + col
            if col < num_cols-1:
                pairs.append((node1, node1+1))
                pair_inds[(node1, node1+1)] = ind
                ind += 1
    for row in range(num_rows):
        for col in range(num_cols):
            node1 = num_cols*row + col
            if row < num_rows-1:
                pairs.append((node1, node1+num_cols))
                pair_inds[(node1, node1+num_cols)] = ind
                ind += 1
    args_dict = {'pair_inds':pair_inds, 'pot_div_factor':args.pot_div_factor, 'tie_pairs':args.tie_pairs}
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.shift_pots:
        args_dict['shift_pots'] = True
    val_scheduler = lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.5)
    pw_params = {
        'batch_size':10000, 
        'num_epochs':50,
        'l_rate':1e-2, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':0.0, 
        'mp_itrs':args.mp_itrs,
        'use_loss_augmented_inference':False,
        'inf_loss':scaled_identity_diff,
        'val_scheduler':val_scheduler, 
        'checkpoint_dir':args.working_dir,
        'task_loss':calculate_accuracy,
        'test_data':val_data,
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
        'global_activation':args.global_activation,
        'global_batchnorm':args.global_bn,
        'global_layernorm':args.global_ln,
        'train_avg_thresh':args.train_avg_thresh,
        'test_avg_thresh':args.test_avg_thresh,
        'train_mp_interval':args.train_mp_interval,
        'test_mp_interval':args.test_mp_interval,
        'train_lambda_l_rate':args.train_lambda_l_rate,
        'train_max_globals_l_rate':args.train_max_globals_l_rate,
        'test_lambda_l_rate':args.test_lambda_l_rate,
        'test_max_globals_l_rate':args.test_max_globals_l_rate,
        'first_bn':args.first_bn,
        'reinit_interval':args.reinit_interval,
        'ignore_global_top':args.ignore_global_top,
        'global_top_full':args.global_top_full,
        'global_top_activation':args.global_top_activation,
        'diff_update':args.diff_update,
        'random_init':args.random_init,
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
    pw_params['window_size'] = 100

    if args.model == 'unary':
        args_dict['pair_inds'] = {}
        graph = Graph(nodes, [], 2, HorsesPotentialModel, args_dict, False)
        model = PairwiseModel([graph], len(nodes), 2, pw_params)
        pw_params['save_checkpoints'] = False
    elif args.model == 'pairwise':
        full_graph = Graph(nodes, pairs, 2, HorsesFeatureModel, args_dict, False)
        model = PairwiseModel([full_graph], len(nodes), 2, pw_params)
    elif args.model == 'global_cnn':
        graphs = []
        if args.use_gt:
            if args.use_res:
                pw_params['global_model'] = HorsesCNNTop_GTRes
            else:
                pw_params['global_model'] = HorsesCNNTop_GT
            pw_params['global_inputs'] = ['other_obs', 'data_masks']
        elif args.use_res:
            if args.mlp_init == 'v2':
                pw_params['global_model'] = HorsesCNNTop_Res_NoObs
            else:
                if args.use_global_beliefs:
                    pw_params['global_model'] = HorsesCNNTop_Res_Beliefs
                    pw_params['global_beliefs'] = True
                else:
                    pw_params['global_model'] = HorsesCNNTop_Res
                pw_params['global_inputs'] = ['other_obs']
        else:
            pw_params['global_model'] = HorsesCNNTop
        nodes_graph = Graph(nodes, [], 2, HorsesFeatureModel, args_dict, False) 
        if args.pretrain is not None and not args.load_global:
            full_graph = Graph(nodes, pairs, 2, HorsesFeatureModel, args_dict, False)
            pw_model = PairwiseModel([full_graph], len(nodes), 2, pw_params)
            pw_model.load(args.pretrain)
        graphs.append(nodes_graph)
        for ind, pair in enumerate(pairs):
            graphs.append(Graph([], [pair], 2, HorsesFeatureModel, args_dict, False))
            if args.pretrain is not None and not args.load_global:
                if args.tie_pairs:
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model.data)
                else:
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind*4:(ind+1)*4].data)

        if args.interleave:
            model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 2, pw_params)
        else:
            model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, pw_params)
        if args.load_global:
            model.load(args.pretrain)

    print(pw_params) 
    start = time.time()

    obj,  train_results, return_vals = model.train(train_data, None, pw_params)
    return_vals['diff_vals'] = None
    end = time.time()
    train_time = end-start
    start = time.time()
    val_results = model.test(val_data, pw_params)
    end = time.time()
    val_time = end-start
    train_test_results = model.test(train_data, pw_params)
    train_loss = calculate_accuracy(train_data, train_test_results)
    val_loss = calculate_accuracy(val_data, val_results)
    print("TRAIN LOSS: ",train_loss)
    print("VAL LOSS: ",val_loss)
    print("TRAIN TIME: ",train_time)
    print("VAL TIME: ",val_time)

    if args.model == 'unary':
        test_data = HorsesSegDataset(args.img_dir, args.label_dir, TEST, load=args.load_data, masks_path=train_masks_path)
        model.init_dataset(test_data, model.mp_graphs, args.use_loss_aug)
        print("Saving train features...")
        save_features(model, train_data, args.batch_size, args.train_feats_file)
        print("Saving val features...")
        save_features(model, val_data, args.batch_size, args.val_feats_file)
        print("Saving test features...")
        save_features(model, test_data, args.batch_size, args.test_feats_file)
