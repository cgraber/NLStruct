from deepstruct.datasets import *
from deepstruct.models import *
import deepstruct.models.modelconf
import argparse

FULL = 1
TRAIN = 2
VAL = 3
TEST = 4

class BibtexDataset(BaseDataset):
    def __init__(self, mode, features_path, labels_path, masks_path=None):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        if mode == TRAIN:
            self.features = self.features[:3660, :]
            self.labels = self.labels[:3660, :]
        elif mode == VAL:
            self.features = self.features[3660:, :]
            self.labels = self.labels[3660:, :]

        super(BibtexDataset, self).__init__(len(self.features), masks_path)

    def __getitem__(self, idx):
        stuff = super(BibtexDataset, self).__getitem__(idx)
        return (self.labels[self.ord[idx],:], self.features[self.ord[idx],:]) + stuff

class DeliciousDataset(BaseDataset):
    def __init__(self, mode, features_path, labels_path, masks_path=None):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        if mode == TRAIN:
            self.features = self.features[:-3185, :]
            self.labels = self.labels[:-3185, :]
        elif mode == VAL:
            self.features = self.features[-3185:, :]
            self.labels = self.labels[-3185:, :]

        super(DeliciousDataset, self).__init__(len(self.features), masks_path)

    def __getitem__(self, idx):
        stuff = super(DeliciousDataset, self).__getitem__(idx)
        return (self.labels[idx,:], self.features[idx,:]) + stuff



class BookmarksDataset(BaseDataset):
    def __init__(self, mode, features_path, labels_path, masks_path=None):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        if mode == TRAIN:
            self.features = self.features[:48000, :]
            self.labels = self.labels[:48000, :]
        elif mode == VAL:
            self.features = self.features[48000:60000, :]
            self.labels = self.labels[48000:60000, :]
        elif mode == TEST:
            self.features = self.features[60000:, :]
            self.labels = self.labels[60000:, :]

        super(BookmarksDataset, self).__init__(len(self.features), masks_path)

    def __getitem__(self, idx):
        stuff = super(BookmarksDataset, self).__getitem__(idx)
        return (self.labels[idx,:], self.features[idx,:]) + stuff

class BibtexUnaryDataset(torch.utils.data.Dataset):
    def __init__(self, mode, features_path, labels_path):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        if mode == TRAIN:
            self.features = self.features[:3660, :]
            self.labels = self.labels[:3660, :]
        elif mode == VAL:
            self.features = self.features[3660:, :]
            self.labels = self.labels[3660:, :]

    def __getitem__(self, idx):
        return self.features[idx,:].float(), self.labels[idx,:].long()

    def __len__(self):
        return len(self.features)

class DeliciousUnaryDataset(torch.utils.data.Dataset):
    def __init__(self, mode, features_path, labels_path):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)

        if mode == TRAIN:
            self.features = self.features[:-3185, :]
            self.labels = self.labels[:-3185, :]
        elif mode == VAL:
            self.features = self.features[-3185:, :]
            self.labels = self.labels[-3185:, :]

    def __getitem__(self, idx):
        return self.features[idx,:].float(), self.labels[idx,:].long()

    def __len__(self):
        return len(self.features)



class BookmarksUnaryDataset(torch.utils.data.Dataset):
    def __init__(self, mode, features_path, labels_path):
        self.features = torch.load(features_path)
        self.labels = torch.load(labels_path)
        if mode == TRAIN:
            self.features = self.features[:48000, :]
            self.labels = self.labels[:48000, :]
        elif mode == VAL:
            self.features = self.features[48000:60000, :]
            self.labels = self.labels[48000:60000, :]
        elif mode == TEST:
            self.features = self.features[60000:, :]
            self.labels = self.labels[60000:, :]

    def __getitem__(self, idx):
        return self.features[idx,:].float(), self.labels[idx,:].long()

    def __len__(self):
        return len(self.features)


class MLCPotentialModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(MLCPotentialModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.pair_inds = args_dict['pair_inds']
        self.pot_div_factor = args_dict.get('pot_div_factor', 1.0)
        self.pair_diags = args_dict.get('pair_diags', False)
        self.pair_one = args_dict.get('pair_one', False)
        if len(node_regions) > 0:
            self.unary_model = build_unary_model(args_dict['unary_features'], args_dict['unary_hidden_layers'], args_dict['unary_hidden_size'], len(self.node_regions)*2)
            self.add_module('unary', self.unary_model)
        if len(pair_regions) > 0:
            #self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(10.0))
            if self.pair_diags or self.pair_one:
                #self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions),2).uniform_(-0.1, 0.1))
                self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions),2).fill_(1.0))
            else:
                #self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).uniform_(-0.1, 0.1))
                self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(pair_regions)*num_vals*num_vals).fill_(1.0))


    def set_observations(self, observations):
        self.num_observations = len(observations)
        self.observations = Variable(observations).float()

    def forward(self):
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, self.num_potentials))
        if len(self.node_regions) > 0:
            result[:, :len(self.node_regions)*self.num_vals] = self.unary_model(self.observations)
        if len(self.pair_regions) > 0:
            if self.pair_diags:
                tmp = modelconf.tensor_mod.FloatTensor(len(self.pair_regions), 4)
                tmp[:, 0] = self.pair_model[:, 0]
                tmp[:, 1] = self.pair_model[:, 1]
                tmp[:, 2] = self.pair_model[:, 1]
                tmp[:, 3] = self.pair_model[:, 0]
                result[:, len(self.node_regions)*self.num_vals:] = tmp.view(1, -1).repeat(self.num_observations,1)
            elif self.pair_one:
                tmp = modelconf.tensor_mod.FloatTensor(len(self.pair_regions), 4)
                tmp[:, 0] = self.pair_model[:, 0]
                tmp[:, 1] = self.pair_model[:, 0]
                tmp[:, 2] = self.pair_model[:, 0]
                tmp[:, 3] = self.pair_model[:, 1]
                result[:, len(self.node_regions)*self.num_vals:] = tmp.view(1, -1).repeat(self.num_observations,1)

            else:
                result[:, len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,1)
        return result/self.pot_div_factor

    '''
    def parameters(self):
        if len(self.pair_regions) > 0:
            return [self.pair_model]
        else:
            return super(MLCPotentialModel, self).parameters()
    '''


def build_unary_model(num_features, num_hidden_layers, hidden_size, output_size, use_dropout=False):
    layers = [torch.nn.Linear(num_features, hidden_size),
              torch.nn.ReLU(inplace=True)]
    if use_dropout:
        layers.append(torch.nn.Dropout())
    for i in range(num_hidden_layers-1):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
        layers.append(torch.nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(torch.nn.Dropout())
    layers.append(torch.nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

def compute_obj(model, batch, label_proportion):
    feats = batch[0]
    labels = Variable(batch[1].view(-1))
    inp = model(Variable(feats))
    inp = inp.view(-1, 2)
    #return torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.024, 0.976]))(inp, labels)
    return torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([label_proportion, 1-label_proportion]))(inp, labels)
    '''
    labels = Variable(batch[1])
    inp = model(Variable(feats))
    return torch.nn.MultiLabelSoftMarginLoss()(inp, labels)
    '''

def test_on_batch(model, batch):
    preds = np.zeros((2,2))
    feats = batch[0]
    #label = batch[1].view(-1, 159).numpy()
    label = batch[1].numpy()
    val, ind = torch.max(model(Variable(feats)).view(-1,2), 1)
    predict = ind.view(len(batch[0]), -1).data.numpy()
    #predict = torch.nn.Sigmoid()(model(Variable(feats))).round().data.numpy()
    correct = (predict*label).sum(1)
    precision = correct/(predict.sum(1) + 0.0000001)
    recall = correct/(label.sum(1)+0.0000001)

    return precision.sum(), recall.sum()

def test(model, dataloader, batch_size):
    model.eval()
    model.eval()
    num_data = 0.0
    precision = 0
    recall = 0
    for batch in dataloader:
        new_p, new_r = test_on_batch(model, batch)
        precision += new_p
        recall += new_r
        num_data += len(batch[0])
    precision /= num_data
    recall /= num_data
    f1 = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1


def train_unary(model, train_data, params):
    l_rate = params.get('l_rate', 1e-4)
    weight_decay = params.get('weight_decay', 0)
    checkpoint_dir = params.get('checkpoint_dir', 'tmp/')
    batch_size = params.get('batch_size', 10)
    validation_data = params.get('validation_data', None)
    verbose = params.get('verbose', False)
    val_interval = params.get('val_interval', 20)
    training_scheduler = params.get('training_scheduler', lambda opt: 
            torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda epoch:1.0/math.sqrt(epoch) if epoch > 0 else 1.0))
    num_epochs = params.get('num_epochs', 20)
    task_loss = params.get('task_loss', None)
    momentum = params.get('momentum', 0.)
    if modelconf.USE_GPU:
        train_data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True)
        if validation_data is not None:
            val_data_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=4, pin_memory=True)
    else:
        train_data_loader = DataLoader(train_data, batch_size=batch_size)
        if validation_data is not None:
            val_data_loader = DataLoader(validation_data, batch_size=batch_size)
    if params.get('use_adam', False):
        model_optimizer = torch.optim.Adam(model.parameters(), lr = params['l_rate'], weight_decay = weight_decay)
    else:
        model_optimizer = torch.optim.SGD(model.parameters(), lr = params['l_rate'], weight_decay = weight_decay, momentum=momentum)
    if training_scheduler is not None:
        training_scheduler = training_scheduler(model_optimizer)
    end = start = 0 
    epoch = 0
    train_results = []
    total_1 = 0.
    total_elem = 0.
    for batch in train_data_loader:
        total_1 += batch[1].sum().item()
        total_elem += (len(batch[1])*len(batch[1][0]))
    label_proportion = total_1/total_elem
        
    while epoch < num_epochs:
        epoch += 1
        print("EPOCH", epoch, (end-start))
        if training_scheduler is not None:
            training_scheduler.step()
        start = time.time() 
        model.train()
        for batch_ind,batch in enumerate(train_data_loader):
            if modelconf.USE_GPU:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda(0
            obj = compute_obj(model, batch, label_proportion)
            print("\tBATCH %d OF %d: %f"%(batch_ind+1, len(train_data_loader), obj.data[0]))
            obj.backward()
            model_optimizer.step()

        end = time.time()
        if epoch%val_interval == 0 or verbose:
            model.eval()
            itrain_score = test(model, train_data, batch_size)
            if validation_data is not None:
                val_score = test(model, validation_data_loader, batch_size)
                train_results.append((train_score, val_score))
                print("TRAIN RESULTS: ",train_score)
                print("VALIDATION RESULTS: ",val_score)
            else:
                train_results.append(train_score)
                print("TRAIN RESULTS: ",train_results[-1])
    return train_results

class TransformedMLPModel(nn.Module):
    def __init__(self, input_width, params):
        super(TransformedMLPModel, self).__init__()
        self.use_residual = params.get('use_residual', False)
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
        layer1 = nn.Linear(input_width, params['global_hidden_size'])
        layer2 = nn.Linear(params['global_hidden_size'], input_width)
        if params.get('global_layernorm', None):
            self.global_model = nn.Sequential(
                    layer1,
                    nn.LayerNorm(params['global_hidden_size']),
                    activation(),
                    layer2,
                )
        else:
            self.global_model = nn.Sequential(
                    layer1,
                    activation(),
                    layer2,
                )

    def forward(self, inp):
        result = self.global_model(inp)
        if self.use_residual:
            result = result+inp
        return result

class TransformedMLPModelResidual(nn.Module):
    def __init__(self, input_width, params):
        super(TransformedMLPModelResidual, self).__init__()
        num_global_layers = params.get('num_global_layers', 1)
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
        self.blocks = []
        for ind in range(num_global_layers):
            layer1 = nn.Linear(input_width, input_width)
            layer2 = nn.Linear(input_width, input_width)
            if params.get('global_layernorm', None):
                block = nn.Sequential(
                        layer1,
                        nn.LayerNorm(params['global_hidden_size']),
                        activation(),
                        layer2,
                    )
            else:
                block = nn.Sequential(
                        layer1,
                        activation(),
                        layer2,
                    )
            self.add_module('block%d'%ind, block)
            self.blocks.append(block)


    def forward(self, inp):
        for block in self.blocks:
            inp = block(inp)+inp
        return inp




class build_initialized_mlp_global_model_v1(nn.Module):
    def __init__(self, num_graphs, params):
        super(build_initialized_mlp_global_model_v1, self).__init__()
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
        self.div_factor = params.get('global_div_factor', 1.0)
        layer1 = nn.Linear(num_graphs, params['global_hidden_size'])
        layer1.weight.data.fill_(0.0)
        for i in range(num_graphs):
            layer1.weight.data[i, i] = 1.0
        layer1.bias.data.fill_(1.0)

        layer2 = nn.Linear(params['global_hidden_size'], 1)
        layer2.weight.data.fill_(0.0)
        for i in range(num_graphs):
            layer2.weight.data[0, i] = 1.0
        layer2.bias.data.fill_(0.0)
        
        if params.get('global_batchnorm', None):
            self.global_model = nn.Sequential(
                        layer1,
                        nn.BatchNorm1d(params['global_hidden_size']),
                        activation(),
                        layer2
                    )
        elif params.get('global_layernorm', None):
            self.global_model = nn.Sequential(
                        layer1,
                        nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False),
                        activation(),
                        layer2
                    )
        else:
            self.global_model = nn.Sequential(
                        layer1,
                        activation(),
                        layer2
                    )
    def forward(self, inp):
        return self.global_model(inp/self.div_factor)

class GlobalFeatureModel(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel, self).__init__()
        if 'global_activation' in params:
            if params['global_activation'] == 'sigmoid':
                activation = lambda: nn.Sigmoid()
            elif params['global_activation'] == 'relu':
                activation = lambda: nn.ReLU()
            elif params['global_activation'] == 'hardtanh':
                activation = lambda: nn.Hardtanh()
            elif params['global_activation'] == 'tanh':
                activation = lambda: nn.Tanh()
            else:
                raise Exception("Activation type not valid: ",params['global_activation'])
        else:
            activation = lambda: nn.Sigmoid()

        input_size = num_graphs + params['num_features']
        layers = []
        if params.get('first_bn', False):
            layers.append(nn.BatchNorm1d(input_size))
        elif params.get('first_ln', False):
            layers.append(nn.LayerNorm(input_size, elementwise_affine=False))
        layers.append(nn.Linear(input_size, params['global_hidden_size']))
        if params.get('global_bn', False):
            layers.append(nn.BatchNorm1d(params['global_hidden_size']))
        elif params.get('global_layernorm', False):
            layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
        layers.append(activation())
        if params.get('use_dropout', False):
            layers.append(nn.Dropout())
        layers.append(nn.Linear(params['global_hidden_size'], 1))
        if params.get('top_sigmoid', False):
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, pots, feats):
        inp = torch.cat([pots, feats.float()], dim=1)
        return self.model(inp)

class GlobalFeatureModel_Res(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel_Res, self).__init__()
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

        input_size = num_graphs + params['num_features']
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if params.get('first_bn', False):
                layers.append(nn.BatchNorm1d(input_size))
            if ind == 0:
                layers.append(nn.Linear(input_size, params['global_hidden_size']))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            if params.get('global_bn', False):
                layers.append(nn.BatchNorm1d(params['global_hidden_size']))
            layers.append(activation())
            if ind == params['global_hidden_size']-1:
                layers.append(nn.Linear(params['global_hidden_size'], input_size))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            block = nn.Sequential(*layers)
            self.add_module('block%d'%ind, block)
            self.blocks.append(block)
        self.use_top = params.get('use_global_top', False)
        if self.use_top:
            self.top = nn.Sequential(
                    nn.Sigmoid(),
                    #nn.Linear(params['global_hidden_size'], num_graphs),
                    nn.Linear(params['global_hidden_size'], 1),
                )

    def forward(self, pots, feats):
        inp = torch.cat([pots, feats.float()], dim=1)
        for block in self.blocks:
            inp = block(inp) + inp
        if self.use_top:
            return self.top(inp).sum(dim=1).unsqueeze(1)
        else:
            return inp.sum(dim=1).unsqueeze(1)

class GlobalFeatureModel_Resv2(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel_Resv2, self).__init__()
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
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']
        if params.get('global_layernorm', False):
            print("USING LAYERNORM IN TOP")
            self.use_ln = True
            self.unary_ln = nn.LayerNorm(self.num_unary)
            self.pair_ln = nn.LayerNorm(self.num_pair)
        else:
            self.use_ln = False
        input_size = num_graphs + params['num_features']
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if params.get('first_bn', False):
                layers.append(nn.BatchNorm1d(input_size))
            layers.append(nn.Linear(input_size, params['global_hidden_size']))
            if params.get('global_bn', False):
                layers.append(nn.BatchNorm1d(params['global_hidden_size']))
            layers.append(activation())
            layers.append(nn.Linear(params['global_hidden_size'], num_graphs))
            block = nn.Sequential(*layers)
            self.add_module('block%d'%ind, block)
            self.blocks.append(block)
        if params['global_activation'] == 'relu':
            self.use_top = True
            self.top = nn.Sequential(
                nn.Sigmoid(),
                nn.Linear(num_graphs, 1)
            )
        else:
            self.use_top = False

    def forward(self, pots, feats):
        if self.use_ln:
            new_unary = self.unary_ln(pots[:, :self.num_unary])
            new_pair = self.pair_ln(pots[:, self.num_unary:])
            pots = torch.cat([new_unary, new_pair], dim=1)
        small_inp = pots
        inp = torch.cat([pots, feats.float()], dim=1)
        for block in self.blocks:
            small_inp = block(inp) + small_inp
            inp = torch.cat([small_inp, feats.float()], dim=1)
        if self.use_top:
            return self.top(small_inp)
        else:
            return small_inp.sum(dim=1).unsqueeze(1)

class GlobalFeatureModel_Resv3(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel_Resv3, self).__init__()
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
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']
        self.num_feats = params['num_features']

        self.unary_block = nn.Sequential(
            nn.Linear(self.num_unary, self.num_unary),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_unary, self.num_unary),
        )
        self.pair_block = nn.Sequential(
            nn.Linear(self.num_pair, self.num_pair),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_pair, self.num_pair),
        )
        self.feat_block = nn.Sequential(
            nn.Linear(self.num_feats, self.num_feats),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_feats, self.num_feats),
        )
        self.combined_block = nn.Sequential(
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
            nn.Sigmoid(),
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
        )

    def forward(self, pots, feats):
        feats = feats.float()
        unary = pots[:, :self.num_unary]
        pair = pots[:, self.num_unary:]
        new_u = self.unary_block(unary) + unary
        new_p = self.pair_block(pair) + pair
        new_f = self.feat_block(feats) + feats
        inp = torch.cat([new_u, new_p, new_f], dim=1)
        result = self.combined_block(inp)
        return result.sum(dim=1).unsqueeze(1)

class GlobalFeatureModel_Resv4(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel_Resv4, self).__init__()
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
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']
        self.num_feats = params['num_features']

        self.unary_block = nn.Sequential(
            nn.Linear(self.num_unary, self.num_unary),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_unary, self.num_unary),
        )
        self.pair_block = nn.Sequential(
            nn.Linear(self.num_pair, self.num_pair),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_pair, self.num_pair),
        )
        self.feat_block = nn.Sequential(
            nn.Linear(self.num_feats, self.num_feats),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_feats, self.num_feats),
        )
        self.combined_block = nn.Sequential(
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
            nn.ReLU(inplace=True),
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
        )
        self.top = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
        )

    def forward(self, pots, feats):
        feats = feats.float()
        unary = pots[:, :self.num_unary]
        pair = pots[:, self.num_unary:]
        new_u = self.unary_block(unary) + unary
        new_p = self.pair_block(pair) + pair
        new_f = self.feat_block(feats) + feats
        inp = torch.cat([new_u, new_p, new_f], dim=1)
        result = self.combined_block(inp)+inp
        return self.top(result).sum(dim=1).unsqueeze(1)

class GlobalFeatureModel_Res_Beliefs(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalFeatureModel_Res_Beliefs, self).__init__()
        if 'global_activation' in params:
            if params['global_activation'] == 'sigmoid':
                activation = lambda: nn.Sigmoid()
            elif params['global_activation'] == 'relu':
                activation = lambda: nn.ReLU()
            elif params['global_activation'] == 'hardtanh':
                activation = lambda: nn.Hardtanh()
            elif params['global_activation'] == 'tanh':
                activation = lambda: nn.Tanh()
            else:
                raise Exception("Activation type not valid: ",params['global_activation'])
        else:
            activation = lambda: nn.Sigmoid()
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']
        self.num_feats = params['num_features']

        self.unary_transform = nn.Sequential(
            nn.Linear(2*self.num_unary, self.num_unary),
            activation(),
            nn.Linear(self.num_unary, self.num_unary),
        )
        self.pair_transform = nn.Sequential(
            nn.Linear(2*self.num_pair, self.num_pair),
            activation(),
            nn.Linear(self.num_pair, self.num_pair),
        )
        self.feat_transform = nn.Sequential(
            nn.Linear(self.num_feats, self.num_feats),
            activation(),
            nn.Linear(self.num_feats, self.num_feats),
        )
        self.combined_block = nn.Sequential(
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
            activation(),
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
        )
        self.top = nn.Sequential(
            activation(),
            nn.Linear(num_graphs+self.num_feats, num_graphs+self.num_feats),
        )

    def forward(self, pots, beliefs, feats):
        feats = feats.float()
        unary = torch.cat([pots[:, :self.num_unary], beliefs[:, :self.num_unary]], dim=1)
        pair = torch.cat([pots[:, self.num_unary:], beliefs[:, self.num_unary:]], dim=1)
        new_u = self.unary_transform(unary)
        new_p = self.pair_transform(pair)
        new_f = self.feat_transform(feats)
        inp = torch.cat([new_u, new_p, new_f], dim=1)
        result = self.combined_block(inp)+inp
        return self.top(result).sum(dim=1).unsqueeze(1)







class GlobalModel_Res(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_Res, self).__init__()
        if 'global_activation' in params:
            if params['global_activation'] == 'sigmoid':
                activation = lambda: nn.Sigmoid()
            elif params['global_activation'] == 'relu':
                activation = lambda: nn.ReLU()
            elif params['global_activation'] == 'hardtanh':
                activation = lambda: nn.Hardtanh()
            elif params['global_activation'] == 'leaky_relu':
                activation = lambda: nn.LeakyReLU(negative_slope=0.25)
            else:
                raise Exception("Activation type not valid: ",params['global_activation'])
        else:
            activation = lambda: nn.Sigmoid()

        input_size = num_graphs
        if params.get('first_bn', False):
            self.first_transform = nn.BatchNorm1d(input_size)
        elif params.get('first_ln', False):
            self.first_transform = nn.LayerNorm(input_size, elementwise_affine=False)
        else:
            self.first_transform = None
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0:
                layers.append(nn.Linear(input_size, params['global_hidden_size']))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            if params.get('global_bn', False):
                layers.append(nn.BatchNorm1d(params['global_hidden_size']))
            elif params.get('global_layernorm', False):
                layers.append(nn.LayerNorm(params['global_hidden_size']))
            layers.append(activation())
            if params.get('use_dropout', False):
                layers.append(nn.Dropout())
            if ind == params['global_hidden_size']-1:
                layers.append(nn.Linear(params['global_hidden_size'], input_size))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            block = nn.Sequential(*layers)
            self.add_module('block%d'%ind, block)
            self.blocks.append(block)
        self.use_top = params.get('use_global_top', False)
        if self.use_top:
            top_layers = []
            if params.get('last_ln', False):
                top_layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
            top_layers.append(nn.Tanh())
            if params.get('use_dropout', False):
                top_layers.append(nn.Dropout())
            top_layers.append(nn.Linear(params['global_hidden_size'], 1))
            self.top = nn.Sequential(*top_layers)

    def forward(self, pots):
        if self.first_transform is not None:
            inp = self.first_transform(pots)
        else:
            inp = pots
        for block in self.blocks:
            inp = block(inp) + inp
        if self.use_top:
            #return self.top(inp).sum(dim=1).unsqueeze(1)
            return self.top(inp)
        else:
            return inp.sum(dim=1).unsqueeze(1)

class GlobalModel_Res_GT(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_Res_GT, self).__init__()
        if 'global_activation' in params:
            if params['global_activation'] == 'sigmoid':
                activation = lambda: nn.Sigmoid()
            elif params['global_activation'] == 'relu':
                activation = lambda: nn.ReLU()
            elif params['global_activation'] == 'hardtanh':
                activation = lambda: nn.Hardtanh()
            elif params['global_activation'] == 'leaky_relu':
                activation = lambda: nn.LeakyReLU(negative_slope=0.25)
            else:
                raise Exception("Activation type not valid: ",params['global_activation'])
        else:
            activation = lambda: nn.Sigmoid()

        input_size = num_graphs
        if params.get('first_bn', False):
            self.first_transform = nn.BatchNorm1d(input_size)
        elif params.get('first_ln', False):
            self.first_transform = nn.LayerNorm(input_size, elementwise_affine=False)
        else:
            self.first_transform = None
        self.blocks = []
        for ind in range(params['num_global_layers']):
            layers = []
            if ind == 0:
                layers.append(nn.Linear(2*input_size, params['global_hidden_size']))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            if params.get('global_bn', False):
                layers.append(nn.BatchNorm1d(params['global_hidden_size']))
            elif params.get('global_layernorm', False):
                layers.append(nn.LayerNorm(params['global_hidden_size']))
            layers.append(activation())
            if params.get('use_dropout', False):
                layers.append(nn.Dropout())
            if ind == params['global_hidden_size']-1:
                layers.append(nn.Linear(params['global_hidden_size'], input_size))
            else:
                layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
            block = nn.Sequential(*layers)
            self.add_module('block%d'%ind, block)
            self.blocks.append(block)
        self.use_top = params.get('use_global_top', False)
        if self.use_top:
            top_layers = []
            if params.get('last_ln', False):
                top_layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
            top_layers.append(activation())
            if params.get('use_dropout', False):
                top_layers.append(nn.Dropout())
            top_layers.append(nn.Linear(params['global_hidden_size'], 1))
            self.top = nn.Sequential(*top_layers)

    def forward(self, pots, gt):
        inp = torch.cat([pots, gt], dim=1)
        if self.first_transform is not None:
            inp = self.first_transform(inp)
        for block in self.blocks:
            inp = block(inp) + inp
        if self.use_top:
            #return self.top(inp).sum(dim=1).unsqueeze(1)
            return self.top(inp)
        else:
            return inp.sum(dim=1).unsqueeze(1)


class GlobalModel_Resv2(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_Resv2, self).__init__()
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

        in_width = params['num_unary_potentials']+params['num_pair_potentials']
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']

        self.unary_block = nn.Sequential(
                nn.Linear(params['num_unary_potentials'], params['num_unary_potentials']),
                activation(),
                nn.Linear(params['num_unary_potentials'], params['num_unary_potentials']),
            )
        self.pair_block = nn.Sequential(
            nn.Linear(params['num_pair_potentials'], params['num_pair_potentials']),
            activation(),
            nn.Linear(params['num_pair_potentials'], params['num_pair_potentials']),
        )
        self.combined = nn.Sequential(
            nn.Linear(in_width, params['global_hidden_size']),
            activation(),
            nn.Linear(params['global_hidden_size'], 1))


    def forward(self, inp):
        unary = self.unary_block(inp[:, :self.num_unary]) + inp[:, :self.num_unary]
        pair = self.pair_block(inp[:, self.num_unary:]) + inp[:, self.num_unary:]
        return self.combined(torch.cat([unary, pair], dim=1))


class build_initialized_mlp_global_model_v2(nn.Module):
    def __init__(self, num_graphs, params):
        super(build_initialized_mlp_global_model_v2, self).__init__()
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
            layers.append(nn.LayerNorm(num_graphs, elementwise_affine=False))
        layers.append(nn.Linear(num_graphs, params['global_hidden_size']))
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
            #top_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
            layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
            layers.append(nn.Tanh())
            layers.append(nn.Linear(params['global_hidden_size'], 1))
        else:
            top_layer = nn.Linear(params['global_hidden_size'], 1)
            if max_val is not None:
                top_layer.weight.data.uniform_(-1*max_val, max_val)
                top_layer.bias.data.fill_(0.0)
            layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, inp):
        result = self.global_model(inp)
        if self.use_global_top:
            result = result.sum(dim=1).unsqueeze(1)
        return result
'''
def build_initialized_mlp_global_model_v2(num_graphs, params):
    if 'global_activation' in params:
        if params['global_activation'] == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params['global_activation'] == 'relu':
            activation = lambda: nn.ReLU()
        else:
            raise Exception("Activation type not valid: ",params['global_activation'])
    else:
        activation = lambda: nn.Sigmoid()
    layer1 = nn.Linear(num_graphs, params['global_hidden_size'])
    layer1.weight.data.uniform_(-0.1, 0.1)
    layer1.bias.data.fill_(0.1)

    layer2 = nn.Linear(params['global_hidden_size'], 1)
    layer2.weight.data.uniform_(-0.1, 0.1)
    layer2.bias.data.fill_(0.0)
    
    global_model = nn.Sequential(
                layer1,
                activation(),
                layer2
            )
    return global_model
'''


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
        self.pot_ln = nn.LayerNorm(num_graphs, elementwise_affine=False)
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
            #top_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
            layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
        else:
            top_layer = nn.Linear(params['global_hidden_size'], 1)
            if max_val is not None:
                top_layer.weight.data.uniform_(-1*max_val, max_val)
                top_layer.bias.data.fill_(0.0)
            layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, pots, gt):
        pots = self.pot_ln(pots)
        inp = torch.cat([pots, 2*(gt-0.5)], dim=1)
        #inp = torch.cat([pots, gt], dim=1)
        result = self.global_model(inp)
        if self.use_global_top:
            result = result.sum(dim=1).unsqueeze(1)
        return result

class GlobalModel_GT_v2(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_GT_v2, self).__init__()
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
        self.pot_ln = nn.LayerNorm(num_graphs, elementwise_affine=False)
        layers = []
        if params.get('first_ln', False):
            layers.append(nn.LayerNorm(num_graphs, elementwise_affine=False))
        layers.append(nn.Linear(num_graphs, params['global_hidden_size']))
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
        top_layer = nn.Linear(params['global_hidden_size'], num_graphs)
        if max_val is not None:
            top_layer.weight.data.uniform_(-1*max_val, max_val)
            top_layer.bias.data.fill_(0.0)
        layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, pots, gt):
        pots = self.pot_ln(pots)
        #inp = torch.cat([pots, gt], dim=1)
        result = self.global_model(pots)
        return (-1*(result - gt)**2).sum(dim=1).unsqueeze(1)



class GlobalLinear_GT(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalLinear_GT, self).__init__()
        self.linear = nn.Linear(2*num_graphs, 1)

    def forward(self, pots, gt):
        inp = torch.cat([pots, 2*(gt-0.5)], dim=1)
        return self.linear(inp)

class GlobalModel_Beliefs(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_Beliefs, self).__init__()
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
        if params.get('first_ln', False):
            self.pot_transform = nn.LayerNorm(num_graphs, elementwise_affine=False)
            self.bel_transform = nn.LayerNorm(num_graphs, elementwise_affine=False)
        else:
            self.pot_transform = None
            self.bel_transform = None
        max_val = params.get('global_init_val', None)
        layer1 = nn.Linear(2*num_graphs, params['global_hidden_size'])
        if max_val is not None:
            layer1.weight.data.uniform_(-1*max_val, max_val)
            layer1.bias.data.fill_(max_val)

        if params.get('global_batchnorm', None):
            layers = [layer1, nn.BatchNorm1d(params['global_hidden_size']), activation()]
        elif params.get('global_layernorm', None):
            layers = [layer1, nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False), activation()]
            #layers = [nn.LayerNorm(num_graphs, elementwise_affine=False), layer1, activation()]
        else:
            layers = [layer1, activation()]
        if params.get('use_dropout', False):
            layers.append(nn.Dropout())

        for i in range(num_hidden_layers - 1):
            new_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
            if max_val is not None:
                new_layer.weight.data.uniform_(-1*max_val, max_val)
                new_layer.bias.data.fill_(max_val)
            layers.append(new_layer)
            #if params.get('global_batchnorm', None):
            #    layers.append(nn.batchnorm1d(params['global_hidden_size']))
            #elif params.get('global_layernorm', None):
            #    layers.append(nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False))
            layers.append(activation())
            if params.get('use_dropout', False):
                layers.append(nn.Dropout())
        self.use_global_top = params.get('use_global_top', False)
        if self.use_global_top:
            top_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
        else:
            top_layer = nn.Linear(params['global_hidden_size'], 1)
            if max_val is not None:
                top_layer.weight.data.uniform_(-1*max_val, max_val)
                top_layer.bias.data.fill_(0.0)
        layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, pots, beliefs):
        if self.pot_transform is not None:
            pots = self.pot_transform(pots)
            beliefs = self.bel_transform(beliefs)
        inp = torch.cat([pots, beliefs], dim=1)
        result = self.global_model(inp)
        if self.use_global_top:
            result = result.sum(dim=1).unsqueeze(1)
        return result

class GlobalModel_GT_Beliefs(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_GT_Beliefs, self).__init__()
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
        #self.pot_transform = nn.LayerNorm(num_graphs, elementwise_affine=False)
        num_hidden_layers = params.get('num_global_layers', 1)
        max_val = params.get('global_init_val', None)
        layers = []
        if params.get('first_ln', False):
            layers.append(nn.LayerNorm(3*num_graphs, elementwise_affine=False))
        layers.append(nn.Linear(3*num_graphs, params['global_hidden_size']))
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
            #top_layer = nn.Linear(params['global_hidden_size'], params['global_hidden_size'])
            layers.append(nn.Linear(params['global_hidden_size'], params['global_hidden_size']))
        else:
            top_layer = nn.Linear(params['global_hidden_size'], 1)
            if max_val is not None:
                top_layer.weight.data.uniform_(-1*max_val, max_val)
                top_layer.bias.data.fill_(0.0)
            layers.append(top_layer)
        
        self.global_model = nn.Sequential(*layers)
        #self.model = nn.Linear(1,1)

    def forward(self, pots, beliefs, gt):
        #return (-1*((beliefs-gt)**2)).sum(dim=1).unsqueeze(1)
        #pots = self.pot_transform(pots)
        #inp = torch.cat([pots, 2*(gt-0.5)], dim=1)
        inp = torch.cat([pots, 2*(beliefs-0.5), 2*(gt-0.5)], dim=1)
        result = self.global_model(inp)
        if self.use_global_top:
            result = result.sum(dim=1).unsqueeze(1)
        return result

class GlobalModel_GT_Beliefs_v2(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_GT_Beliefs_v2, self).__init__()
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
        #self.pot_transform = nn.LayerNorm(num_graphs, elementwise_affine=False)
        if params.get('use_simple_top', False):
            self.simple_top = True
            self.model = nn.Linear(1, 1)
            return
        self.simple_top = False
        num_hidden_layers = params.get('num_global_layers', 1)
        max_val = params.get('global_init_val', None)
        layers = []
        if params.get('first_ln', False):
            layers.append(nn.LayerNorm(num_graphs, elementwise_affine=False))
        layers.append(nn.Linear(num_graphs, params['global_hidden_size']))
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
        layers.append(nn.Linear(params['global_hidden_size'], num_graphs))
        
        self.global_model = nn.Sequential(*layers)

    def forward(self, pots, beliefs, gt):
        if self.simple_top:
            return (-1*((beliefs-gt)**2)).sum(dim=1).unsqueeze(1)
        #pots = self.pot_transform(pots)
        #inp = torch.cat([pots, 2*(gt-0.5)], dim=1)
        #inp = torch.cat([pots, 2*(beliefs-0.5), 2*(gt-0.5)], dim=1)
        result = self.global_model(beliefs)
        return (-1*((result-gt)**2)).sum(dim=1).unsqueeze(1)



class MLP_Model_v3(nn.Module):

    def __init__(self, num_graphs, params):
        super(MLP_Model_v3, self).__init__()
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
        self.num_unary = params['num_unary_potentials']
        self.num_pair = params['num_pair_potentials']
        self.unary_transform = nn.Linear(self.num_unary, self.num_unary)
        self.pair_transform = nn.Linear(self.num_pair, self.num_pair)
        in_width = params['num_unary_potentials']+params['num_pair_potentials']
        if params.get('global_layernorm', None):
            self.unary_transform = nn.Sequential(
                nn.Linear(params['num_unary_potentials'], params['num_unary_potentials']),
                nn.LayerNorm(params['num_unary_potentials'], elementwise_affine=False),
                activation(),
            )
            self.pair_transform = nn.Sequential(
                nn.Linear(params['num_pair_potentials'], params['num_pair_potentials']),
                nn.LayerNorm(params['num_pair_potentials'], elementwise_affine=False),
                activation(),
            )
            self.combined = nn.Sequential(
                nn.Linear(in_width, params['global_hidden_size']),
                nn.LayerNorm(params['global_hidden_size'], elementwise_affine=False),
                activation(),
                nn.Linear(params['global_hidden_size'], 1))
        else:
            self.unary_transform = nn.Sequential(
                nn.Linear(params['num_unary_potentials'], params['num_unary_potentials']),
                activation(),
            )
            self.pair_transform = nn.Sequential(
                nn.Linear(params['num_pair_potentials'], params['num_pair_potentials']),
                activation(),
            )
            self.combined = nn.Sequential(
                nn.Linear(in_width, params['global_hidden_size']),
                activation(),
                nn.Linear(params['global_hidden_size'], 1))


    def forward(self, inp):
        unary = self.unary_transform(inp[:, :self.num_unary])
        pair = self.pair_transform(inp[:, self.num_unary:])
        return self.combined(torch.cat([unary, pair], dim=1))



def calculate_accuracy(dataset, found):
    
    gt = np.array([datum[0].numpy() for datum in dataset])
    found = np.array(found)

    correct = (gt * found).sum(1)
    precision = correct / (found.sum(1) + 0.0000001)
    recall = correct / (gt.sum(1) + 0.0000001)
    precision = precision.mean()
    recall = recall.mean()
    f1 = 2*precision*recall / (precision+recall)

    #preds = [np.zeros((2,2)) for _ in range(159)]
    '''
    for true, guess in zip(correct, found):
        for ind, (true_p, guess_p) in enumerate(zip(true, guess)):
            preds[ind][true_p, guess_p] += 1
    precision = 0
    recall = 0
    for ind in xrange(159):
        precision += preds[ind][1, 1]/(preds[ind][1, 1] + preds[ind][0, 1])
        recall += preds[ind][1, 1]/(preds[ind][1, 1] + preds[ind][1, 0])
    precision /= 159
    recall /= 159
    '''
    '''
    for true, guess in zip(correct, found):
        preds = np.zeros((2,2))
        for true_p, guess_p in zip(true, guess):
            preds[true_p, guess_p] += 1
            precision += preds[1, 1]/(preds[1, 1] + preds[0, 1])
        recall += preds[1, 1]/(preds[1, 1] + preds[1, 0])
    precision /= len(correct)
    recall /= len(correct)
    f1 = 2*(precision*recall)/(precision+recall)
    '''

    return precision, recall, f1




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run experiment for multilabel classification datasets')
    parser.add_argument('dataset', choices=['bibtex', 'bookmarks', 'delicious'])
    parser.add_argument('model', choices=['unary', 'unary_v2', 'pairwise', 'pairwise_transformed', 'global_mlp', 'global_linear_gt'])
    parser.add_argument('working_dir')
    parser.add_argument('feature_file')
    parser.add_argument('label_file')
    parser.add_argument('--combined', action='store_true')
    parser.add_argument('-p', '--pretrain')
    parser.add_argument('--load_global', action='store_true')
    parser.add_argument('--unary_model')
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
    parser.add_argument('--mp_itrs', type=int, default=1000)
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
    parser.add_argument('--global_div_factor', type=float, default=1.0)

    args = parser.parse_args()
    
    if args.gpu:
        modelconf.use_gpu()
    if args.dataset == 'bibtex':
        if args.model == 'unary':
            dataset = BibtexUnaryDataset
        else:
            dataset = BibtexDataset
        
        if args.combined:
            train_data = dataset(FULL, args.feature_file, args.label_file)
            val_data = None
        else:
            train_data = dataset(TRAIN, args.feature_file, args.label_file)
            val_data = dataset(VAL, args.feature_file, args.label_file)
        num_labels = 159
        num_features = 1836
    elif args.dataset == 'bookmarks':
        if args.model == 'unary':
            dataset = BookmarksUnaryDataset
        else:
            dataset = BookmarksDataset

        train_data = dataset(TRAIN, args.feature_file, args.label_file)
        val_data = dataset(VAL, args.feature_file, args.label_file)
        num_features = 2150
        num_labels = 208
    elif args.dataset == 'delicious':
        if args.model == 'unary':
            dataset = DeliciousUnaryDataset
        else:
            dataset = DeliciousDataset
        
        if args.combined:
            train_data = dataset(FULL, args.feature_file, args.label_file)
            val_data = None
        else:
            train_data = dataset(TRAIN, args.feature_file, args.label_file)
            val_data = dataset(VAL, args.feature_file, args.label_file)
        num_labels = 983
        num_features = 500

    def scaled_identity_diff(true_val, other_val):
        return float(true_val != other_val)/args.loss_aug_div_factor

    if args.model == 'unary':
        train_params = {
            'num_epochs':args.num_epochs,
            'l_rate':args.l_rate,
            'batch_size':args.batch_size,
            'weight_decay':args.weight_decay,
            'use_adam':args.use_adam,
            'momentum':args.momentum,
            'validation_data':val_data,
            'val_interval':args.val_interval,
        }
        if args.no_l_rate_decay:
            train_params['training_scheduler'] = None

        model = build_unary_model(num_features, args.unary_hidden_layers, args.unary_hidden_size, num_labels*2, args.use_dropout)
        print("UNARY MODEL: ",model)
        if args.unary_model is not None:
            model.load_state_dict(torch.load(args.unary_model)) 
        if val_data is not None:
            test_result = test(model, val_data, args.batch_size)
            print("STARTING LOSS: ",test_result)
        train_unary(model, train_data,train_params)
        if val_data is not None:
            print("STARTING TEST")
            test_result = test(model, val_data, args.batch_size)
            print("VALIDATION RESULT: ",test_result)

        torch.save(model.state_dict(), os.path.join(args.working_dir, 'final_model'))
        sys.exit(0)
        
    if args.dataset == 'bibtex':
        num_features = 1836
        nodes = list(range(159))
        #pairs = [(19, 70), (19, 71), (33, 66), (33, 67), (36, 68), (36, 84), (36, 105), (36, 106), (37, 52), (43, 75), (43, 152), (48, 140), (57, 84), (66, 67), (68, 84), (68, 105), (68, 106), (68, 118), (68, 121), (70, 71), (71, 80), (75, 152), (77, 119), (77, 136), (83, 84), (84, 88), (84, 89), (84, 90), (84, 105), (84, 106), (88, 89), (88, 90), (88, 105), (88, 106), (88, 126), (88, 127), (89, 90), (89, 105), (89, 106), (90, 105), (90, 106), (90, 126), (90, 127), (96, 97), (96, 124), (96, 130), (96, 156), (96, 157), (97, 124), (97, 130), (97, 156), (97, 157), (100, 139), (103, 104), (105, 106), (118, 121), (119, 136), (122, 124), (124, 130), (124, 156), (124, 157), (126, 127), (129, 130), (130, 156), (130, 157), (137, 138), (156, 157), (28, 101), (45, 63), (24, 26), (28, 48), (28, 140), (78, 137), (78, 138), (83, 120), (84, 120), (88, 141), (89, 126), (89, 127), (90, 141), (28, 109), (29, 141), (42, 43), (42, 75), (81, 109), (88, 116), (89, 116), (90, 116), (96, 137), (96, 138), (97, 137), (97, 138), (116, 126), (116, 127), (11, 135), (23, 57), (23, 84), (24, 135), (33, 133), (36, 137), (36, 138), (42, 47), (68, 137), (68, 138), (99, 135), (103, 137), (103, 138), (104, 137), (104, 138), (41, 108), (59, 84), (59, 105), (59, 106), (81, 86), (84, 135), (105, 135), (106, 135), (135, 137), (135, 138), (15, 152), (77, 121), (83, 133), (84, 133), (119, 121), (121, 136), (6, 68), (6, 118), (6, 121), (11, 42), (11, 48), (11, 140), (19, 66), (19, 67), (19, 109), (19, 135), (36, 110), (42, 48), (42, 140), (48, 101), (51, 63), (51, 77), (51, 119), (57, 130), (63, 77), (63, 119), (66, 70), (66, 71), (67, 70), (67, 71), (68, 110), (70, 109), (70, 135), (71, 109), (71, 135), (84, 130), (98, 120), (101, 140), (108, 120), (120, 123), (124, 137), (124, 138), (126, 141), (127, 141), (137, 156), (137, 157), (138, 156), (138, 157), (4, 109), (11, 120), (14, 123), (17, 51), (17, 63), (17, 77), (17, 119), (20, 116), (24, 29), (26, 29), (28, 81), (29, 57), (29, 84), (29, 88), (29, 90), (29, 113), (48, 120), (57, 92), (57, 100), (62, 116), (84, 92), (84, 100), (85, 137), (85, 138), (120, 140), (134, 157), (6, 77), (6, 83), (6, 84), (6, 119), (11, 89), (12, 15), (12, 37), (15, 37), (15, 77), (15, 119), (17, 43), (17, 75), (22, 37), (24, 99), (28, 136), (31, 135), (36, 48), (36, 81), (36, 101), (36, 103), (36, 104), (37, 79), (37, 109), (37, 153), (42, 57), (42, 84), (42, 105), (42, 106), (42, 109), (42, 116), (42, 121), (44, 108), (48, 68), (52, 109), (57, 105), (57, 106), (57, 109), (57, 116), (57, 121), (68, 81), (68, 101), (68, 103), (68, 104), (73, 121), (77, 86), (77, 152), (78, 85), (78, 103), (78, 104), (84, 109), (84, 116), (84, 121), (101, 124), (101, 156), (101, 157), (105, 109), (105, 116), (105, 121), (106, 109), (106, 116), (106, 121), (109, 116), (109, 121), (116, 121), (116, 155), (119, 152), (124, 135), (135, 156), (135, 157), (3, 6), (3, 68), (3, 118), (3, 121), (6, 136), (44, 120), (57, 137), (57, 138), (68, 77), (68, 119), (68, 136), (77, 118), (83, 98), (83, 108), (83, 123), (84, 98), (84, 108), (84, 123), (84, 137), (84, 138), (112, 120), (118, 119), (118, 136), (120, 133), (12, 135), (15, 66), (15, 67), (17, 109), (36, 135), (42, 101), (43, 109), (66, 133), (67, 133), (68, 135), (73, 109), (75, 109), (101, 116), (103, 135)]
        pairs = [(134, 144), (134, 146), (134, 149), (134, 143), (41, 134), (52, 131), (134, 151), (49, 81), (122, 156), (44, 63), (88, 134), (10, 49), (61, 63), (107, 134), (36, 84), (3, 63), (134, 150), (134, 142), (134, 138), (75, 84), (10, 81), (134, 153), (86, 131), (113, 134), (36, 75), (134, 141), (5, 44), (54, 139), (134, 148), (23, 63), (3, 61), (75, 83), (97, 129), (24, 134), (134, 147), (132, 134), (114, 134), (97, 134), (48, 131), (134, 145), (63, 65), (9, 10), (5, 63), (44, 65), (36, 106), (96, 129), (3, 44), (27, 77), (99, 134), (57, 75), (122, 124), (101, 131), (42, 83), (26, 112), (44, 64), (50, 134), (38, 62), (3, 65), (6, 29), (32, 134), (92, 134), (56, 95), (97, 144), (113, 150), (66, 155), (52, 117), (5, 65), (43, 75), (1, 95), (9, 85), (126, 134), (11, 131), (16, 83), (3, 23), (44, 61), (5, 64), (31, 154), (29, 55), (3, 5), (9, 49), (127, 134), (29, 53), (90, 134), (57, 84), (9, 69), (24, 144), (38, 134), (23, 44), (63, 76), (104, 122), (54, 129), (10, 63), (104, 124), (1, 56), (79, 131), (41, 91), (132, 151), (6, 55), (53, 55), (24, 97), (103, 104), (91, 134), (122, 158), (6, 53), (1, 72), (23, 61), (69, 85), (96, 134), (63, 64), (72, 95), (107, 153), (124, 156), (61, 76), (10, 45), (56, 72), (47, 134), (40, 77), (86, 101), (104, 156), (103, 124), (70, 80), (42, 84), (10, 125), (61, 65), (128, 129), (93, 122), (42, 75), (5, 61), (36, 105), (96, 128), (16, 77), (96, 97), (27, 40), (121, 156), (84, 106), (36, 68), (117, 131), (75, 94), (34, 87), (53, 93), (35, 80), (22, 128), (21, 54), (36, 43), (108, 120), (41, 149), (39, 54), (9, 63), (83, 84), (93, 156), (105, 134), (99, 146), (151, 153), (64, 65), (9, 81), (129, 139), (6, 93), (23, 65), (55, 117), (134, 135), (58, 134), (43, 106), (68, 75), (24, 138), (93, 104), (134, 137), (36, 131), (39, 139), (97, 128), (54, 152), (10, 74), (104, 123), (58, 149), (1, 122), (75, 106), (43, 84), (21, 139), (70, 104), (96, 144), (131, 155), (36, 57), (4, 125), (52, 134), (3, 64), (75, 104), (14, 41), (36, 118), (87, 93), (11, 36), (129, 134), (36, 141), (41, 97), (36, 83), (87, 156), (29, 93), (27, 126), (55, 93), (3, 76), (41, 146), (18, 152), (25, 60), (138, 144), (113, 132), (4, 131), (125, 131), (2, 26), (35, 47), (7, 35), (129, 157), (9, 45), (109, 134), (22, 129), (104, 139), (84, 118), (68, 84), (139, 152), (7, 47), (66, 70), (57, 83), (128, 130), (36, 101), (54, 122), (145, 153), (6, 131), (6, 122), (97, 138), (70, 93), (41, 126), (41, 143), (138, 146), (19, 75), (16, 27), (0, 26), (93, 103), (27, 134), (54, 157), (95, 122), (129, 144), (7, 102), (11, 52), (41, 88), (75, 141), (71, 72), (41, 144), (83, 94), (34, 93), (52, 143), (1, 124), (88, 141), (46, 117), (36, 42), (107, 126), (6, 96), (5, 23), (27, 140), (54, 124), (115, 156), (54, 104), (93, 95), (6, 125), (24, 96), (41, 148), (133, 134), (6, 33), (66, 129), (88, 145), (84, 141), (107, 145), (115, 122), (122, 123), (6, 129), (48, 117), (70, 116), (91, 126), (88, 146), (4, 6), (33, 87), (139, 157), (95, 124), (103, 122), (56, 122), (114, 144), (50, 142), (1, 156), (62, 118), (54, 93), (129, 138), (33, 91), (41, 129), (34, 67), (4, 91), (88, 142), (127, 143), (70, 75), (129, 156), (56, 124), (36, 60), (42, 118), (141, 144), (8, 156), (122, 157), (91, 125), (77, 140), (79, 86), (61, 64), (156, 157), (6, 54), (21, 129), (23, 76), (88, 144), (24, 129), (1, 93), (30, 93), (115, 124), (70, 71), (36, 73), (18, 51), (54, 123), (46, 131), (54, 156), (19, 89), (45, 63), (92, 126), (1, 54), (85, 118), (0, 112), (96, 130), (52, 88), (4, 81), (90, 91), (22, 130), (48, 104), (41, 141), (88, 96), (36, 138), (38, 146), (27, 75), (33, 34), (12, 129), (92, 149), (97, 143), (33, 93), (1, 6), (68, 106), (131, 138), (41, 153), (4, 90), (32, 145), (17, 104), (93, 124), (6, 104), (72, 93), (18, 104), (69, 134), (122, 139), (18, 93), (83, 116), (20, 158), (52, 101), (63, 119), (129, 130), (70, 87), (48, 70), (124, 139), (52, 79), (68, 83), (107, 113), (18, 129), (63, 85), (67, 121), (20, 93), (40, 102), (106, 141), (109, 146), (18, 54), (28, 77), (41, 127), (20, 21), (70, 156), (66, 104), (18, 135), (91, 148), (29, 33), (37, 131), (92, 145), (39, 152), (72, 124), (7, 118), (33, 125), (43, 68), (123, 139), (141, 149), (6, 19), (34, 156), (87, 122), (2, 112), (40, 140), (88, 152), (21, 39), (2, 93), (39, 104), (117, 152), (124, 157), (66, 140), (23, 45), (40, 75), (70, 155), (50, 146), (19, 83), (104, 129), (88, 153), (85, 104), (12, 157), (18, 87), (114, 146), (50, 88), (6, 134), (22, 96), (30, 122), (20, 75), (129, 140), (75, 102), (33, 53), (96, 141), (152, 157), (98, 133), (41, 99), (1, 98), (114, 143), (88, 97), (89, 134), (70, 109), (59, 96), (129, 131), (6, 90), (81, 131), (20, 104), (48, 155), (75, 110), (30, 72), (107, 146), (30, 43), (11, 101), (6, 152), (41, 138), (132, 138), (101, 105), (91, 143), (123, 124), (26, 98), (138, 151), (123, 157), (20, 54), (89, 152), (88, 126), (6, 91), (45, 69), (108, 134), (35, 131), (72, 122), (117, 155), (88, 90), (97, 141), (75, 156), (52, 86), (75, 138), (107, 138), (18, 39), (53, 70), (89, 131), (114, 141), (6, 124), (75, 116), (118, 141), (27, 127), (6, 97), (49, 63), (105, 147), (84, 138), (27, 91), (56, 156), (19, 129), (93, 123), (6, 139), (132, 150), (90, 126), (126, 149), (81, 104), (126, 145), (41, 96), (1, 103), (57, 141), (6, 21), (88, 107), (42, 57), (14, 88), (43, 157), (84, 105), (106, 131), (96, 138)]
    elif args.dataset == 'bookmarks':
        num_features = 2150
        nodes = list(range(208))
        pairs = [(89, 109), (144, 145), (124, 170), (170, 182), (31, 42), (42, 87), (197, 198), (31, 182), (182, 198), (103, 170), (31, 82), (182, 197), (42, 82), (62, 170), (13, 198), (143, 160), (87, 170), (87, 138), (100, 124), (100, 170), (138, 170), (116, 149), (36, 199), (173, 196), (60, 175), (53, 92), (175, 198), (86, 197), (127, 173), (127, 196), (0, 197), (13, 15), (23, 24), (54, 92), (42, 170), (86, 182), (53, 54), (165, 198), (31, 87), (169, 198), (42, 182), (23, 35), (24, 35), (59, 134), (74, 78), (8, 35), (170, 197), (123, 160), (23, 110), (24, 45), (35, 45), (87, 182), (24, 110), (31, 170), (8, 23), (2, 198), (8, 45), (23, 45), (8, 24), (116, 172), (160, 197), (45, 110), (110, 147), (35, 110), (13, 197), (159, 197), (86, 198), (8, 110), (43, 138), (155, 182), (175, 176), (19, 117), (12, 98), (109, 110), (14, 15), (170, 181), (170, 204), (170, 198), (155, 197), (61, 170), (124, 182), (43, 170), (13, 14), (67, 155), (41, 199), (1, 179), (48, 97), (47, 87), (197, 199), (170, 191), (60, 198), (2, 88), (75, 180), (160, 161), (155, 156), (13, 182), (2, 197), (76, 199), (138, 146), (181, 182), (182, 199), (15, 198), (138, 197), (146, 147), (102, 124), (165, 197), (84, 85), (41, 182), (31, 86), (29, 198), (60, 176), (146, 182), (37, 179), (159, 160), (138, 182), (37, 171), (24, 81), (36, 76), (61, 182), (62, 103), (82, 87), (37, 193), (28, 198), (23, 81), (81, 110), (181, 198), (1, 37), (13, 86), (150, 198), (45, 81), (155, 198), (165, 182), (35, 81), (166, 198), (13, 119), (86, 155), (103, 126), (150, 182), (90, 110), (199, 200), (147, 153), (41, 197), (175, 182), (75, 186), (165, 175), (22, 138), (124, 138), (40, 175), (13, 150), (3, 197), (90, 147), (8, 81), (160, 198), (138, 186), (114, 115), (75, 187), (30, 170), (37, 72), (197, 206), (40, 198), (87, 124), (13, 170), (171, 193), (25, 110), (76, 197), (133, 134), (146, 155), (146, 197), (116, 125), (86, 170), (169, 175), (103, 124), (181, 197), (2, 182), (61, 124), (165, 169), (118, 165), (88, 197), (60, 182), (1, 147), (90, 109), (165, 170), (41, 193), (31, 91), (176, 198), (59, 79), (43, 197), (147, 155), (3, 198), (14, 198), (27, 138), (52, 137), (17, 175), (37, 136), (42, 47), (75, 182), (29, 165), (100, 188), (6, 103), (7, 41), (124, 197), (37, 41), (88, 138), (60, 94), (61, 197), (87, 191), (166, 175), (109, 147), (30, 154), (198, 202), (155, 157), (198, 199), (13, 175), (116, 130), (3, 160), (18, 175), (138, 141), (13, 178), (17, 198), (147, 197), (48, 147), (51, 179), (60, 165), (13, 147), (37, 51), (1, 51), (86, 147), (178, 198), (31, 100), (40, 182), (43, 87), (18, 198), (171, 179), (15, 182), (121, 198), (75, 100), (67, 198), (119, 198), (1, 48), (165, 168), (1, 171), (87, 197), (100, 182), (147, 182), (61, 198), (147, 198), (13, 165), (17, 166), (85, 157), (2, 138), (5, 87), (40, 60), (17, 165), (67, 182), (121, 182), (160, 195), (13, 155), (99, 182), (18, 19), (168, 198), (18, 117), (67, 105), (36, 197), (121, 197), (84, 157), (28, 170), (15, 86), (41, 86), (169, 182), (126, 170), (60, 197), (28, 165), (186, 199), (42, 206), (202, 203), (182, 201), (31, 138), (155, 175), (42, 138), (180, 182), (176, 182), (37, 86), (86, 179), (87, 206), (124, 198), (167, 175), (128, 160), (182, 204), (75, 146), (132, 133), (146, 170), (198, 201), (143, 197), (123, 143), (146, 186), (7, 68), (150, 197), (36, 200), (48, 49), (18, 165), (178, 197), (43, 182), (18, 182), (18, 60), (62, 182), (67, 197), (197, 200), (41, 68), (14, 182), (60, 169), (41, 170), (182, 206), (25, 147), (146, 153), (28, 29), (23, 147), (29, 197), (149, 182), (13, 202), (15, 119), (147, 179), (61, 62), (37, 53), (2, 170), (41, 198), (41, 97), (41, 48), (37, 147), (175, 197), (147, 175), (24, 147), (3, 12), (160, 182), (13, 29), (160, 206), (170, 202), (28, 197), (150, 206), (100, 204), (55, 56), (45, 147), (36, 41), (35, 147), (88, 199), (71, 110), (62, 124), (98, 146), (59, 198), (98, 138), (170, 206), (13, 41), (40, 167), (197, 202), (138, 198), (13, 169), (8, 147), (62, 204), (41, 138), (169, 197), (124, 181), (19, 207), (28, 182), (143, 206), (170, 178), (17, 18), (69, 191), (1, 193), (147, 170), (5, 138), (41, 147), (155, 170), (125, 148), (40, 176), (53, 193), (111, 160), (167, 198), (143, 159), (61, 121), (86, 99), (138, 206), (1, 72), (2, 199), (147, 160), (162, 197), (86, 119), (60, 147), (10, 115), (29, 169), (46, 170), (41, 171), (15, 197), (165, 166), (86, 124), (192, 198), (5, 197), (191, 197), (36, 88), (20, 146), (28, 169), (86, 146), (41, 190), (13, 60), (169, 170), (36, 186), (179, 193), (30, 152), (146, 198), (155, 165), (43, 191), (0, 147), (13, 199), (48, 182), (166, 182), (88, 182), (165, 176), (162, 182), (80, 134), (13, 98), (87, 181), (169, 202), (9, 160), (88, 198), (26, 124), (36, 182), (0, 198), (9, 198), (76, 200), (81, 147), (37, 101), (29, 182), (40, 169), (61, 86), (170, 199), (166, 169), (29, 147), (119, 182), (128, 147), (119, 178), (119, 150), (86, 169), (1, 86), (18, 176), (86, 121), (112, 204), (37, 48), (86, 199), (15, 165), (178, 182), (51, 171), (86, 177), (97, 182), (7, 37), (86, 168), (98, 198), (108, 119), (18, 40), (143, 182), (94, 175), (3, 206), (160, 170), (135, 170), (37, 99), (18, 166), (158, 170), (94, 98), (95, 146), (86, 175), (119, 136), (3, 13), (86, 165), (47, 170), (160, 202), (124, 204), (177, 182), (118, 170), (57, 182), (116, 182), (11, 147), (27, 170), (15, 37), (28, 202), (21, 198), (182, 202), (147, 178), (41, 43), (13, 181), (139, 160), (87, 160), (31, 206), (43, 146), (7, 80), (98, 124), (169, 176), (17, 60), (60, 155), (64, 77), (79, 170), (100, 146), (99, 155), (60, 166), (103, 182), (155, 176), (13, 160), (61, 138), (68, 170), (46, 182), (15, 150), (13, 176), (97, 147), (6, 126), (32, 160), (86, 178), (87, 98), (13, 67), (147, 169), (22, 42), (41, 75), (15, 75), (12, 94), (63, 77), (41, 179), (25, 109), (155, 181), (100, 138), (30, 100), (146, 199), (15, 41), (40, 166), (118, 197), (1, 53), (108, 192), (168, 169), (48, 146), (72, 193), (98, 197), (1, 99), (143, 195), (22, 170), (9, 94), (98, 170), (76, 186), (13, 40), (31, 47), (43, 124), (161, 197), (13, 28), (48, 153), (30, 138), (75, 138), (33, 116), (75, 170), (20, 101), (112, 170), (17, 182), (20, 98), (48, 198), (68, 80), (2, 87), (99, 147), (13, 37), (76, 182), (13, 108), (147, 165), (1, 11), (7, 133), (99, 179), (9, 147), (186, 197), (9, 197), (198, 206), (87, 186), (5, 198), (48, 179), (17, 176), (158, 182), (76, 88), (94, 147), (121, 162), (195, 197), (76, 146), (146, 149), (28, 147), (23, 25), (97, 146), (124, 202), (108, 198), (24, 25), (13, 21), (119, 177), (14, 150), (98, 147), (80, 133), (21, 86), (41, 146), (22, 31), (15, 178), (57, 170), (89, 147), (58, 65), (95, 138), (10, 114), (138, 199), (25, 90), (170, 194), (39, 170), (136, 179), (43, 199), (87, 143), (182, 200), (104, 170), (0, 165), (72, 153), (40, 165), (2, 86), (132, 134), (124, 135), (58, 63), (18, 207), (51, 193), (2, 43), (1, 146), (159, 161), (25, 35), (43, 198), (48, 178), (61, 181), (27, 197), (13, 48), (93, 98), (123, 159), (34, 182), (118, 182), (67, 86), (36, 138), (28, 124), (197, 201), (13, 136), (59, 133), (116, 140), (59, 132), (37, 182), (6, 170), (86, 108), (76, 138), (88, 146), (160, 186), (72, 80), (12, 93), (167, 169), (13, 124), (186, 206), (15, 170), (138, 200), (40, 155), (121, 149), (15, 180), (40, 197), (25, 45), (177, 197), (17, 169), (1, 136), (39, 182), (100, 186), (150, 170), (9, 13), (14, 175), (86, 190), (29, 202), (1, 153), (98, 155), (29, 170), (49, 97), (15, 155), (72, 101), (43, 181), (67, 170), (72, 136), (48, 202), (150, 155), (75, 133), (31, 70), (21, 197), (72, 179), (30, 182), (41, 180), (39, 146), (177, 178), (27, 87), (86, 193), (48, 72), (28, 175), (143, 161), (165, 181), (138, 181), (117, 207), (91, 100), (75, 197), (86, 150), (89, 90), (111, 143), (53, 179), (7, 134), (3, 143), (7, 115), (121, 170), (30, 124), (86, 181), (98, 182), (2, 41), (123, 128), (147, 202), (146, 160), (2, 181), (72, 171), (168, 182), (115, 170), (72, 149), (46, 57), (7, 72), (83, 170), (8, 25), (43, 88), (135, 138), (27, 43), (165, 178), (181, 191), (37, 168), (69, 186), (13, 166), (34, 63), (195, 206), (11, 146), (146, 206), (9, 183), (2, 36), (41, 178), (22, 82), (14, 197), (61, 100), (186, 200), (182, 194), (0, 13), (70, 196), (36, 146), (15, 99), (183, 198), (41, 200), (34, 41), (47, 182), (176, 197), (70, 173), (70, 127), (37, 119), (83, 197), (175, 181), (30, 146), (86, 171), (135, 197), (41, 72), (86, 180), (119, 197), (31, 140), (76, 170), (40, 150), (170, 192), (2, 13), (13, 138), (170, 175), (7, 34), (68, 182), (129, 170), (86, 166), (146, 154), (21, 106), (99, 193), (48, 86), (46, 61), (99, 198), (29, 175), (75, 86), (100, 103), (61, 155), (60, 86), (147, 194), (51, 136), (123, 197), (198, 200), (69, 75), (166, 176), (27, 124), (156, 197), (18, 155), (39, 155), (100, 112), (7, 63), (48, 124), (10, 170), (2, 146), (1, 101), (29, 124), (21, 119), (97, 198), (115, 142), (160, 175), (31, 124), (194, 197), (155, 162), (119, 153), (119, 155), (7, 13), (90, 179), (13, 18), (20, 140), (5, 182), (119, 170), (2, 67), (72, 147), (34, 198), (92, 101), (84, 140), (68, 199), (91, 104), (9, 175), (13, 115), (15, 179), (57, 197), (155, 160), (48, 170), (82, 206), (1, 41), (1, 155), (115, 192), (170, 186), (150, 175), (21, 182), (51, 72), (17, 40), (136, 171), (186, 198), (3, 181), (61, 115), (75, 177), (181, 201), (66, 98), (48, 99), (76, 206), (156, 198), (26, 170), (59, 80), (87, 146), (60, 160), (9, 60), (48, 193), (2, 150), (1, 97), (105, 198), (20, 61), (155, 169), (15, 136), (34, 197), (113, 170), (47, 138), (2, 5), (16, 20), (146, 178), (41, 186), (143, 146), (99, 197), (18, 169), (59, 182), (124, 169), (48, 75), (1, 90), (108, 115), (162, 191), (59, 175), (51, 53), (43, 206), (67, 156), (86, 136), (150, 162), (198, 203), (135, 182), (1, 182), (107, 153), (155, 201), (61, 146), (153, 178), (119, 147), (61, 68), (2, 124), (108, 142), (156, 182), (159, 202), (89, 110), (108, 165), (15, 177), (28, 60), (13, 146), (68, 124), (14, 165), (146, 180), (2, 169), (107, 146), (194, 198), (60, 98), (153, 154), (61, 165), (80, 132), (150, 181), (40, 59), (31, 204), (4, 103), (94, 198), (124, 126), (86, 138), (5, 67), (177, 180), (29, 86), (42, 124), (147, 171), (30, 86), (88, 200), (182, 190), (96, 170), (68, 134), (130, 160), (13, 97), (58, 197), (147, 174), (146, 200), (124, 147), (31, 146), (14, 170), (61, 204), (183, 197), (182, 193), (162, 198), (31, 158), (48, 61), (39, 87), (31, 125), (2, 61), (168, 197), (5, 43), (190, 199), (99, 146), (3, 170), (62, 181), (26, 135), (60, 170), (53, 99), (135, 199), (0, 182), (83, 91), (15, 175), (36, 86), (41, 76), (141, 170), (48, 180), (162, 170), (169, 181), (76, 160), (177, 198), (20, 72), (124, 146), (76, 198), (76, 86), (68, 194), (13, 30), (53, 171), (13, 177), (63, 182), (44, 193), (10, 192), (124, 206), (25, 155), (29, 118), (13, 34), (29, 60), (95, 182), (95, 170), (15, 21), (159, 198), (29, 168), (47, 82), (14, 75), (13, 99), (30, 197), (5, 170), (27, 182), (86, 174), (168, 179), (28, 86), (72, 86), (30, 153), (147, 181), (158, 197), (22, 124), (133, 170)]
    elif args.dataset == 'delicious':
        num_features = 500
        nodes = list(range(983))
        pairs = [(809, 879), (879, 941), (247, 700), (452, 733), (733, 879), (941, 942), (733, 941), (452, 897), (99, 102), (240, 941), (941, 946), (378, 879), (733, 897), (809, 941), (247, 941), (240, 946), (99, 941), (378, 809), (941, 947), (733, 809), (452, 876), (700, 941), (700, 809), (488, 941), (240, 733), (879, 942), (247, 733), (99, 240), (99, 733), (700, 733), (247, 809), (378, 941), (897, 898), (733, 876), (636, 809), (247, 947), (452, 898), (946, 947), (247, 879), (65, 240), (240, 879), (878, 879), (700, 947), (733, 898), (700, 879), (168, 700), (452, 879), (240, 247), (247, 946), (168, 247), (700, 897), (636, 879), (99, 942), (99, 879), (205, 240), (189, 809), (876, 897), (452, 941), (733, 946), (247, 897), (452, 809), (240, 478), (879, 946), (381, 809), (488, 879), (733, 740), (897, 941), (452, 700), (381, 879), (378, 381), (240, 947), (247, 452), (471, 733), (378, 733), (240, 700), (205, 386), (275, 378), (700, 946), (240, 809), (240, 452), (240, 423), (240, 942), (733, 947), (733, 942), (205, 941), (378, 636), (99, 452), (240, 897), (809, 897), (99, 946), (99, 809), (809, 942), (942, 946), (99, 621), (879, 947), (102, 941), (809, 878), (879, 897), (488, 942), (540, 809), (37, 941), (488, 809), (205, 879), (876, 879), (636, 700), (221, 946), (221, 941), (488, 733), (632, 941), (99, 205), (102, 240), (809, 967), (67, 733), (189, 879), (65, 205), (221, 240), (99, 876), (897, 946), (247, 636), (168, 941), (240, 378), (168, 733), (733, 881), (99, 226), (275, 879), (897, 947), (205, 733), (740, 941), (275, 809), (876, 898), (247, 942), (378, 942), (99, 488), (878, 941), (189, 733), (740, 879), (67, 452), (452, 946), (942, 947), (876, 941), (296, 733), (386, 387), (809, 946), (99, 860), (452, 947), (636, 941), (898, 941), (378, 632), (809, 860), (733, 860), (387, 456), (99, 247), (858, 860), (452, 881), (221, 947), (700, 898), (99, 478), (65, 423), (378, 878), (65, 99), (183, 941), (386, 456), (67, 881), (102, 733), (471, 941), (247, 898), (454, 941), (37, 942), (733, 770), (809, 947), (168, 947), (168, 897), (240, 740), (183, 942), (809, 858), (632, 879), (553, 809), (168, 809), (770, 879), (540, 636), (431, 733), (378, 488), (99, 101), (65, 478), (99, 183), (803, 942), (240, 898), (102, 226), (879, 967), (99, 897), (102, 621), (471, 879), (275, 381), (378, 946), (205, 378), (860, 941), (700, 942), (240, 876), (99, 471), (454, 946), (168, 879), (733, 858), (99, 223), (102, 942), (733, 738), (99, 700), (553, 643), (221, 247), (770, 941), (454, 947), (247, 378), (378, 700), (879, 918), (478, 941), (65, 733), (740, 946), (621, 733), (99, 947), (122, 941), (226, 621), (860, 879), (661, 664), (99, 122), (99, 378), (898, 946), (431, 452), (168, 452), (67, 897), (809, 876), (381, 636), (99, 386), (189, 378), (501, 941), (189, 858), (809, 898), (471, 621), (183, 803), (423, 946), (632, 942), (478, 946), (247, 454), (65, 941), (643, 809), (58, 553), (423, 941), (168, 240), (540, 879), (37, 247), (240, 454), (636, 733), (168, 946), (37, 879), (67, 876), (205, 478), (809, 918), (183, 733), (664, 665), (879, 898), (661, 665), (471, 809), (205, 942), (803, 941), (205, 809), (99, 575), (37, 946), (876, 881), (37, 700), (101, 102), (37, 947), (67, 99), (102, 879), (488, 632), (221, 454), (431, 897), (65, 386), (247, 876), (378, 740), (488, 860), (876, 946), (122, 733), (275, 276), (315, 386), (575, 924), (183, 879), (881, 897), (189, 452), (240, 664), (575, 621), (240, 470), (226, 471), (378, 967), (898, 947), (698, 879), (386, 941), (68, 733), (621, 860), (189, 860), (221, 733), (553, 879), (381, 967), (858, 879), (223, 733), (58, 643), (247, 501), (240, 386), (454, 700), (452, 535), (99, 881), (378, 471), (381, 941), (122, 240), (423, 879), (102, 860), (58, 809), (205, 484), (501, 947), (67, 941), (37, 240), (535, 876), (276, 378), (386, 733), (501, 700), (221, 700), (478, 733), (452, 942), (452, 540), (67, 247), (101, 941), (271, 733), (700, 876), (275, 941), (65, 664), (632, 733), (476, 733), (245, 700), (423, 733), (526, 733), (99, 858), (878, 942), (245, 247), (501, 946), (122, 942), (189, 190), (205, 223), (860, 942), (189, 941), (205, 387), (65, 946), (102, 205), (540, 733), (65, 470), (378, 452), (240, 488), (102, 946), (102, 809), (454, 733), (456, 457), (378, 423), (378, 918), (435, 452), (378, 540), (621, 941), (575, 941), (240, 471), (102, 223), (67, 700), (470, 665), (65, 223), (535, 733), (791, 792), (531, 700), (168, 898), (61, 809), (470, 664), (37, 809), (77, 606), (667, 941), (541, 733), (803, 879), (183, 240), (190, 809), (471, 488), (221, 942), (205, 452), (205, 423), (897, 942), (223, 240), (247, 881), (189, 381), (858, 941), (189, 897), (102, 488), (122, 879), (67, 68), (99, 803), (99, 740), (240, 860), (99, 898), (247, 531), (535, 698), (643, 879), (488, 946), (700, 881), (102, 478), (223, 386), (221, 879), (189, 967), (183, 809), (470, 661), (37, 501), (205, 456), (876, 947), (431, 876), (387, 457), (108, 733), (632, 809), (500, 700), (452, 740), (221, 897), (386, 471), (488, 770), (61, 879), (37, 733), (924, 926), (102, 452), (381, 878), (386, 484), (205, 488), (740, 897), (189, 540), (733, 739), (423, 478), (535, 879), (102, 471), (247, 740), (189, 636), (540, 897), (423, 470), (540, 700), (484, 733), (101, 942), (381, 918), (65, 378), (452, 858), (435, 879), (531, 879), (189, 700), (99, 221), (271, 700), (355, 941), (223, 621), (296, 526), (247, 500), (205, 946), (881, 941), (102, 183), (575, 606), (240, 501), (189, 488), (553, 555), (555, 643), (378, 947), (386, 457), (189, 471), (263, 452), (240, 792), (240, 661), (339, 799), (68, 99), (519, 733), (378, 575), (700, 755), (471, 575), (454, 897), (698, 809), (168, 636), (65, 102), (58, 555), (378, 386), (67, 240), (636, 878), (240, 665), (621, 858), (770, 942), (122, 567), (667, 700), (189, 275), (102, 575), (102, 386), (454, 879), (740, 947), (247, 667), (247, 878), (575, 733), (423, 809), (247, 271), (488, 858), (223, 575), (500, 809), (809, 849), (240, 392), (431, 898), (600, 606), (101, 879), (636, 947), (555, 809), (221, 452), (271, 897), (247, 488), (471, 770), (276, 809), (452, 488), (471, 632), (488, 621), (275, 967), (240, 275), (878, 918), (65, 471), (276, 879), (183, 488), (740, 809), (104, 108), (65, 661), (247, 755), (275, 636), (471, 860), (739, 740), (413, 941), (378, 897), (204, 365), (667, 947), (99, 456), (315, 471), (122, 488), (65, 665), (77, 600), (700, 719), (271, 809), (205, 860), (65, 879), (719, 755), (271, 452), (315, 575), (68, 881), (478, 942), (698, 876), (488, 575), (378, 553), (240, 621), (168, 454), (501, 942), (102, 876), (501, 879), (183, 378), (386, 924), (924, 941), (452, 698), (168, 501), (700, 878), (773, 809), (190, 858), (617, 618), (99, 567), (452, 636), (102, 122), (240, 355), (205, 471), (452, 860), (221, 501), (170, 700), (99, 423), (733, 763), (240, 636), (423, 740), (719, 756), (226, 941), (435, 733), (501, 733), (168, 221), (226, 733), (740, 898), (247, 540), (171, 942), (168, 942), (413, 879), (58, 879), (247, 719), (226, 386), (539, 733), (275, 733), (531, 733), (381, 553), (531, 941), (700, 740), (77, 575), (488, 803), (740, 942), (228, 733), (755, 756), (99, 664), (452, 454), (598, 924), (698, 733), (68, 452), (190, 733), (423, 664), (102, 858), (170, 247), (567, 941), (738, 879), (488, 947), (122, 809), (488, 700), (773, 879), (879, 917), (378, 924), (667, 809), (381, 540), (26, 876), (667, 879), (876, 942), (433, 435), (809, 881), (375, 700), (740, 876), (398, 860), (37, 168), (501, 897), (99, 398), (332, 339), (386, 879), (99, 484), (55, 700), (171, 941), (398, 858), (470, 941), (488, 878), (77, 378), (189, 247), (67, 898), (240, 666), (435, 876), (471, 858), (99, 387), (398, 733), (733, 917), (378, 606), (435, 809), (429, 698), (276, 381), (205, 632), (65, 740), (636, 967), (531, 809), (99, 632), (621, 809), (541, 879), (636, 897), (223, 471), (526, 897), (667, 946), (61, 381), (575, 942), (386, 575), (393, 395), (205, 470), (205, 664), (296, 879), (700, 756), (247, 375), (245, 809), (365, 727), (67, 809), (171, 879), (575, 879), (386, 398), (226, 575), (240, 881), (61, 378), (205, 315), (115, 941), (433, 452), (858, 897), (452, 526), (226, 860), (205, 924), (556, 621), (37, 221), (849, 879), (55, 247), (386, 632), (378, 770), (99, 535), (67, 947), (189, 398), (733, 878), (454, 942), (183, 621), (189, 553), (122, 860), (535, 881), (636, 942), (700, 860), (37, 99), (378, 860), (189, 876), (617, 809), (471, 740), (540, 849), (879, 924), (636, 946), (65, 452), (221, 898), (102, 247), (398, 809), (296, 738), (204, 727), (531, 636), (37, 897), (245, 733), (67, 879), (452, 471), (617, 879), (240, 575), (104, 733), (918, 967), (223, 226), (739, 879), (423, 897), (240, 878), (332, 799), (168, 245), (470, 879), (183, 223), (26, 452), (540, 967), (454, 501), (664, 733), (247, 756), (221, 478), (381, 733), (226, 240), (240, 739), (183, 205), (674, 879), (541, 941), (899, 924), (315, 387), (878, 946), (68, 247), (898, 942), (205, 897), (122, 621), (223, 484), (275, 471), (539, 941), (99, 556), (632, 878), (386, 488), (26, 733), (205, 355), (99, 924), (190, 860), (68, 897), (423, 666), (733, 803), (296, 452), (381, 643), (733, 964), (240, 421), (205, 398), (386, 395), (101, 946), (431, 809), (733, 924), (240, 381), (65, 466), (62, 240), (65, 392), (429, 535), (99, 189), (99, 665), (881, 898), (168, 876), (205, 792), (189, 898), (378, 643), (664, 941), (205, 575), (664, 879), (275, 946), (122, 876), (205, 226), (37, 378), (469, 470), (240, 466), (423, 452), (67, 946), (296, 740), (65, 387), (183, 471), (68, 941), (575, 632), (221, 740), (738, 941), (809, 924), (65, 456), (621, 942), (183, 575), (240, 632), (435, 897), (68, 876), (381, 488), (555, 879), (99, 539), (879, 881), (488, 897), (739, 941), (733, 967), (168, 531), (452, 501), (700, 858), (168, 378), (65, 897), (122, 452), (247, 860), (386, 942), (618, 809), (223, 456), (378, 858), (540, 909), (878, 967), (478, 740), (122, 881), (488, 636), (65, 575), (470, 946), (315, 456), (205, 275), (221, 876), (223, 941), (99, 168), (433, 879), (99, 315), (122, 974), (275, 918), (719, 941), (674, 675), (205, 876), (59, 809), (102, 378), (560, 879), (275, 740), (315, 924), (68, 700), (65, 226), (205, 665), (661, 879), (168, 170), (436, 860), (122, 946), (733, 977), (452, 849), (700, 714), (315, 941), (755, 941), (296, 378), (575, 860), (454, 898), (183, 860), (436, 858), (68, 240), (189, 436), (190, 879), (452, 478), (621, 879), (575, 598), (99, 770), (189, 205), (423, 661), (339, 783), (560, 809), (247, 471), (67, 122), (99, 470), (263, 733), (99, 636), (858, 876), (101, 240), (739, 946), (540, 906), (478, 879), (122, 247), (99, 698), (205, 878), (803, 809), (423, 898), (240, 567), (471, 946), (99, 972), (189, 918), (924, 942), (99, 661), (205, 661), (102, 947), (698, 881), (61, 553), (777, 809), (226, 858), (99, 878), (470, 733), (67, 942), (423, 665), (99, 454), (665, 879), (205, 858), (738, 770), (122, 560), (226, 315), (171, 809), (223, 478), (275, 423), (275, 488), (471, 792), (674, 809), (387, 941), (77, 879), (606, 924), (102, 700), (37, 452), (240, 442), (240, 858), (240, 905), (471, 897), (102, 897), (25, 567), (392, 941), (575, 809), (436, 809), (65, 666), (378, 531), (500, 879), (67, 535), (809, 909), (296, 897), (240, 924), (189, 240), (471, 700), (275, 553), (618, 879), (226, 456), (223, 387), (488, 876), (170, 733), (398, 452), (806, 942), (386, 478), (484, 881), (381, 700), (431, 879), (941, 978), (59, 879), (115, 809), (61, 643), (263, 897), (296, 763), (122, 698), (664, 946), (440, 733), (621, 738), (733, 849), (205, 381), (247, 381), (540, 941), (471, 738), (478, 947), (392, 946), (240, 791), (421, 423), (240, 469), (101, 809), (783, 799), (433, 809), (575, 899), (500, 636), (240, 336), (488, 617), (315, 621), (355, 879), (378, 600), (378, 470), (621, 682), (65, 421), (65, 275), (166, 941), (719, 947), (632, 770), (471, 606), (296, 471), (355, 946)]

    pair_inds = {}
    for ind, pair in enumerate(pairs):
        pair_inds[pair] = ind
    args_dict = {'pair_inds':pair_inds, 'pot_div_factor':args.pot_div_factor, 'unary_hidden_layers':args.unary_hidden_layers, 'unary_hidden_size':args.unary_hidden_size, 'unary_features':num_features, 'pair_diags':args.pair_diags, 'pair_one':args.pair_one}
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.shift_pots:
        args_dict['shift_pots'] = True
    params = {}

    pw_params = {
        'batch_size':10000, 
        'num_epochs':50,
        'l_rate':1e-2, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':args.mp_eps, 
        'mp_itrs':args.mp_itrs,
        'use_loss_augmented_inference':False,
        'inf_loss':scaled_identity_diff,
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
        'global_div_factor':args.global_div_factor,
    }
    if args.use_val_scheduler is not None:
        pw_params['val_scheduler'] = lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, args.use_val_scheduler)
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
        if args.pair_only:
            pw_params['l_rate'] = {'unary':0., 'pair':args.l_rate}
        else:
            pw_params['l_rate'] = args.l_rate
    if args.batch_size is not None:
        pw_params['batch_size'] = args.batch_size
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.reinit is not None:
        pw_params['reinit'] = args.reinit
    pw_params['window_size'] = 100

    if args.model == 'unary_v2':
        graph = Graph(nodes, [], 2, MLCPotentialModel, args_dict, False)
        model = PairwiseModel([graph], len(nodes), 2, pw_params)
    elif args.model == 'pairwise':
        full_graph = Graph(nodes, pairs, 2, MLCPotentialModel, args_dict, False)
        model = PairwiseModel([full_graph], len(nodes), 2, pw_params)
        if args.unary_model is not None:
            #new_unary = build_unary_model(num_features, args.unary_hidden_layers, args.unary_hidden_size, num_labels*2)
            #new_unary.load_state_dict(torch.load(args.unary_model))
            #full_graph.potential_model.unary_model = new_unary_model
            full_graph.potential_model.unary_model.load_state_dict(torch.load(args.unary_model))
        elif args.pretrain is not None:
            model.load(args.pretrain)
    elif args.model == 'pairwise_transformed':
        if args.use_residual:
            pw_params['global_model'] = TransformedMLPModelResidual
        else:
            pw_params['global_model'] = TransformedMLPModel
        pw_params['global_hidden_size'] = args.global_hidden_size
        pw_params['global_activation'] = args.global_activation
        full_graph = Graph(nodes, pairs, 2, MLCPotentialModel, args_dict, False)
        model = PairwiseModelTransformed([full_graph], len(nodes), 2, pw_params)
        if args.pretrain is not None:
            pw_model = PairwiseModel([full_graph], len(nodes), 2, params)
            pw_model.load(args.pretrain)
    elif args.model == 'global_mlp' or args.model == 'global_linear_gt':
        if args.model == 'global_linear_gt':#
            pw_params['global_model'] = GlobalLinear_GT
            pw_params['global_inputs'] = ['data_masks']
        elif args.use_gt:
            if args.use_global_beliefs:
                if args.mlp_init == 'v2':
                    pw_params['global_model'] = GlobalModel_GT_Beliefs_v2
                else:
                    pw_params['global_model'] = GlobalModel_GT_Beliefs
                pw_params['global_inputs'] = ['data_masks']
                pw_params['global_beliefs'] = True
            elif args.use_residual:
                pw_params['global_model'] = GlobalModel_Res_GT
                pw_params['global_inputs'] = ['data_masks']
            elif args.mlp_init == 'v2':
                pw_params['global_model'] = GlobalModel_GT_v2
                pw_params['global_inputs'] = ['data_masks']
            else:
                pw_params['global_model'] = GlobalModel_GT
                pw_params['global_inputs'] = ['data_masks']
        elif args.use_feats:
            if args.use_residual:
                if args.use_global_beliefs:
                    pw_params['global_model'] = GlobalFeatureModel_Res_Beliefs
                    pw_params['global_beliefs'] = True
                elif args.mlp_init == 'v2':
                    pw_params['global_model'] = GlobalFeatureModel_Resv2
                elif args.mlp_init == 'v3':
                    pw_params['global_model'] = GlobalFeatureModel_Resv3
                elif args.mlp_init == 'v4':
                    pw_params['global_model'] = GlobalFeatureModel_Resv4
                else:
                    pw_params['global_model'] = GlobalFeatureModel_Res
            else:
                pw_params['global_model'] = GlobalFeatureModel
            pw_params['num_features'] = num_features 
            pw_params['global_inputs'] = ['observations']
        elif args.use_residual:
            if args.mlp_init == 'v2':
                pw_params['global_model'] = GlobalModel_Resv2
            else:
                pw_params['global_model'] = GlobalModel_Res
        elif args.use_global_beliefs:
            pw_params['global_model'] = GlobalModel_Beliefs
            pw_params['global_beliefs'] = True
        elif args.mlp_init is None:
            pw_params['global_model'] = build_mlp_global_model
        elif args.mlp_init == 'v1':
            pw_params['global_model'] = build_initialized_mlp_global_model_v1
        elif args.mlp_init == 'v2':
            pw_params['global_model'] = build_initialized_mlp_global_model_v2
        elif args.mlp_init == 'v3':
            pw_params['global_model'] = MLP_Model_v3
        if args.global_hidden_size is None:
            pw_params['global_hidden_size'] = 10
        else:
            pw_params['global_hidden_size'] = args.global_hidden_size
        if args.global_activation is not None:
            pw_params['global_activation'] = args.global_activation
        pw_params['global_batchnorm'] = args.global_bn
        graphs = []
        nodes_graph = Graph(nodes, [], 2, MLCPotentialModel, args_dict, False)
        graphs.append(nodes_graph)
        if args.pretrain is not None and not args.load_global:
            full_graph = Graph(nodes, pairs, 2, MLCPotentialModel, args_dict, False)
            if args.unary_model is not None:
                full_graph.potential_model.unary_model.load_state_dict(torch.load(args.unary_model))
            pw_model = PairwiseModel([full_graph], len(nodes), 2, params)

            pw_model.load(args.pretrain)
            graphs[0].potential_model.unary_model = pw_model.graphs[0].potential_model.unary_model
        elif args.unary_model is not None and not args.load_global:
            graphs[0].potential_model.unary_model.load_state_dict(torch.load(args.unary_model))
        for ind,pair in enumerate(pairs):
            graphs.append(Graph([], [pair], 2, MLCPotentialModel, args_dict, False))
            if args.pretrain is not None and not args.load_global:
                if args.pair_diags or args.pair_one:
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind:(ind+1), :].data)
                else:
                    graphs[-1].potential_model.pair_model = torch.nn.Parameter(pw_model.graphs[0].potential_model.pair_model[ind*4:(ind+1)*4].data)

        if args.interleave:
            model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 2, pw_params)
        else:
            model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, pw_params)
        if args.load_global:
            model.load(args.pretrain)
            graphs[0].potential_model.unary_model.load_state_dict(torch.load(args.unary_model))



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

