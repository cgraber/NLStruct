from deepstruct.models import *
from deepstruct.datasets import *
import argparse, os

import deepstruct.models.modelconf

TRAIN = 0
VAL = 1
TEST = 2

class WordsDataset(BaseDataset):
    def __init__(self, data_dir, mode, masks_path=None):
        if mode == TRAIN:
            path = os.path.join(data_dir, 'train/')
            data_len = 1000
        elif mode == VAL:
            path = os.path.join(data_dir, 'val/')
            data_len = 200
        elif mode == TEST:
            path = os.path.join(data_dir, 'test/')
            data_len = 200
        super(WordsDataset, self).__init__(data_len, masks_path)
        self.observations = []
        self.labels = []
        for i in range(data_len):
            tmp_path = os.path.join(path, str(i))
            label_path = os.path.join(tmp_path, 'label.txt')
            with open(label_path, 'r') as fin:
                self.labels.append([int(label.strip()) for label in fin.readlines()])
            datum = []
            for j in range(5):
                img_path = os.path.join(tmp_path, '%d.png'%j)
                img = torch.from_numpy(skimage.io.imread(img_path, as_grey=True).flatten()).float()
                img.div_(255)
                datum.append(img)
            self.observations.append(torch.stack(datum))

    def __getitem__(self, idx):
        stuff = super(WordsDataset, self).__getitem__(idx)
        return (self.labels[idx], self.observations[idx]) + stuff


class WordsPotentialModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(WordsPotentialModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.img_size = args_dict['img_size']
        self.hidden_size = args_dict['hidden_size']
        self.fixed_unaries = args_dict.get('fixed_unaries', False)
        print("USING FIXED UNARIES: ",self.fixed_unaries)
        if len(node_regions) > 0:
            self.unary_model = nn.Sequential(
                nn.Linear(self.img_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.num_vals),
            )
        if len(pair_regions) > 0:
            self.pair_model = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(num_vals*num_vals).fill_(0.0))
        if 'linear_top' in args_dict and args_dict['linear_top'] == True:
            self.top = torch.nn.Parameter(modelconf.tensor_mod.FloatTensor(len(self.node_regions)+int(len(self.pair_regions) > 0)).fill_(1.0))

            self.linear_top = True
        else:
            self.linear_top = False
        if 'finetune' in args_dict and args_dict['finetune'] == True:
            self.finetune = True
        else:
            self.finetune = False

    def parameters(self):
        if self.linear_top and not self.finetune:
            return [self.top]
        elif self.fixed_unaries:
            return [self.pair_model]
        else:
            return super(WordsPotentialModel, self).parameters()

    def load_classifier(self, classifier_path):
        self.unary_model = torch.load(classifier_path)

    def set_observations(self, observations):
        self.num_observations = len(observations)
        #self.observations = torch.stack([Variable(torch.stack(obs)) for obs in zip(*observations)]).float()
        self.observations = Variable(observations)

    def forward(self):
        num_potentials = len(self.node_regions)*self.num_vals + len(self.pair_regions)*self.num_vals*self.num_vals
        result = Variable(modelconf.tensor_mod.FloatTensor(self.num_observations, num_potentials))
        for ind,orig_node_region in enumerate(self.original_node_regions):
            if self.linear_top:
                result[:, ind*self.num_vals:(ind+1)*self.num_vals] = self.unary_model(self.observations[:,orig_node_region,:])*self.top[ind]
            else:
                result[:, ind*self.num_vals:(ind+1)*self.num_vals] = self.unary_model(self.observations[:,orig_node_region,:])/100.0
        if len(self.pair_regions) > 0:
            if self.linear_top:
                result[:,len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,1)*self.top[len(self.node_regions)]
            else:
                result[:,len(self.node_regions)*self.num_vals:] = self.pair_model.repeat(self.num_observations,len(self.pair_regions))

        return result

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

class GlobalModel_Beliefs_v1(nn.Module):
    def __init__(self, num_graphs, params):
        super(GlobalModel_Beliefs_v1, self).__init__()
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
        layer1 = nn.Linear(2*num_graphs, params['global_hidden_size'])
        layer1.weight.data.fill_(0.0)
        for i in range(2*num_graphs):
            layer1.weight.data[i, i] = 1.0
        layer1.bias.data.fill_(0.0)

        layer2 = nn.Linear(params['global_hidden_size'], 1)
        layer2.weight.data.fill_(0.0)
        for i in range(2*num_graphs):
            layer2.weight.data[0, i] = 1.0
        layer2.bias.data.fill_(0.0)
        
        self.global_model = nn.Sequential(
                    layer1,
                    activation(),
                    layer2,
                )
    def forward(self, pots, beliefs):
        inp = torch.cat([pots, beliefs/100.], dim=1)
        result = self.global_model(inp)
        return result


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




def plot_task_losses(dir_name, model_name, return_vals):
    train_task_losses = return_vals['train_task_losses']
    test_task_losses = return_vals['test_task_losses']
    test_task_losses = list(zip(*test_task_losses))

    plt.figure()
    plt.clf()
    train_word_acc = [loss[0] for loss in train_task_losses]
    train_char_acc = [loss[1] for loss in train_task_losses]
    test_word_acc = [loss[0] for loss in test_task_losses[1]]
    test_char_acc = [loss[1] for loss in test_task_losses[1]]
    plt.plot(list(range(len(train_task_losses))), train_word_acc, label='Train Word Accuracy')
    plt.plot(list(range(len(train_task_losses))), train_char_acc, label='Train Char Accuracy')
    plt.plot(test_task_losses[0], test_word_acc, label='Test Word Accuracy')
    plt.plot(test_task_losses[0], test_char_acc, label='Test Char Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    path = os.path.join(dir_name, 'task_losses_%s.pdf'%(model_name))
    plt.savefig(path)
    path = os.path.join(dir_name, 'task_losses_%s.png'%(model_name))
    plt.savefig(path)
    path = os.path.join(dir_name, 'train_task_losses_%s.csv'%(model_name))
    save_data(path, list(range(len(train_task_losses))), train_word_acc, train_char_acc)
    path = os.path.join(dir_name, 'test_task_losses_%s.csv'%(model_name))
    save_data(path, test_task_losses[0], test_word_acc, test_char_acc)


def calculate_accuracies(dataset, found):
    correct_chars = 0.0
    correct_words = 0.0
    correct = [datum[0] for datum in dataset]
    for true, guess in zip(correct, found):
        correct_words += int(true == guess)
        correct_chars += sum([int(l1 == l2) for l1, l2 in zip(true, guess)])
    return correct_words/len(correct), correct_chars/(len(correct)*5)

if __name__ == '__main__':
    bit_depth = 2**4
    parser = argparse.ArgumentParser(description = 'Run experiment for words dataset')
    parser.add_argument('model', choices=['pairwise', 'pairwise_linear', 'global_linear', 'global_mlp'])
    parser.add_argument('graph', choices=['chain', 'hops'])
    parser.add_argument('data_directory')
    parser.add_argument('working_dir')
    parser.add_argument('-p', '--pretrain')
    parser.add_argument('--global_lr', type=float)
    parser.add_argument('--graph_lr', type=float)
    parser.add_argument('--train_interleaved_itrs', type=int)
    parser.add_argument('--test_interleaved_itrs', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'hardtanh'])
    parser.add_argument('--mlp_init', choices=['v1', 'v2'])
    parser.add_argument('--train_max_globals_l_rate', type=float)
    parser.add_argument('--train_lambda_l_rate', type=float)
    parser.add_argument('--test_max_globals_l_rate', type=float)
    parser.add_argument('--test_lambda_l_rate', type=float)
    parser.add_argument('--no_l_rate_decay', action='store_true')
    parser.add_argument('--l_rate', type=float)
    parser.add_argument('--train_masks_path')
    parser.add_argument('--test_masks_path')
    parser.add_argument('--load_masks', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--wide_top', action='store_true')
    parser.add_argument('--split_pairs', action='store_true')
    parser.add_argument('--use_loss_aug', action='store_true')
    parser.add_argument('--interleave', action='store_true')
    parser.add_argument('--loss_aug_div_factor', type=float, default=1.0)
    parser.add_argument('--load_classifier')
    parser.add_argument('--fix_unaries', action='store_true')
    parser.add_argument('--use_pd', action='store_true')
    parser.add_argument('--reinit', action='store_true')
    parser.add_argument('--mp_itrs', type=int, default=10)
    parser.add_argument('--train_avg_thresh', type=int, default=-1)
    parser.add_argument('--test_avg_thresh', type=int, default=-1)
    parser.add_argument('--test_mp_interval', type=int, default=-1)
    parser.add_argument('--train_mp_interval', type=int, default=-1)
    parser.add_argument('--val_interval', type=int, default=10)
    parser.add_argument('--use_global_beliefs', action='store_true')
    parser.add_argument('--global_div_factor', type=float, default=1.0)

    args = parser.parse_args()

    if args.gpu == True:
        modelconf.use_gpu()
    if args.load_masks:
        train_masks_path = args.train_masks_path
        test_masks_path = args.test_masks_path
    else:
        train_masks_path = test_masks_path = None
    train_data = WordsDataset(args.data_directory, TRAIN, train_masks_path)
    #val_data = WordsDataset(args.data_directory, VAL)
    val_data = None
    test_data = WordsDataset(args.data_directory, TEST, test_masks_path)
    
    nodes = [0,1,2,3,4]
    if args.graph == 'chain':
        pairs = [(0,1), (1,2), (2,3), (3,4)]
    elif args.graph == 'hops':
        pairs = [(0, 1), (0,2), (1, 2), (1, 3), (2, 3), (2, 4), (3,4)]

    args_dict = {'img_size':28*28, 'hidden_size':128}
    params = {}
    val_scheduler = None

    def scaled_identity_diff(true_val, other_val):
        if true_val == other_val:
            return 0.0
        else:
            return 1.0/args.loss_aug_div_factor

    pw_params = {
        'batch_size':50, 
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
        'task_loss':calculate_accuracies,
        'test_data':test_data,
        'train_masks_path':args.train_masks_path,
        'train_mp_interval':args.train_mp_interval,
        'test_mp_interval':args.test_mp_interval,
        'train_lambda_l_rate':args.train_lambda_l_rate,
        'train_max_globals_l_rate':args.train_max_globals_l_rate,
        'test_lambda_l_rate':args.test_lambda_l_rate,
        'test_max_globals_l_rate':args.test_max_globals_l_rate,
        'test_masks_path':args.test_masks_path,
        'val_interval':args.val_interval,
        'use_global_beliefs':args.use_global_beliefs,
        'global_div_factor':args.global_div_factor,
    }
    if args.use_loss_aug:
        pw_params['use_loss_augmented_inference'] = True
    if args.use_pd:
        pw_params['use_pd'] = True
    if args.wide_top:
        pw_params['wide_top'] = True
    if args.batch_size is not None:
        pw_params['batch_size'] = args.batch_size
    if args.no_l_rate_decay:
        pw_params['training_scheduler'] = None
    if args.num_epochs is not None:
        pw_params['num_epochs'] = args.num_epochs
    if args.l_rate is not None:
        pw_params['l_rate'] = args.l_rate
    if args.reinit is not None:
        pw_params['reinit'] = args.reinit
    global_params = pw_params.copy()
    global_params['train_interleaved_itrs'] = 500
    global_params['test_interleaved_itrs'] = 500
    global_params['global_lr'] = 1e-2
    global_params['graph_lr'] = 5e-2
    global_params['window_size'] = 100

    if args.model == 'pairwise' or args.model == 'pairwise_linear':
        if args.fix_unaries is not None:
            args_dict['fixed_unaries'] = args.fix_unaries
        if args.model == 'pairwise_linear':
            args_dict['linear_top'] = True
            if pw_params['l_rate'] != 0.0:
                args_dict['finetune'] = True
            else:
                args_dict['finetune'] = False
                pw_params['l_rate'] = global_params['global_lr']
        full_graph = Graph(nodes, pairs, 26, WordsPotentialModel, args_dict, False)
        if args.pretrain is not None:
            args_dict['linear_top'] = False
            preload_graph = Graph(nodes, pairs, 26, WordsPotentialModel, args_dict, False)
            tmp_model = PairwiseModel([preload_graph], len(nodes), 26, params)
            tmp_model.load(args.pretrain)
            full_graph.potential_model.unary_model = tmp_model.graphs[0].potential_model.unary_model
            full_graph.potential_model.pair_model = tmp_model.graphs[0].potential_model.pair_model
        elif args.load_classifier is not None:
            full_graph.potential_model.load_classifier(args.load_classifier)
        pw_model = PairwiseModel([full_graph], len(nodes), 26, params)
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
        train_word_acc, train_char_acc = calculate_accuracies(train_data, train_test_results)
        test_word_acc, test_char_acc = calculate_accuracies(test_data, test_results)

        print("TRAIN:")
        print("\tWORD ACCURACY: ",train_word_acc)
        print("\tCHAR ACCURACY: ",train_char_acc)
        print("TEST: ")
        print("\tWORD ACCURACY: ",test_word_acc)
        print("\tCHAR ACCURACY: ",test_char_acc)

        print("TRAIN TIME: ",train_time)
        print("TEST TIME: ",test_time)
        plot_task_losses(args.working_dir, 'baseline', return_vals)
        graph_results(args.working_dir, 'baseline', return_vals)

    else: 
        if args.model == 'global_linear':    
            global_params['global_model'] = build_linear_global_model
            global_params['interleaved_itrs'] = 1000
            global_params['global_lr'] = 1e-2
            global_params['graph_lr'] = 0
        elif args.model == 'global_quad':
            global_params['global_model'] = QuadModel
            global_params['global_lr'] = 1e-3
            if args.pretrain is None:
                global_params['graph_lr'] = 1e-2
            else:
                global_params['graph_lr'] = 1e-3
        elif args.model == 'global_mlp':
            if args.use_global_beliefs:
                if args.mlp_init == 'v1':
                    global_params['global_model'] = GlobalModel_Beliefs_v1
                else:
                    global_params['global_model'] = GlobalModel_Beliefs
                global_params['global_beliefs'] = True
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
        elif args.model == 'dual_quad':
            global_params['global_model'] = QuadModel
            global_params['global_lr'] = 1e-1
            global_params['graph_lr'] = 1e-2
        elif args.model == 'dual_linear':
            global_params['global_model'] = build_linear_global_model
            global_params['global_lr'] = 1e-1
            global_params['graph_lr'] = 1e-2
        if args.train_interleaved_itrs is not None:
            global_params['train_interleaved_itrs'] = args.train_interleaved_itrs
        if args.test_interleaved_itrs is not None:
            global_params['test_interleaved_itrs'] = args.test_interleaved_itrs
        if args.global_lr is not None:
            global_params['global_lr'] = args.global_lr
        if args.graph_lr is not None:
            global_params['graph_lr'] = args.graph_lr
        print("PARAMS: ",global_params)
        graphs = []
        for node in nodes:
            node_graph = Graph([node], [], 26, WordsPotentialModel, args_dict, False)
            graphs.append(node_graph)
        if args.pretrain is not None:
            full_graph = Graph(nodes, pairs, 26, WordsPotentialModel, args_dict, False)
            pw_model = PairwiseModel([full_graph], len(nodes), 26, params)
            pw_model.load(args.pretrain)
            graphs[0].potential_model.unary_model = pw_model.graphs[0].potential_model.unary_model
        for node_graph in graphs[1:]:
            node_graph.potential_model.unary_model = graphs[0].potential_model.unary_model
        if args.split_pairs:
            for pair in pairs:
                pair_graph = Graph([], [pair], 26, WordsPotentialModel, args_dict, False)
                graphs.append(pair_graph)
        else:
            graphs.append(Graph([], pairs, 26, WordsPotentialModel, args_dict, False))
        if args.pretrain is not None:
            graphs[len(nodes)].potential_model.pair_model = pw_model.graphs[0].potential_model.pair_model
        if args.split_pairs:
            for pair_graph in graphs[len(nodes)+1:]:
                pair_graph.potential_model.pair_model = graphs[len(nodes)].potential_model.pair_model
        if args.model == 'global_linear' or args.model == 'global_mlp':
            if args.interleave:
                global_model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 26, global_params)
            else:
                global_model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 26, global_params)
        start = time.time()
        obj, train_results, return_vals = global_model.train(train_data, None, global_params)
        end = time.time()
        train_time = end-start

        #graph_results(args.working_dir, args.model, return_vals)

        start = time.time()
        test_results = global_model.test(test_data, global_params)
        end = time.time()

        train_test_results = global_model.test(train_data, global_params)
        test_time = end-start
        for datum, result in zip(test_data, test_results):
            print("\tCORRECT: ",datum[0])
            print("\tFOUND:   ", result)
        train_word_acc, train_char_acc = calculate_accuracies(train_data, train_test_results)
        test_word_acc, test_char_acc = calculate_accuracies(test_data, test_results)
        print("TRAIN:")
        print("\tWORD ACCURACY: ",train_word_acc)
        print("\tCHAR ACCURACY: ",train_char_acc)
        print("TEST: ")
        print("\tWORD ACCURACY: ",test_word_acc)
        print("\tCHAR ACCURACY: ",test_char_acc)

        print("TRAIN TIME: ",train_time)
        print("TEST TIME: ",test_time)

        #plot_task_losses(args.working_dir, args.model, return_vals)
