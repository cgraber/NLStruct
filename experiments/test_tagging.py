from . import tagging_experiment
from . import train_tagging_baseline as tag_base

import sys, argparse

from deepstruct.models import *
import deepstruct.models.modelconf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run experiment for Flickr tagging dataset')
    parser.add_argument('model', choices=['unary', 'pairwise', 'global_linear', 'global_mlp'])
    parser.add_argument('working_dir')
    parser.add_argument('model_path')
    parser.add_argument('--img_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--train_feat_file')
    parser.add_argument('--val_feat_file')
    parser.add_argument('--test_feat_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--val_label_file')
    parser.add_argument('--test_label_file')

    parser.add_argument('--global_lr', type=float)
    parser.add_argument('--graph_lr', type=float)
    parser.add_argument('--train_interleaved_itrs', type=int, default=100)
    parser.add_argument('--test_interleaved_itrs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'hardtanh', 'tanh', 'leaky_relu'])
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
    parser.add_argument('--use_global_top', action='store_true')
    parser.add_argument('--diff_update', action='store_true')
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--load_pots', action='store_true')


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
        val_masks_path = args.val_masks_path
        test_masks_path = args.test_masks_path
    else:
        train_masks_path = None
        val_masks_path = None
        test_masks_path = None


    dataset_type = tagging_experiment.FULL
    train_data = tagging_experiment.FlickrTaggingDataset_Features(dataset_type, args.train_feat_file, args.label_dir, args.train_label_file, tagging_experiment.TRAIN, load=load, masks_path=test_masks_path, images_folder=args.img_dir)
    val_data = tagging_experiment.FlickrTaggingDataset_Features(dataset_type, args.val_feat_file, args.label_dir, args.val_label_file, tagging_experiment.VAL, load=load, masks_path=val_masks_path, images_folder=args.img_dir)
    test_data = tagging_experiment.FlickrTaggingDataset_Features(dataset_type, args.test_feat_file, args.label_dir, args.test_label_file, tagging_experiment.TEST, load=load, masks_path=test_masks_path, images_folder=args.img_dir)

    nodes = list(range(24))
    pairs = []
    pair_inds = {}
    ind = 0
    for node1 in nodes:
        for node2 in nodes[node1+1:]:
            pairs.append((node1, node2))
            pair_inds[(node1, node2)] = ind
            ind += 1

    args_dict = {'pair_inds':pair_inds, 'pot_div_factor':args.pot_div_factor}

    params = {
        'batch_size':10000, 
        'num_epochs':50,
        'l_rate':1e-2, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':args.mp_eps, 
        'mp_itrs':args.mp_itrs,
        'use_loss_augmented_inference':False,
        'checkpoint_dir':args.working_dir,
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
        'use_global_top':args.use_global_top,
        'diff_update':args.diff_update,
        'use_dropout':args.use_dropout,
        'global_hidden_size':args.global_hidden_size,
        'global_activation':args.global_activation,
    }

    if args.wide_top:
        params['wide_top'] = True
    if args.load_max_globals:
        params['load_max_globals'] = True
    if args.max_globals_save_path:
        params['max_globals_save_path'] = args.max_globals_save_path
    if args.use_loss_aug:
        params['use_loss_augmented_inference'] = True
    if args.no_l_rate_decay:
        params['training_scheduler'] = None
    if args.l_rate_div:
        params['training_scheduler'] = lambda opt: torch.optim.lr_scheduler.StepLR(opt, 10, gamma=0.5)
    if args.num_epochs is not None:
        params['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.reinit is not None:
        params['reinit'] = args.reinit

   

    if args.model == 'unary':
        graph = Graph(nodes, [], 2, tagging_experiment.FlickrBaselineModel, args_dict, False)
        graph.potential_model.load_classifier(args.model_path)
        model = PairwiseModel([graph], len(nodes), 2, params)

    elif args.model == 'pairwise' or args.model == 'pairwise_linear':
        if args.load_pots:
            full_graph = Graph(nodes, pairs, 2, tagging_experiment.FlickrFixedModel, args_dict, False)
            load_graph = Graph(nodes, pairs, 2, tag_base.FlickrPotentialModel, args_dict, False)
            load_model = PairwiseModel([load_graph], len(nodes), 2, params)
            load_model.load(args.model_path)
            full_graph.potential_model.pair_model = load_graph.potential_model.pair_model
        else:
            full_graph = Graph(nodes, pairs, 2, tagging_experiment.FlickrBaselineModel, args_dict, False)
        model = PairwiseModel([full_graph], len(nodes), 2, params)
    else:
        if args.model == 'global_linear':    
            params['global_model'] = build_linear_global_model
        elif args.model == 'global_mlp':
            if args.mlp_init is None:
                params['global_model'] = build_mlp_global_model
            elif args.mlp_init == 'v1':
                params['global_model'] = tagging_experiment.build_initialized_mlp_global_model_v1
            elif args.mlp_init == 'v2':
                params['global_model'] = tagging_experiment.build_initialized_mlp_global_model_v2
            if args.global_hidden_size is None:
                params['global_hidden_size'] = 10
            else:
                params['global_hidden_size'] = args.global_hidden_size
            if args.global_activation is not None:
                params['global_activation'] = args.global_activation
            if args.train_interleaved_itrs is not None:
                params['train_interleaved_itrs'] = args.train_interleaved_itrs
            if args.test_interleaved_itrs is not None:
                params['test_interleaved_itrs'] = args.test_interleaved_itrs
        graphs = []
        if args.load_pots:
            nodes_graph = Graph(nodes, [], 2, tagging_experiment.FlickrFixedModel, args_dict, False)
        else:
            nodes_graph = Graph(nodes, [], 2, tagging_experiment.FlickrBaselineModel, args_dict, False)
        graphs.append(nodes_graph)
        for ind,pair in enumerate(pairs):
            if args.load_pots:
                graphs.append(Graph([], [pair], 2, tagging_experiment.FlickrFixedModel, args_dict, False))
                graphs[-1].potential_model.pair_model = torch.nn.Parameter(torch.FloatTensor(4))
            else:
                graphs.append(Graph([], [pair], 2, tagging_experiment.FlickrBaselineModel, args_dict, False))
        if args.interleave:
            model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 2, params)
        else:
            model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, params)

    print("PARAMS: ",params)
    if args.model not in ['unary', 'pairwise']:
        model.load(args.model_path)

    if args.model != 'pairwise':
        print("INITIALIZING MAX GLOBALS")
        mp_graph = [fastmp.FastMP(len(model.node_regions), model.num_vals, model.pair_regions, np.zeros(model.num_potentials, dtype=float))] 
        model.init_dataset(train_data, mp_graph, False)
        train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], collate_fn=datasets.collate_batch)
        model.init_max_globals(train_dataloader, params['mp_eps'], True)

    print("STARTING TESTING")
    train_results = model.test(train_data, params)
    train_loss = tagging_experiment.calculate_hamming_loss(train_data, train_results)
    val_results = model.test(val_data, params)
    val_loss = tagging_experiment.calculate_hamming_loss(val_data, val_results)
    test_results = model.test(test_data, params)
    test_loss = tagging_experiment.calculate_hamming_loss(test_data, test_results)
    print("TRAIN LOSS: ", train_loss)
    print("VAL LOSS: ", val_loss)
    print("TEST LOSS: ", test_loss)
