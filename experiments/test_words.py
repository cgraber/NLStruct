from . import words_experiment, train_words_pots
import sys, argparse

from deepstruct.models import *
import deepstruct.models.modelconf

if __name__ == '__main__':
    bit_depth = 2**4
    parser = argparse.ArgumentParser(description = 'Test model on words dataset')
    parser.add_argument('model', choices=['unary', 'pairwise', 'global_linear', 'global_mlp'])
    parser.add_argument('graph', choices=['chain', 'hops'])
    parser.add_argument('data_directory')
    parser.add_argument('working_dir')
    parser.add_argument('model_path')
    parser.add_argument('--global_lr', type=float)
    parser.add_argument('--graph_lr', type=float)
    parser.add_argument('--train_interleaved_itrs', type=int, default=100)
    parser.add_argument('--test_interleaved_itrs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'hardtanh', 'tanh'])
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
    parser.add_argument('--mp_itrs', type=int, default=10)
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
    parser.add_argument('--split_pairs', action='store_true')



    args = parser.parse_args()

    if args.gpu == True:
        modelconf.use_gpu()
    if args.load_masks:
        train_masks_path = args.train_masks_path
        test_masks_path = args.test_masks_path
        val_masks_path = args.val_masks_path
    else:
        train_masks_path = test_masks_path = val_masks_path = None
    
    nodes = [0,1,2,3,4]
    if args.graph == 'chain':
        pairs = [(0,1), (1,2), (2,3), (3,4)]
    elif args.graph == 'hops':
        pairs = [(0, 1), (0,2), (1, 2), (1, 3), (2, 3), (2, 4), (3,4)]

    args_dict = {'img_size':28*28, 'hidden_size':128}

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
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size

    if args.test_interleaved_itrs is not None:
        params['test_interleaved_itrs'] = args.test_interleaved_itrs
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd

    if args.model == 'unary':
        #This one has a slightly different testing procedure
        test_data = train_words_pots.WordsDataset(args.data_directory, words_experiment.VAL)
        print("BEGINNING TESTING OF UNARY MODEL")
        model = torch.load(args.model_path)
        char_acc, word_acc = train_words_pots.test(model, test_data, params['batch_size'])
        print("RESULTS: ")
        print("\tCHARACTER ACCURACY: ",char_acc)
        print("\tWORD ACCURACY: ",word_acc)
        sys.exit(0)

    train_data = words_experiment.WordsDataset(args.data_directory, words_experiment.TRAIN, train_masks_path)
    val_data = words_experiment.WordsDataset(args.data_directory, words_experiment.TEST, val_masks_path)
    #Yes, I am loading the "validation" set. This is on purpose
    test_data = words_experiment.WordsDataset(args.data_directory, words_experiment.VAL, test_masks_path)
    
    if args.model == 'pairwise':
        full_graph = Graph(nodes, pairs, 26, words_experiment.WordsPotentialModel, args_dict, False)
        model = PairwiseModel([full_graph], len(nodes), 26, params)
    else: 
        if args.model == 'global_linear':    
            params['global_model'] = build_linear_global_model
        elif args.model == 'global_mlp':
            if args.mlp_init is None:
                params['global_model'] = build_mlp_global_model
            elif args.mlp_init == 'v1':
                params['global_model'] = words_experiment.build_initialized_mlp_global_model_v1
            elif args.mlp_init == 'v2':
                params['global_model'] = words_experiment.build_initialized_mlp_global_model_v2
            if args.global_hidden_size is None:
                params['global_hidden_size'] = 10
            else:
                params['global_hidden_size'] = args.global_hidden_size
            if args.global_activation is not None:
                params['global_activation'] = args.global_activation
        
        graphs = []
        for node in nodes:
            node_graph = Graph([node], [], 26, words_experiment.WordsPotentialModel, args_dict, False)
            graphs.append(node_graph)
        for node_graph in graphs[1:]:
            node_graph.potential_model.unary_model = graphs[0].potential_model.unary_model
        if args.split_pairs:
            for pair in pairs:
                pair_graph = Graph([], [pair], 26, words_experiment.WordsPotentialModel, args_dict, False)
                graphs.append(pair_graph)
            for pair_graph in graphs[len(nodes)+1:]:
                pair_graph.potential_model.pair_model = graphs[len(nodes)].potential_model.pair_model
        else:
            graphs.append(Graph([], pairs, 26, words_experiment.WordsPotentialModel, args_dict, False))
        if args.interleave:
            model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 26, params)
            print("USING INTERLEAVED")
        else:
            model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 26, params)

    model.load(args.model_path)

    if args.model != 'pairwise':
        print("INITIALIZING MAX GLOBALS")
        mp_graph = [fastmp.FastMP(len(model.node_regions), model.num_vals, model.pair_regions, np.zeros(model.num_potentials, dtype=float))] 
        model.init_dataset(train_data, mp_graph, False)
        train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], collate_fn=datasets.collate_batch)
        model.init_max_globals(train_dataloader, params['mp_eps'], True)

    print("STARTING TESTING")
    print("TRAIN DATA:")
    start = time.time()
    train_results = model.test(train_data, params)
    train_word_acc, train_char_acc = words_experiment.calculate_accuracies(train_data, train_results)
    end = time.time()
    train_time = (end-start)

    print("VAL DATA: ")
    start = time.time()
    val_results = model.test(val_data, params)
    val_word_acc, val_char_acc = words_experiment.calculate_accuracies(val_data, val_results)
    end = time.time()
    val_time = (end-start)

    print("TEST DATA: ")
    start = time.time()
    test_results = model.test(test_data, params)
    test_word_acc, test_char_acc = words_experiment.calculate_accuracies(test_data, test_results)
    end = time.time()
    test_time = (end-start)

    print("FINAL RESULTS:")
    print("TRAINING:")
    print("\tWORD ACCURACY: ",train_word_acc)
    print("\tCHAR ACCURACY: ",train_char_acc)
    print("\tTIME: ", train_time)
    print("VAL:")
    print("\tWORD ACCURACY: ",val_word_acc)
    print("\tCHAR ACCURACY: ",val_char_acc)
    print("\tTIME: ", val_time)
    print("TEST:")
    print("\tWORD ACCURACY: ",test_word_acc)
    print("\tCHAR ACCURACY: ",test_char_acc)
    print("\tTIME: ", test_time)
