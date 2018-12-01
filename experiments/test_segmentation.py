from . import segmentation_experiment as seg

import sys, argparse
from deepstruct.models import *
import deepstruct.models.modelconf

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
    parser.add_argument('--test_interleaved_itrs', type=int)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'hardtanh'])
    parser.add_argument('--test_max_globals_l_rate', type=float)
    parser.add_argument('--test_lambda_l_rate', type=float)
    parser.add_argument('--test_avg_thresh', type=int, default=-1)
    parser.add_argument('--test_mp_interval', type=int, default=-1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--load_masks', action='store_true')
    parser.add_argument('--train_masks_path')
    parser.add_argument('--val_masks_path')
    parser.add_argument('--mlp_init', choices=['v1', 'v2'])
    parser.add_argument('--load_max_globals', action='store_true')
    parser.add_argument('--max_globals_save_path')
    parser.add_argument('--wide_top', action='store_true')
    parser.add_argument('--interleave', action='store_true')
    parser.add_argument('--use_pd', action='store_true')
    parser.add_argument('--use_res', action='store_true')
    parser.add_argument('--pot_div_factor', type=float, default=1.0)
    parser.add_argument('--shift_pots', action='store_true')
    parser.add_argument('--global_init_val', type=float, default=0.001)
    parser.add_argument('--reinit', action='store_true')
    parser.add_argument('--num_global_layers', type=int, default=1)
    parser.add_argument('--mp_itrs', type=int, default=1000)
    parser.add_argument('--tie_pairs', action='store_true', default=False)
    parser.add_argument('--ignore_global_top', action='store_true')
    parser.add_argument('--global_top_full', action='store_true')
    parser.add_argument('--use_gt', action='store_true')

    args = parser.parse_args()
    if args.img_dir == None and args.train_img_file == None:
        print("ERROR: Must specify either an image directory or image file")
        sys.exit(1)
    if args.label_dir == None and args.train_label_file == None:
        print("ERROR: Must specify either a label directory or label file")
        sys.exit(1)
    if args.gpu:
        modelconf.use_gpu()
    train_masks_path = None
    test_masks_path = None

    save_img_file = os.path.join(args.img_dir, 'preprocessed_imgs')
    save_label_file = os.path.join(args.label_dir, 'preprocessed_imgs')

    train_data = seg.HorsesSegDataset_Features(args.img_dir, args.train_feats_file, args.label_dir,  seg.TRAIN, load=args.load_data)
    val_data = seg.HorsesSegDataset_Features(args.img_dir, args.val_feats_file, args.label_dir, seg.VAL, load=args.load_data)
    test_data = seg.HorsesSegDataset_Features(args.img_dir, args.test_feats_file, args.label_dir, seg.TEST, load=args.load_data)

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

    params = {
        'batch_size':10000, 
        'interleaved_itrs':10, 
        'print_MAP':False, 
        'mp_eps':0.0, 
        'mp_itrs':args.mp_itrs,
        'global_init_val':args.global_init_val,
        'num_global_layers':args.num_global_layers,
        'keep_graph_order':True,
        'test_interleaved_itrs':args.test_interleaved_itrs,
        'use_pd':args.use_pd,
        'test_lambda_l_rate':args.test_lambda_l_rate,
        'test_max_globals_l_rate':args.test_max_globals_l_rate,
        'test_avg_thresh':args.test_avg_thresh,
        'test_mp_interval':args.test_mp_interval,
        'ignore_global_top':args.ignore_global_top,
        'global_top_full':args.global_top_full,
    }
    if args.wide_top:
        params['wide_top'] = True
    if args.load_max_globals:
        params['load_max_globals'] = True
    if args.max_globals_save_path:
        params['max_globals_save_path'] = args.max_globals_save_path
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.use_pd is not None:
        args_dict['use_pd'] = args.use_pd
    if args.reinit is not None:
        params['reinit'] = args.reinit
    params['window_size'] = 100

    if args.model == 'unary':
        args_dict['pair_inds'] = {}
        graph = Graph(nodes, [], 2, seg.HorsesFeatureModel, args_dict, False)
        model = PairwiseModel([graph], len(nodes), 2, params)
    elif args.model == 'pairwise':
        full_graph = Graph(nodes, pairs, 2, seg.HorsesFeatureModel, args_dict, False)
        model = PairwiseModel([full_graph], len(nodes), 2, params)
        model.load(args.pretrain)
    elif args.model == 'global_cnn':
        graphs = []
        if args.use_res:
            if args.use_gt:
                params['global_model'] = seg.HorsesCNNTop_GTRes
                params['global_inputs'] = ['other_obs', 'data_masks']
            else:
                params['global_model'] = seg.HorsesCNNTop_Res
                params['global_inputs'] = ['other_obs']
        else:
            params['global_model'] = seg.HorsesCNNTop
        nodes_graph = Graph(nodes, [], 2, seg.HorsesFeatureModel, args_dict, False) 
        graphs.append(nodes_graph)
        for ind, pair in enumerate(pairs):
            graphs.append(Graph([], [pair], 2, seg.HorsesFeatureModel, args_dict, False))
        if args.interleave:
            model = GlobalPairwiseModel_AveragingInterleaved(graphs, len(nodes), 2, params)
        else:
            model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, params)
        model.load(args.pretrain)
        print("INITIALIZING MAX GLOBALS")
        mp_graph = [fastmp.FastMP(len(model.node_regions), model.num_vals, model.pair_regions, np.zeros(model.num_potentials, dtype=float))] 
        model.init_dataset(train_data, mp_graph, False)
        train_dataloader = DataLoader(train_data, batch_size=params['batch_size'], collate_fn=datasets.collate_batch)
        model.init_max_globals(train_dataloader, params['mp_eps'], True)



    print("Testing on val data...")
    val_results = model.test(val_data, params)
    val_acc, val_iu = seg.calculate_accuracy(val_data, val_results)
    print("VAL:")
    print("\tPIXEL ACC: ",val_acc)
    print("\tMEAN IU:   ",val_iu)

    print("Testing on test data...")
    test_results = model.test(test_data, params)
    test_acc, test_iu = seg.calculate_accuracy(test_data, test_results)
    print("TEST:")
    print("\tPIXEL ACC: ",test_acc)
    print("\tMEAN IU:   ",test_iu)

    print("Testing on training data...")
    train_results = model.test(train_data, params)
    train_acc, train_iu = seg.calculate_accuracy(train_data, train_results)

    print("FINAL RESULTS:")
    print("TRAIN:")
    print("\tPIXEL ACC: ",train_acc)
    print("\tMEAN IU:   ",train_iu)
    print("VAL:")
    print("\tPIXEL ACC: ",val_acc)
    print("\tMEAN IU:   ",val_iu)
    print("TEST:")
    print("\tPIXEL ACC: ",test_acc)
    print("\tMEAN IU:   ",test_iu)



