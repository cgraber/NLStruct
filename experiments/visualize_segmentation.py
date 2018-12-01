from . import segmentation_experiment as seg
from PIL import Image
from deepstruct.models import *
import deepstruct.models.modelconf
import sys
import numpy as np
import argparse



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




    args = parser.parse_args()

    if args.gpu == True:
        modelconf.use_gpu()

    train_data = seg.HorsesSegDataset_Features(args.img_dir, args.train_feats_file, args.label_dir,  seg.TRAIN, load=args.load_data)
    val_data = seg.HorsesSegDataset_Features(args.img_dir, args.val_feats_file, args.label_dir, seg.VAL, load=args.load_data)

    num_rows = num_cols = 64
    nodes = list(range(num_rows*num_cols))
    pairs = []
    pair_inds = {}
    ind = 0
    for col in range(num_cols):
        for row in range(num_rows):
            node1 = num_cols*row + col
            if col < num_cols-1:
                pairs.append((node1, node1+1))
                pair_inds[(node1, node1+1)] = ind
                ind += 1
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


    val_results = model.test(val_data, params)
    
    '''
    for ind, val_result in enumerate(val_results[:20]):
        result = np.array(val_result, dtype=np.uint8).reshape((64, 64))*255
        im = Image.fromarray(result, 'L')
        path = os.path.join(args.working_dir, "im_%d.png"%ind)
        im.save(path)
    '''
    print(val_data[6][0])
    result = np.array(val_data[6][0], dtype=np.uint8).reshape((64, 64))*255
    im = Image.fromarray(result, 'L')
    path = os.path.join(args.working_dir, 'gt.png')
    im.save(path)
