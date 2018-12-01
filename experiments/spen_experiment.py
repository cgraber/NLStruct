from deepstruct.models import *
from  . import tagging_experiment as tag
import argparse, sys

import deepstruct.models.modelconf as modelconf

class GlobalPairwiseModel_SPEN(GlobalPairwiseModel):
    def init_max_globals(self, dataloader, eps, is_train, save_path=None, load=False):
        pass #Only for global model

    def assignment2beliefs(self, assignment):
        node_beliefs = torch.stack([1-assignment, assignment], dim=2).view(len(assignment), -1)
        pair_beliefs = Variable(modelconf.tensor_mod.FloatTensor(len(assignment),len(self.pair_regions)*self.num_vals*self.num_vals))
        #offset = 2*len(self.node_regions)
        for ind,(node1,node2) in enumerate(self.pair_regions):
            unaries_1 = node_beliefs[:, 2*node1:2*(node1+1)]
            unaries_2 = node_beliefs[:, 2*node2:2*(node2+1)]
            pair_beliefs[:,4*ind:4*(ind+1)] = (unaries_2.unsqueeze(2)*unaries_1.unsqueeze(1)).contiguous().view(len(assignment), -1)
        return torch.cat([node_beliefs, pair_beliefs], dim=1)

    def inference_step(self, batch, epoch, num_iters, mp_itrs, eps, is_train, return_obj=False):
        if self.wide_top:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs = False, update_graphs=True, detach_pots=True)
            pots = aggregate_pots[1]
        else:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=True, detach_pots=True)
            pots = None
        assignment = Variable(modelconf.tensor_mod.FloatTensor(len(batch), len(self.node_regions)).uniform_(0., 1.), requires_grad=True)
        opt = torch.optim.SGD([assignment], lr=0.1)#, momentum=0.95)
        past_obj = float('inf')

        beliefs = self.assignment2beliefs(assignment)
        obj = -1*self.global_model(pots*beliefs).sum()
        for i in range(num_iters):
            obj.backward()
            opt.step()

            assignment.data.clamp_(0., 1.)
            beliefs = self.assignment2beliefs(assignment)
            obj = -1*self.global_model(pots*beliefs).sum()
            diff = abs(past_obj - obj.data[0])
            past_obj = obj.data[0]
            print("\t%d, %f, %f"%(i, past_obj, diff))

        batch.beliefs.data.copy_(beliefs.data)
        diff_vals = []   
        y_updates = []
        print("DONE")
        print("FINAL INF OBJ: ",past_obj)
        return None, diff_vals, y_updates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Compare SPEN against our method for trained tagging model')
    parser.add_argument('working_dir')
    parser.add_argument('model_path')
    parser.add_argument('--img_dir')
    parser.add_argument('--label_dir')
    parser.add_argument('--train_feat_file')
    parser.add_argument('--train_label_file')
    parser.add_argument('--val_feat_file')
    parser.add_argument('--val_label_file')
    parser.add_argument('--test_feat_file')
    parser.add_argument('--test_label_file')
    parser.add_argument('--test_interleaved_itrs', type=int, default=500)
    parser.add_argument('--global_hidden_size', type=int)
    parser.add_argument('--global_activation', choices=['sigmoid', 'relu', 'hardtanh', 'tanh'])
    parser.add_argument('--max_globals_l_rate', type=float, default=1e-1)
    parser.add_argument('--lambda_l_rate', type=float, default=1e-1)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--load_masks', action='store_true')
    parser.add_argument('--val_masks_path')
    parser.add_argument('--mlp_init', choices=['v1', 'v2'])
    parser.add_argument('--use_pd', action='store_true', default=False)
    parser.add_argument('--pot_div_factor', type=float, default=1.0)
    parser.add_argument('--random_init', type=float, default=10.0)

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
        val_masks_path = args.val_masks_path
    else:
        val_masks_path = None

    dataset_type = tag.FULL
    train_data = tag.FlickrTaggingDataset_Features(dataset_type, args.train_feat_file, args.label_dir, args.train_label_file, tag.TRAIN, load=load, masks_path=val_masks_path, images_folder=args.img_dir)
    val_data = tag.FlickrTaggingDataset_Features(dataset_type, args.val_feat_file, args.label_dir, args.val_label_file, tag.VAL, load=load, masks_path=val_masks_path, images_folder=args.img_dir)
    test_data = tag.FlickrTaggingDataset_Features(dataset_type, args.test_feat_file, args.label_dir, args.test_label_file, tag.TEST, load=load, masks_path=val_masks_path, images_folder=args.img_dir)

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
        'batch_size':args.batch_size, 
        'interleaved_itrs':10, 
        'mp_eps':0.0, 
        'mp_itrs':100,
        'checkpoint_dir':args.working_dir,
        'test_interleaved_itrs':args.test_interleaved_itrs,
        'lambda_l_rate':args.lambda_l_rate,
        'max_globals_l_rate':args.max_globals_l_rate,
        'use_pd':args.use_pd,
        'wide_top':True,
        'global_hidden_size':args.global_hidden_size,
        'global_activation':args.global_activation,
        'pot_div_factor':args.pot_div_factor,
        'random_init':args.random_init,
    }

    if args.mlp_init is None:
        params['global_model'] = tag.build_mlp_global_model
    elif args.mlp_init == 'v1':
        params['global_model'] = tag.build_initialized_mlp_global_model_v1
    elif args.mlp_init == 'v2':
        params['global_model'] = tag.build_initialized_mlp_global_model_v2

    graphs = []
    nodes_graph = Graph(nodes, [], 2, tag.FlickrFixedModel, args_dict, False)
    graphs.append(nodes_graph)
    for ind,pair in enumerate(pairs):
        graphs.append(Graph([], [pair], 2, tag.FlickrFixedModel, args_dict, False))
        graphs[-1].potential_model.pair_model = torch.nn.Parameter(torch.FloatTensor(4))

    model = GlobalPairwiseModel_SPEN(graphs, len(nodes), 2, params)
    model.load(args.model_path)
    spen_train_losses = []
    spen_val_losses = []
    spen_test_losses = []
    start = time.time()
    for i in range(5):
        print("##############################")
        print("RUNNING SPEN EXPERIMENT ITR %d"%i)
        print("##############################")
        train_results = model.test(train_data, params)
        train_loss = tag.calculate_hamming_loss(train_data, train_results)
        print("SPEN TRAIN LOSS: ", train_loss)
        spen_train_losses.append(train_loss)
    for i in range(5):
        print("##############################")
        print("RUNNING SPEN EXPERIMENT ITR %d"%i)
        print("##############################")
        val_results = model.test(val_data, params)
        val_loss = tag.calculate_hamming_loss(val_data, val_results)
        print("SPEN VAL LOSS: ", val_loss)
        spen_val_losses.append(val_loss)
    for i in range(5):
        print("##############################")
        print("RUNNING SPEN EXPERIMENT ITR %d"%i)
        print("##############################")
        test_results = model.test(test_data, params)
        test_loss = tag.calculate_hamming_loss(test_data, test_results)
        print("SPEN TEST LOSS: ", test_loss)
        spen_test_losses.append(test_loss)
    ''' 
    model = GlobalPairwiseModel_Averaging(graphs, len(nodes), 2, params)
    model.load(args.model_path)
    avg_losses = []

    for i in xrange(5):
        print "##############################"
        print "RUNNING AVG EXPERIMENT ITR %d"%i
        print "##############################"
        val_results = model.test(val_data, params)
        val_loss = tag.calculate_hamming_loss(val_data, val_results)
        print "AVG VAL LOSS: ", val_loss
        avg_losses.append(val_loss)

    pd_losses = []
    params['use_pd'] = True
    for i in xrange(5):
        print "##############################"
        print "RUNNING PD EXPERIMENT ITR %d"%i
        print "##############################"
        val_results = model.test(val_data, params)
        val_loss = tag.calculate_hamming_loss(val_data, val_results)
        print "PD VAL LOSS: ", val_loss
        pd_losses.append(val_loss)
    end = time.time()
    '''
    print("FINAL RESULTS:")
    print("SPEN")
    print("TRAIN:")
    print("\tAVG LOSS: ",np.mean(spen_train_losses))
    print("\tstd dev: ", np.std(spen_train_losses))
    print("VAL:")
    print("\tAVG LOSS: ",np.mean(spen_val_losses))
    print("\tstd dev: ", np.std(spen_val_losses))
    print("TEST:")
    print("\tAVG LOSS: ",np.mean(spen_test_losses))
    print("\tstd dev: ", np.std(spen_test_losses))
    #print "Averaging"
    #print "\tAVG LOSS: ",np.mean(avg_losses)
    #print "\tstd dev: ", np.std(avg_losses)
    #print "PD"
    #print "\tAVG LOSS: ",np.mean(pd_losses)
    #print "\tstd dev: ", np.std(pd_losses)
    #print ""
    #print "TIME: ",(end-start)
