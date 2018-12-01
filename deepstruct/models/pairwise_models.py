import sys, math, time, collections, shutil, os, random
from copy import copy
import dill as pickle
import numpy as np
import torch
import torch.optim.lr_scheduler
import torch.cuda
from . import modelconf

np.random.seed(1)
torch.manual_seed(1)
random.seed(1)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
from deepstruct.fastmp import fastmp
from deepstruct.datasets import datasets
flag = False

#Auxiliary function - we need a different iterator order than the default for fastmp
def product(*args, **kwds):
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [[y] + x for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


class BasePairwiseModel(object):
    """The base class used to implement common methods for the provided structured models"""

    def __init__(self, graphs, num_nodes, num_vals, params):
        self.graphs = graphs
        self.num_nodes = num_nodes
        self.num_vals = num_vals
        self.C_R = 1
        self.use_loss_augmented_inference = False
        
    def save(self, fout):
        raise NotImplementedError

    def load(self, fin):
        raise NotImplementedError

    def get_vals(self, region):
        """Returns the possible assignments of values to a specified region"""
        if type(region) == tuple:
            return product(range(self.num_vals), repeat=2)
        else:
            return range(self.num_vals)

    def init_max_globals(self, dataloader, eps, is_train, save_path=None, load=False):
        pass #Only for global model

    def calculate_obj(self, batch, potentials, mp_eps, use_data, normalize):
        raise NotImplementedError

    def calculate_potentials(self, batch, eps, is_train, volatile=False):
        raise NotImplementedError

    def update_potentials(self, batch, eps, model_optimizer):
        """Calculates the training objective and takes a step of the provided optimizer"""
        self.set_to_eval(False)
        if modelconf.USE_GPU:
            orig_lambdas = batch.lambdas
            orig_max_globals = batch.max_globals
            batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
            batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]
        pots = self.calculate_potentials(batch, eps, True)

        model_optimizer.zero_grad()
        obj,_ = self.calculate_obj(batch, pots, eps, True, True)
        print("FIRST OBJ: ",obj.item())
        obj.backward()
        model_optimizer.step()

        self.set_to_eval(True)
        pots = self.calculate_potentials(batch, eps, True)
        new_obj,_ = self.calculate_obj(batch, pots, eps, True, True)
        print("SECOND OBJ: ",new_obj.item())
        if modelconf.USE_GPU:
            batch.lambdas = orig_lambdas
            batch.max_globals = orig_max_globals
        return new_obj, obj.item() - new_obj.item()

    def get_belief(self, mp_graph, region, val):
        """Calculates the correct index to extract a belief from the given graph"""
        region_ind = self.region_ind_dict[region]
        if type(val) == tuple:
            val_ind = val[1]*self.num_vals + val[0]
        else:
            val_ind = val
        return mp_graph.get_belief(region_ind, val_ind)

    def update_beliefs(self, mp_graphs, eps, potentials=None):
        """Calculates the beliefs given the provided potentials and the current messages"""
        if potentials is not None:
            self.update_mp_graph_potentials(mp_graphs, potentials)
        fastmp.update_beliefs(mp_graphs, eps)

    def update_mp_graph_potentials(self, mp_graphs, potentials):
        """Updates the potential array pointers within fastmp"""
        for graph_ind, mp_graph in enumerate(mp_graphs):
            pots = potentials[graph_ind, :].data.cpu().numpy()
            mp_graph.update_potentials(pots)

    def update_belief_pointers(self, mp_graphs, beliefs):
        """Updates the belief array pointers within fastmp"""
        for graph_ind, mp_graph in enumerate(mp_graphs):
            belief = beliefs[graph_ind, :].data.cpu().numpy()
            mp_graph.update_belief_pointer(belief)

    def inference_step(self, batch, epoch, num_iters, eps, is_train, return_obj=False):
        raise NotImplementedError

    def init_batch(self, batch, mp_graphs, beliefs, test=False):
        """Sets up the model to be able to properly work on the provided batch.

        This includes:
        - Updating the message pointers within fastmp
        - Setting the fields of the batch object correctly
        """
        for graph in self.graphs:
            graph.set_data(batch.data)
            graph.set_observations(batch.observations)
        if mp_graphs is not None:
            for idx,msg in enumerate(batch.msgs):
                if test:
                    new_msg = self.messages = np.zeros(mp_graphs[idx].get_num_msgs(), dtype=float)
                    mp_graphs[idx].update_msgs(new_msg)
                else:
                    mp_graphs[idx].update_msgs(msg)
                mp_graphs[idx].update_beliefs_pointer(beliefs[idx, :].data.numpy())

            batch.mp_graphs = mp_graphs[:len(batch)]
        if beliefs is not None:
            batch.beliefs = beliefs[:len(batch), :]

    def get_all_parameters(self):
        raise NotImplementedError

    def get_inf_opts(self, batch, epoch):
        raise NotImplementedError

    def get_data_masks(self, datum):
        return [Variable(graph.get_data_mask(datum), requires_grad=False) for graph in self.graphs]

    def init_dataset(self, dataset, mp_graphs, use_loss_augmented_inf, top_width=None, load_loss_aug=False, loss_aug_save_path=None, masks_save_path=None):
        """Initializes a dataset properly to work with this framework.

        This includes:
        - Initializing retained variables (lambda, messages, y) for each datum
        - Precomputing the losses used for loss-augmented inference
        - Precomputing the data masks used to compute the data scores in the loss function
        """
        if top_width == None:
            dataset.init(len(self.graphs), mp_graphs[0].get_num_msgs())
        else:
            dataset.init(top_width, mp_graphs[0].get_num_msgs())
        if use_loss_augmented_inf:
            if load_loss_aug:
                dataset.loss_augmentations = torch.load(loss_aug_save_path)
            else:
                #Precalculate for every possible value
                losses = []
                for data_val in range(self.num_vals):
                    result = Variable(modelconf.tensor_mod.FloatTensor(self.num_vals), requires_grad=False)
                    losses.append(result)
                    for other_val in range(self.num_vals):
                        result[other_val] = self.inf_loss(data_val, other_val)

                #Now get full loss augmentations for each data point
                for idx,datum in enumerate(dataset):
                    result = Variable(modelconf.tensor_mod.FloatTensor(self.num_potentials).fill_(0.0), requires_grad=False)
                    dataset.loss_augmentations[idx] = result
                    for region_ind, region_val in enumerate(datum[0]):
                        result[region_ind*self.num_vals:(region_ind+1)*self.num_vals] = losses[region_val]
                if loss_aug_save_path != None:
                    loss_augs = torch.stack(dataset.loss_augmentations)
                    torch.save(loss_augs, loss_aug_save_path)


        if dataset.data_masks != None:
            return # This indicates masks were precomputed/loaded
        data_masks = []
        print("CREATING MASKS")
        for entry_ind, entry in enumerate(dataset):
            if entry_ind % 1000 == 0:
                print("ENTRY: %d/%d"%(entry_ind+1, len(dataset)))
            data_masks.append(self.get_data_masks(entry[0]))
        print("DONE")

        if masks_save_path is not None:
            print("SAVING MASKS")
            torch.save(data_masks, masks_save_path)
            print("DONE")
        dataset.data_masks = data_masks

    def save_training_checkpoint(self, checkpoint_folder, epoch, optimizer, scheduler, dataset):
        """Save new checkpoint for training, including optimizer and model"""
        folder = os.path.join(checkpoint_folder, "epoch_%d/"%int(epoch))
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.save(os.path.join(folder, "model"))
        dataset.save_checkpoint(os.path.join(folder, "dataset_info"))
        with open(os.path.join(folder, "training_info"), "wb") as fout:
            pickle.dump([epoch, optimizer.param_groups], fout)

        
    def resume_training(self, dataset, checkpoint_folder='tmp/'):
        """Given a checkpoint folder, loads everything necessary to resume training"""

        # Find most recent checkpoint
        dirs = [name for name in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, name))]
        folder = dirs[0]
        val = int(folder.split("_")[1])
        for option in dirs[1:]:
            new_val = int(option.split("_")[1])
            if new_val > val:
                folder = option
                val = new_val

        # Load checkpoint
        folder = os.path.join(checkpoint_folder, folder)
        self.load(os.path.join(folder, "model"))
        dataset.load_checkpoint(os.path.join(folder, "dataset_info"))
        with open(os.path.join(folder, "training_info"), "rb") as fin:
            resume_info=pickle.load(fin)
        return self.train(dataset, resume_info=resume_info)

    def get_model_optimizer(self, params):
        raise NotImplementedError

    def run_first_inf(self, data_loader, mp_graphs, beliefs, num_itrs, mp_itrs, mp_eps):
        for batch_ind,batch in enumerate(data_loader):
            #print "\tBATCH %d OF %d"%(batch_ind+1, len(train_data_loader))
            self.init_batch(batch, mp_graphs, beliefs)
            _, new_diff_vals, new_y = self.inference_step(batch, 1, num_itrs, mp_itrs, mp_eps, True)
            if batch_ind == 0:
                diff_vals = new_diff_vals
                y_vals = new_y
        return None,diff_vals, y_vals

    def set_to_eval(self, is_eval):
        for graph in self.graphs:
            if is_eval:
                graph.potential_model.eval()
            else:
                graph.potential_model.train()

    def train(self, train_data, validation_data, params, resume_info=None):
        """Runs the training procedure using the specified data and parameters"""

        l_rate = params.get('l_rate', 1e-4)
        checkpoint_dir = params.get('checkpoint_dir', 'tmp/')
        self.batch_size = batch_size = params.get('batch_size', 10)
        epsilon = params.get('epsilon', 1.0) 
        use_early_stopping = params.get('use_early_stopping', False)
        training_scheduler = params.get('training_scheduler', lambda opt: 
                torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda epoch:1.0/math.sqrt(epoch) if epoch > 0 else 1.0))
        val_scheduler = params.get('val_scheduler', None)
        self.MAP_epsilon = params.get('MAP_epsilon', 1e-2)
        self.use_regularization = params.get('use_regularization', False)
        self.record_inf = params.get('record_inf', False)
        self.train_interleaved_itrs = train_interleaved_itrs = params.get('train_interleaved_itrs', 10)
        self.test_interleaved_itrs = test_interleaved_itrs = params.get('test_interleaved_itrs', 10)
        num_epochs = params.get('num_epochs', 20)
        print_beliefs = params.get('print_beliefs', False)
        print_MAP = params.get('print_MAP', False)
        self.mp_itrs = mp_itrs = params.get('mp_itrs', 100)
        self.mp_eps = mp_eps = params.get('mp_eps', 0.0)
        self.use_loss_augmented_inference = params.get('use_loss_augmented_inference', False)
        self.inf_loss = params.get('inf_loss', squared_diff)
        task_loss = params.get('task_loss', None)
        test_data = params.get('test_data', None)
        train_loss_data = copy(train_data)
        train_masks_path = params.get('train_masks_path', None)
        test_masks_path = params.get('test_masks_path', None)
        load_loss_aug = params.get('load_loss_aug', False)
        loss_aug_save_path = params.get('loss_aug_save_path', None)
        load_max_globals = params.get('load_max_globals', False)
        max_globals_save_path = params.get('max_globals_save_path', None)
        val_interval = params.get('val_interval', 10)
        save_checkpoints = params.get('save_checkpoints', True)
        shuffle_data = params.get('shuffle_data', False)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.mp_graphs = [fastmp.FastMP(len(self.node_regions), self.num_vals, self.pair_regions, np.zeros(self.num_potentials, dtype=float)) for _ in range(batch_size)]
        belief_size = self.mp_graphs[0].get_num_beliefs()
        beliefs = Variable(torch.FloatTensor(self.batch_size, belief_size), requires_grad=False)
        if modelconf.USE_GPU:
            print("PINNING BELIEFS")
            beliefs.data.pin_memory()
        if resume_info is not None:
            epoch, param_groups = resume_info
            param_groups[0]['params'] = self.get_all_parameters()
            model_optimizer = torch.optim.SGD(param_groups)
        else:
            model_optimizer = self.get_model_optimizer(params)
            self.init_dataset(train_data, self.mp_graphs, self.use_loss_augmented_inference, load_loss_aug=load_loss_aug, loss_aug_save_path=loss_aug_save_path, masks_save_path=train_masks_path)
            epoch = 0

        print("CREATING DATA LOADER")
        train_data_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=datasets.collate_batch, pin_memory=modelconf.USE_GPU)
        print("INITIALIZING MAX GLOBALS")
        self.init_max_globals(train_data_loader, mp_eps, True, save_path=max_globals_save_path, load=load_max_globals)
        print("DONE")
        if training_scheduler is not None:
            training_scheduler = training_scheduler(model_optimizer)
            training_scheduler.last_epoch = epoch
        if val_scheduler is not None:
            val_scheduler = val_scheduler(model_optimizer)
            val_scheduler.last_epoch = epoch

        if validation_data is not None:
            self.init_dataset(validation_data, self.use_loss_augmented_inference)
            validation_data_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=datasets.collate_batch)
            validation_score = new_validation_score = self.calculate_full_data_obj(validation_data_loader, self.mp_graphs, beliefs, mp_eps)
            best_obj = start_obj
            self.save(os.path.join(checkpoint_dir, 'best_validation'))
        if test_data is not None:
            self.init_dataset(test_data, self.mp_graphs, False, masks_save_path=test_masks_path)
            test_data_loader = DataLoader(test_data, batch_size=1, collate_fn = datasets.collate_batch, pin_memory=modelconf.USE_GPU)

        validation_patience = 0

        diff = epsilon + 1
        print("STARTING LEARNING")
        start = end = 0
        exited_early=False
        
        train_values = []
        val_values = []

        diff_vals = []
        y_updates = []
        train_task_losses = []
        test_task_losses = []

        print("TASK LOSS STUFF")
        if task_loss is not None:
            _, new_diff_vals, new_y = self.run_first_inf(train_data_loader, self.mp_graphs, beliefs, train_interleaved_itrs, mp_itrs, mp_eps)
            diff_vals.append(new_diff_vals)
            y_updates.append(new_y)
            train_results = self.get_MAP_assignment(train_data, self.mp_graphs, beliefs, batch_size, mp_eps)
            train_task_losses.append(task_loss(train_data, train_results))
            print "TRAIN TASK LOSSES: ",train_task_losses[-1]
            if test_data is not None:
                test_results = self.test(test_data, params, test_masks_path)
                test_task_losses.append((epoch, task_loss(test_data, test_results)))
                print("TEST TASK LOSSES: ",test_task_losses[-1])

        itrs = -1
        total_time = 0
        validation_scores = []
        while epoch < num_epochs:
            #NEW VERSION
            epoch += 1
            print("EPOCH", epoch, (end-start))
            if training_scheduler is not None:
                training_scheduler.step()
            if shuffle_data:
                train_data.shuffle()
            start = time.time() 
            for batch_ind,batch in enumerate(train_data_loader):
                self.init_batch(batch, self.mp_graphs, beliefs)
                _, new_diff_vals, new_y = self.inference_step(batch, epoch, train_interleaved_itrs, mp_itrs, mp_eps, True)

                diff_vals.append(new_diff_vals)
                y_updates.append(new_y)

                train_results = self.get_batch_MAP_assignment(batch, mp_eps, True)
                train_task_losses.append(task_loss(batch, train_results))
                print("TRAIN TASK LOSSES: ",train_task_losses[-1])

                new_obj, diff = self.update_potentials(batch, mp_eps, model_optimizer)
                current_obj = new_obj
                diff = 0

                print("NEW TRAIN TASK LOSSES: ",task_loss(batch, self.get_batch_MAP_assignment(batch, mp_eps, True)))

                print("\tMINI BATCH %d of %d: %f"%(batch_ind+1, len(train_data_loader), new_obj.item()))
                train_values.append((epoch*len(train_data_loader)+batch_ind, new_obj.item()))
            end = time.time()
            total_time += (end-start)
            if epoch%val_interval == 0 and save_checkpoints:
                print("Saving...")
                self.save_training_checkpoint(checkpoint_dir, epoch, model_optimizer, training_scheduler, train_data)
                print("Done")
            if task_loss is not None and epoch%val_interval == 0:
                #If you want to compute loss on training data, uncomment the following lines
                #train_results = self.get_MAP_assignment(train_data, self.mp_graphs, beliefs, batch_size, mp_eps)
                #train_results = self.test(train_loss_data, params)
                #train_task_losses.append(task_loss(train_data, train_results))
                #print "TRAIN TASK LOSSES: ",train_task_losses[-1]
                #train_task_losses.append((0,0))

                if test_data is not None:
                    test_results = self.test(test_data, params, test_masks_path)
                    test_task_losses.append((epoch*len(train_data_loader), task_loss(test_data, test_results)))
                    print("TEST TASK LOSSES: ",test_task_losses[-1])

                    if val_scheduler is not None:
                        validation_scores.append(test_task_losses[-1][-1][-1])        
                        if len(validation_scores) > 6:
                            validation_scores.pop(0)
                        print("CURRENT VAL SCORES: ",validation_scores)
                        if len(validation_scores) == 6:
                            if sum(validation_scores[:3])/3 >= sum(validation_scores[3:])/3:
                                print("RUNNING VALIDATION SCHEDULER")
                                val_scheduler.step()
        if use_early_stopping and (exited_early or validation_patience != 0):
            self.load(os.path.join(checkpoint_dir, 'best_validation'))
            train_values = train_values[:-1*validation_patience]
            val_values = val_values[:-1*validation_patience]
        self.save(os.path.join(checkpoint_dir, "final_model"))
        print("FINAL EPOCH TIME: ",(end-start))
        print("TOTAL TRAIN TIME: ",total_time)
        
        results = self.get_MAP_assignment(train_data, self.mp_graphs, beliefs, batch_size, print_beliefs)
        if print_MAP:
            for datum,result in zip(train_data, results):
                print("CORRECT: ",datum[0])
                print("FOUND:   ", result)
        return_vals = {}
        return_vals['train_vals'] = train_values
        return_vals['diff_vals'] = diff_vals
        return_vals['y_updates'] = y_updates
        if task_loss is not None:
            return_vals['train_task_losses'] = train_task_losses
            if test_data is not None:
                return_vals['test_task_losses'] = test_task_losses
        if validation_data is not None:
            return current_obj, validation_score.item(), results, return_vals
        else:
            return current_obj, results, return_vals

    def get_batch_MAP_assignment(self, batch, mp_eps, is_train):
        """Compute the MAP assignment for the current batch given the current message values
           (i.e. message passing is not run here)"""
        if modelconf.USE_GPU:
            orig_lambdas = batch.lambdas
            orig_max_globals = batch.max_globals
            batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
            batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]

        pots = self.calculate_potentials(batch, mp_eps, is_train, volatile=True)
        self.update_beliefs(batch.mp_graphs, mp_eps, pots)
        results = []
        for ind in range(len(batch)):
            final_assignment = [0]*self.num_nodes
            results.append(final_assignment)
            for node_region in range(self.num_nodes):
                max_val = -float("inf")
                max_assignment = None
                for node_val in self.get_vals(node_region):
                    new_belief = self.get_belief(batch.mp_graphs[ind], node_region, node_val)
                    if new_belief > max_val:
                        max_val = new_belief
                        max_assignment = node_val
                final_assignment[node_region] = max_assignment
        return results

        
    def get_MAP_assignment(self, dataset, mp_graphs, beliefs, batch_size, mp_eps, print_beliefs=False):
        """Compute the MAP assignment for the dataset given the current message values
           (i.e. message passing is not run here)"""
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=datasets.collate_batch, pin_memory=modelconf.USE_GPU)
        results = []
        for batch in data_loader:
            self.init_batch(batch, mp_graphs, beliefs)
            if modelconf.USE_GPU:
                orig_lambdas = batch.lambdas
                orig_max_globals = batch.max_globals
                batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
                batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]

            pots = self.calculate_potentials(batch, mp_eps, False, volatile=True)
            self.update_beliefs(batch.mp_graphs, mp_eps, pots)
            for ind in range(len(batch)):
                final_assignment = [0]*self.num_nodes
                results.append(final_assignment)
                if print_beliefs:
                    print("CORRECT: ",batch.item())
                    print("OBS: ",batch.observations[0])
                for node_region in range(self.num_nodes):
                    if print_beliefs:
                        print("REGION: ",node_region)
                    max_val = -float("inf")
                    max_assignment = None
                    for node_val in self.get_vals(node_region):
                        if print_beliefs:
                            print("\tbelief %d: "%node_val, self.get_belief(batch.mp_graphs[ind], node_region,node_val))
                        new_belief = self.get_belief(batch.mp_graphs[ind], node_region, node_val)
                        if new_belief > max_val:
                            max_val = new_belief
                            max_assignment = node_val
                    final_assignment[node_region] = max_assignment

                if print_beliefs:
                    for pair_region in self.pair_regions:
                        print("REGION: ",pair_region)
                        for pair_val in self.get_vals(pair_region):
                            print("\tbelief", pair_val, ": ", self.get_belief(batch.mp_graphs[ind], pair_region, pair_val))
        return results


    def test(self, dataset, params, masks_path=None):
        self.use_regularization = params.get('use_regularization', False)
        self.record_inf = params.get('record_inf', False)
        self.test_interleaved_itrs = test_interleaved_itrs = params.get('test_interleaved_itrs', 10)
        num_epochs = params.get('num_epochs', 20)
        print_beliefs = params.get('print_beliefs', False)
        print_MAP = params.get('print_MAP', False)
        self.mp_itrs = mp_itrs = params.get('mp_itrs', 100)
        self.mp_eps = mp_eps = params.get('mp_eps', 0.0)
        self.use_loss_augmented_inference = params.get('use_loss_augmented_inference', False)
        self.inf_loss = params.get('inf_loss', squared_diff)
        task_loss = params.get('task_loss', None)
        test_data = params.get('test_data', None)

        mp_eps = params['mp_eps']
        batch_size = params['batch_size']
        MAP_epsilon = params.get('MAP_epsilon', 1e-2)
        self.record_inf = params.get('record_inf', False)
        results = []
        self.test_mp_graphs = [fastmp.FastMP(self.num_nodes, self.num_vals, self.pair_regions, np.zeros(self.num_potentials, dtype=float)) for _ in range(batch_size)]
        belief_size = self.test_mp_graphs[0].get_num_beliefs()
        beliefs = Variable(torch.FloatTensor(batch_size, belief_size), requires_grad=False)
         
        self.init_dataset(dataset, self.test_mp_graphs, False, masks_save_path=masks_path)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=datasets.collate_batch, pin_memory=modelconf.USE_GPU)
        self.init_max_globals(data_loader, mp_eps, False)
        for batch in data_loader:
            self.init_batch(batch, self.test_mp_graphs, beliefs)
            self.do_MAP_inf(batch, test_interleaved_itrs, mp_itrs, mp_eps, MAP_epsilon)
            for ind in range(len(batch)):
                # Find MAP Configuration
                final_assignment = [0]*self.num_nodes
                results.append(final_assignment)
                for node_region in range(self.num_nodes):
                    max_val = -float("inf")
                    max_assignment = None
                    for node_val in self.get_vals(node_region):
                        new_belief = batch.beliefs[ind, self.potential_ind_dict[node_region][node_val]].item()
                        if new_belief > max_val:
                            max_val = new_belief
                            max_assignment = node_val
                    final_assignment[node_region] = max_assignment
            
        return results

class PairwiseModel(BasePairwiseModel):
    """This is the implementation of a Deep Structured Model"""

    def __init__(self, graphs, num_nodes, num_vals, params):
        super(PairwiseModel, self).__init__(graphs, num_nodes, num_vals, params)
        if len(graphs) > 1:
            raise ValueError('PairwiseModel only accepts one graph as argument')
        self.graph = graphs[0]

        self.node_regions = self.graph.node_regions
        self.pair_regions = self.graph.pair_regions
        self.potential_ind_dict = self.graph.potential_ind_dict
        self.region_ind_dict = self.graph.region_ind_dict
        self.num_potentials = self.num_vals*len(self.node_regions) + self.num_vals*self.num_vals*len(self.pair_regions)
        self.num_node_potentials = self.num_vals*len(self.node_regions)

    def save(self, fout):
        torch.save(self.graph.potential_model.state_dict(), fout)

    def load(self, fin):
        self.graph.potential_model.load_state_dict(torch.load(fin))

    def build_objective(self, mp_graphs):
        return LearningObjective(self.graph.node_regions, self.graph.pair_regions, self.num_vals, mp_graphs, self.C_R)

    def print_params(self):
        print("MODEL PARAMS:")
        for param in self.graph.potential_model.parameters():
            print(param)

    def calculate_potentials(self, batch, eps, is_train, volatile=False):
        tmp_pots = self.graph.calculate_potentials()
        pots = Variable(modelconf.tensor_mod.FloatTensor(len(batch), self.num_potentials), volatile=volatile)
        if len(self.node_regions) > 0:
            pots[:, :self.num_node_potentials] = tmp_pots[:, :self.num_node_potentials]
        if len(self.pair_regions) > 0:
            if self.graph.tie_pairwise:
                pots[:, self.num_node_potentials:] = tmp_pots[:, self.num_node_potentials:].repeat(1, len(self.pair_regions))
            else:
                pots[:, self.num_node_potentials:] = tmp_pots[:, self.num_node_potentials:]

        if is_train and self.use_loss_augmented_inference:
            pots = pots + torch.stack(batch.loss_augmentations)
        self.update_beliefs(batch.mp_graphs, eps, pots)
        return pots

    def calculate_obj(self, batch, pots, mp_eps, use_data, normalize):
        if modelconf.USE_GPU:
            beliefs = batch.beliefs.cuda(async=True)
        else:
            beliefs = batch.beliefs
        entropy = 0.0
        if mp_eps > 0:
            for mp_graph in batch.mp_graphs:
                entropy += mp_graph.get_entropy()

        obj = Variable(modelconf.tensor_mod.FloatTensor(1).fill_(entropy*mp_eps))
        obj += torch.nn.ReLU()(pots*beliefs - pots*batch.data_masks[0]).sum()
        if normalize:
            obj /= len(batch)
        if self.use_regularization:
            for param in self.graph.potential_model.parameters():
                obj += (param*param).sum()*0.5
        return obj,None

    def get_all_parameters(self):
        return self.graph.potential_model.parameters()

    def get_model_optimizer(self, params):
        if type(params['l_rate']) == dict:
            unary = []
            pair = []
            for name, param in self.graph.potential_model.named_parameters():
                if name.startswith('unary'):
                    unary.append(param)
                elif name.startswith('pair'):
                    pair.append(param)
                else:
                    raise Exception('Encountered invalid potential model param: %s'%name)
            param_grps = [{'params':unary, 'lr': params['l_rate']['unary']},
                          {'params':pair,  'lr': params['l_rate']['pair']}]
        else:
            param_grps = [{'params':self.get_all_parameters(), 'lr':params['l_rate']}]
        if params.get('use_adam', False):
            model_optimizer = torch.optim.Adam(
                    param_grps,
                    weight_decay=params.get('weight_decay',0)
                )
        else:
            model_optimizer = torch.optim.SGD(
                    param_grps,
                    momentum=params.get('momentum',0.),
                    weight_decay=params.get('weight_decay',0)
                )
        return model_optimizer



    def get_inf_opts(self, batch, epoch):
        return None

    def get_inf_schedulers(self, inf_opts):
        return []

    def inference_step(self, batch, epoch, num_iters, mp_itrs, eps, is_train, return_obj=False):
        self.set_to_eval(True)
        obj_vals = []
        pots = self.calculate_potentials(batch, eps, is_train, volatile=True)
        if self.record_inf:
            for i in range(num_iters):
                fastmp.runmp(batch.mp_graphs, 1, eps);
                pots = self.calculate_potentials(batch, eps, is_train)
                obj,_ = self.calculate_obj(batch, pots, eps, False, True)
                obj_vals.append(obj.item())
        else:
            fastmp.runmp(batch.mp_graphs, mp_itrs, eps)
        if return_obj:
            pots = self.calculate_potentials(batch, eps, is_train)
            obj,_ = self.calculate_obj(batch, pots, eps, True, True)
            return obj, num_iters, None
        return None, num_iters, obj_vals

    def do_MAP_inf(self, batch, interleaved_itrs, mp_itrs, mp_eps, MAP_epsilon):
        past_obj = diff = float('inf')
        itr = 0.0

        # Run MAP MP
        while diff > MAP_epsilon:
            itr += 1.0
            print("\t",itr, past_obj, diff)
            current_obj,count,_ = self.inference_step(batch, itr, interleaved_itrs, mp_itrs, mp_eps, False, True)
            current_obj = current_obj.item()
            diff = abs(current_obj - past_obj)
            past_obj = current_obj
            if itr >= 1000:
                break

class GlobalPairwiseModel(BasePairwiseModel):
    """This class is the base for the NLStruct model. The primary difference between the two version is inference:
    one runs message passing during every iteration, and one does it only at the beginning/end"""
    def __init__(self, graphs, num_nodes, num_vals, params):
        super(GlobalPairwiseModel, self).__init__(graphs, num_nodes, num_vals, params)

        self.use_pd = params.get('use_pd', False)
        self.pd_theta = params.get('pd_theta', 1.)
        self.reinit = params.get('reinit', False)
        self.record_interval = params.get('record_interval', 500)
        self.random_init = params.get('random_init', None)
        self.train_avg_thresh = params.get('train_avg_thresh', -1)
        self.test_avg_thresh = params.get('test_avg_thresh', -1)
        self.train_max_globals_l_rate = params.get('train_max_globals_l_rate', 1e-1)
        self.train_lambda_l_rate = params.get('train_lambda_l_rate', 1e-1)
        self.test_max_globals_l_rate = params.get('test_max_globals_l_rate', 1e-1)
        self.test_lambda_l_rate = params.get('test_lambda_l_rate', 1e-1)
        self.global_inputs = params.get('global_inputs', [])
        self.global_beliefs = params.get('global_beliefs', False)
        self.diff_update = params.get('diff_update', False)

        if 'wide_top' in params:
            self.wide_top = params['wide_top']
        else:
            self.wide_top = False

        # Register all regions
        if params.get('keep_graph_order', False):
            self.region2graphs = collections.OrderedDict()
        else:
            self.region2graphs = {}
        for graph_ind, graph in enumerate(self.graphs):
            for region in (graph.original_node_regions + graph.original_pair_regions):
                if region not in self.region2graphs:
                    self.region2graphs[region] = []
                self.region2graphs[region].append(graph_ind)
        self.num_regions = len(self.region2graphs)
        self.region_ind_dict = {}
        self.potential_ind_dict = collections.defaultdict(dict)
        self.node_regions = list(range(num_nodes))
        self.pair_regions = []
        potential_ind = 0

        #Currently grouping all node regions at start of indices - might not be necessary
        for node in range(num_nodes):
            self.region_ind_dict[node] = node
            for val in self.get_vals(node):
                self.potential_ind_dict[node][val] = potential_ind
                potential_ind += 1
        region_ind = num_nodes
        for region in self.region2graphs:
            if type(region) == tuple:
                self.pair_regions.append(region)
                self.region_ind_dict[region] = region_ind
                region_ind += 1
                for val in self.get_vals(region):
                    self.potential_ind_dict[region][val] = potential_ind
                    potential_ind += 1
        self.regions = self.node_regions + self.pair_regions
        self.num_potentials = potential_ind
        self.C_R = 1

        if self.wide_top:
            self.top_width = self.num_potentials
        else:
            self.top_width = len(self.graphs)
        params['num_unary_potentials'] = len(self.node_regions)*self.num_vals
        params['num_pair_potentials'] = len(self.pair_regions)*self.num_vals*self.num_vals

        if 'global_model' in params:
            self.global_model = params['global_model'](self.top_width, params)
            print("USING GLOBAL MODEL: ",self.global_model)
        else:
            print("USING DEFAULT GLOBAL MODEL")
            self.global_model = QuadModel(self.top_width, params)
        if modelconf.USE_GPU:
            self.global_model.cuda()

    def set_to_eval(self, is_eval):
        super(GlobalPairwiseModel, self).set_to_eval(is_eval)
        if is_eval:
            self.global_model.eval()
        else:
            self.global_model.train()
    
    def get_data_masks(self, datum):
        if self.wide_top:
            data_mask = torch.zeros(self.num_potentials)
            for region_ind, region in enumerate(self.regions):
                if type(region) == tuple:
                    r1, r2 = region
                    assignment = (int(datum[r1]), int(datum[r2]))
                else:
                    assignment = int(datum[region])
                potential_ind = self.potential_ind_dict[region][assignment]
                data_mask[potential_ind] += 1
            return [data_mask]
        else:
            return super(GlobalPairwiseModel, self).get_data_masks(datum)
        
    def save(self, file_path):
        with open(file_path, "wb") as fout:
            result = [self.global_model.state_dict()]
            for i,graph in enumerate(self.graphs):
                result.append(graph.potential_model.state_dict())
            torch.save(result, fout)

    def load(self, file_path):
        with open(file_path, "rb") as fin:
            params = torch.load(fin)
        self.global_model.load_state_dict(params[0])
        for graph, param in zip(self.graphs, params[1:]):
            graph.potential_model.load_state_dict(param)

    def init_max_globals(self, dataloader, eps, is_train, save_path=None, load=False):
        total = modelconf.tensor_mod.FloatTensor(self.top_width).fill_(0.0)
        num_data = 0.0
        if is_train:
            for batch_ind, batch in enumerate(dataloader):
                print("\t%d of %d"%(batch_ind+1, len(dataloader)))
                self.init_batch(batch, None, None)
                if load:
                    print("\t\tLoading...")
                    path = save_path + "_%d"%batch_ind
                    data_score_input = torch.load(path)
                    print("\t\tDone")
                else:
                    print("\t\tComputing...")
                    if modelconf.USE_GPU:
                        orig_lambdas = batch.lambdas
                        orig_max_globals = batch.max_globals
                        batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
                        batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]
                    pots = self.calculate_potentials(batch, eps, is_train, volatile=True, update_beliefs=False, update_graphs=True, detach_pots=True)
                    if self.wide_top:
                        pots = pots[1]
                    data_score_input = self.get_data_score_input(batch, pots)
                    print("\t\tDone.")
                    if save_path is not None:
                        print("\t\tSaving...")
                        path = save_path + "_%d"%batch_ind
                        torch.save(data_score_input, path)
                        print("\t\tDone.")
                for glob_ind, glob in enumerate(batch.max_globals):
                    glob.data[:self.num_potentials].copy_(data_score_input[glob_ind, :].data[:self.num_potentials])
                    total += data_score_input[glob_ind, :].data[:self.num_potentials]
                    num_data += 1
            self.max_global_init = total / num_data
            if self.global_beliefs:
                self.max_global_init = torch.cat([self.max_global_init, modelconf.tensor_mod.FloatTensor(self.top_width).fill_(1.0)])
        elif self.random_init is None:
            for batch in dataloader:
                self.init_batch(batch, None, None)
                for glob_ind, glob in enumerate(batch.max_globals):
                    glob.data.copy_(self.max_global_init)

    def call_global_model_on_y(self, batch, y):
        if self.global_beliefs:
            inputs = [y[:, :self.num_potentials], y[:, self.num_potentials:]]
        else:
            inputs = [y]
        if 'observations' in self.global_inputs:
            inputs.append(batch.observations)
        if 'other_obs' in self.global_inputs:
            inputs.append(batch.other_obs)
        if 'data_masks' in self.global_inputs:
            inputs.append(batch.data_masks[0])
        return self.global_model(*inputs)

    def call_global_model_on_pots(self, batch, pots, beliefs=None):
        if self.global_beliefs:
            inputs = [pots, beliefs]
        else:
            inputs = [pots]
        if 'observations' in self.global_inputs:
            inputs.append(batch.observations)
        if 'other_obs' in self.global_inputs:
            inputs.append(batch.other_obs)
        if 'data_masks' in self.global_inputs:
            inputs.append(batch.data_masks[0])
        return self.global_model(*inputs)


    def build_inf_obj(self, batch):
        global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
        lambdas = torch.stack(batch.lambdas)
        max_globals = torch.stack(batch.max_globals)
        return global_scores.squeeze() - (lambdas*max_globals).sum(1)

    def get_data_score_input(self, batch, potentials):
        if self.wide_top:
            if self.global_beliefs:
                beliefs = Variable(batch.data_masks[0], requires_grad=False)
                data_score_input = torch.cat([potentials*beliefs, beliefs], dim=1)
            else:
                data_score_input = potentials*Variable(batch.data_masks[0], requires_grad=False)
        else:
            data_score_input = Variable(modelconf.tensor_mod.FloatTensor(len(batch), self.top_width))
            for graph_ind, graph in enumerate(self.graphs):
                data_score_input[:, graph_ind] = (graph.potentials*batch.data_masks[graph_ind]).sum(1)
        return data_score_input

    def calculate_obj(self, batch, aggregate_potentials, mp_eps, use_data, normalize):
        if use_data and self.diff_update:
            pass
        else:
            global_inf_obj = self.build_inf_obj(batch)
        data_scores = 0.0
        if self.wide_top:
            aggregate_potentials, pots = aggregate_potentials
        else:
            pots = None
        if modelconf.USE_GPU:
            beliefs = batch.beliefs.cuda(async=True)
        else:
            beliefs = batch.beliefs
        if use_data:
            inp = self.get_data_score_input(batch, pots)
            if self.global_beliefs:
                data_scores = self.call_global_model_on_pots(batch, inp[:, :self.num_potentials], inp[:, self.num_potentials:])
            else:
                data_scores = self.call_global_model_on_pots(batch, inp)
            if self.diff_update:

                obj = self.call_global_model_on_pots(batch, pots, beliefs).sum(1) - data_scores.squeeze()
                if self.use_loss_augmented_inference:
                    obj = obj + (torch.stack(batch.loss_augmentations)*beliefs).sum(1)
                obj = torch.nn.ReLU()(obj).sum()
            else:
                obj = torch.nn.ReLU()(global_inf_obj + (aggregate_potentials*beliefs).sum(1) - data_scores.squeeze()).sum()
            
        else:
            obj = global_inf_obj.sum() + (aggregate_potentials*beliefs).sum()
        if mp_eps > 0:
            entropy = 0.0
            for mp_graph in batch.mp_graphs:
                entropy += mp_graph.get_entropy()
            obj += entropy * mp_eps
        if normalize:
            obj /= len(batch)
            data_scores = data_scores/len(batch)
        return obj, data_scores


    def calculate_potentials(self, batch, eps, is_train, volatile=False, update_beliefs=True, update_graphs=True, detach_pots=False):
        pots = []
        for graph in self.graphs:
            if update_graphs:
                pots.append(graph.calculate_potentials(detach_pots))
            else:
                pots.append(graph.potentials)
        if self.wide_top:
            combined_pots = AggregatePotentials(len(batch.observations), self.num_vals, self.num_potentials, self.potential_ind_dict, self.regions, self.graphs, False)(*pots)
            if self.global_beliefs:
                lambds = torch.stack(batch.lambdas)
                aggregate_pots = combined_pots * lambds[:, :self.num_potentials] + lambds[:, self.num_potentials:]
            else:
                aggregate_pots = combined_pots * torch.stack(batch.lambdas)
        else:
            inputs = batch.lambdas + pots
            aggregate_pots = AggregatePotentials(len(batch.observations), self.num_vals, self.num_potentials, self.potential_ind_dict, self.regions, self.graphs, True)(*inputs)
        if is_train and self.use_loss_augmented_inference:
            aggregate_pots = aggregate_pots + torch.stack(batch.loss_augmentations)
        if update_beliefs:
            self.update_beliefs(batch.mp_graphs, eps, aggregate_pots)
        if volatile:
            aggregate_pots.detach_()
            aggregate_pots.volatile=True
        if self.wide_top:
            return [aggregate_pots, combined_pots]
        else:
            return aggregate_pots

    def get_local_model_params(self):
        result = []
        for graph in self.graphs:
            result.extend(list(graph.potential_model.parameters()))
        return result

    def inference_step(self, batch, epoch, num_iters, eps, is_train, return_obj=False):
        raise NotImplementedError
        
    def do_MAP_inf(self, batch, interleaved_itrs, mp_itrs, mp_eps, MAP_epsilon): 
        objs = []
        diff = past_val = float('inf')
        itr = 0
        current_obj,count,_ = self.inference_step(batch, 1, interleaved_itrs, mp_itrs, mp_eps, False, True)

    def get_batch_MAP_assignment(self, batch, mp_eps, is_train):
        if modelconf.USE_GPU:
            orig_lambdas = batch.lambdas
            orig_max_globals = batch.max_globals
            batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
            batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]

        pots = self.calculate_potentials(batch, mp_eps, is_train, volatile=True)
        if self.wide_top:
            pots = pots[0]
        self.update_beliefs(batch.mp_graphs, mp_eps, pots)
        results = []
        for ind in range(len(batch)):
            final_assignment = [0]*self.num_nodes
            results.append(final_assignment)
            for node_region in range(self.num_nodes):
                max_val = -float("inf")
                max_assignment = None
                for node_val in self.get_vals(node_region):
                    new_belief = self.get_belief(batch.mp_graphs[ind], node_region, node_val)
                    if new_belief > max_val:
                        max_val = new_belief
                        max_assignment = node_val
                final_assignment[node_region] = max_assignment
        return results
            
    def get_MAP_assignment(self, dataset, mp_graphs, beliefs, batch_size, mp_eps, print_beliefs=False):
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=datasets.collate_batch, pin_memory=modelconf.USE_GPU)
        results = []
        for batch in data_loader:
            self.init_batch(batch, mp_graphs, beliefs)
            if modelconf.USE_GPU:
                orig_lambdas = batch.lambdas
                orig_max_globals = batch.max_globals
                batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
                batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]

            pots = self.calculate_potentials(batch, mp_eps, False, volatile=True)
            if self.wide_top:
                pots = pots[0]
            self.update_beliefs(batch.mp_graphs, mp_eps, pots)
            for ind in range(len(batch)):
                final_assignment = [0]*self.num_nodes
                results.append(final_assignment)
                if print_beliefs:
                    print("CORRECT: ",batch.item())
                    print("OBS: ",batch.observations[0])
                for node_region in range(self.num_nodes):
                    if print_beliefs:
                        print("REGION: ",node_region)
                    max_val = -float("inf")
                    max_assignment = None
                    for node_val in self.get_vals(node_region):
                        if print_beliefs:
                            print("\tbelief %d: "%node_val, self.get_belief(batch.mp_graphs[ind], node_region,node_val))
                        new_belief = self.get_belief(batch.mp_graphs[ind], node_region, node_val)
                        if new_belief > max_val:
                            max_val = new_belief
                            max_assignment = node_val
                    final_assignment[node_region] = max_assignment

                if print_beliefs:
                    for pair_region in self.pair_regions:
                        print("REGION: ",pair_region)
                        for pair_val in self.get_vals(pair_region):
                            print("\tbelief", pair_val, ": ", self.get_belief(batch.mp_graphs[ind], pair_region, pair_val))
        return results

    def get_graph_parameters(self):
        params = []
        for graph in self.graphs:
            params.extend(list(graph.potential_model.parameters()))
        return params

    def get_all_parameters(self):
        params = list(self.global_model.parameters())
        #params = []
        for graph in self.graphs:
            params.extend(list(graph.potential_model.parameters()))
        return params

    def init_dataset(self, dataset, mp_graphs, use_loss_augmented_inf, load_loss_aug=False, loss_aug_save_path=None, masks_save_path=None):
        if self.global_beliefs:
            super(GlobalPairwiseModel, self).init_dataset(dataset, mp_graphs, use_loss_augmented_inf, top_width=2*self.top_width, load_loss_aug=load_loss_aug, loss_aug_save_path=loss_aug_save_path, masks_save_path=masks_save_path)
        else:
            super(GlobalPairwiseModel, self).init_dataset(dataset, mp_graphs, use_loss_augmented_inf, top_width=self.top_width, load_loss_aug=load_loss_aug, loss_aug_save_path=loss_aug_save_path, masks_save_path=masks_save_path)

    def get_inf_opts(self, batch, epoch):
        lambda_opt = torch.optim.SGD(batch.lambdas, 0.5)
        max_globals_opt = torch.optim.SGD(batch.max_globals, -1*self.max_globals_l_rate)
        return lambda_opt, max_globals_opt

    def get_model_optimizer(self, params):
        if params.get('use_adam', False):
            model_optimizer = torch.optim.Adam([
                {'params':self.global_model.parameters(), 'lr':params['global_lr']},
                {'params':self.get_graph_parameters(), 'lr':params['graph_lr']},
            ])
        else:
            model_optimizer = torch.optim.SGD([
                    {'params':self.global_model.parameters(), 'lr':params['global_lr']},
                    {'params':self.get_graph_parameters(), 'lr':params['graph_lr']},
                ], momentum = params.get('momentum', 0.), weight_decay=params.get('weight_decay',0))
        return model_optimizer

    def print_params(self):
        print("GLOBAL MODEL PARAMS:")
        for param in self.global_model.parameters():
            print(param)
        for ind,graph in enumerate(self.graphs):
            print("LOCAL MODEL %d PARAMS:"%ind)
            for param in graph.potential_model.parameters():
                print(param)


class GlobalPairwiseModel_Averaging(GlobalPairwiseModel):
    """This version of the NLStruct model runs message passing at every iteration of inference. As such, it
    is much slower. """

    def inference_step(self, batch, epoch, num_iters, mp_itrs, eps, is_train, return_obj=False):
        self.set_to_eval(True)
        if modelconf.USE_GPU:
            orig_lambdas = batch.lambdas
            orig_max_globals = batch.max_globals
            batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
            batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]
        for lambd in batch.lambdas:
            lambd.requires_grad = True
        for glob in batch.max_globals:
            glob.requires_grad = True

        if is_train:
            lambda_l_rate = self.train_lambda_l_rate
            max_globals_l_rate = self.train_max_globals_l_rate
            avg_thresh = self.train_avg_thresh
        else:
            lambda_l_rate = self.test_lambda_l_rate
            max_globals_l_rate = self.test_max_globals_l_rate
            avg_thresh = self.test_avg_thresh

          
        lambdas_opt = torch.optim.SGD(batch.lambdas, lr=lambda_l_rate)
        max_globals_opt = torch.optim.SGD(batch.max_globals, lr=max_globals_l_rate)
        count = 0
        y_updates = []
        diff_vals = []
        total_obj = float('inf')

        total_obj_diff = float('inf')
        score_diff = 1

        avg_lambdas = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.lambdas[0])).fill_(0.0)
        avg_lambdas.add_(torch.stack(batch.lambdas).data)
        current_avg_lambdas = avg_lambdas/1
        avg_max_globals = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.max_globals[0])).fill_(0.0)
        avg_max_globals.add_(torch.stack(batch.max_globals).data)
        current_avg_max_globals = avg_max_globals/1
        avg_max_pots = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.lambdas[0])).fill_(0.0)


        lambda_updates = []
        max_global_updates = []
        max_pots_updates = []
        past_avg_lambdas = avg_lambdas
        past_avg_max_globals = avg_max_globals
        avg_count = 1

        if self.wide_top:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs = False, update_graphs=True, detach_pots=True)
            pots = aggregate_pots[1]
        else:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=True, detach_pots=True)
            pots = None

        data_score_input = self.get_data_score_input(batch, pots)
        data_scores = self.call_global_model_on_pots(batch, data_score_input)
        if is_train:
            self.max_global_init = (data_score_input.sum(0)/len(data_score_input)).data
            if self.reinit:
                for glob_ind, glob in enumerate(batch.max_globals):
                    glob.data.copy_(data_score_input[glob_ind, :].data)
                for lambd_ind, lambd in enumerate(batch.lambdas):
                    lambd.data.copy_(modelconf.tensor_mod.FloatTensor(len(lambd)).fill_(0.0))

        if self.random_init is not None:
            for glob in batch.max_globals:
                glob.data.uniform_(-1*self.random_init, self.random_init)
            for lambd in batch.lambdas:
                lambd.data.uniform_(-1*self.random_init, self.random_init)

        map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
        new_total_obj = map_obj.item()

        tmp_lambdas = batch.lambdas
        batch.lambdas = [Variable(current_avg_lambdas[i,:]) for i in range(len(current_avg_lambdas))]
        if self.wide_top:
            avg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs = False, update_graphs=False, detach_pots=False)
            avg_agg_pots = avg_pots[0]
        else:
            avg_agg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
        batch.lambdas = tmp_lambdas

        total_obj_diff = abs(new_total_obj - total_obj)
        total_obj = new_total_obj
        unavg_global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
        global_scores = self.call_global_model_on_y(batch, Variable(current_avg_max_globals))
        score_diff = (global_scores - data_scores).sum().item()
        unavg_score_diff = (unavg_global_scores - data_scores).sum().item()
        all_score_diffs = (global_scores - data_scores).data.cpu().numpy()
        if modelconf.USE_GPU:
            beliefs = batch.beliefs.cuda(async=True)
        else:
            beliefs = batch.beliefs
        norm_diff = ((current_avg_lambdas*current_avg_max_globals).sum() - (beliefs*avg_agg_pots).sum())/len(batch)
        norm_diff = norm_diff.item()

        diff_vals.append((score_diff, norm_diff, all_score_diffs, unavg_score_diff, new_total_obj-data_scores.sum().item()))
        y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), new_total_obj))
        avg_lambdas_diff = (current_avg_lambdas-past_avg_lambdas).norm(2,1).sum()/len(batch)
        past_avg_lambdas = current_avg_lambdas
        avg_max_globals_diff = (current_avg_max_globals-past_avg_max_globals).norm(2,1).sum()/len(batch)
        past_avg_max_globals = current_avg_max_globals

        if self.use_pd:
            lambda_bar = torch.stack(batch.lambdas).detach()
            lambda_bar.requires_grad = False

        while count < num_iters:# or score_diff < 0:
            count += 1

            # Local potential inference
            if self.wide_top:
                aggregate_pots[0] = aggregate_pots[1]*torch.stack(batch.lambdas)
                if is_train and self.use_loss_augmented_inference:
                    aggregate_pots[0] += torch.stack(batch.loss_augmentations)
                self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots[0])
            else:
                aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots)
            fastmp.runmp(batch.mp_graphs, mp_itrs, eps)
            self.update_beliefs(batch.mp_graphs, eps)

            if modelconf.USE_GPU:
                beliefs = batch.beliefs.cuda(async=True)
            else:
                beliefs = batch.beliefs
            if self.use_pd:
                h = pots.data*beliefs.data
                prev_max_globals = torch.stack(batch.max_globals).detach()
                prev_max_globals.requires_grad = False
                prev_obj = float('inf')
                diff = 1
                i = 0
                while diff > 1e-4 and i < 10:
                    i += 1
                    max_globals_opt.zero_grad()
                    tmp_max_globals = torch.stack(batch.max_globals)
                    tmp = tmp_max_globals - prev_max_globals + max_globals_l_rate*lambda_bar
                    obj = -1*self.call_global_model_on_y(batch, tmp_max_globals).sum() + (tmp*tmp).sum()/(2*max_globals_l_rate)
                    obj.backward()
                    max_globals_opt.step()
                    diff = abs(prev_obj - obj.item())
                    prev_obj = obj.item()
                lambdas_new = torch.stack(batch.lambdas).data - lambda_l_rate*(h - torch.stack(batch.max_globals).data)
                lambda_bar = Variable(2*lambdas_new - torch.stack(batch.lambdas).data, requires_grad=False)
                for lambd_ind, lambd in enumerate(batch.lambdas):
                    lambd.data.copy_(lambdas_new[lambd_ind, :])
                
            else:
                # Y update
                for i in range(1):
                    max_globals_opt.zero_grad()
                    inf_obj = self.build_inf_obj(batch).sum()*-1
                    inf_obj.backward()
                    max_globals_opt.step()
                batch.lambdas[0].requires_grad = True

                # Lambda update
                batch.max_globals[-1].requires_grad = False
                if self.wide_top:
                    all_lambds = torch.stack(batch.lambdas).data
                    new_lambds = all_lambds - lambda_l_rate * (pots.data*beliefs.data - torch.stack(batch.max_globals).data)
                    for lambd_ind, lambd in enumerate(batch.lambdas):
                        lambd.data.copy_(new_lambds[lambd_ind, :])
                else:
                    lambdas_opt.zero_grad()
                    map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
                    map_obj.backward()
                    lambdas_opt.step()
                batch.max_globals[-1].requires_grad = True

            if count > avg_thresh:
                avg_count += 1
                avg_lambdas.add_(torch.stack(batch.lambdas).data)
                current_avg_lambdas = avg_lambdas/avg_count
                avg_max_globals.add_(torch.stack(batch.max_globals).data)
                current_avg_max_globals = avg_max_globals/avg_count
            else:
                avg_lambdas = torch.stack(batch.lambdas).data
                current_avg_lambdas = avg_lambdas/avg_count
                avg_max_globals = torch.stack(batch.max_globals).data
                current_avg_max_globals = avg_max_globals/avg_count


            if count%self.record_interval == 0:
                aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
                new_total_obj = map_obj.item()

                tmp_lambdas = batch.lambdas
                tmp_globs = batch.max_globals
                batch.lambdas = [Variable(current_avg_lambdas[i,:]) for i in range(len(current_avg_lambdas))]
                batch.max_globals = [Variable(current_avg_max_globals[i, :]) for i in range(len(current_avg_max_globals))]
                avg_agg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                avg_obj,_ = self.calculate_obj(batch, avg_agg_pots, eps, False, False)
                if self.wide_top:
                    avg_agg_pots = avg_agg_pots[0]
                batch.lambdas = tmp_lambdas
                batch.max_globals = tmp_globs
                total_obj_diff = abs(new_total_obj - total_obj)
                total_obj = new_total_obj
                unavg_global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
                global_scores = self.call_global_model_on_y(batch, Variable(current_avg_max_globals))
                score_diff = (global_scores - data_scores).sum().item()
                unavg_score_diff = (unavg_global_scores - data_scores).sum().item()
                all_score_diffs = (global_scores - data_scores).data.cpu().numpy()
                if modelconf.USE_GPU:
                    beliefs = batch.beliefs.cuda(async=True)
                else:
                    beliefs = batch.beliefs
                norm_diff = ((current_avg_lambdas*current_avg_max_globals).sum() - (beliefs*avg_agg_pots).sum())/len(batch)
                norm_diff = norm_diff.item()

                diff_vals.append((score_diff, norm_diff, all_score_diffs, unavg_score_diff, new_total_obj-data_scores.sum().item()))
                y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), avg_obj.item()))
                avg_lambdas_diff = (current_avg_lambdas-past_avg_lambdas).norm(2,1).sum()/len(batch)
                past_avg_lambdas = current_avg_lambdas
                avg_max_globals_diff = (current_avg_max_globals-past_avg_max_globals).norm(2,1).sum()/len(batch)
                past_avg_max_globals = current_avg_max_globals

                if count%100 == 0:
                    print("\t",count, new_total_obj, total_obj_diff, score_diff, global_scores.data[0,0].item())
                    print("\t\t", avg_lambdas_diff.item(), avg_max_globals_diff.item()) 
                    print("\t\tSCORE DIFF: ",score_diff)

        for i,lambd in enumerate(batch.lambdas):
            lambd.data.copy_(current_avg_lambdas[i,:])
        for i,max_global in enumerate(batch.max_globals):
            max_global.data.copy_(current_avg_max_globals[i,:])
        aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
        if self.wide_top:
            self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots[0])
        else:
            self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots)
        fastmp.runmp(batch.mp_graphs, mp_itrs, eps)
        self.update_beliefs(batch.mp_graphs, eps)
        if len(diff_vals) > 1:
            diff_vals = diff_vals[:-1]
            y_updates = y_updates[:-1]
        if modelconf.USE_GPU:
            for orig_l, lambd in zip(orig_lambdas, batch.lambdas):
                orig_l.data.copy_(lambd.data)
            for orig_g, glob in zip(orig_max_globals, batch.max_globals):
                orig_g.data.copy_(glob.data)
        if modelconf.USE_GPU:
            beliefs = batch.beliefs.cuda(async=True)
        else:
            beliefs = batch.beliefs
        final_obj = self.call_global_model_on_pots(batch, beliefs*aggregate_pots[1])
        print("FINAL INF OBJ: ",final_obj.sum())
        if return_obj:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train)
            map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, True)
            if modelconf.USE_GPU:
                batch.lambdas = orig_lambdas
                batch.max_globals = orig_max_globals
            return map_obj, diff_vals, y_updates
        if modelconf.USE_GPU:
            batch.lambdas = orig_lambdas
            batch.max_globals = orig_max_globals
        return None, diff_vals, y_updates


class GlobalPairwiseModel_AveragingInterleaved(GlobalPairwiseModel_Averaging):
    """This version of the NLStruct only runs message passing at the beginning/end of inference.
    Our experiments indicated that this still worked well in practice, and hence this is the version
    presented in the paper"""

    def __init__(self, graphs, num_nodes, num_vals, params):
        super(GlobalPairwiseModel_AveragingInterleaved, self).__init__(graphs, num_nodes, num_vals, params)
        self.train_mp_interval = params.get('train_mp_interval', -1)
        self.test_mp_interval = params.get('test_mp_interval', -1)
        self.reinit_interval = params.get('reinit_interval', -1)

    def inference_step(self, batch, epoch, num_iters, mp_itrs, eps, is_train, return_obj=False):
        self.set_to_eval(True)
        if modelconf.USE_GPU:
            orig_lambdas = batch.lambdas
            orig_max_globals = batch.max_globals
            batch.lambdas = [lambd.cuda(async=True).detach() for lambd in batch.lambdas]
            batch.max_globals = [glob.cuda(async=True).detach() for glob in batch.max_globals]
        for lambd in batch.lambdas:
            lambd.requires_grad = True
        for glob in batch.max_globals:
            glob.requires_grad = True

        if is_train:
            lambda_l_rate = self.train_lambda_l_rate
            max_globals_l_rate = self.train_max_globals_l_rate
            avg_thresh = self.train_avg_thresh
            mp_interval = self.train_mp_interval

        else:
            lambda_l_rate = self.test_lambda_l_rate
            max_globals_l_rate = self.test_max_globals_l_rate
            avg_thresh = self.test_avg_thresh
            mp_interval = self.test_mp_interval

            
        lambdas_opt = torch.optim.SGD(batch.lambdas, lr=lambda_l_rate)

        max_globals_opt = torch.optim.SGD(batch.max_globals, lr=max_globals_l_rate)
        count = 0
        y_updates = []
        diff_vals = []
        total_obj = float('inf')

        total_obj_diff = float('inf')
        score_diff = 1

        avg_lambdas = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.lambdas[0])).fill_(0.0)
        avg_lambdas.add_(torch.stack(batch.lambdas).data)
        current_avg_lambdas = avg_lambdas/1
        avg_max_globals = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.max_globals[0])).fill_(0.0)
        avg_max_globals.add_(torch.stack(batch.max_globals).data)
        current_avg_max_globals = avg_max_globals/1
        avg_max_pots = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.lambdas[0])).fill_(0.0)


        lambda_updates = []
        max_global_updates = []
        max_pots_updates = []
        past_avg_lambdas = avg_lambdas
        past_avg_max_globals = avg_max_globals
        count_thresh = 0
        avg_count = 1

        if self.wide_top:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs = False, update_graphs=True, detach_pots=True)
            pots = aggregate_pots[1]
        else:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=True, detach_pots=True)
            pots = None

        data_score_input = self.get_data_score_input(batch, pots)
        if self.global_beliefs:
            data_scores = self.call_global_model_on_pots(batch, data_score_input[:, :self.num_potentials], data_score_input[:, self.num_potentials:])
        else:
            data_scores = self.call_global_model_on_pots(batch, data_score_input)
        if is_train:
            self.max_global_init = (data_score_input.sum(0)/len(data_score_input)).data
            if self.reinit or (self.reinit_interval != -1 and epoch%self.reinit_interval == 0):
                print("REINITIALIZING INFERENCE VARS")
                for glob_ind, glob in enumerate(batch.max_globals):
                    glob.data.copy_(self.max_global_init)
                for lambd_ind, lambd in enumerate(batch.lambdas):
                    lambd.data.copy_(modelconf.tensor_mod.FloatTensor(len(lambd)).fill_(1.0))

        if self.random_init is not None:
            for glob in batch.max_globals:
                glob.data.uniform_(-1*self.random_init, self.random_init)
            for lambd in batch.lambdas:
                lambd.data.uniform_(-1*self.random_init, self.random_init)

        map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
        new_total_obj = map_obj.item()

        tmp_lambdas = batch.lambdas
        batch.lambdas = [Variable(current_avg_lambdas[i,:]) for i in range(len(current_avg_lambdas))]
        if self.wide_top:
            avg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs = False, update_graphs=False, detach_pots=False)
            avg_agg_pots = avg_pots[0]
        else:
            avg_agg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
        global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
        batch.lambdas = tmp_lambdas

        total_obj_diff = abs(new_total_obj - total_obj)
        total_obj = new_total_obj
        unavg_global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
        score_diff = (global_scores - data_scores).sum().item()
        unavg_score_diff = (unavg_global_scores - data_scores).sum().item()
        all_score_diffs = (global_scores - data_scores).data.cpu().numpy()
        if modelconf.USE_GPU:
            beliefs = batch.beliefs.cuda(async=True)
        else:
            beliefs = batch.beliefs
        norm_diff = ((current_avg_lambdas*current_avg_max_globals).sum() - (beliefs*avg_agg_pots).sum())/len(batch)
        norm_diff = norm_diff.item()

        diff_vals.append((score_diff, norm_diff, all_score_diffs, unavg_score_diff, new_total_obj-data_scores.sum().item()))
        if self.use_pd:
            y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), new_total_obj, batch.lambdas[0].data[0].item()))
        else:
            y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), new_total_obj))
        avg_lambdas_diff = (current_avg_lambdas-past_avg_lambdas).norm(2,1).sum()/len(batch)
        past_avg_lambdas = current_avg_lambdas
        avg_max_globals_diff = (current_avg_max_globals-past_avg_max_globals).norm(2,1).sum()/len(batch)
        past_avg_max_globals = current_avg_max_globals

        if self.use_pd:
            print("USING PD")
            lambda_bar = torch.stack(batch.lambdas).detach()
            lambda_bar.requires_grad = False
        
        while count < num_iters:
            if (mp_interval == -1 and count == 0) or (mp_interval != -1 and count % mp_interval == 0):
                if self.wide_top:
                    if self.global_beliefs:
                        lambds = torch.stack(batch.lambdas)
                        aggregate_pots[0] = aggregate_pots[1]*lambds[:, :self.num_potentials] + lambds[:, self.num_potentials:]
                    else:
                        aggregate_pots[0] = aggregate_pots[1]*torch.stack(batch.lambdas)
                    if is_train and self.use_loss_augmented_inference:
                        aggregate_pots[0] += torch.stack(batch.loss_augmentations)
                    self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots[0])
                else:
                    aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                    self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots)

                print("COUNT %d, DOING MP"%count)
                fastmp.runmp(batch.mp_graphs, mp_itrs, eps)
                self.update_beliefs(batch.mp_graphs, eps)
                if modelconf.USE_GPU:
                    beliefs = batch.beliefs.cuda(async=True)
                else:
                    beliefs = batch.beliefs
                if self.global_beliefs:
                    belief_pots = torch.cat([aggregate_pots[1]*beliefs, beliefs], dim=1) 
                else:
                    belief_pots = aggregate_pots[1]*beliefs


            count += 1

            
            if self.use_pd:
                h = belief_pots.data
                prev_max_globals = torch.stack(batch.max_globals).detach()
                prev_max_globals.requires_grad = False
                prev_obj = float('inf')
                diff = 1
                i = 0
                while diff > 1e-4 and i < 10:
                    i += 1
                    max_globals_opt.zero_grad()
                    tmp_max_globals = torch.stack(batch.max_globals)
                    tmp = tmp_max_globals - prev_max_globals + max_globals_l_rate*lambda_bar
                    obj = -1*self.call_global_model_on_y(batch, tmp_max_globals).sum() + (tmp*tmp).sum()/(2*max_globals_l_rate)
                    obj.backward()
                    max_globals_opt.step()
                    diff = abs(prev_obj - obj.item())
                    prev_obj = obj.item()
                lambdas_new = torch.stack(batch.lambdas).data - lambda_l_rate*(h - torch.stack(batch.max_globals).data)
                lambda_bar = Variable(lambdas_new + self.pd_theta*(lambdas_new - torch.stack(batch.lambdas).data), requires_grad=False)
                for lambd_ind, lambd in enumerate(batch.lambdas):
                    lambd.data.copy_(lambdas_new[lambd_ind, :])

            else:
                # Y update
                for i in range(1):
                    max_globals_opt.zero_grad()

                    #Multiplying by -1 is necessary to maximize the objective
                    inf_obj = self.build_inf_obj(batch).sum()*-1
                    inf_obj.backward()
                    max_globals_opt.step()
                batch.lambdas[0].requires_grad = True

                # Lambda update
                batch.max_globals[-1].requires_grad = False
                if self.wide_top:
                    all_lambds = torch.stack(batch.lambdas).data
                    new_lambds = all_lambds - lambda_l_rate * (belief_pots.data - torch.stack(batch.max_globals).data)
                    for lambd_ind, lambd in enumerate(batch.lambdas):
                        lambd.data.copy_(new_lambds[lambd_ind, :])
                else:
                    lambdas_opt.zero_grad()
                    map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
                    map_obj.backward()
                    lambdas_opt.step()
                batch.max_globals[-1].requires_grad = True

            if (mp_interval == -1 and count > avg_thresh) or (mp_interval != -1 and (count-1)%mp_interval > avg_thresh):
                avg_count += 1
                avg_lambdas.add_(torch.stack(batch.lambdas).data)
                current_avg_lambdas = avg_lambdas/avg_count
                avg_max_globals.add_(torch.stack(batch.max_globals).data)
                current_avg_max_globals = avg_max_globals/avg_count
            else:
                avg_lambdas = torch.stack(batch.lambdas).data
                current_avg_lambdas = avg_lambdas/avg_count
                avg_max_globals = torch.stack(batch.max_globals).data
                current_avg_max_globals = avg_max_globals/avg_count

            if (mp_interval == -1 and count == num_iters) or (mp_interval != -1 and count%mp_interval == 0): 
                print("COUNT %d, COPYING VALS"%count)
                for i,lambd in enumerate(batch.lambdas):
                    lambd.data.copy_(current_avg_lambdas[i,:])
                for i,max_global in enumerate(batch.max_globals):
                    max_global.data.copy_(current_avg_max_globals[i,:])

                avg_lambdas = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.lambdas[0])).fill_(0.0)
                avg_lambdas.add_(torch.stack(batch.lambdas).data)
                current_avg_lambdas = avg_lambdas/1
                avg_max_globals = modelconf.tensor_mod.FloatTensor(len(batch), len(batch.max_globals[0])).fill_(0.0)
                avg_max_globals.add_(torch.stack(batch.max_globals).data)
                current_avg_max_globals = avg_max_globals/1
                avg_count = 1

                if self.use_pd:
                    lambda_bar = torch.stack(batch.lambdas).detach()
                    lambda_bar.requires_grad = False


            if count%self.record_interval == 0:
                aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, False)
                new_total_obj = map_obj.item()

                tmp_lambdas = batch.lambdas
                tmp_globs = batch.max_globals
                batch.lambdas = [Variable(current_avg_lambdas[i,:]) for i in range(len(current_avg_lambdas))]
                batch.max_globals = [Variable(current_avg_max_globals[i, :]) for i in range(len(current_avg_max_globals))]
                avg_agg_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
                avg_obj,_ = self.calculate_obj(batch, avg_agg_pots, eps, False, False)
                if self.wide_top:
                    avg_agg_pots = avg_agg_pots[0]
                global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
                batch.lambdas = tmp_lambdas
                batch.max_globals = tmp_globs

                total_obj_diff = abs(new_total_obj - total_obj)
                total_obj = new_total_obj
                unavg_global_scores = self.call_global_model_on_y(batch, torch.stack(batch.max_globals))
                score_diff = (global_scores - data_scores).sum().item()
                unavg_score_diff = (unavg_global_scores - data_scores).sum().item()
                all_score_diffs = (global_scores - data_scores).data.cpu().numpy()
                if modelconf.USE_GPU:
                    beliefs = batch.beliefs.cuda(async=True)
                else:
                    beliefs = batch.beliefs
                norm_diff = ((current_avg_lambdas*current_avg_max_globals).sum() - (beliefs*avg_agg_pots).sum())/len(batch)
                norm_diff = norm_diff.item()

                diff_vals.append((score_diff, norm_diff, all_score_diffs, unavg_score_diff, new_total_obj-data_scores.sum().item()))
                if self.use_pd:
                    y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), avg_obj.item(), lambda_bar[0,0].item()))
                else:
                    y_updates.append((count, new_total_obj, norm_diff, batch.lambdas[0].data[0].item(), batch.max_globals[0].data[0].item(), current_avg_lambdas[0,0].item(), current_avg_max_globals[0,0].item(), avg_obj.item()))
                avg_lambdas_diff = (current_avg_lambdas-past_avg_lambdas).norm(2,1).sum()/len(batch)
                past_avg_lambdas = current_avg_lambdas
                avg_max_globals_diff = (current_avg_max_globals-past_avg_max_globals).norm(2,1).sum()/len(batch)
                past_avg_max_globals = current_avg_max_globals

                if count%100 == 0:
                    print("\t",count, new_total_obj, total_obj_diff, score_diff, global_scores.data[0,0].item())
                    print("\t\t", avg_lambdas_diff.item(), avg_max_globals_diff.item()) 
                    print("\t\tSCORE DIFF: ",score_diff)
        aggregate_pots = self.calculate_potentials(batch, eps, is_train, update_beliefs=False, update_graphs=False)
        if self.wide_top:
            self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots[0])
        else:
            self.update_mp_graph_potentials(batch.mp_graphs, aggregate_pots)
        fastmp.runmp(batch.mp_graphs, mp_itrs, eps)
        self.update_beliefs(batch.mp_graphs, eps)
        if len(diff_vals) > 1:
            diff_vals = diff_vals[:-1]
            y_updates = y_updates[:-1]
        if modelconf.USE_GPU:
            for orig_l, lambd in zip(orig_lambdas, batch.lambdas):
                orig_l.data.copy_(lambd.data)
            for orig_g, glob in zip(orig_max_globals, batch.max_globals):
                orig_g.data.copy_(glob.data)


        if return_obj:
            aggregate_pots = self.calculate_potentials(batch, eps, is_train)
            map_obj,_ = self.calculate_obj(batch, aggregate_pots, eps, False, True)
            if modelconf.USE_GPU:
                batch.lambdas = orig_lambdas
                batch.max_globals = orig_max_globals
            return map_obj, diff_vals, y_updates
        if modelconf.USE_GPU:
            batch.lambdas = orig_lambdas
            batch.max_globals = orig_max_globals
        return None, diff_vals, y_updates



#############################################################
###################GLOBAL MODEL STUFF########################
#############################################################

def build_linear_global_model(num_graphs, params):
    global_model = nn.Linear(num_graphs, 1, bias=False)
    global_model.weight.data.fill_(1.0)
    return global_model

class LinearWithSigmoid(torch.nn.Module):
    def __init__(self, num_graphs, params):
        super(LinearWithSigmoid, self).__init__()
        self.linear = nn.Linear(num_graphs, 1, bias=False)
        self.linear.weight.data.fill_(1.0)
        self.model = torch.nn.Sequential(self.linear, torch.nn.Sigmoid())

    def forward(self, inp):
        return self.model(inp)

def build_mlp_global_model(num_graphs, params):
    if 'global_activation' in params:
        if params['global_activation'] == 'sigmoid':
            activation = lambda: nn.Sigmoid()
        elif params['global_activation'] == 'relu':
            activation = lambda: nn.ReLU()
        else:
            raise Exception("Activation type not valid: ",params['global_activation'])
    else:
        activation = lambda: nn.Sigmoid()
    global_model = nn.Sequential(
                nn.Linear(num_graphs, params['global_hidden_size']),
                activation(),
                nn.Linear(params['global_hidden_size'], 1)
            )
    return global_model

    
class QuadFunc(torch.autograd.Function):
    def __init__(self):
        super(QuadFunc, self).__init__()

    def forward(self, variables, quad_weights, linear_weights):
        self.save_for_backward(variables, quad_weights, linear_weights)
        result = modelconf.tensor_mod.FloatTensor(len(variables), 1)
        for i in range(len(variables)):
            result[i,0] = (variables[i,:]**2).view(-1).dot(quad_weights.view(-1))+ variables[i,:].view(-1).dot(linear_weights)
        return result

    def backward(self, grad_output):
        variables, quad_weights, linear_weights = self.saved_tensors
        grad_vars = modelconf.tensor_mod.FloatTensor(variables.size()).fill_(0.0)
        for i in range(len(variables)):
            grad_vars[i,:] = (2*variables[i,:]*quad_weights+linear_weights)*grad_output[i]
        grad_quad_weights = ((variables**2)*grad_output).sum(0)
        grad_linear_weights = (variables*grad_output).sum(0)
        return grad_vars, grad_quad_weights, grad_linear_weights

class QuadModel(nn.Module):
    def __init__(self, num_vars, params):
        super(QuadModel, self).__init__()
        self.quad_weights = nn.Parameter(torch.FloatTensor(num_vars).fill_(-1))
        self.linear_weights = nn.Parameter(torch.FloatTensor(num_vars).fill_(1))

    def forward(self, variables):
        return QuadFunc()(variables, self.quad_weights, self.linear_weights)


##############################################################################
################## LOSSES FOR LOSS-AUGMENTED INFERENCE #######################
def squared_diff(true_val, other_val):
    return (true_val-other_val)**2

def hamming_diff(true_val, other_val):
    return abs(true_val - other_val)/10.0

def identity_diff(true_val, other_val):
    if true_val == other_val:
        return 0.0
    else:
        return 0.01


##############################################################################
################### FUNCTIONS USED IN ABOVE MODELS ###########################
##############################################################################


class AggregatePotentials(torch.autograd.Function):
    """A function that takes one or more graphs and calculates one potential vector aggregating all of them

    Pieces of this code suport the possibility of using a variety of individual graphs and weighing these 
    individually (that is, instead of containing one lambda variable per assignment per region, there's only
    one lambda per entire graph); however, this is an idea we ended up not pursuing. This implementation
    is an artifact of this"""
    def __init__(self, num_data, num_vals, num_potentials, potential_ind_dict, regions, graphs, use_lambda):
        super(AggregatePotentials, self).__init__()
        self.num_data = num_data
        self.num_vals = num_vals
        self.num_pair_vals = self.num_vals*self.num_vals
        self.num_potentials = num_potentials
        self.potential_ind_dict = potential_ind_dict
        self.regions = regions
        self.graphs = graphs
        self.use_lambda = use_lambda

    def get_potential_offset(self, region):
        if type(region) == tuple:
            start = self.potential_ind_dict[region][(0,0)]
        else:
            start = self.potential_ind_dict[region][0]
        return start

    def get_num_vals(self, region):
        if type(region) == tuple:
            return self.num_pair_vals
        else:
            return self.num_vals

    def forward(self, *inputs):
        self.save_for_backward(*inputs)
        if self.use_lambda:
            lambdas = torch.stack(inputs[:self.num_data])
            potentials = inputs[self.num_data:]

            result = modelconf.tensor_mod.FloatTensor(self.num_data, self.num_potentials).fill_(0.0)
            for graph_ind, graph in enumerate(self.graphs):
                tmp_lambdas = lambdas[:, graph_ind].unsqueeze(1).expand(len(lambdas), self.num_pair_vals)
                for region in graph.original_regions:
                    agg_pot_offset = self.get_potential_offset(region)
                    graph_offset = graph.get_potential_offset(region)
                    if type(region) == tuple:
                        result[:, agg_pot_offset:(agg_pot_offset+self.num_pair_vals)] += tmp_lambdas * potentials[graph_ind][:, graph_offset:(graph_offset+self.num_pair_vals)]
                    else:
                        result[:, agg_pot_offset:(agg_pot_offset+self.num_vals)] += tmp_lambdas[:, :self.num_vals]*potentials[graph_ind][:, graph_offset:(graph_offset+self.num_vals)]
        else:
            potentials = inputs
            result = modelconf.tensor_mod.FloatTensor(self.num_data, self.num_potentials).fill_(0.0)
            for graph_ind, graph in enumerate(self.graphs):
                for region in graph.original_regions:
                    agg_pot_offset = self.get_potential_offset(region)
                    graph_offset = graph.get_potential_offset(region)
                    if type(region) == tuple:
                        result[:, agg_pot_offset:(agg_pot_offset+self.num_pair_vals)] += potentials[graph_ind][:, graph_offset:(graph_offset+self.num_pair_vals)]
                    else:
                        result[:, agg_pot_offset:(agg_pot_offset+self.num_vals)] += potentials[graph_ind][:, graph_offset:(graph_offset+self.num_vals)]

        return result

    def backward(self, grad_output):
        inputs = self.saved_tensors
        if self.use_lambda:
            lambdas = torch.stack(inputs[:self.num_data])
            potentials = inputs[self.num_data:]

            grad_lambdas = modelconf.tensor_mod.FloatTensor(lambdas.size()).fill_(0.0)
            grad_pots = []
            for potential in potentials:
                grad_pots.append(modelconf.tensor_mod.FloatTensor(potential.size()))
            for graph_ind, graph in enumerate(self.graphs):
                tmp_lambdas = lambdas[:, graph_ind].unsqueeze(1).expand(len(lambdas), self.num_pair_vals)
                for region in graph.original_regions:
                    graph_offset = graph.get_potential_offset(region)
                    agg_offset = self.get_potential_offset(region)
                    if type(region) == tuple:
                        grad_lambdas[:, graph_ind] += (potentials[graph_ind][:, graph_offset:(graph_offset+self.num_pair_vals)]*grad_output[:, agg_offset:(agg_offset+self.num_pair_vals)]).sum(1)
                        grad_pots[graph_ind][:, graph_offset:(graph_offset+self.num_pair_vals)] = tmp_lambdas*grad_output[:, agg_offset:(agg_offset+self.num_pair_vals)]
                    else:
                        grad_lambdas[:, graph_ind] += (potentials[graph_ind][:, graph_offset:(graph_offset+self.num_vals)]*grad_output[:, agg_offset:(agg_offset+self.num_vals)]).sum(1)
                        grad_pots[graph_ind][:, graph_offset:(graph_offset+self.num_vals)] = tmp_lambdas[:, :self.num_vals]*grad_output[:, agg_offset:(agg_offset+self.num_vals)]
            return tuple([grad_lambdas[i,:].squeeze() for i in range(self.num_data)] +
                      grad_pots)
        else:
            potentials = inputs
            grad_pots = []
            for potential in potentials:
                grad_pots.append(modelconf.tensor_mod.FloatTensor(potential.size()))
            for graph_ind, graph in enumerate(self.graphs):
                for region in graph.original_regions:
                    graph_offset = graph.get_potential_offset(region)
                    agg_offset = self.get_potential_offset(region)
                    if type(region) == tuple:
                        grad_pots[graph_ind][:, graph_offset:(graph_offset+self.num_pair_vals)] = grad_output[:,agg_offset:(agg_offset+self.num_pair_vals)]
                    else:
                        grad_pots[graph_ind][:, graph_offset:(graph_offset+self.num_vals)] = grad_output[:, agg_offset:(agg_offset+self.num_vals)]
            end = time.time()
            return tuple(grad_pots)


