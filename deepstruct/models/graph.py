from . import modelconf
import torch
from torch.autograd import Variable
import torch.nn as nn
import collections, time

#Auxiliary function - we need a different iterator order than the default
def product(*args, **kwds):
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [[y] + x for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


class Graph(object):
    def __init__(self, node_regions, pair_regions, num_vals, potential_model, model_args, tie_pairwise):
        self.original_node_regions = node_regions
        self.original_pair_regions = pair_regions
        self.original_regions = node_regions + pair_regions
        self.tie_pairwise = tie_pairwise
        # We need to be able to translate between original "names" and the internal representation
        self.node_regions = []
        self.pair_regions = []
        self.region2internal = {}
        self.internal2region = {}
        for idx, original_node_region in enumerate(self.original_node_regions):
            self.node_regions.append(idx)
            self.region2internal[original_node_region] = idx
            self.internal2region[idx] = original_node_region
        for original_node_1, original_node_2 in self.original_pair_regions:
            if original_node_1 in self.region2internal:
                node1 = self.region2internal[original_node_1]
            else:
                node1 = len(self.region2internal)
                self.region2internal[original_node_1] = node1
                self.internal2region[node1] = original_node_1
            if original_node_2 in self.region2internal:
                node2 = self.region2internal[original_node_2]
            else:
                node2 = len(self.region2internal)
                self.region2internal[original_node_2] = node2
                self.internal2region[node2] = original_node_2
            self.pair_regions.append((node1, node2))
        self.regions = self.node_regions + self.pair_regions

        self.num_vals = num_vals
        self.neighbors = collections.defaultdict(list)
        for pair_region in self.pair_regions:
            for node in pair_region:
                self.neighbors[node].append(pair_region)
        self.region_ind_dict = {}
        self.potential_ind_dict = {}
        ind = 0
        for ind,region in enumerate(self.node_regions + self.pair_regions):
            self.region_ind_dict[region] = ind
        self.num_regions = ind+1
        ind = 0

        #node potentials are a function of the observation - can be different at every node
        for node_region in self.node_regions:
            self.potential_ind_dict[node_region] = {}
            for val in self.get_vals(node_region):
                self.potential_ind_dict[node_region][val] = ind
                ind += 1

        if len(self.pair_regions) > 0:
            if tie_pairwise:
                #pair potentials are only a function of value - same for every pair
                pair_inds = {}
                for val in self.get_vals((0,0)):
                    pair_inds[val] = ind
                    ind += 1
                for pair_region in self.pair_regions:
                    self.potential_ind_dict[pair_region] = pair_inds
            else:
                for pair_region in self.pair_regions:
                    self.potential_ind_dict[pair_region] = {}
                    for val in self.get_vals(pair_region):
                        self.potential_ind_dict[pair_region][val] = ind
                        ind += 1
        self.num_potentials = ind

        self.potential_model = potential_model(self.node_regions, self.pair_regions, self.original_node_regions, self.original_pair_regions, num_vals, self.region_ind_dict, self.potential_ind_dict, self.num_potentials, model_args)
        if modelconf.USE_GPU:
            self.potential_model.cuda()

    def __contains__(self, region):
        return region in self.region2interal

    def get_features(self, datum):
        feature = [None]*self.num_regions
        for region in self.region_ind_dict:
            region_ind = self.region_ind_dict[region]
            if type(region) == tuple:
                feature[region_ind] = (datum[region[0]], datum[region[1]])
            else:
                feature[region_ind] = datum[region]
        return feature
        
    def set_observations(self, observations):
        self.observations = observations
        self.potential_model.set_observations(self.observations)

    def get_data_mask(self, datum):
        data_mask = torch.zeros(self.num_potentials)
        for region_ind, region in enumerate(self.regions):
            if type(region) == tuple:
                r1, r2 = region
                assignment = (datum[self.internal2region[r1]].item(), datum[self.internal2region[r2]].item())
            else:
                assignment = datum[self.internal2region[region]].item()
            potential_ind = self.potential_ind_dict[region][assignment]
            data_mask[potential_ind] += 1
        return data_mask


    def set_data(self, data):
        self.data = data
        
    def calculate_potentials(self, detach=False):
        self.potentials = self.potential_model()
        if detach:
            self.potentials.detach_()
        return self.potentials

    def get_potential(self, region, assignment, obs_feature_ind):
        ind = self.potential_ind_dict[self.convert2internal(region)][assignment]
        return self.potentials[obs_feature_ind, ind]

    def convert2internal(self, region):
        if type(region) == tuple:
            return (self.region2internal[region[0]], self.region2internal[region[1]])
        else:
            return self.region2internal[region]

    def get_data_scores(self):
        return (self.potentials*self.data_mask).sum(1)

    def get_vals(self, region):
        if type(region) == tuple:
            return product(range(self.num_vals), repeat=2)
        else:
            return range(self.num_vals)

    def get_potential_offset(self, region):
        region = self.convert2internal(region)
        if type(region) == tuple:
            return self.potential_ind_dict[region][(0,0)]
        else:
            return self.potential_ind_dict[region][0]

    def get_region_potentials(self, region):
        region = self.convert2internal(region)
        if type(region) == tuple:
            start = self.potential_ind_dict[region][(0,0)]
            return self.potentials[:, start:(start+self.num_vals*self.num_vals)]
        else:
            start = self.potential_ind_dict[region][0]
            return self.potentials[:, start:(start+self.num_vals)]

class BasePotentialModel(nn.Module):   
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(BasePotentialModel, self).__init__()
        self.node_regions = node_regions
        self.pair_regions = pair_regions
        self.original_node_regions = original_node_regions
        self.original_pair_regions = original_pair_regions
        self.num_vals = num_vals
        self.num_pair_vals = self.num_vals*self.num_vals
        self.potential_ind_dict = potential_ind_dict
        self.region_ind_dict = region_ind_dict
        self.num_potentials = num_potentials
        self.args_dict = args_dict

    def get_vals(self, region):
        if type(region) == tuple:
            return product(range(self.num_vals), repeat=2)
        else:
            return range(self.num_vals)

    def forward(self):
        raise NotImplementedError

    def set_observations(self, observations):
        self.observations = observations


class LinearFunc(torch.autograd.Function):
    def __init__(self, node_regions, pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, potential_func_cache):
        super(LinearFunc, self).__init__()
        self.node_regions = node_regions
        self.pair_regions = pair_regions
        self.num_vals = num_vals
        self.potential_ind_dict = potential_ind_dict
        self.region_ind_dict = region_ind_dict
        self.num_potentials = num_potentials
        self.potential_func_cache = potential_func_cache

    def get_vals(self, region):
        if type(region) == tuple:
            return product(range(self.num_vals), repeat=2)
        else:
            return range(self.num_vals)

    def forward(self, weights):
        #print "\t\t\tLINEAR FUNCTION FORWARD"
        start = time.time()
        result = modelconf.tensor_mod.FloatTensor(self.potential_func_cache.size())
        #result = torch.DoubleTensor(len(self.obs_features), self.num_potentials)
        tmp_ind = len(self.node_regions)*self.num_vals
        if len(self.node_regions) > 0:
            result[:, :tmp_ind] = self.potential_func_cache[:, :tmp_ind] * weights[0]
        if len(self.pair_regions) > 0:
            result[:, tmp_ind:] = self.potential_func_cache[:, tmp_ind:] * weights[1]
        end = time.time()
        #print "\t\t\t\tLINEAR FUNCTION FORWARD TIME: ",(end-start)
        return result

    def backward(self, grad_output):
        #print "\t\t\tLINEAR FUNCTION BACKWARD"
        start = time.time()
        result = modelconf.tensor_mod.FloatTensor(2).fill_(0.0)
        #result = torch.zeros(2).double()
        tmp_ind = len(self.node_regions)*self.num_vals
        if len(self.node_regions) > 0:
            result[0] = (grad_output[:, :tmp_ind]*self.potential_func_cache[:, :tmp_ind]).sum()
        if len(self.pair_regions) > 0:
            result[1] = (grad_output[:, tmp_ind:]*self.potential_func_cache[:, tmp_ind:]).sum()
        end = time.time()
        #print "\t\t\t\tLINEAR FUNCtion BACKWARD TIME: ",(end-start)
        #print "LINEAR FUNCTION GRADS: "
        #print "\tGRAD_WEIGHT: ",result.numpy()
        return result

class LinearModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(LinearModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        #self.weights = nn.Parameter(2*torch.rand(2)-1)
        self.weights = nn.Parameter(modelconf.tensor_mod.FloatTensor(2).fill_(1))
        if "potential_func" in args_dict:
            self.potential_func = args_dict['potential_func']
        else:
            def potential_func(region, assignment, observation, region_ind_dict):
                if type(region) == tuple:
                    return 2*float(assignment[0] == assignment[1])-1
                    #return float(assignment[0] == assignment[1])
                    #return float(-abs(assignment[0] - assignment[1]))
                else:
                    return float(-abs(observation[region] - assignment))
            self.potential_func = potential_func

    def set_observations(self, observations):
        super(LinearModel, self).set_observations(observations)

        #Cache potential function calculation to save time
        self.potential_func_cache = modelconf.tensor_mod.FloatTensor(len(observations), self.num_potentials)
        for obs_ind, observation in enumerate(self.observations):
            for node_region in self.node_regions:
                for val in self.get_vals(node_region):
                    potential_ind = self.potential_ind_dict[node_region][val]
                    self.potential_func_cache[obs_ind, potential_ind] = self.potential_func(node_region, val, observation, self.region_ind_dict)
        if len(self.pair_regions) > 0:
            tmp = modelconf.tensor_mod.FloatTensor(self.num_vals*self.num_vals)
            for ind,val in enumerate(self.get_vals(self.pair_regions[0])):
                tmp[ind] = self.potential_func(self.pair_regions[0], val, observation, self.region_ind_dict)
            self.potential_func_cache[:, self.num_potentials-self.num_pair_vals:] = tmp.repeat(len(observations), 1)

    def forward(self):
        #print "STARTING FORWARD"
        start = time.time()
        result = LinearFunc(self.node_regions, self.pair_regions, self.num_vals, self.region_ind_dict, self.potential_ind_dict, self.num_potentials, self.potential_func_cache)(self.weights)
        end = time.time()
        #print "FORWARD TIME: ",(end-start)
        return result

class IndicatorModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(IndicatorModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.ind_region = args_dict['ind_region']
        self.weight = nn.Parameter(modelconf.tensor_mod.FloatTensor([1.0]))

    def forward(self):
        result = Variable(torch.from_numpy(-1*np.ones((len(obs_features), self.num_potentials))).float())*self.weight[0]
        if self.ind_region is not None:
            if type(self.ind_region) == tuple:
                val = (1,1)
            else:
                val = 1
            potential_ind = self.potential_ind_dict[self.ind_region][val]
            for obs_ind, obs_feature in enumerate(self.obs_features):
                #potential_ind = self.potential_ind_dict[self.ind_region][obs_feature[self.region_ind_dict[self.ind_region]]]
                result[obs_ind, potential_ind] = 1.0*self.weight[0]
        return result
        
class MLPModel(nn.Module):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(MLPModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.hidden_size = args_dict['hidden_size']

        if len(node_regions) > 0:
            self.node_model = nn.Sequential(
                nn.Linear(num_vals, self.hidden_size),
                nn.Sigmoid(),
                nn.Linear(self.hidden_size, num_vals)
            )
        if len(pair_regions) > 0:
            self.pair_model = nn.Sequential(
                nn.Linear(num_vals*2, self.hidden_size),
                nn.Sigmoid(),
                nn.Linear(self.hidden_size, 1)
            )

                

    def assignment2input(self, region, obs_features, assignment):
        if type(region) == tuple:
            result = Variable(torch.from_numpy(np.zeros((len(obs_features), self.num_vals*2))).float())
            for ind, obs_feature in enumerate(obs_features):
                result[ind, assignment[0]] = 1
                result[ind, self.num_vals + assignment[1]] = 1
        else:
            result = Variable(torch.from_numpy(np.zeros((len(obs_features), self.num_vals))).float())
            for ind, obs_feature in enumerate(obs_features):
                result[ind, obs_feature[region]] = 1
        return result

    def forward(self):
        start = time.time()
        #result = Variable(torch.FloatTensor(len(obs_features), self.num_potentials))
        result = Variable(modelconf.tensor_mod.zeros(len(self.obs_features), self.num_potentials))
        for node_region in self.node_regions:
            potential_ind = self.potential_ind_dict[node_region][0]
            input_val = self.assignment2input(node_region, obs_features, None)
            result[:, potential_ind:(potential_ind+self.num_vals)] = self.node_model(input_val)
        if len(self.pair_regions) > 0:
            for assignment in product(range(self.num_vals), repeat=2):
                potential_ind = self.potential_ind_dict[self.pair_regions[0]][assignment]
                input_val = self.assignment2input(self.pair_regions[0], obs_features, assignment)
                result[:, potential_ind] = self.pair_model(input_val)
        end = time.time()
        #print "FORWARD TIME: ",(end-start)
        return result

class CNNMLPModel(BasePotentialModel):
    def __init__(self, node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict):
        super(CNNMLPModel, self).__init__(node_regions, pair_regions, original_node_regions, original_pair_regions, num_vals, region_ind_dict, potential_ind_dict, num_potentials, args_dict)
        self.filter_size = args_dict['filter_size']
        self.img_width = args_dict['img_width']
        self.img_height = args_dict['img_height']
        self.hidden_size = args_dict['hidden_size']
        if len(self.node_regions) > 0:
            self.unary_model = nn.Conv2d(num_vals, num_vals, self.filter_size, padding=(self.filter_size-1)/2)
        if len(pair_regions) > 0:
            self.pair_model = nn.Sequential(
                nn.Linear(num_vals*2, self.hidden_size),
                nn.Sigmoid(),
                nn.Linear(self.hidden_size, 1)
            )

            # Build input for pair model
            self.pair_input = Variable(modelconf.tensor_mod.FloatTensor(num_vals*num_vals, 2*num_vals).fill_(0.0))
            for ind,val in enumerate(self.get_vals(self.pair_regions[0])):
                self.pair_input[ind,val[0]] = 1
                self.pair_input[ind,val[1]] = 1

    def forward(self):
        if len(self.node_regions) > 0:
            tmp = modelconf.tensor_mod.FloatTensor(self.observations).view(len(self.observations), self.img_height, self.img_width)
            result = Variable(modelconf.tensor_mod.FloatTensor(len(self.observations), self.num_vals, self.img_height, self.img_width))
            for i in range(self.num_vals):
                result[:, i, :, :] = tmp.eq(i)
            node_potentials = self.unary_model(result)
            node_potentials = node_potentials.permute(0,2,3,1).contiguous().view(len(self.observations),-1)
            if len(self.pair_regions) == 0:
                return node_potentials
        if len(self.pair_regions) > 0:
            pair_potentials = self.pair_model(self.pair_input).view(1,-1).repeat(len(self.observations), 1)
            #pair_potentials = Variable(tensor_mod.FloatTensor(len(self.observations), self.num_vals*self.num_vals).fill_(0.0))
            if len(self.node_regions) == 0:
                return pair_potentials
        return torch.cat([node_potentials, pair_potentials], dim=1)

