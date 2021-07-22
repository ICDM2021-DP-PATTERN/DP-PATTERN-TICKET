from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import sys
import pickle
import collections
from numpy import linalg as LA
import yaml
import datetime
import operator
import random

# from tensorboardX import SummaryWriter
import numpy as np
import scipy.misc

class ADMM:
    def __init__(self, model, file_name, rho=0.001):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.rho = rho
        self.rhos = {}

        self.init(file_name, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        if not isinstance(config, str):
            raise Exception("filename must be a str")
        with open(config, "r") as stream:
            try:
                raw_dict = yaml.load(stream)
                self.prune_ratios = raw_dict['prune_ratios']
                for k, v in self.prune_ratios.items():
                    self.rhos[k] = self.rho
                for (name, W) in model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                    # if(len(W.size()) == 4):
                    #     if name not in self.prune_ratios:
                    #         continue
                    #     self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    #     self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z


            except yaml.YAMLError as exc:
                print(exc)


def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def weight_pruning(args, name, weight, prune_ratio):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    percent = prune_ratio * 100
    if (args.sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "channel"):
        shape = weight.shape
        print("channel pruning...", weight.shape)
        weight3d = weight.reshape(shape[0], shape[1], -1)
        channel_l2_norm = LA.norm(weight3d, 2, axis=(0,2))
        percentile = np.percentile(channel_l2_norm, percent)
        under_threshold = channel_l2_norm <= percentile
        above_threshold = channel_l2_norm > percentile
        weight3d[:,under_threshold,:] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(weight3d.shape, dtype=np.float32)
        for i in range(weight3d.shape[1]):
            expand_above_threshold[:, i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]
        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "pattern"):
        print("pattern pruning...", weight.shape)
        shape = weight.shape

        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]  # 1
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53

        pattern9 = [[1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]  # 125
        pattern10 = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]]  # 6
        pattern11 = [[1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]  # 126
        pattern12 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 0]]  # 10

        if args.patternNum == 4:
            patterns_dict = {1: pattern1,
                             2: pattern2,
                             3: pattern3,
                             4: pattern4
                             }
        elif args.patternNum == 8:
            patterns_dict = {1: pattern1,
                             2: pattern2,
                             3: pattern3,
                             4: pattern4,
                             5: pattern5,
                             6: pattern6,
                             7: pattern7,
                             8: pattern8
                             }
        elif args.patternNum == 12:
            patterns_dict = {1: pattern1,
                             2: pattern2,
                             3: pattern3,
                             4: pattern4,
                             5: pattern5,
                             6: pattern6,
                             7: pattern7,
                             8: pattern8,
                             9: pattern9,
                             10: pattern10,
                             11: pattern11,
                             12: pattern12
                             }

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_kernel = weight[i, j, :, :].copy()
                temp_dict = {}  # store each pattern's norm value on the same weight kernel
                for key, pattern in patterns_dict.items():
                    temp_kernel = current_kernel.copy()
                    for index in pattern:
                        temp_kernel[index[0], index[1]] = 0
                    current_norm = LA.norm(temp_kernel)
                    temp_dict[key] = current_norm
                best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                # print(best_pattern)
                for index in patterns_dict[best_pattern]:
                    weight[i, j, index[0], index[1]] = 0
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        # zeros = weight == 0
        # zeros = zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "filter_balance"): # the connectivity pruning
        print("pruning filter with balanced outputs")

        kth_smallest = prune_ratio  # the percent from script is used to represent k-th smallest l2-norm kernel will be pruned in each filter
        shape = weight.shape
        weight3d = weight.reshape(shape[0], shape[1], -1)
        for i in range(shape[0]):
            kernel_l2norm_list = LA.norm(weight3d[i, :, :], 2, axis=1)
            partial_sorted_index = np.argpartition(kernel_l2norm_list,
                                                   kth_smallest)  # list of all indices, but partially sorted
            kth_smallest_index = partial_sorted_index[:kth_smallest]  # indices of k-th smallest l2-norm
            for idx in kth_smallest_index:
                weight3d[i, idx, :] = 0
            non_zeros = weight != 0
            non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    elif (args.sparsity_type == "two_filter_balance_1"):
        pattern1 = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]  # 3
        pattern2 = [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]]  # 12
        pattern3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]  # 65
        pattern4 = [[0, 2], [1, 2], [2, 0], [2, 1], [2, 2]]  # 120

        pattern5 = [[0, 0], [0, 2], [2, 0], [2, 1], [2, 2]]
        pattern6 = [[0, 0], [0, 1], [0, 2], [2, 0], [2, 2]]  # 14
        pattern7 = [[0, 0], [0, 2], [1, 0], [2, 0], [2, 2]]  # 44
        pattern8 = [[0, 0], [0, 2], [1, 2], [2, 0], [2, 2]]  # 53


        patterns_dict = {1: pattern1,
                         2: pattern2,
                         3: pattern3,
                         4: pattern4,
                         5: pattern5,
                         6: pattern6,
                         7: pattern7,
                         8: pattern8
                         }

        print("pruning two filter with balanced outputs -- step 1: group aligned pattern", name)

        shape = weight.shape
        numFilter = shape[0]

        weight2d = weight.reshape(shape[0], -1)
        filter_L2 = LA.norm(weight2d, 2, axis=1)
        weight3d = weight.reshape(shape[0], shape[1], -1)

        filter_index_dict = {}
        for index, l2_item in enumerate(filter_L2):
            filter_index_dict[index] = l2_item
        filter_index_dict = sorted(filter_index_dict.items(), key=lambda k: [k[1], k[0]])
        filter_index_dict = collections.OrderedDict(filter_index_dict)
        sorted_filter_index = list(filter_index_dict.keys())

        if os.path.exists("./{}.pkl".format(name)):
            os.remove("./{}.pkl".format(name))
        afile = open(r"./{}.pkl".format(name), 'wb')
        pickle.dump(sorted_filter_index, afile)
        afile.close()

        for i, (filter_idx, _) in enumerate(filter_index_dict.items()):
            if i % 4 == 0:
                first_idx = filter_idx
                second_idx = list(filter_index_dict.keys())[i + 1]
                third_idx = list(filter_index_dict.keys())[i + 2]
                forth_idx = list(filter_index_dict.keys())[i + 3]
                temp = np.array([weight3d[first_idx, :, :], weight3d[second_idx, :, :], weight3d[third_idx, :, :], weight3d[forth_idx, :, :]])

                """add aligned pattern prune for this current pair before aligned connectivity prune"""
                temp = temp.reshape([temp.shape[0], temp.shape[1], 3, 3])
                for k in range(temp.shape[1]):  # loop channel
                    current_channel = temp[:, k, :, :].copy()
                    temp_dict = {}  # store each pattern's norm value on the same weight kernel
                    for key, pattern in patterns_dict.items():
                        temp_channel = current_channel.copy()
                        for j in range(temp_channel.shape[0]):  # loop every kernel in a channel
                            for index in pattern:
                                temp_channel[j, :][index[0], index[1]] = 0
                        current_norm = LA.norm(temp_channel)
                        temp_dict[key] = current_norm
                    best_pattern = max(temp_dict.items(), key=operator.itemgetter(1))[0]
                    for index in patterns_dict[best_pattern]:
                        temp[:, k, index[0], index[1]] = 0
                temp = temp.reshape([temp.shape[0], temp.shape[1], -1])

                """aligned connectivity prune"""
                if percent == 0:
                    weight3d[first_idx] = temp[0]
                    weight3d[second_idx] = temp[1]
                    weight3d[third_idx] = temp[2]
                    weight3d[forth_idx] = temp[3]
                    continue
                channel_l2_norm = LA.norm(temp, 2, axis=(0, 2))
                if i <= numFilter / 4:
                    percentile = np.percentile(channel_l2_norm, percent / 1)
                elif numFilter / 4 < i <= numFilter / 2:
                    percentile = np.percentile(channel_l2_norm, percent / 1)
                elif numFilter / 2 < i <= numFilter:
                    percentile = np.percentile(channel_l2_norm, percent)
                under_threshold = channel_l2_norm <= percentile
                above_threshold = channel_l2_norm > percentile
                temp[:, under_threshold, :] = 0

                weight3d[first_idx] = temp[0]
                weight3d[second_idx] = temp[1]
                weight3d[third_idx] = temp[2]
                weight3d[forth_idx] = temp[3]

        weight = weight3d.reshape(shape)

        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        return torch.from_numpy(non_zeros).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise SyntaxError("Unknown sparsity type")

def hard_prune(args, ADMM, model, option=None):
    """
    hard_pruning, or direct masking
    Args:
         model: contains weight tensors in cuda

    """

    print("hard pruning")
    for (name, W) in model.named_parameters():
        if name not in ADMM.prune_ratios:  # ignore layers that do not have rho
            continue
        cuda_pruned_weights = None
        if option == None:
            _, cuda_pruned_weights = weight_pruning(args, name, W, ADMM.prune_ratios[name])  # get sparse model in cuda

        elif option == "random":
            _, cuda_pruned_weights = random_pruning(args, W, ADMM.prune_ratios[name])

        elif option == "l1":
            _, cuda_pruned_weights = L1_pruning(args, W, ADMM.prune_ratios[name])
        else:
            raise Exception("not implmented yet")
        W.data = cuda_pruned_weights  # replace the data field in variable



def admm_initialization(args, ADMM, model):
    if not args.admm:
        return
    for i, (name, W) in enumerate(model.named_parameters()):
        if name in ADMM.prune_ratios:
            _, updated_Z = weight_pruning(args, name, W, ADMM.prune_ratios[name])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            ADMM.ADMM_Z[name] = updated_Z


def z_u_update(args, ADMM, model, device, train_loader, optimizer, epoch, data, batch_idx, writer):
    if not args.admm:
        return

    if epoch != 1 and (epoch - 1) % args.admm_epoch == 0 and batch_idx == 0:
        for i, (name, W) in enumerate(model.named_parameters()):
            if name not in ADMM.prune_ratios:
                continue
            Z_prev = None
            if (args.verbose):
                Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
            ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            _, updated_Z = weight_pruning(args, name, ADMM.ADMM_Z[name],
                                          ADMM.prune_ratios[name])  # equivalent to Euclidean Projection
            ADMM.ADMM_Z[name] = updated_Z
            if (args.verbose):
                if writer:
                    writer.add_scalar('layer:{} W(k+1)-Z(k+1)'.format(name),
                                      torch.sqrt(torch.sum((W - ADMM.ADMM_Z[name]) ** 2)).item(), epoch)
                    writer.add_scalar('layer:{} Z(k+1)-Z(k)'.format(name),
                                      torch.sqrt(torch.sum((ADMM.ADMM_Z[name] - Z_prev) ** 2)).item(), epoch)
                # print ("at layer {}. W(k+1)-Z(k+1): {}".format(name,torch.sqrt(torch.sum((W-ADMM.ADMM_Z[name])**2)).item()))
                # print ("at layer {}, Z(k+1)-Z(k): {}".format(name,torch.sqrt(torch.sum((ADMM.ADMM_Z[name]-Z_prev)**2)).item()))
            ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)


def append_admm_loss(args, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm_loss = {}

    if args.admm:

        for i, (name, W) in enumerate(model.named_parameters()):  ## initialize Z (for both weights and bias)
            if name not in ADMM.prune_ratios:
                continue

            admm_loss[name] = 0.5 * ADMM.rhos[name] * (torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
    mixed_loss = 0
    mixed_loss += ce_loss
    for k, v in admm_loss.items():
        mixed_loss += v
    return ce_loss, admm_loss, mixed_loss


def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    we increase it by 1.9x every admm epoch
    After 10 admm updates, the rho will be 0.91

    """

    ADMM.rhos[name] *= 2


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default 
    admm epoch is 9)

    """
    admm_epoch = args.admm_epoch
    lr = None
    if epoch % admm_epoch == 0:
        lr = args.lr
    else:
        admm_epoch_offset = epoch % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.lr * (0.1 ** (admm_epoch_offset // admm_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def zero_masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)
        if name in config.prune_ratios:
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)
            masks[name] = torch.from_numpy(indices).cuda()
    config.zero_masks = masks


def masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():
        if name in config.prune_ratios:
            above_threshold, pruned_weight = weight_pruning(args, W, config.prune_ratios[name])
            W.data = pruned_weight
            masks[name] = above_threshold

    config.masks = masks



