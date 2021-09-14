import copy

import torch
from torch import nn
from torch.nn.utils import prune

from models.dynamic_pruning.gatedconv import GatedConv


def soft_prune_step(network, prune_rate):
    for i in range(len(network.features)):
        if isinstance(network.features[i], GatedConv):

            kernel = network.features[i].conv.weight.data
            sum_of_kernel = torch.sum(torch.abs(kernel.reshape(kernel.size(0), -1)), dim=1)
            _, args = torch.sort(sum_of_kernel)
            soft_prune_list = args[:int(round(kernel.size(0) * prune_rate))].tolist()
            for j in soft_prune_list:
                network.features[i].conv.weight.data[j] = torch.zeros_like(network.features[i].conv.weight.data[j])

    return network


def static_pruning(net, pruning_ratio=0.3):
    '''
    :param net: DNN
    :param preserve_ratio: preserve rate
    :return: newnet (nn.Module): a newnet contain mask that help prune network's weight
    '''
    #
    if not isinstance(net, nn.Module):
        print('Invalid input. Must be nn.Module')
        return
    newnet = copy.deepcopy(net)
    for name, module in newnet.named_modules():
        if isinstance(module, nn.Conv2d):

            prune.ln_structured(module, name='weight', amount=float(pruning_ratio), n=2, dim=0)


    return newnet
