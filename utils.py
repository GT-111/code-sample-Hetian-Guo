import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def build_mlps(in_channel, mlp_channels=None,activation ='relu', ret_before_act=False, without_norm=False):
    layers = nn.Sequential()
    assert isinstance(mlp_channels, list)
    num_layers = len(mlp_channels)
    
    if activation == 'relu':
        activation_layer = nn.ReLU()
    elif activation == 'sigmoid':
        activation_layer = nn.Sigmoid()
    elif activation == 'leakyrelu':
        activation_layer = nn.LeakyReLU()

    for layer in range(num_layers):
        if layer + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[layer], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[layer], bias=True), activation_layer]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[layer], bias=False), nn.BatchNorm1d(mlp_channels[layer]), activation_layer])
            c_in = mlp_channels[layer]

    return layers


def softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0)
    return soft_max_1d.transpose(axis, 0)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(logits.size()).float()
    gumbel_noise = - torch.log(eps - torch.log(U + eps))
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    y_soft =  softmax(y / tau, axis=-1)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y
