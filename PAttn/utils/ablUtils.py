import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

def perturb_sequence(batch_x , shuffle_type , patch_size = 16 , mask_ratio= 0.2  ):
    '''
        batch_x : shape [256, 336, 1]
        perturb time series input 
        sf_all : shuffle the whole sequnece 
        sf_half : shuffle first halp sequnece 
        ex-half : exchange first and second half 
    '''
    assert shuffle_type in ['sf_all' , 'sf_half' , 'ex_half' ,'sf_patchs' , 'masking']
    if shuffle_type == 'sf_all':
        perm = torch.randperm(batch_x.size(1))
        return batch_x[:, perm, :]
    if shuffle_type == 'sf_half':
        mid_point = batch_x.size(1) // 2
        pre_half = batch_x[:, :mid_point, :]
        post_half = batch_x[:, mid_point:, :]
        perm = torch.randperm(pre_half.size(1))
        shuffled_pre_half = pre_half[:, perm, :]
        return torch.cat((shuffled_pre_half, post_half), dim=1)
    if shuffle_type == 'ex_half':
        mid_point = batch_x.size(1) // 2
        pre_half = batch_x[:, :mid_point, :]
        post_half = batch_x[:, mid_point:, :]
        return torch.cat((post_half, pre_half), dim=1)
    if shuffle_type =='sf_patchs':
        num_patches= (batch_x.size(1)  // patch_size )
        shuffle_indices = torch.randperm(num_patches)
        shuffled_ts = [batch_x[:, i*patch_size:(i+1)*patch_size, :] for i in shuffle_indices]
        if  num_patches * patch_size < batch_x.size(1):
            shuffled_ts.append(batch_x[:, num_patches*patch_size:, :])
        return torch.cat(shuffled_ts , dim=1)
    if shuffle_type =='masking':
        input_length  = batch_x.size(1)
        num_to_mask = int(input_length * mask_ratio )
        mask_indices = torch.randperm(input_length)[:num_to_mask]
        masked_tensor = batch_x.clone()
        masked_tensor[:, mask_indices, :] = 0
        return masked_tensor
        
# arrayTS = torch.rand(1, 32, 1)
# print(arrayTS[0,:,0])
# arrayTS = perturb_sequence(arrayTS , 'masking' , patch_size = 4 , mask_ratio= 0.8  )
# print(arrayTS[0,:,0])
