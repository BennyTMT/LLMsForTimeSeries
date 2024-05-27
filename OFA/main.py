from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.NLinear import NLinear
from models.DLinear_plus import DLinearPlus


    
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random
    
warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
parser.add_argument('--all', type=int, default=0)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--train_ratio', type=float, default=1.0 , required=False)
parser.add_argument('--save_file_name', type=str, default=None)
parser.add_argument('--gpu_loc', type=int, default=1)
parser.add_argument('--n_scale', type=float, default=-1)
parser.add_argument('--method', type=str, default='')


args = parser.parse_args()

if args.save_file_name is not None : 
    log_fine_name = args.save_file_name

device_address = 'cuda:'+str(args.gpu_loc)


# log_fine_name='NLinear_336_96.txt'

def select_optimizer(model ,args):
    param_dict = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": args.learning_rate}
    ]
    model_optim = optim.Adam([param_dict[1]], lr=args.learning_rate)
    loss_optim = optim.Adam([param_dict[0]], lr=args.learning_rate)
    return model_optim, loss_optim

        
SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}
mses = []
maes = []
print(args.model_id)
for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    # path = os.path.join(args.checkpoints, setting)
    path = './checkpoints/' +  args.model_id + '_'+ str(ii)
    if not os.path.exists(path):
        os.makedirs(path)

    if os.path.exists(path + '/' + 'checkpoint.pth'):
        print(args.model_id , ' has done!!')
        continue
        
    if args.freq == 0:
        args.freq = 'h'
        
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    device = torch.device(device_address)

    time_now = time.time()
    train_steps = len(train_loader)

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'NLinear':
        model = NLinear(args)
        model.to(device)
    elif args.model == 'DLinear_plus':
        model = DLinearPlus(args ,  device )
        print(device)
        model.to(device)
    else:
        model = GPT4TS(args, device , log_fine_name = log_fine_name)
        
    params = model.parameters()
    
    if 'ofa' in args.model_id : 
        model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    else : 
        model_optim, loss_optim = select_optimizer(model ,args)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    if args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()

    if 'ofa' in args.model_id : 
        criterion = nn.MSELoss()
    else : 
        criterion = nn.L1Loss()
    '''
        Note : 
        Original Repo use l2 loss, but I can not reach paper`s results with it 
        when I change it to l1 loss, it does. And LLaTA uses l1 as well. 
    '''
    criterion = nn.L1Loss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    is_first = True 
    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            
            if is_first : 
                print(args.data_path, batch_x.shape  , batch_y.shape)
                is_first=False
                
            iter_count += 1
            model_optim.zero_grad()
            if 'ofa' not in args.model_id :  loss_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            # print(batch_x.shape, batch_y.shape , )
            outputs = model(batch_x, ii)
            
            # print(batch_x.shape , batch_y.shape  , outputs)
            if 'ofa' in args.model_id : 
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
            # else:
            #     outputs = outputs[:, :, -args.pred_len:]
            #     batch_y = batch_y[:, :, -args.pred_len:].to(device)
                
            assert outputs.shape == batch_y.shape
            
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()
            if 'ofa' not in args.model_id :  loss_optim.step()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        
        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        # with open(log_fine_name , 'a') as f : 
        #     f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}\n".format(
        #     epoch + 1, train_steps, train_loss, vali_loss))
            
        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
                    
    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    mses.append(round(mse,5))
    maes.append(round(mae,5))

if len(maes)==0 : exit()
maes = np.array(maes)
mses = np.array(mses)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))
    
with open(log_fine_name , 'a') as f : 
    f.write("{}\n".format(args.model_id))
    # f.write("mae{}\n".format(str(maes)))
    # f.write("mse{}\n".format(str(mses)))
    f.write("mae:{:.4f}, std:{:.4f} ---- mse:{:.4f}, std:{:.4f}\n".format(np.mean(maes), np.std(maes) , np.mean(mses), np.std(mses)))
        
print(log_fine_name)
# os.system('rm -r '+ path )
            
