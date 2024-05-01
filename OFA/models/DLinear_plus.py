import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from  models.Attention import MultiHeadAttention
    
class Encoder_LLaTA(nn.Module):
    def __init__(self, input_dim , hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_LLaTA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        return x 

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinearPlus(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs, device):
        super(DLinearPlus, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_size = configs.patch_size 
        self.stride = configs.patch_size //2 
        
        self.d_model = configs.d_model
        self.method = configs.method

        # multi : 7 or 1
        if self.method == 'single_linr':
            kernel_size = 25
            self.decompsition = series_decomp(kernel_size)
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
        elif self.method == 'single_linr_decp':
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Residual = nn.Linear(self.seq_len,self.pred_len)
        elif self.method == 'multi_linr_trsf':
            self.in_layer = Encoder_LLaTA(self.seq_len , hidden_dim=self.d_model)
            self.out_layer = nn.Linear(self.d_model, self.pred_len)
            
        elif self.method == 'multi_decp_trsf':
            self.in_layers =  nn.ModuleList()
            self.out_layers =  nn.ModuleList()
            for _ in range(3):
                self.in_layers.append(Encoder_LLaTA(self.seq_len , hidden_dim=self.d_model))
                self.out_layers.append(nn.Linear(self.d_model, self.pred_len))
                
        elif self.method == 'multi_patch_attn':
            self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
            self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
            self.in_layer = nn.Linear(self.patch_size, self.d_model)
            self.basic_attn = MultiHeadAttention(d_model =self.d_model )
            self.out_layer = nn.Linear(self.d_model * self.patch_num, configs.pred_len)
                        
        elif self.method ==  'multi_patch_decp':
            # (256,7 ,3 ,96)
            self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 2
            self.padding_patch_layer = nn.ReplicationPad1d((0,  self.stride)) 
            print('d_model' , self.d_model , ' patch_num' , self.patch_num)
            self.in_layers =  nn.ModuleList()
            self.basic_attns =  nn.ModuleList()
            self.out_layers =  nn.ModuleList()
            for _ in range(3):
                self.in_layers.append(nn.Linear(self.patch_size, self.d_model))
                self.basic_attns.append(MultiHeadAttention(d_model =self.d_model ))
                self.out_layers.append(nn.Linear(self.d_model * self.patch_num, configs.pred_len))

    def norm(self, x, dim =0, means= None , stdev=None):
        if means is not None :  
            return x * stdev + means
        else : 
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=dim, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x /= stdev
            return x , means ,  stdev 

    def forward(self, x, itr):
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
        # print(self.method)
        # [ Batch, Channel, Dec ,  Inp_len ]
        # [256, 1 ,336 ]  x  
        #  It is actually DLinear 
        if self.method == 'single_linr':
            # print(self.method , x.shape)
            # # x: [Batch, Input length, Channel]
            x = x.permute(0,2,1)
            seasonal_init, trend_init = self.decompsition(x)
            seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            x = seasonal_output + trend_output
            return x
            
        elif self.method == 'single_linr_decp':
            # [256, 3, 336]
            trend_init , seasonal_init ,  residual_init = x[:,0:1,:] , x[:,1:2,:] , x[:,2:3,:] 
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            residual_output = self.Linear_Residual(residual_init)
            return trend_output + seasonal_output + residual_output
            
        elif self.method == 'multi_linr_trsf':
            # [256, 7, 336]
            x , means, stdev  = self.norm(x , dim=2)
            outputs = self.in_layer(x)
            x = self.out_layer(outputs)
            x  = self.norm(x , means=means, stdev=stdev )
            return x

        elif self.method == 'multi_decp_trsf':
            # [256, 7 ,3 , 336]
            inputs = x 
            outs =[]
            # print(inputs.shape)  torch.Size([512, 7, 3, 336])
            for i in range(3):
                x = inputs[:,:,i,:]
                x , means, stdev  = self.norm(x , dim=2)
                # print(x.shape)  torch.Size([512, 7, 336])  
                outputs = self.in_layers[i](x)
                # print(outputs.shape) torch.Size([512, 7, 768]) 
                x = self.out_layers[i](outputs)
                # print(x.shape) torch.Size([512, 7, 96]) 
                x  = self.norm(x , means=means, stdev=stdev )
                outs.append(x)
            return outs[0] + outs[1] + outs[2]

        elif self.method == 'multi_patch_attn':
            B , C = x.size(0) , x.size(1)
            # [256, 7, 336]
            x , means, stdev  = self.norm(x , dim=2)
            # [256, 7, 344]
            x = self.padding_patch_layer(x)
            # print('x3' ,x.shape) [256, 7, 12, 16]
            x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
            # [256, 7, 12, 768]
            x = self.in_layer(x)
            
            x =  rearrange(x, 'b c m l -> (b c) m l')
            # print(x.shape)
            x , _ = self.basic_attn( x ,x ,x )
            
            x =  rearrange(x, '(b c) m l -> b c (m l)' , b=B , c=C)
            # print(x.shape)
            x = self.out_layer(x)
            x  = self.norm(x , means=means, stdev=stdev )
            return x  
            
        elif self.method == 'multi_patch_decp':
            # [256, 7, 3 , 336]
            B  , C = x.size(0) , x.size(1) 
            assert x.size(1) == 7 and x.size(2) == 3 
            outs = []
            for i in range(3):
                # [256, 7, 336 ]
                xi = x[:,:,i,:]
                
                xi , means, stdev  = self.norm(xi , dim=2)
                xi = self.padding_patch_layer(xi)
                xi = xi.unfold(dimension=-1, size=self.patch_size, step=self.stride)

                xi = self.in_layers[i](xi)
                xi =  rearrange(xi, 'b c m l -> (b c) m l')
                # print(xi.shape)
                xi , _ = self.basic_attns[i]( xi ,xi ,xi )
                xi =  rearrange(xi, '(b c) m l -> b c (m l)' , b=B , c=C)
                # print(xi.shape)
                # [256, 7, pre_len ]
                xi = self.out_layers[i](xi)
                xi  = self.norm(xi , means=means, stdev=stdev )
                outs.append(xi)

            return outs[0] + outs[1] + outs[2]