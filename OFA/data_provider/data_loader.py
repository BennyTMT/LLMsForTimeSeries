import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.tools import convert_tsf_to_dataframe
import warnings
from pathlib import Path
from statsmodels.tsa.seasonal import STL
from typing import Tuple
import matplotlib.pyplot as plt
import random 

warnings.filterwarnings('ignore')

def decompose(
    x: torch.Tensor, period: int = 7
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose input time series into trend, seasonality and residual components using STL.

    Args:
        x (torch.Tensor): Input time series. Shape: (1, seq_len).
        period (int, optional): Period of seasonality. Defaults to 7.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Decomposed components. Shape: (1, seq_len).
    """
    x = x.squeeze(0).cpu().numpy()
    decomposed = STL(x, period=period).fit()
    trend = decomposed.trend.astype(np.float32)
    seasonal = decomposed.seasonal.astype(np.float32)
    residual = decomposed.resid.astype(np.float32)
    return (
        torch.from_numpy(trend).unsqueeze(0),
        torch.from_numpy(seasonal).unsqueeze(0),
        torch.from_numpy(residual).unsqueeze(0),
    )

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 percent=100, max_len=-1, train_all=False, train_ratio = 1.0 ,model_id ='' ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_size_ratio = 1.0 
        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_ratio = train_ratio 
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()

        self.model_id = model_id
        
        self.period = 24 
        self.channel= 7
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
            
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def draw_decompose(self, x , trend, seasonal, residual):
        plt.figure(figsize=(10, 6))  # Optional: Specifies the figure size
        # Plot each array
        x = x.reshape(-1,)
        trend = trend.reshape(-1,)
        seasonal = seasonal.reshape(-1,)
        residual = residual.reshape(-1,)
        print(x.shape , trend.shape)
        
        plt.plot(x, label='x A')
        plt.plot(trend, label='trend B')
        plt.plot(seasonal, label='seasonal C')
        plt.plot(residual, label='residual D')
        ii = random.randint(0,100)
        # Adding labels
        plt.xlabel('Index')  # Assuming the index represents the x-axis
        plt.ylabel('Value')  # The y-axis label
        plt.title('Plot of Four Arrays')  # Title of the plot
        plt.legend()
        plt.savefig(f'/p/selfdrivingpj/projects_time/NeurIPS2023-One-Fits-All/Long-term_Forecasting/figures/{ii}.jpg')
        plt.cla()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [int(12 * 30 * 24 * self.train_ratio), 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # (17420, 7) 
        # print(data.shape)
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        print(self.set_type , len(self.data_x))
                
    def __getitem__(self, index):
        
        '''
            single_linr  single_linr_decp multi_linr_att  multi_patch multi_patch_attn multi_patch_decp
        '''
        
        if 'multi' in self.model_id and 'decp' in self.model_id:
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            components= []
            for i in range(self.channel):
                (trend, seasonal, residual) = decompose(x[i:i+1] , period=self.period)
                component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
                components.append(component)
            components = torch.stack(components)
            # [batch , 7 , 3 , seq_len ]
            return components , y  , seq_x_mark, seq_y_mark
        elif 'multi' in self.model_id and ('decp'  not in self.model_id): 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            return x , y ,  seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]

            (trend, seasonal, residual) = decompose(x , period=self.period)
            # self.draw_decompose(x.numpy() , trend.numpy(), seasonal.numpy(), residual.numpy())
            component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
            return component , y  , seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' not in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]
            
            return x, y, seq_x_mark, seq_y_mark
        elif 'ofa' in  self.model_id :
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 percent=100, max_len=-1, train_all=False  , model_id = ''):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        self.model_id = model_id 
        self.period = 60 
        self.channel= 7
        if 'multi' in self.model_id:
            self.enc_in =1 
        else : 
            self.enc_in = self.data_x.shape[-1]
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if 'multi' in self.model_id and 'decp' in self.model_id:
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            components= []
            for i in range(self.channel):
                (trend, seasonal, residual) = decompose(x[i:i+1] , period=self.period)
                component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
                components.append(component)
            components = torch.stack(components)
            # [batch , 7 , 3 , seq_len ]
            return components , y  , seq_x_mark, seq_y_mark
        elif 'multi' in self.model_id and ('decp'  not in self.model_id): 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            return x , y ,  seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]

            (trend, seasonal, residual) = decompose(x , period=self.period)
            # self.draw_decompose(x.numpy() , trend.numpy(), seasonal.numpy(), residual.numpy())
            component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
            return component , y  , seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' not in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]
            
            return x, y, seq_x_mark, seq_y_mark
            
        elif 'ofa' in  self.model_id :
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            return seq_x, seq_y, seq_x_mark, seq_y_mark
        
    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=10, max_len=-1, train_all=False , train_ratio=1.0 , model_id=''):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.model_id= model_id
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        if 'weather' in data_path:
            # per 10min
            self.period = 24
            self.channel= 21
        if 'traffic' in data_path:
            # per hour 
            self.period = 24
            self.channel= 862
        if 'electricity' in data_path:
            # per hour 
            self.period = 24
            self.channel= 321
        if 'illness' in data_path:
            # 1week
            self.period = 12
            self.channel= 7
                     
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        print(len(self.data_x))
        
    def __getitem__(self, index):
        '''
            single_linr         multi_decp_trsf     single_linr_decp 
            multi_linr_trsf     multi_patch_attn    multi_patch_decp

        '''
        if 'multi' in self.model_id and 'decp' in self.model_id:
            exit()
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            components= []
            for i in range(x.shape[0]):
                (trend, seasonal, residual) = decompose(x[i:i+1] , period=self.period)
                component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
                components.append(component)
            components = torch.stack(components)
            # [batch , 7 , 3 , seq_len ]
            return components , y  , seq_x_mark, seq_y_mark
        elif 'multi' in self.model_id and ('decp'  not in self.model_id): 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [7, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [7, pred_len]
            return x , y ,  seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]

            (trend, seasonal, residual) = decompose(x , period=self.period)
            # self.draw_decompose(x.numpy() , trend.numpy(), seasonal.numpy(), residual.numpy())
            component = torch.concat([trend, seasonal, residual], dim=0)  # [3, seq_len]
            return component , y  , seq_x_mark, seq_y_mark
        elif 'single' in self.model_id and  'decp' not in self.model_id: 
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            x = torch.tensor(seq_x, dtype=torch.float).transpose(1, 0)  # [1, seq_len]
            y = torch.tensor(seq_y, dtype=torch.float).transpose(1, 0)  # [1, pred_len]
            
            return x, y, seq_x_mark, seq_y_mark

        elif 'ofa' in  self.model_id :
            feat_id = index // self.tot_len
            s_begin = index % self.tot_len
            
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            
            return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # print(len(self.data_x) ,   self.seq_len , self.pred_len  )
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None,
                 percent=None, train_all=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='Daily',
                 percent=10, max_len=-1, train_all=False):
        
        self.train_all = train_all
        
        self.seq_len = size[0]
        self.pred_len = size[2]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(os.path.join(self.root_path,
                                                                                                                              self.data_path))
        self.freq = frequency
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            if self.train_all:
                border1s = [0,          0,          train_len-self.seq_len]
                border2s = [train_len,  train_len,  _len]
            else:
                border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
                border2s = [train_len - self.pred_len,  train_len,                                _len]
            border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len
            # print("_len = {}".format(_len))
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        if self.train_all:
            border1s = [0,          0,          train_len-self.seq_len]
            border2s = [train_len,  train_len,  _len]
        else:
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
        border2s[0] = (border2s[0] - self.seq_len) * self.percent // 100 + self.seq_len

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.set_type == 2:
            s_end = -self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        # if self.set_type == 2:
        #     print("data_x.shape = {}, data_y.shape = {}".format(data_x.shape, data_y.shape))

        return data_x, data_y, data_x, data_y

    def __len__(self):
        if self.set_type == 0:
            # return self.tot_len
            return min(self.max_len, self.tot_len)
        else:
            return self.tot_len
