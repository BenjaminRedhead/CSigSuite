import os
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Tuple

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, label_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]

        return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y)

    def __len__(self):
        # Ensure length is non-negative
        l = len(self.data) - self.seq_len - self.pred_len + 1
        return max(0, l)

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class ETTDataModule(BaseDataModule):
    def __init__(self, root_path, data_path, freq='h', size=None, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.root_path = root_path
        self.data_path = data_path
        self.freq = freq
        self.scale = scale
        self.seq_len, self.label_len, self.pred_len = size if size else (96, 48, 96)
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        if self.freq == 'h':
            # FIX: Added max(0, ...) to prevent negative indices
            border1s = [0, max(0, 12*30*24 - self.seq_len), max(0, 12*30*24 + 4*30*24 - self.seq_len)]
            border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
        else:
            border1s = [0, max(0, 12*30*24*4 - self.seq_len), max(0, 12*30*24*4 + 4*30*24*4 - self.seq_len)]
            border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]

        df_data = df_raw.drop(columns=['date'])
        
        # Ensure contiguous memory
        raw_values = np.ascontiguousarray(df_data.values)

        if self.scale:
            train_data = raw_values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(raw_values)
        else:
            data = raw_values

        data = np.ascontiguousarray(data)

        self.train_ds = TimeSeriesDataset(data[border1s[0]:border2s[0]], self.seq_len, self.label_len, self.pred_len)
        self.val_ds = TimeSeriesDataset(data[border1s[1]:border2s[1]], self.seq_len, self.label_len, self.pred_len)
        self.test_ds = TimeSeriesDataset(data[border1s[2]:border2s[2]], self.seq_len, self.label_len, self.pred_len)

class EWELDDataModule(BaseDataModule):
    def __init__(self, root_path, data_path, freq='m', size=None, scale=True, 
                 split_ratio=(0.7, 0.1, 0.2), target='Temperature(F)_1', **kwargs):
        super().__init__(**kwargs)
        self.root_path = root_path
        self.data_path = data_path
        self.freq = freq.lower() 
        self.scale = scale
        self.split_ratio = split_ratio
        self.target = target
        self.seq_len, self.label_len, self.pred_len = size if size else (96, 48, 96)
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 0. Handle Frequency Sampling
        if self.freq == 'h':
            print("EWELD: Downsampling from 15-min to Hourly (taking every 4th sample).")
            df_raw = df_raw.iloc[::4, :].copy().reset_index(drop=True)
        
        # 1. Clean data
        if 'date' in df_raw.columns:
            df_raw.drop(columns=['date'], inplace=True)

        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
            df_raw = df_raw[cols + [self.target]]
        
        df_raw = df_raw.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)

        # 2. Split calculation
        n = len(df_raw)
        num_train = int(n * self.split_ratio[0])
        num_test = int(n * self.split_ratio[2])
        num_vali = n - num_train - num_test
        
        # FIX: Added max(0, ...) to prevent negative indices
        border1s = [0, max(0, num_train - self.seq_len), max(0, n - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_vali, n]

        # Ensure contiguous memory
        raw_values = np.ascontiguousarray(df_raw.values)

        # 3. Scaling
        if self.scale:
            self.scaler.fit(raw_values[border1s[0]:border2s[0]])
            data = self.scaler.transform(raw_values)
        else:
            data = raw_values
        
        data = np.ascontiguousarray(data)

        self.train_ds = TimeSeriesDataset(data[border1s[0]:border2s[0]], self.seq_len, self.label_len, self.pred_len)
        self.val_ds = TimeSeriesDataset(data[border1s[1]:border2s[1]], self.seq_len, self.label_len, self.pred_len)
        self.test_ds = TimeSeriesDataset(data[border1s[2]:border2s[2]], self.seq_len, self.label_len, self.pred_len)

class CustomDataModule(BaseDataModule):
    def __init__(self, root_path, data_path, size=None, scale=True, 
                 split_ratio=(0.7, 0.1, 0.2), target='OT', **kwargs):
        super().__init__(**kwargs)
        self.root_path = root_path
        self.data_path = data_path
        self.scale = scale
        self.split_ratio = split_ratio
        self.target = target
        self.seq_len, self.label_len, self.pred_len = size if size else (96, 48, 96)
        self.scaler = StandardScaler()

    def setup(self, stage=None):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        for c in ['date', 'Date']:
            if c in cols: cols.remove(c)
        
        if self.target in cols:
            cols.remove(self.target)
            df_raw = df_raw[cols + [self.target]]
        else:
            df_raw = df_raw[cols]

        df_raw = df_raw.apply(pd.to_numeric, errors='coerce').ffill().fillna(0.0)

        num_train = int(len(df_raw) * self.split_ratio[0])
        num_test = int(len(df_raw) * self.split_ratio[2])
        num_vali = len(df_raw) - num_train - num_test
        
        # FIX: Added max(0, ...) to prevent negative indices which cause empty arrays
        border1s = [0, max(0, num_train - self.seq_len), max(0, len(df_raw) - num_test - self.seq_len)]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        # Ensure contiguous memory
        raw_values = np.ascontiguousarray(df_raw.values)

        if self.scale:
            self.scaler.fit(raw_values[border1s[0]:border2s[0]])
            data = self.scaler.transform(raw_values)
        else:
            data = raw_values

        data = np.ascontiguousarray(data)

        self.train_ds = TimeSeriesDataset(data[border1s[0]:border2s[0]], self.seq_len, self.label_len, self.pred_len)
        self.val_ds = TimeSeriesDataset(data[border1s[1]:border2s[1]], self.seq_len, self.label_len, self.pred_len)
        self.test_ds = TimeSeriesDataset(data[border1s[2]:border2s[2]], self.seq_len, self.label_len, self.pred_len)

def data_factory(config):
    data_map = {
        'ETT': ETTDataModule,
        'EWELD': EWELDDataModule,
        'custom': CustomDataModule,
    }
    
    data_class = data_map.get(config['data'], CustomDataModule)
    
    params = {
        'root_path': config['root_path'],
        'data_path': config['data_path'],
        'size': [config['seq_len'], config['label_len'], config['pred_len']],
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'scale': config.get('scale', True),
    }
    
    if config['data'] == 'ETT':
        params['freq'] = config.get('freq', 'h')
    elif config['data'] == 'EWELD':
        params['freq'] = config.get('freq', 'm')
        params['target'] = config.get('target', 'Temperature(F)_1')
        params['split_ratio'] = config.get('split_ratio', (0.7, 0.1, 0.2))
    else:
        params['target'] = config.get('target', 'OT')
        params['split_ratio'] = config.get('split_ratio', (0.7, 0.1, 0.2))

    return data_class(**params)