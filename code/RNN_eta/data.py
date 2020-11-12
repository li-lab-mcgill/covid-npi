#/usr/bin/python

import os, pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class COVID_Eta_Dataset(Dataset):
    def __init__(self, eta, cnpis, cnpi_mask):
        super().__init__()

        self.eta = eta

        self.cnpis = cnpis
        self.cnpi_mask = cnpi_mask

    def __len__(self):
        return self.eta.shape[0]

    def __getitem__(self, country_idx):
        batch_eta = self.eta[country_idx]
        labels = self.cnpis[country_idx]
        mask = self.cnpi_mask[country_idx, :, :]

        return batch_eta, labels, mask

class COVID_Eta_Data_Module(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.path = os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']))

    def pickle_load(self, filename):
        with open(filename, "rb") as file:
            data = pickle.load(file)
        return [np.array(arr) for arr in data]

    def fetch(self, name):
        if name == 'train':
            token_file = os.path.join(self.path, 'bow_tr_tokens.pkl')
            count_file = os.path.join(self.path, 'bow_tr_counts.pkl')
            time_file = os.path.join(self.path, 'bow_tr_timestamps.pkl')
            source_file = os.path.join(self.path, 'bow_tr_sources.pkl')
        elif name == 'valid':
            token_file = os.path.join(self.path, 'bow_va_tokens.pkl')
            count_file = os.path.join(self.path, 'bow_va_counts.pkl')
            time_file = os.path.join(self.path, 'bow_va_timestamps.pkl')
            source_file = os.path.join(self.path, 'bow_va_sources.pkl')
        else:
            token_file = os.path.join(self.path, 'bow_ts_tokens.pkl')
            count_file = os.path.join(self.path, 'bow_ts_counts.pkl')
            time_file = os.path.join(self.path, 'bow_ts_timestamps.pkl')
            source_file = os.path.join(self.path, 'bow_ts_sources.pkl') 
        
        with open(token_file, 'rb') as file:
            tokens = pickle.load(file)
        with open(count_file, 'rb') as file:
            counts = pickle.load(file)
        
        with open(time_file, 'rb') as file:
            times = pickle.load(file)

        with open(source_file, 'rb') as file:
            sources = pickle.load(file)

        return {'tokens': tokens, 'counts': counts, 'times': times, 'sources': sources}

    def prepare_data(self):
        # load eta
        eta = torch.from_numpy(np.load(os.path.join(self.configs['eta_path'], 'eta.npy')))
        eta = torch.softmax(eta, dim=-1)

        # load cnpis
        cnpis_file = os.path.join(self.path, 'cnpis.pkl')
        with open(cnpis_file, 'rb') as file:
            cnpis = pickle.load(file)
        num_cnpis = cnpis.shape[-1]
        cnpis = torch.from_numpy(cnpis)
        # load mask
        cnpi_mask_file = os.path.join(self.path, 'cnpi_mask.pkl')
        with open(cnpi_mask_file, 'rb') as file:
            cnpi_mask = pickle.load(file)
        cnpi_mask = torch.from_numpy(cnpi_mask).type('torch.LongTensor')
        cnpi_mask = cnpi_mask.unsqueeze(-1).expand(cnpis.size())    # match cnpis' shape to apply masking

        # construct training and validation datasets
        self.train_dataset = COVID_Eta_Dataset(eta, cnpis, cnpi_mask)
        self.eval_dataset = COVID_Eta_Dataset(eta, cnpis, 1 - cnpi_mask)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['batch_size'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=self.configs['batch_size'], shuffle=True)