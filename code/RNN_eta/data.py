#/usr/bin/python

import os, pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class COVID_Eta_Dataset(Dataset):
    def __init__(self, eta, cnpis, cnpi_mask, forecast=False, random_baseline=False):
        super().__init__()

        self.eta = eta

        self.cnpis = cnpis
        self.cnpi_mask = cnpi_mask
        self.num_timestamps = cnpis.shape[1]

        self.forecast = forecast
        self.random_baseline = random_baseline

    def __len__(self):
        return self.eta.shape[0]

    def __getitem__(self, country_idx):
        if not self.forecast:
            batch_eta = self.eta[country_idx]
        else:
            batch_eta = self.eta[country_idx, :self.num_timestamps - 1, :]

        if self.random_baseline:
            batch_eta = batch_eta[:, torch.randperm(batch_eta.shape[-1], dtype=torch.long)]

        labels = self.cnpis[country_idx]
        mask = self.cnpi_mask[country_idx]
        
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

    def prepare_data(self):
        # load eta
        eta = torch.from_numpy(np.load(os.path.join(self.configs['eta_path'], 'eta.npy')))
        eta = torch.softmax(eta, dim=-1)
        self.num_sources = eta.shape[0]

        # load cnpis
        cnpis_file = os.path.join(self.path, 'cnpis.pkl')
        with open(cnpis_file, 'rb') as file:
            cnpis = pickle.load(file)
        num_cnpis = cnpis.shape[-1]
        cnpis = torch.from_numpy(cnpis)
        # load mask
        cnpi_mask_file = os.path.join(self.path, 'cnpi_mask_forecast.pkl') \
            if self.configs['forecast'] else os.path.join(self.path, 'cnpi_mask.pkl')
        with open(cnpi_mask_file, 'rb') as file:
            cnpi_mask = pickle.load(file)
        if self.configs['forecast']:
            assert cnpi_mask[:, int(cnpi_mask.shape[1] / 2): ].sum() == 0, \
                "not all second-half time points are masked for evaluation"
        cnpi_mask = torch.from_numpy(cnpi_mask).type('torch.LongTensor')
        cnpi_mask = cnpi_mask.unsqueeze(-1).expand(cnpis.size())    # match cnpis' shape to apply masking

        if self.configs['one_npi_per_model']:
            cnpis = cnpis[:, :, self.configs['current_cnpi']].unsqueeze(dim=-1)
            cnpi_mask = cnpi_mask[:, :, self.configs['current_cnpi']].unsqueeze(dim=-1)

        # construct training and validation datasets
        self.train_dataset = COVID_Eta_Dataset(eta, cnpis, cnpi_mask, forecast=self.configs['forecast'], random_baseline=self.configs['random_baseline'])
        self.eval_dataset = COVID_Eta_Dataset(eta, cnpis, cnpi_mask, forecast=self.configs['forecast'], random_baseline=self.configs['random_baseline'])
        self.test_dataset = COVID_Eta_Dataset(eta, cnpis, cnpi_mask, forecast=self.configs['forecast'], random_baseline=self.configs['random_baseline'])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        # evaluate all countries at once
        return DataLoader(self.eval_dataset, batch_size=self.num_sources, shuffle=True, num_workers=4)

    def test_dataloader(self):
        # evaluate all countries at once
        return DataLoader(self.test_dataset, batch_size=self.num_sources, shuffle=True, num_workers=4)