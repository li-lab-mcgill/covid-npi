#/usr/bin/python

import os, pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np

class COVID_Dataset(Dataset):
    def __init__(self, source_to_timestamp_to_tokens, source_to_timestamp_to_counts, \
        cnpis, cnpi_mask, vocab_size, num_times, num_sources):
        super().__init__()

        self.source_to_timestamp_to_tokens = source_to_timestamp_to_tokens
        self.source_to_timestamp_to_counts = source_to_timestamp_to_counts

        self.cnpis = cnpis
        self.cnpi_mask = cnpi_mask

        self.vocab_size = vocab_size
        self.num_times = num_times
        self.num_sources = num_sources

    def __len__(self):
        return self.num_sources

    # def __getitem__(self, idxs):
    #     # idxs are country indices
    #     batch_size = len(idxs)
    #     data_batch = np.zeros((batch_size, self.num_times, self.vocab_size))
    #     masks_batch = []
    #     times_batch = np.zeros((batch_size, ))
    #     sources_batch = np.zeros((batch_size, ))

    #     for i, country_idx in enumerate(idxs):    
    #         for time_idx, tokens in self.source_to_timestamp_to_tokens[country_idx].items():
    #             bow = np.zeros(self.vocab_size)
    #             for idx, token in tokens:
    #                 bow[token] += self.source_to_timestamp_to_counts[country_idx][time_idx][idx]
    #             data_batch[i, time_idx] += bow
        
    #     data_batch = torch.from_numpy(data_batch).float() 
    #     labels_batch = self.cnpis[idxs]
    #     mask_batch = self.cnpi_mask[idxs]

    #     return data_batch, labels_batch, mask_batch
    def __getitem__(self, country_idx):
        data = np.zeros((self.num_times, self.vocab_size))
        for time_idx, tokens in self.source_to_timestamp_to_tokens[country_idx].items():
            bow = np.zeros(self.vocab_size)
            for idx, token in enumerate(tokens):
                bow[token] += self.source_to_timestamp_to_counts[country_idx][time_idx][idx]
            data[time_idx] += bow

        data = torch.from_numpy(data).float()
        labels = self.cnpis[country_idx]
        mask = self.cnpi_mask[country_idx, :, :]

        return data, labels, mask

class COVID_Data_Module(pl.LightningDataModule):
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
        # load data as in MixMedia

        train_dict = self.fetch('train')
        valid_dict = self.fetch('valid')
        test_dict = self.fetch('test')

        # merge them together since the split in on (country, time) pairs
        data_dict = {
            'tokens': np.concatenate([train_dict['tokens'], valid_dict['tokens'], test_dict['tokens']]),
            'counts': np.concatenate([train_dict['counts'], valid_dict['counts'], test_dict['counts']]),
            'times': np.concatenate([train_dict['times'], valid_dict['times'], test_dict['times']]),
            'sources': np.concatenate([train_dict['sources'], valid_dict['sources'], test_dict['sources']]),
        }

        # construct a dict of countries and their indices
        source_to_idx = {}
        for idx, source_idx in enumerate(data_dict['sources']):
            if source_idx in source_to_idx:
                source_to_idx[source_idx].append(idx)
            else:
                source_to_idx[source_idx] = []

        # get num_times and num_sources
        all_timestamps = pickle.load(open(os.path.join(self.path, 'timestamps.pkl'), 'rb'))
        num_times = len(all_timestamps)
        self.num_sources = len(source_to_idx)

        # construct dicts of time stamps and docs
        source_to_timestamp_to_tokens = {}
        source_to_timestamp_to_counts = {}
        for source_idx, idxs in tqdm(source_to_idx.items()):
            source_to_timestamp_to_tokens[source_idx] = {}
            source_to_timestamp_to_counts[source_idx] = {}
            for idx in idxs:
                if data_dict['times'][idx] in source_to_timestamp_to_tokens[source_idx]:
                    source_to_timestamp_to_tokens[source_idx][data_dict['times'][idx]] += \
                        data_dict['tokens'][idx]
                    source_to_timestamp_to_counts[source_idx][data_dict['times'][idx]] += \
                        data_dict['counts'][idx]
                else:
                    source_to_timestamp_to_tokens[source_idx][data_dict['times'][idx]] = []
                    source_to_timestamp_to_counts[source_idx][data_dict['times'][idx]] = []

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
        self.train_dataset = COVID_Dataset(source_to_timestamp_to_tokens, source_to_timestamp_to_counts, \
            cnpis, cnpi_mask, self.configs['vocab_size'], num_times, self.num_sources)
        self.eval_dataset = COVID_Dataset(source_to_timestamp_to_tokens, source_to_timestamp_to_counts, \
            cnpis, cnpi_mask, self.configs['vocab_size'], num_times, self.num_sources)
        self.test_dataset = COVID_Dataset(source_to_timestamp_to_tokens, source_to_timestamp_to_counts, \
            cnpis, cnpi_mask, self.configs['vocab_size'], num_times, self.num_sources)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['batch_size'], shuffle=True)

    def val_dataloader(self):
        # evaluate all countries at once
        return DataLoader(self.eval_dataset, batch_size=self.num_sources, shuffle=False)
    
    def test_dataloader(self):
        # evaluate all countries at once
        return DataLoader(self.test_dataset, batch_size=self.num_sources, shuffle=False)