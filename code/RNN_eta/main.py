#/usr/bin/python

import os, argparse, pickle, time, json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
# from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data import COVID_Eta_Data_Module

import wandb
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='RNN CNPI prediction baseline')

    # data io params
    parser.add_argument('--dataset', type=str, help='name of corpus')
    parser.add_argument('--data_path', type=str, help='directory containing data')
    parser.add_argument('--eta_path', type=str, help='directory containing eta from mixmedia')
    parser.add_argument('--save_path', type=str, help='path to save results')

    # training configs
    parser.add_argument('--test', action='store_true', help='test only')
    parser.add_argument('--batch_size', type=int, default=128, help='number of documents in a batch for training')
    parser.add_argument('--min_df', type=int, default=10, help='to get the right data..minimum document frequency')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--mode', type=str, default='train', help='train or eval model')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')
    parser.add_argument('--one_npi_per_model', action='store_true', help='train separate models for each npi')

    # model configs
    parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of mapping')
    parser.add_argument('--hidden_size', type=int, default=128, help='rnn hidden size')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_layers', type=int, default=1, help='number of rnn layers')
    parser.add_argument('--embed_topic', action='store_true', \
        help='embed topics with learnable embeddings')
    parser.add_argument('--embed_topic_with_alpha', action='store_true', \
        help='embed topics with embeddings from mixmedia')
    parser.add_argument('--checkpoint', type=str, help='checkpoint to evaluate. only effective in test mode')

    return parser.parse_args()

class RNN_CNPI_EtaModel(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        if self.configs['embed_topic']:
            topic_emb_size = self.configs['emb_size']
            self.topic_embed = nn.Linear(self.configs['num_topics'], topic_emb_size, bias=False)
            lstm_input_size = topic_emb_size
        elif configs['embed_topic_with_alpha']:
            topic_emb_size = self.configs['emb_size']
            self.topic_embed = nn.Linear(self.configs['num_topics'], topic_emb_size, bias=False)
            # use alpha (topics x embedding size)
            self.topic_embed.weight = nn.Parameter(self.configs['alpha'].clone().T)
            self.topic_embed.weight.requires_grad = False
            lstm_input_size = topic_emb_size
        else:
            lstm_input_size = self.configs['num_topics']

        self.rnn = nn.LSTM(lstm_input_size, hidden_size=self.configs['hidden_size'], bidirectional=False, \
            dropout=self.configs['dropout'], num_layers=self.configs['num_layers'], batch_first=True)

        if self.configs['one_npi_per_model']:
            self.rnn_out = nn.Linear(self.configs['hidden_size'], 1, bias=True)
        else:
            self.rnn_out = nn.Linear(self.configs['hidden_size'], self.configs['num_cnpis'], bias=True)

    def forward(self, batch_eta):
        if self.configs['embed_topic'] or self.configs['embed_topic_with_alpha']:
            batch_eta = self.topic_embed(batch_eta.view(-1, batch_eta.shape[-1]))\
                .view(batch_eta.shape[0], batch_eta.shape[1], -1)
        # batch_eta: batch_size x times_span x (mapped_size or num_topics)
        rnn_hidden = self.rnn(batch_eta)[0]
        return self.rnn_out(rnn_hidden)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.configs['lr'], weight_decay=self.configs['wdecay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=10, min_lr=1e-7)
        scheduler = {
            'scheduler': scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            # 'monitor': 'val_checkpoint_on'
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        batch_eta, batch_labels, batch_mask = batch
        batch_predictions = self(batch_eta)
        return F.binary_cross_entropy_with_logits(batch_predictions * batch_mask, batch_labels * batch_mask)

    def compute_top_k_recall_prec(self, labels, predictions, k=5, metric='recall'):
        '''
        inputs:
        - labels: tensor, (number of samples, number of classes)
        - predictions: tensor, (number of samples, number of classes)
        - k
        - metric: recall or prec
        output:
        - top-k recall or precision of the batch
        '''
        assert metric in ['recall', 'prec'], 'metric is either recall or prec'

        # remove ones without positive labels
        has_pos_labels = labels.sum(1) != 0
        labels = labels[has_pos_labels, :]
        predictions = predictions[has_pos_labels, :]
        idxs = torch.argsort(predictions, dim=1, descending=True)[:, 0: k]
        if metric == 'recall':
            return (torch.gather(labels, 1, idxs).sum(1) / labels.sum(1)).mean()
        else:
            return (torch.gather(labels, 1, idxs).sum(1) / k).mean()
    
    def validation_step(self, batch, batch_idx):
        batch_eta, batch_labels, batch_mask = batch
        batch_predictions = self(batch_eta)
        batch_mask = 1 - batch_mask
        batch_predictions_masked = batch_predictions * batch_mask
        batch_labels_masked = batch_labels * batch_mask

        val_loss = F.binary_cross_entropy_with_logits(batch_predictions_masked, batch_labels_masked)

        self.log_dict({'val_loss': val_loss, 'epoch': self.current_epoch, 'step': self.global_step})
        self.logger.experiment.log({'val_loss': val_loss, 'epoch': self.current_epoch, 'step': self.global_step})

        if not self.configs['one_npi_per_model']:
            top_k_recalls = {
                1: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1),
                3: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3),
                5: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5),
                10: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10),
                }
            top_k_recalls_log = {f"recall/{key}": value for key, value in top_k_recalls.items()}
            top_k_recalls_log.update({'epoch': self.current_epoch, 'step': self.global_step})
            # self.log_dict(top_k_recalls_log)
            self.logger.experiment.log(top_k_recalls_log)
            top_k_precs = {
                1: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 1, 'prec'),
                3: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 3, 'prec'),
                5: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 5, 'prec'),
                10: self.compute_top_k_recall_prec(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                    batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]), 10, 'prec'),
                }
            top_k_precs_log = {f"prec/{key}": value for key, value in top_k_precs.items()}
            top_k_precs_log.update({'epoch': self.current_epoch, 'step': self.global_step})
            # self.log_dict(top_k_precs_log)
            self.logger.experiment.log(top_k_precs_log)
            top_k_f1s = {
                k: (2 * top_k_recalls[k] * top_k_precs[k]) / \
                    (top_k_recalls[k] + top_k_precs[k]) for k in [1, 3, 5, 10]
                }
            top_k_f1s_log = {f"f1/{key}": value for key, value in top_k_f1s.items()}
            top_k_f1s_log.update({'epoch': self.current_epoch, 'step': self.global_step})
            # self.log_dict(top_k_f1s_log)
            self.logger.experiment.log(top_k_f1s_log)
            # return results

    def compute_auprc_breakdown(self, labels, predictions, average=None):
        '''
        inputs:
        - labels: tensor, (number of samples, number of classes)
        - predictions: tensor, (number of samples, number of classes)
        - average: None or str, whether to take the average
        output:
        - auprcs: array, (number of classes) if average is None, or scalar otherwise
        '''
        if labels.shape[1] > 1:
            # remove ones without positive labels
            has_pos_labels = labels.sum(1) != 0
            labels = labels[has_pos_labels, :]
            predictions = predictions[has_pos_labels, :]
        
        labels = labels.cpu().numpy()
        if labels.size == 0:    # empty
            return np.nan
        predictions = predictions.cpu().numpy()
        return average_precision_score(labels, predictions, average=average)

    def get_cnpi_auprcs(self, batch_bows, batch_labels, batch_mask, breakdown_by='measure'):
        assert breakdown_by in ['measure', 'source'], 'can only breankdown by measure or source'

        batch_predictions = self(batch_bows)
        batch_mask = 1 - batch_mask
        batch_predictions_masked = batch_predictions * batch_mask
        batch_labels_masked = batch_labels * batch_mask

        if breakdown_by == 'measure':
            return self.compute_auprc_breakdown(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]))
        else:
            return np.array([self.compute_auprc_breakdown(batch_labels_masked[source_idx, :, :], \
                batch_predictions_masked[source_idx, :, :], average='micro') \
                    for source_idx in range(batch_labels_masked.shape[0])])

    def test_step(self, batch, batch_idx):
        batch_bows, batch_labels, batch_mask = batch
        # breakdown by measure
        cnpi_auprcs_breakdown = self.get_cnpi_auprcs(batch_bows, batch_labels, batch_mask)
        if self.configs['one_npi_per_model']:
            cnpi_auprcs_breakdown_out = {label_idx_to_label[self.configs['current_cnpi']]: cnpi_auprcs_breakdown}
        else:
            cnpi_auprcs_breakdown_out = {label_idx_to_label[label_idx]: cnpi_auprcs_breakdown[label_idx] \
                    for label_idx, auprc in enumerate(cnpi_auprcs_breakdown) if not np.isnan(auprc)}
        
        auprc_df = pd.DataFrame.from_dict(cnpi_auprcs_breakdown_out, \
            orient='index')
        auprc_df.columns = ['auprc']
        auprc_df["measure"] = auprc_df.index
        wandb_table = wandb.Table(dataframe=auprc_df)
        self.logger.experiment.log({"AUPRC breakdown": wandb_table})

        # breakdown by source
        cnpi_auprcs_breakdown_source = self.get_cnpi_auprcs(batch_bows, batch_labels, batch_mask, breakdown_by='source')
        cnpi_auprcs_breakdown_source_out = {source_idx_to_source[source_idx]: auprc \
                for source_idx, auprc in enumerate(cnpi_auprcs_breakdown_source) if not np.isnan(auprc)}
        
        auprc_source_df = pd.DataFrame.from_dict(cnpi_auprcs_breakdown_source_out, \
            orient='index')
        auprc_source_df.columns = ['auprc']
        auprc_source_df["source"] = auprc_source_df.index
        auprc_source_df["measure"] = label_idx_to_label[self.configs['current_cnpi']]
        wandb_table = wandb.Table(dataframe=auprc_source_df)
        self.logger.experiment.log({"AUPRC breakdown by source": wandb_table})

        return cnpi_auprcs_breakdown_out, cnpi_auprcs_breakdown_source_out

if __name__ == '__main__':
    args = parse_args()
    configs = vars(args)

    if args.test:
        if not args.checkpoint:
            raise Exception("no checkpoint provided")
        print(f"Testing with checkpoint {args.checkpoint}")

    assert not (configs['embed_topic'] and configs['embed_topic_with_alpha']), \
        "can not learn embedding if using alpha"
    if configs['embed_topic_with_alpha']:
        print("WARNING: emb_size is ineffective, will use alpha's dimension")

    time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {time_stamp}")

    if not os.path.exists(os.path.join(configs['save_path'], time_stamp)):
        os.makedirs(os.path.join(configs['save_path'], time_stamp))

    # save configs
    with open(os.path.join(configs['save_path'], time_stamp, 'configs.json'), 'w') as file:
        json.dump(configs, file)

    eta = np.load(os.path.join(configs['eta_path'], 'eta.npy'))
    configs['num_topics'] = eta.shape[-1]
    del eta

    if configs['embed_topic_with_alpha']:
        configs['alpha'] = torch.from_numpy(np.load(os.path.join(configs['eta_path'], "alpha.npy")))
        configs['emb_size'] = configs['alpha'].shape[-1]

    with open(os.path.join(os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']), 'cnpis.pkl')), 'rb') as file:
        cnpis = pickle.load(file)
    configs['num_cnpis'] = cnpis.shape[-1]
    del cnpis

    # load label_maps for auprc breakdown
    with open(os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']), 'labels_map.pkl'), 'rb') as file:
        labels_map = pickle.load(file)  
    label_idx_to_label = {value: key for key, value in labels_map.items()}

    # load sources_maps for auprc breakdown
    with open(os.path.join(configs['data_path'], 'min_df_{}'.format(configs['min_df']), 'sources_map.pkl'), 'rb') as file:
        sources_map = pickle.load(file) 
    source_idx_to_source = {value: key for key, value in sources_map.items()}

    ## set seed
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)

    if not configs['one_npi_per_model']:
        # initiate data module
        data_module = COVID_Eta_Data_Module(configs)

        # initiate model
        model = RNN_CNPI_EtaModel(configs)
        if args.test:
            checkpoint = torch.load(args.checkpoint)
            # model = RNN_CNPI_BaseModel.load_from_checkpoint(
            #     checkpoint_path=args.checkpoint,
            # )
            model.load_state_dict(checkpoint['state_dict'])

        # train
        # tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/{time_stamp}")
        tags = ["RNN on eta", args.dataset, configs['eta_path'].split('/')[-1]]
        if configs['embed_topic']:
            tags.append('Train topic embeddings')
        if configs['embed_topic_with_alpha']:
            tags.append('Use alpha embeddings')
        if args.test:
            tags.append("Test")
        wb_logger = pl_loggers.WandbLogger(
            name=f"{time_stamp}",
            project="covid",
            tags=tags,
            )
        wb_logger.log_hyperparams(args)
        trainer = pl.Trainer(
            gradient_clip_val=args.clip, 
            max_epochs=args.epochs, 
            gpus=1, 
            logger=wb_logger,
            weights_save_path=os.path.join(configs['save_path'], time_stamp)
            )
        if not args.test:
            trainer.fit(model, data_module)
            trainer.test()
        else:
            trainer.test(model, datamodule=data_module)
        # save predictions
        with torch.no_grad():
            for data, _, _ in data_module.test_dataloader():
                test_predictions = model(data.to(torch.device('cuda')))
            # should be only 1 batch
            torch.save(test_predictions, os.path.join(configs['save_path'], time_stamp, 'test_predictions.pt'))
    else:
        for current_cnpi in range(configs['num_cnpis']):
            configs['current_cnpi'] = current_cnpi

            # initiate data module
            data_module = COVID_Eta_Data_Module(configs)

            # initiate model
            model = RNN_CNPI_EtaModel(configs)
            if args.test:
                checkpoint = torch.load(args.checkpoint)
                # model = RNN_CNPI_BaseModel.load_from_checkpoint(
                #     checkpoint_path=args.checkpoint,
                # )
                model.load_state_dict(checkpoint['state_dict'])

            # train
            # tb_logger = pl_loggers.TensorBoardLogger(f"lightning_logs/{time_stamp}")
            tags = ["RNN on eta", args.dataset, configs['eta_path'].split('/')[-1], "One per NPI"]
            if configs['embed_topic']:
                tags.append('Train topic embeddings')
            if configs['embed_topic_with_alpha']:
                tags.append('Use alpha embeddings')
            if args.test:
                tags.append("Test")
            wb_logger = pl_loggers.WandbLogger(
                name=label_idx_to_label[current_cnpi],
                project="covid",
                tags=tags,
                group=time_stamp,
                )
            wb_logger.log_hyperparams(args)
            trainer = pl.Trainer(
                gradient_clip_val=args.clip, 
                max_epochs=args.epochs, 
                gpus=1, 
                logger=wb_logger,
                weights_save_path=os.path.join(configs['save_path'], time_stamp, f"{current_cnpi}")
                )
            if not args.test:
                trainer.fit(model, data_module)
                trainer.test()
            else:
                trainer.test(model, datamodule=data_module)
            # save predictions
            with torch.no_grad():
                for data, _, _ in data_module.test_dataloader():
                    test_predictions = model(data.to(torch.device('cuda')))
                # should be only 1 batch
                torch.save(test_predictions, \
                    os.path.join(configs['save_path'], time_stamp, f"{current_cnpi}", 'test_predictions.pt'))