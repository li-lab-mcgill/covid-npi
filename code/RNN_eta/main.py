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
    parser.add_argument('--forecast', action='store_true', help='train to forecast')
    parser.add_argument('--teacher_force', action='store_true', help='teacher forcing. only effective when forecasting')
    parser.add_argument('--random_baseline', action='store_true', help='randomly permute topics as a baseline')
    parser.add_argument('--use_proto_loss', action='store_true', help='use prototypical loss')
    parser.add_argument('--update_rate', type=float, default=0.5, \
        help='update rate for updating prototypes across time. prototypes <- update_rate * new_prototypes + (1 - update_rate) * prototypes')

    # model configs
    parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of mapping')
    parser.add_argument('--no_bi_lstm', action='store_true', help='not using bidirectional lstm')
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

        if self.configs['teacher_force']:
            lstm_input_size += self.configs['num_cnpis']

        self.rnn = nn.LSTM(lstm_input_size, hidden_size=self.configs['hidden_size'], bidirectional=self.configs['bi_lstm'], \
            dropout=self.configs['dropout'], num_layers=self.configs['num_layers'], batch_first=True)
        self.rnn_out_size = self.configs['hidden_size'] if not self.configs['bi_lstm'] else self.configs['hidden_size'] * 2

        if self.configs['one_npi_per_model']:
            self.output = nn.Linear(self.rnn_out_size, 1, bias=True)
        else:
            self.output = nn.Linear(self.rnn_out_size, \
                self.rnn_out_size if self.configs['use_proto_loss'] else self.configs['num_cnpis'], bias=True)

        if self.configs['use_proto_loss']:
            self.prototypes = torch.zeros(self.configs['num_cnpis'], self.rnn_out_size)

    def forward(self, batch_eta, batch_label=None, rnn_hidden_state=None, rnn_cell_state=None):
        if self.configs['embed_topic'] or self.configs['embed_topic_with_alpha']:
            batch_eta = self.topic_embed(batch_eta.view(-1, batch_eta.shape[-1]))\
                .view(batch_eta.shape[0], batch_eta.shape[1], -1)
        # batch_eta: batch_size x times_span x (mapped_size or num_topics)
        if not self.configs['teacher_force']:
            rnn_out_state = self.rnn(batch_eta)[0]
            return self.output(rnn_out_state)
        else:
            # batch_label could either be true npis of the previous time point (in training)
            # or predicted npi probabilities (in testing)
            # when teacher forcing, generate predictions one at a time
            # so time span is 1, and the model is iteratively called
            batch_input = torch.cat([batch_eta, batch_label.type(torch.float)], dim=-1)
            rnn_out_state, (rnn_hidden_state, rnn_cell_state) = self.rnn(batch_input, (rnn_hidden_state, rnn_cell_state))
            return self.output(rnn_out_state), rnn_hidden_state, rnn_cell_state

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

        # clamp mask into no larger than 1
        batch_mask = torch.clamp(batch_mask, max=1)

        # proto loss requires batch_mask of different size for prediction
        batch_pred_mask = batch_mask if not self.configs['use_proto_loss'] else \
            batch_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, self.rnn_out_size)

        if not self.configs['forecast']:
            batch_predictions = self(batch_eta)
            # return F.binary_cross_entropy_with_logits(batch_predictions * batch_mask, batch_labels * batch_mask)
            return self.get_loss(batch_labels * batch_mask, batch_predictions * batch_pred_mask)
        else:
            if not self.configs['teacher_force']:
                batch_predictions = self(batch_eta)
            else:
                # teacher forcing
                rnn_hidden_state = rnn_cell_state = \
                    torch.zeros((self.configs['num_layers'], batch_eta.shape[0], self.configs['hidden_size'])).to(self.device)
                # batch_predictions holds all predictions
                batch_predictions = torch.zeros((batch_eta.shape[0], batch_eta.shape[1], \
                    batch_labels.shape[-1] if not self.configs['use_proto_loss'] else batch_eta.shape[1])).to(self.device)
                for time_idx in range(batch_eta.shape[1]):
                    if batch_mask[:, time_idx, :].sum() == 0:   # end of training time points
                        break
                    current_batch_predictions, rnn_hidden_state, rnn_cell_state = \
                        self(batch_eta[:, time_idx, :].unsqueeze(1), batch_labels[:, time_idx, :].unsqueeze(1), \
                            rnn_hidden_state, rnn_cell_state)
                    batch_predictions[:, time_idx, :] = current_batch_predictions.squeeze()
            batch_labels = batch_labels[:, 1: , :]
            batch_mask = batch_mask[:, 1:, :]
            batch_pred_mask = batch_pred_mask[:, 1:, :]
            # return F.binary_cross_entropy_with_logits(batch_predictions * batch_mask, batch_labels * batch_mask)
            return self.get_loss(batch_labels * batch_mask, batch_predictions * batch_pred_mask)

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

    def get_batch_predictions_eval(self, batch_eta, batch_labels, batch_mask):
        # clamp mask into no larger than 1
        batch_mask = torch.clamp(batch_mask, max=1)

        if not self.configs['teacher_force']:
            batch_predictions = self(batch_eta)
        else:
            # teacher forcing
            rnn_hidden_state = rnn_cell_state = \
                torch.zeros((self.configs['num_layers'], batch_eta.shape[0], self.configs['hidden_size'])).to(self.device)
            # batch_predictions holds all predictions
            batch_predictions = torch.zeros((batch_eta.shape[0], batch_eta.shape[1], \
                batch_labels.shape[-1] if not self.configs['use_proto_loss'] else batch_eta.shape[1])).to(self.device)
            for time_idx in range(batch_eta.shape[1]):
                if batch_mask[:, time_idx, :].sum() > 0:   # in training time points
                    current_batch_predictions, rnn_hidden_state, rnn_cell_state = \
                        self(batch_eta[:, time_idx, :].unsqueeze(1), batch_labels[:, time_idx, :].unsqueeze(1), \
                            rnn_hidden_state, rnn_cell_state)
                else:   # in testing time points
                    # use predictions from the previous time point instead of true labels
                    current_batch_predictions, rnn_hidden_state, rnn_cell_state = \
                        self(batch_eta[:, time_idx, :].unsqueeze(1), current_batch_predictions.round(), \
                            rnn_hidden_state, rnn_cell_state)
                current_batch_predictions = torch.sigmoid(current_batch_predictions).detach()
                batch_predictions[:, time_idx, :] = current_batch_predictions.squeeze()

        return batch_predictions
    
    def validation_step(self, batch, batch_idx):
        batch_eta, batch_labels, batch_mask = batch

        # proto loss requires batch_mask of different size for prediction
        batch_pred_mask = batch_mask if not self.configs['use_proto_loss'] else \
            batch_mask[:, :, 0].unsqueeze(-1).expand(-1, -1, self.rnn_out_size)

        batch_predictions_raw = self.get_batch_predictions_eval(batch_eta, batch_labels, batch_mask)

        batch_predictions = self.predict_proba(batch_predictions_raw) if self.configs['use_proto_loss'] \
            else batch_predictions_raw

        if self.configs['forecast']:
            batch_labels = batch_labels[:, 1: , :]
            batch_mask = batch_mask[:, 1:, :]
        batch_predictions_masked = batch_predictions * torch.clamp(1 - batch_mask, min=0)
        batch_labels_masked = batch_labels * torch.clamp(1 - batch_mask, min=0)

        # val_loss = F.binary_cross_entropy_with_logits(batch_predictions_masked, batch_labels_masked)
        val_loss = self.get_loss(batch_labels_masked, batch_predictions_raw * (1 - batch_pred_mask))

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

    def get_cnpi_auprcs(self, batch_eta, batch_labels, batch_mask, breakdown_by='measure'):
        assert breakdown_by in ['measure', 'source'], 'can only breankdown by measure or source'

        batch_predictions = self.get_batch_predictions_eval(batch_eta, batch_labels, batch_mask)
        if self.configs['use_proto_loss']:
            batch_predictions = self.predict_proba(batch_predictions)

        if self.configs['forecast']:
            batch_labels = batch_labels[:, 1: , :]
            batch_mask = batch_mask[:, 1:, :]
        batch_predictions_masked = batch_predictions * torch.clamp(1 - batch_mask, min=0)
        batch_labels_masked = batch_labels * torch.clamp(1 - batch_mask, min=0)

        if breakdown_by == 'measure':
            return self.compute_auprc_breakdown(batch_labels_masked.reshape(-1, batch_labels_masked.shape[-1]), \
                batch_predictions_masked.reshape(-1, batch_predictions_masked.shape[-1]))
        else:
            return np.array([self.compute_auprc_breakdown(batch_labels_masked[source_idx, :, :], \
                batch_predictions_masked[source_idx, :, :], average='micro') \
                    for source_idx in range(batch_labels_masked.shape[0])])

    def get_cnpi_supports(self, labels, mask):
        cnpis_masked = labels * (1 - mask)
        testing_cnpi_cnts = cnpis_masked.view(-1, cnpis_masked.shape[-1]).sum(0).tolist()
        return np.array(testing_cnpi_cnts) / np.sum(testing_cnpi_cnts)

    def test_step(self, batch, batch_idx):
        batch_eta, batch_labels, batch_mask = batch
        # breakdown by measure
        cnpi_auprcs_breakdown = self.get_cnpi_auprcs(batch_eta, batch_labels, batch_mask)
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

        # log average auprc
        if not self.configs['one_npi_per_model']:
            testing_cnpi_cnts_normed = self.get_cnpi_supports(batch_labels, batch_mask)
            self.logger.experiment.log({
                "Average AUPRC, weighted": np.average(cnpi_auprcs_breakdown, weights=testing_cnpi_cnts_normed)
            })
            self.logger.experiment.log({"Average AUPRC, macro": np.mean(cnpi_auprcs_breakdown)})

        # breakdown by source
        cnpi_auprcs_breakdown_source = self.get_cnpi_auprcs(batch_eta, batch_labels, batch_mask, breakdown_by='source')
        cnpi_auprcs_breakdown_source_out = {source_idx_to_source[source_idx]: auprc \
                for source_idx, auprc in enumerate(cnpi_auprcs_breakdown_source) if not np.isnan(auprc)}
        
        auprc_source_df = pd.DataFrame.from_dict(cnpi_auprcs_breakdown_source_out, \
            orient='index')
        auprc_source_df.columns = ['auprc']
        auprc_source_df["source"] = auprc_source_df.index
        if self.configs['one_npi_per_model']:
            auprc_source_df["measure"] = label_idx_to_label[self.configs['current_cnpi']]
        wandb_table = wandb.Table(dataframe=auprc_source_df)
        self.logger.experiment.log({"AUPRC breakdown by source": wandb_table})

        return cnpi_auprcs_breakdown_out, cnpi_auprcs_breakdown_source_out

    def get_loss(self, labels, pred):
        if not self.configs["use_proto_loss"]:
            return F.binary_cross_entropy_with_logits(pred, labels)
        else:
            return self.__proto_loss(labels.reshape(-1, labels.shape[-1]), pred.reshape(-1, pred.shape[-1]))
    
    # below are prototypical loss functions
    def predict_proba(self, pred):
        # pred is of shape (batch size, time points, hidden size), needs to reshape to 2-d then reshape back

        dists = torch.sigmoid(-self.__euclidean_dist(pred.reshape(-1, pred.shape[-1]), self.prototypes.to(self.device)))
        return dists.reshape(pred.shape[0], pred.shape[1], -1)

    def __get_prototypes(self, batch_emb, batch_cnpis):
        prototypes = []
        non_zero_indices = torch.nonzero(batch_cnpis, as_tuple=False)
        for cnpi_idx in range(batch_cnpis.shape[-1]):
            current_non_zero_indices = non_zero_indices[non_zero_indices[:, 1] == cnpi_idx][:, 0]
            if current_non_zero_indices.nelement() == 0:
                prototypes.append(self.prototypes[cnpi_idx].detach().to(self.device))
            else:
                prototypes.append(batch_emb[current_non_zero_indices].mean(dim=0))

        prototypes = (1 - self.configs['update_rate']) * self.prototypes.detach().to(self.device) \
            + self.configs['update_rate'] * torch.stack(prototypes, dim=0)
                
        return prototypes
    
    def __euclidean_dist(self, x, y):
        '''
        Compute euclidean distance between two tensors
        '''
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception(f"shape mismatch: {x.shape}, {y.shape}")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)
    
    def __proto_loss(self, batch_cnpis, batch_emb):
        # batch_cnpis: (number of samples, number of cnpis)
        # batch_emb: (number of samples, hidden size)

        has_pos_labels = batch_cnpis.sum(1) != 0
        batch_cnpis = batch_cnpis[has_pos_labels]
        batch_emb = batch_emb[has_pos_labels]

        if batch_emb.nelement() == 0:
            return 0
        
        if self.training:
            prototypes = self.__get_prototypes(batch_emb, batch_cnpis)
        else:
            prototypes = self.prototypes.detach()
            
        non_zero_indices = torch.nonzero(batch_cnpis, as_tuple=False)
        # DO NOT use torch.cdist to compute distance, it would cause nan in gradient
        dists = self.__euclidean_dist(batch_emb, prototypes.to(self.device))

        # log_denominators = torch.logsumexp(-dists, dim=1)
        probas = 2 * torch.sigmoid(-dists)  # multiply by 2 because -dist is non-positive
        log_denominators = torch.log(torch.sum(probas, dim=1))

        # query_size = log_denominators.shape[0]
        query_size = probas.shape[0]
        neg_log_likelihood = 0
        for query_idx in range(query_size):
            # log_numerator = torch.logsumexp(-dists[[query_idx, non_zero_indices[non_zero_indices[:, 0] == query_idx][:, 1]]], dim=0)
            log_numerator = torch.log(torch.sum(probas[[query_idx, non_zero_indices[non_zero_indices[:, 0] == query_idx][:, 1]]], dim=0))
            neg_log_likelihood -= (log_numerator - log_denominators[query_idx])
                
        self.prototypes = prototypes.detach().to(self.device)
                
        return neg_log_likelihood / query_size

if __name__ == '__main__':
    args = parse_args()
    args.bi_lstm = not args.no_bi_lstm
    configs = vars(args)

    if args.test:
        if not args.checkpoint:
            raise Exception("no checkpoint provided")
        print(f"Testing with checkpoint {args.checkpoint}")

    assert not (configs['embed_topic'] and configs['embed_topic_with_alpha']), \
        "cannot learn embedding if using alpha"
    if configs['embed_topic_with_alpha']:
        print("WARNING: emb_size is ineffective, will use alpha's dimension")

    if configs['teacher_force'] and not configs['forecast']:
        raise Exception("not forecasting, cannot teacher force")

    assert not (configs['bi_lstm'] and configs['teacher_force']), \
        "cannot use bi-LSTM when teacher forcing"

    if configs['one_npi_per_model'] and configs['use_proto_loss']:
        raise NotImplementedError()

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
        tags = ["RNN on eta", args.dataset, configs['eta_path'].split('/')[-1], f"seed: {args.seed}"]
        if configs['embed_topic']:
            tags.append('Train topic embeddings')
        if configs['embed_topic_with_alpha']:
            tags.append('Use alpha embeddings')
        if configs['bi_lstm']:
            tags.append('Bi-LSTM')
        if args.test:
            tags.append("Test")
        if configs['forecast']:
            tags.append("Forecast")
        if configs['teacher_force']:
            tags.append('Teacher forcing')
        if configs['random_baseline']:
            tags.append('Random baseline')
        if configs['use_proto_loss']:
            tags.append('Prototypical loss')
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
            for data, labels, mask in data_module.test_dataloader():
                test_predictions = model.get_batch_predictions_eval(
                    data.to(model.device), labels.to(model.device), mask.to(model.device)
                )
                if model.configs['use_proto_loss']:
                    test_predictions = model.predict_proba(test_predictions)
            # should be only 1 batch
            torch.save(test_predictions, os.path.join(configs['save_path'], time_stamp, 'test_predictions.pt'))
    else:
        for current_cnpi in range(configs['num_cnpis']):
            configs['current_cnpi'] = current_cnpi
            print(f"Current CNPI: {label_idx_to_label[current_cnpi]}")

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
            assert wb_logger.name == label_idx_to_label[current_cnpi]
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
                for data, labels, mask in data_module.test_dataloader():
                    test_predictions = model.get_batch_predictions_eval(
                        data.to(model.device), labels.to(model.device), mask.to(model.device)
                    )
                    if model.configs['use_proto_loss']:
                        test_predictions = model.predict_proba(test_predictions)
                # should be only 1 batch
                torch.save(test_predictions, \
                    os.path.join(configs['save_path'], time_stamp, f"{current_cnpi}", 'test_predictions.pt'))