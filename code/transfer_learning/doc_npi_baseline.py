import os, argparse, time, json
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

import wandb

from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Baselines of predicting document NPIs from BOW")

    # data io args
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing bow and labels")
    parser.add_argument("--save_dir", type=str, required=True, help="root directory to save output")

    # training args
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2048, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="epochs")
    parser.add_argument("--weight_decay", type=float, default=2e-3, help="weight decay")
    parser.add_argument("--hidden_size", type=int, default=256, help="hidden size (only effective when non-linear)")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate (only effective when non-linear)")

    # other args
    parser.add_argument("--train_ratio", type=float, default=0.8, help="training data ratio")
    parser.add_argument("--num_seeds", type=int, default=1, help="number of runs with different seeds")
    parser.add_argument("--save_ckpt", action="store_true", help="save best model checkpoints")
    parser.add_argument("--linear", action="store_true", help="use linear classifier (otherwise, a 2 layer feed-forward network)")
    parser.add_argument("--no_logger", action="store_true", help="not using wandb for logging")

    return parser.parse_args()

def prepare_data(*args):
    out = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            out.append(torch.from_numpy(arg).type(torch.float).to(device))
        else:
            out.append(arg.to(device))
    return out

def build_model(args):
    if args.linear:
        return torch.nn.Linear(
            train_bow.shape[-1],
            train_labels.shape[-1]
        ).to(device)
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(
                train_bow.shape[-1],
                args.hidden_size
            ),
            torch.nn.ReLU(), torch.nn.Dropout(args.dropout),
            torch.nn.Linear(
                args.hidden_size,
                train_labels.shape[-1]
            )
        ).to(device)

@torch.no_grad()
def compute_auprc_breakdown(labels, predictions, average=None):
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

@torch.no_grad()
def get_label_supports(labels):
    label_cnts = labels.sum(0).tolist()
    return np.array(label_cnts) / np.sum(label_cnts)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    config = vars(args)

    exp_time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {exp_time_stamp}")

    # create save directory
    if args.save_ckpt:
        save_dir = os.path.join(args.save_dir, "doc_baselines", "linear" if args.linear else "nonlinear", exp_time_stamp)
        if os.path.exists(save_dir):
            raise Exception(f"saving directory already exist: {save_dir}")
        os.makedirs(save_dir)
        print("Saving directory created")
        with open(os.path.join(save_dir, "configs.json"), "w") as file:
            json.dump(config, file)

    # load bow
    bow = sparse.load_npz(os.path.join(args.data_dir, "bow.npz")).toarray()

    # load labels
    labels = np.load(os.path.join(args.data_dir, "labels.npy"))
    assert bow.shape[0] == labels.shape[0]
    print("Data loaded")

    # data split
    indices = np.arange(bow.shape[0])
    rng = np.random.default_rng(seed=2021)
    rng.shuffle(indices)
    train_indices = indices[: int(len(indices) * args.train_ratio)]
    test_indices = indices[int(len(indices) * args.train_ratio):]
    train_bow, test_bow = bow[train_indices], bow[test_indices]
    train_labels, test_labels = labels[train_indices], labels[test_indices]

    # prepare data
    train_bow, test_bow, train_labels, test_labels = \
        prepare_data(train_bow, test_bow, train_labels, test_labels)
    class_weight = (train_labels.sum() / train_labels.sum(dim=0)) \
        / (train_labels.sum() / train_labels.sum(dim=0)).sum()

    # data loader
    train_loader = torch.utils.data.DataLoader(
        range(train_bow.shape[0]),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        range(test_bow.shape[0]),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False
    )

    # train
    seeds = list(range(2021, 2021 + args.num_seeds))

    all_train_weighted_auprcs = []
    all_train_macro_auprcs = []
    all_test_weighted_auprcs = []
    all_test_macro_auprcs = []

    for seed in tqdm(seeds):
        torch.manual_seed(seed)
        train_weighted_auprcs = []
        train_macro_auprcs = []
        test_weighted_auprcs = []
        test_macro_auprcs = []

        # build model
        if args.linear:
            model = torch.nn.Linear(
                train_bow.shape[-1],
                train_labels.shape[-1]
            ).to(device)
        else:
            model = torch.nn.Sequential(
                torch.nn.Linear(
                    train_bow.shape[-1],
                    args.hidden_size
                ),
                torch.nn.ReLU(), torch.nn.Dropout(args.dropout),
                torch.nn.Linear(
                    args.hidden_size,
                    train_labels.shape[-1]
                )
            ).to(device)

        # wandb logging stuff
        if not args.no_logger:
            tags = ['Doc NPI baseline', 'Linear' if args.linear else 'Nonlinear']
            wandb_run = wandb.init(name=f"{exp_time_stamp}_{seed}", project="covid", group=exp_time_stamp, config=args, tags=tags)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_test_loss = None
        for epoch in tqdm(range(args.epochs), total=args.epochs, disable=True):
            model.zero_grad()
            optimizer.zero_grad()

            # train
            all_preds = []
            all_labels = []
            train_loss = 0
            model.train()
            for batch_idx, idxs in enumerate(train_loader):
                pred = model(train_bow[idxs])
                loss = torch.dot(class_weight,
                            F.binary_cross_entropy_with_logits(pred, train_labels[idxs], reduction="none").mean(dim=0))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                all_preds.append(pred.detach())
                all_labels.append(train_labels[idxs])
            train_loss = train_loss / (batch_idx + 1)

            auprc_breakdown = compute_auprc_breakdown(torch.cat(all_labels, dim=0), torch.cat(all_preds, dim=0))  
            label_cnts_normed = get_label_supports(torch.cat(all_labels, dim=0))

            train_weighted_auprcs.append(np.average(auprc_breakdown, weights=label_cnts_normed))
            train_macro_auprcs.append(np.mean(auprc_breakdown))

            # eval
            all_preds = []
            with torch.no_grad():
                test_loss = 0
                model.eval()
                for batch_idx, idxs in enumerate(test_loader):
                    pred = model(test_bow[idxs])
                    loss = torch.dot(class_weight, \
                            F.binary_cross_entropy_with_logits(pred, test_labels[idxs], reduction="none").mean(dim=0))
                    test_loss += loss.item()
                    all_preds.append(pred.detach())
                test_loss = test_loss / (batch_idx + 1)

                if args.save_ckpt and (best_test_loss is None or test_loss < best_test_loss):
                    best_test_loss = test_loss
                    torch.save(model, os.path.join(save_dir, f"model_{seed}.pt"))
                    torch.save(torch.cat(all_preds, dim=0).detach().cpu(), \
                        os.path.join(save_dir, f"test_preds_{seed}.pt"))

                auprc_breakdown = compute_auprc_breakdown(test_labels, torch.cat(all_preds, dim=0))  
                label_cnts_normed = get_label_supports(test_labels)

                test_weighted_auprcs.append(np.average(auprc_breakdown, weights=label_cnts_normed))
                test_macro_auprcs.append(np.mean(auprc_breakdown))

            # wandb logging
            if not args.no_logger:
                wandb.log({
                    "epoch": epoch, "weighted auprc/train": train_weighted_auprcs[-1], "weighted auprc/test": test_weighted_auprcs[-1],
                    "macro auprc/train": train_macro_auprcs[-1], "macro auprc/test": test_macro_auprcs[-1],
                })
                
        all_train_weighted_auprcs.append(np.array(train_weighted_auprcs))
        all_train_macro_auprcs.append(np.array(train_macro_auprcs))
        all_test_weighted_auprcs.append(np.array(test_weighted_auprcs))
        all_test_macro_auprcs.append(np.array(test_macro_auprcs))

        if not args.no_logger:
            wandb_run.finish()

        del model
        del optimizer

    # print and save results
    print(f"AUPRCs")
    print("Weighted:")
    print(f" - train: {np.max(all_train_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_train_weighted_auprcs, axis=1).std():.3f})")
    print(f" - test: {np.max(all_test_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_test_weighted_auprcs, axis=1).std():.3f})")
    print(f"Macro:")
    print(f" - train: {np.max(all_train_macro_auprcs, axis=1).mean():.3f} ({np.max(all_train_macro_auprcs, axis=1).std():.3f})")
    print(f" - test: {np.max(all_test_macro_auprcs, axis=1).mean():.3f} ({np.max(all_test_macro_auprcs, axis=1).std():.3f})")

    if args.save_ckpt:
        # save the raw auprcs
        np.save(os.path.join(save_dir, "all_train_weighted_auprcs.npy"), np.vstack(all_train_weighted_auprcs))
        np.save(os.path.join(save_dir, "all_test_weighted_auprcs.npy"), np.vstack(all_test_weighted_auprcs))
        np.save(os.path.join(save_dir, "all_train_macro_auprcs.npy"), np.vstack(all_train_macro_auprcs))
        np.save(os.path.join(save_dir, "all_test_macro_auprcs.npy"), np.vstack(all_test_macro_auprcs))

        # save the aggregated auprcs
        results = {
            "weighted": {
                "train": f"{np.max(all_train_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_train_weighted_auprcs, axis=1).std():.3f})",
                "test": f"{np.max(all_test_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_test_weighted_auprcs, axis=1).std():.3f})"
            },
            "macro": {
                "train": f"{np.max(all_train_macro_auprcs, axis=1).mean():.3f} ({np.max(all_train_macro_auprcs, axis=1).std():.3f})",
                "test": f"{np.max(all_test_macro_auprcs, axis=1).mean():.3f} ({np.max(all_test_macro_auprcs, axis=1).std():.3f})"
            }
        }
        with open(os.path.join(save_dir, "results.json"), "w") as file:
            json.dump(results, file)