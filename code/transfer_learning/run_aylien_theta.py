import os, argparse, time, json, pickle
import numpy as np
from tqdm import tqdm

import wandb

from sklearn.metrics import average_precision_score

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Predicting country NPIs from theta")

    # data io args
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing theta and labels")
    parser.add_argument("--save_dir", type=str, required=True, help="root directory to save output")
    parser.add_argument("--load_ckpt_from", type=str, help="the checkpoint to load from")

    # training args
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size")
    parser.add_argument("--epochs", type=int, default=10, help="epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")

    # other args
    parser.add_argument("--mode", type=str, required=True, choices=["from-scratch", "zero-shot", "fine-tuned"])
    parser.add_argument("--num_seeds", type=int, default=1, help="number of runs with different seeds")
    parser.add_argument("--save_ckpt", action="store_true", help="save best model checkpoints")
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

@torch.no_grad()
def merge_and_compute_auprc_scores(labels, predictions, countries, times):
    # labels and predictions are tensors while countries and times are np arrays
    # first merge the labels and predictions of the same (country, time) pair
    country_to_time_to_preds = {}
    country_to_time_to_cnpis = {}

    for idx in range(labels.shape[0]):
        if countries[idx] not in country_to_time_to_preds:
            country_to_time_to_preds[countries[idx]] = {}
            country_to_time_to_cnpis[countries[idx]] = {}
        if times[idx] not in country_to_time_to_preds[countries[idx]]:
            country_to_time_to_preds[countries[idx]][times[idx]] = [predictions[idx]]
            country_to_time_to_cnpis[countries[idx]][times[idx]] = labels[idx]
        else:
            country_to_time_to_preds[countries[idx]][times[idx]].append(predictions[idx])
    for country_idx in country_to_time_to_preds:
        for time_idx in country_to_time_to_preds[country_idx]:
            country_to_time_to_preds[country_idx][time_idx] = \
                torch.mean(torch.stack(country_to_time_to_preds[country_idx][time_idx], dim=0), dim=0)

    # then compute auprcs
    labels_merged = []
    preds_merged = []

    for country_idx in country_to_time_to_preds:
        for time_idx in country_to_time_to_preds[country_idx]:
            labels_merged.append(country_to_time_to_cnpis[country_idx][time_idx])
            preds_merged.append(country_to_time_to_preds[country_idx][time_idx])

    auprc_breakdown = compute_auprc_breakdown(torch.stack(labels_merged, dim=0), torch.stack(preds_merged, dim=0))
    label_cnts_normed = get_label_supports(torch.stack(labels_merged, dim=0))
    return np.average(auprc_breakdown, weights=label_cnts_normed), np.mean(auprc_breakdown)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    config = vars(args)

    if args.mode == "from-scratch" and args.load_ckpt_from:
        raise Exception("cannot load checkpoint when training from scratch")
    if args.mode != "from-scratch" and not args.load_ckpt_from:
        raise Exception("no checkpoint provided")

    exp_time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {exp_time_stamp}")

    # create save directory
    if args.save_ckpt:
        save_dir = os.path.join(args.save_dir, "aylien_theta", args.mode, exp_time_stamp)
        if os.path.exists(save_dir):
            raise Exception(f"saving directory already exist: {save_dir}")
        os.makedirs(save_dir)
        print("Saving directory created")
        with open(os.path.join(save_dir, "configs.json"), "w") as file:
            json.dump(config, file)

    # load thetas
    train_thetas = np.load(os.path.join(args.data_dir, "train_aylien_thetas.npy"))
    test_thetas = np.load(os.path.join(args.data_dir, "test_aylien_thetas.npy"))

    # load alpha
    alpha = np.load(os.path.join(args.data_dir, "alpha.npy"))

    # load countries and times
    countries_dict = {}
    with open(os.path.join(args.data_dir, "train_aylien_countries.pkl"), "rb") as file:
        countries_dict["train"] = np.array(pickle.load(file))
    with open(os.path.join(args.data_dir, "test_aylien_countries.pkl"), "rb") as file:
        countries_dict["test"] = np.array(pickle.load(file))
    times_dict = {}
    with open(os.path.join(args.data_dir, "train_aylien_times.pkl"), "rb") as file:
        times_dict["train"] = np.array(pickle.load(file))
    with open(os.path.join(args.data_dir, "test_aylien_times.pkl"), "rb") as file:
        times_dict["test"] = np.array(pickle.load(file))

    # load cnpis
    train_cnpis = np.load(os.path.join(args.data_dir, "train_aylien_cnpis.npy"))
    test_cnpis = np.load(os.path.join(args.data_dir, "test_aylien_cnpis.npy"))
    assert train_thetas.shape[0] == train_cnpis.shape[0] and test_thetas.shape[0] == test_cnpis.shape[0]
    print("Data loaded")

    # prepare data
    train_thetas, test_thetas, train_cnpis, test_cnpis, alpha = \
        prepare_data(train_thetas, test_thetas, train_cnpis, test_cnpis, alpha)

    # data loader
    train_loader = torch.utils.data.DataLoader(
        range(train_thetas.shape[0]),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        range(test_thetas.shape[0]),
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

    for seed in tqdm(seeds, disable=args.mode != "zero-shot"):
        torch.manual_seed(seed)
        train_weighted_auprcs = []
        train_macro_auprcs = []
        test_weighted_auprcs = []
        test_macro_auprcs = []

        # build model
        if args.load_ckpt_from:
            model = torch.load(os.path.join(args.load_ckpt_from, f"model_{seed}.pt"), map_location=device)
        else:
            model = torch.nn.Linear(
                alpha.shape[-1],
                train_cnpis.shape[-1]
            ).to(device)

        # wandb logging stuff
        if not args.no_logger:
            tags = ['Aylien theta', args.mode]
            wandb_run = wandb.init(name=f"{exp_time_stamp}_{seed}", project="covid", \
                group=exp_time_stamp, config=args, tags=tags)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.mode != "zero-shot":
            best_test_loss = None
            for epoch in tqdm(range(args.epochs), total=args.epochs):
                # train
                model.zero_grad()
                optimizer.zero_grad()

                all_preds = []
                all_labels = []

                # use the concatenation of batched countries/times because the order in training data is not preserved
                all_countries = []
                all_times = []

                train_loss = 0
                model.train()
                for batch_idx, idxs in enumerate(train_loader):
                    pred = model(torch.matmul(train_thetas[idxs], alpha))
                    loss = F.binary_cross_entropy_with_logits(pred, train_cnpis[idxs])
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    all_preds.append(pred.detach())
                    all_labels.append(train_cnpis[idxs])
                    all_countries.append(countries_dict["train"][idxs])
                    all_times.append(times_dict["train"][idxs])
                train_loss = train_loss / (batch_idx + 1)

                train_weighted_auprc, train_macro_auprc = \
                    merge_and_compute_auprc_scores(torch.cat(all_labels, dim=0), torch.cat(all_preds, dim=0), \
                        np.concatenate(all_countries), np.concatenate(all_times))

                train_weighted_auprcs.append(train_weighted_auprc)
                train_macro_auprcs.append(train_macro_auprc)

                # eval
                all_preds = []
                with torch.no_grad():
                    test_loss = 0
                    model.eval()
                    for batch_idx, idxs in enumerate(test_loader):
                        pred = model(torch.matmul(test_thetas[idxs], alpha))
                        loss = F.binary_cross_entropy_with_logits(pred, test_cnpis[idxs])
                        test_loss += loss.item()
                        all_preds.append(pred.detach())
                    test_loss = test_loss / (batch_idx + 1)

                    if args.save_ckpt and (best_test_loss is None or test_loss < best_test_loss):
                        best_test_loss = test_loss
                        torch.save(model, os.path.join(save_dir, f"model_{seed}.pt"))
                        torch.save(torch.cat(all_preds, dim=0).detach().cpu(), \
                            os.path.join(save_dir, f"test_preds_{seed}.pt"))

                    test_weighted_auprc, test_macro_auprc = \
                        merge_and_compute_auprc_scores(test_cnpis, torch.cat(all_preds, dim=0), \
                            countries_dict["test"], times_dict["test"])

                    test_weighted_auprcs.append(test_weighted_auprc)
                    test_macro_auprcs.append(test_macro_auprc)

                # wandb logging
                if not args.no_logger:
                    wandb.log({
                        "epoch": epoch, "weighted auprc/train": train_weighted_auprcs[-1], "weighted auprc/test": test_weighted_auprcs[-1],
                        "macro auprc/train": train_macro_auprcs[-1], "macro auprc/test": test_macro_auprcs[-1],
                    })
        else:
            # eval
            all_preds = []
            with torch.no_grad():
                test_loss = 0
                model.eval()
                for batch_idx, idxs in enumerate(test_loader):
                    pred = model(torch.matmul(test_thetas[idxs], alpha))
                    loss = F.binary_cross_entropy_with_logits(pred, test_cnpis[idxs])
                    test_loss += loss.item()
                    all_preds.append(pred.detach())
                test_loss = test_loss / (batch_idx + 1)

                if args.save_ckpt:
                    best_test_loss = test_loss
                    torch.save(torch.cat(all_preds, dim=0).detach().cpu(), \
                        os.path.join(save_dir, f"test_preds_{seed}.pt"))

                test_weighted_auprc, test_macro_auprc = \
                    merge_and_compute_auprc_scores(test_cnpis, torch.cat(all_preds, dim=0), \
                        countries_dict["test"], times_dict["test"])

                test_weighted_auprcs.append(test_weighted_auprc)
                test_macro_auprcs.append(test_macro_auprc)

            # wandb logging
            if not args.no_logger:
                wandb_run.summary["weighted auprc/test"] = test_weighted_auprcs[-1]
                wandb_run.summary["macro auprc/test"] = test_macro_auprcs[-1]
                
        all_train_weighted_auprcs.append(np.array(train_weighted_auprcs))
        all_train_macro_auprcs.append(np.array(train_macro_auprcs))
        all_test_weighted_auprcs.append(np.array(test_weighted_auprcs))
        all_test_macro_auprcs.append(np.array(test_macro_auprcs))

        if not args.no_logger:
            wandb_run.finish()

        del model
        if args.mode != "zero-shot":
            del optimizer

    # print and save results
    print(f"AUPRCs")
    print("Weighted:")
    if args.mode != "zero-shot":
        print(f" - train: {np.max(all_train_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_train_weighted_auprcs, axis=1).std():.3f})")
    print(f" - test: {np.max(all_test_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_test_weighted_auprcs, axis=1).std():.3f})")
    print(f"Macro:")
    if args.mode != "zero-shot":
        print(f" - train: {np.max(all_train_macro_auprcs, axis=1).mean():.3f} ({np.max(all_train_macro_auprcs, axis=1).std():.3f})")
    print(f" - test: {np.max(all_test_macro_auprcs, axis=1).mean():.3f} ({np.max(all_test_macro_auprcs, axis=1).std():.3f})")

    if args.save_ckpt:
        if args.mode != "zero-shot":
            # save the raw auprcs
            np.save(os.path.join(save_dir, "all_train_weighted_auprcs.npy"), np.vstack(all_train_weighted_auprcs))
            np.save(os.path.join(save_dir, "all_test_weighted_auprcs.npy"), np.vstack(all_test_weighted_auprcs))
            np.save(os.path.join(save_dir, "all_train_macro_auprcs.npy"), np.vstack(all_train_macro_auprcs))
            np.save(os.path.join(save_dir, "all_test_macro_auprcs.npy"), np.vstack(all_test_macro_auprcs))

        # save the aggregated auprcs
        if args.mode != "zero-shot":
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
        else:
            results = {
                "weighted": {
                    "test": f"{np.max(all_test_weighted_auprcs, axis=1).mean():.3f} ({np.max(all_test_weighted_auprcs, axis=1).std():.3f})"
                },
                "macro": {
                    "test": f"{np.max(all_test_macro_auprcs, axis=1).mean():.3f} ({np.max(all_test_macro_auprcs, axis=1).std():.3f})"
                }
            }
        with open(os.path.join(save_dir, "results.json"), "w") as file:
            json.dump(results, file)