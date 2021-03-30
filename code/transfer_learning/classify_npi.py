import os, pickle, argparse, time, json
import numpy as np
from tqdm import tqdm
from itertools import compress

import torch
import torch.nn.functional as F 
from sklearn.metrics import average_precision_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Classify NPIs from topic mixtures")

    # data io args
    parser.add_argument("--eta_dir", type=str, required=True, help="directory containing eta")
    parser.add_argument("--theta_dir", type=str, help="directory containing theta")
    parser.add_argument("--who_label_dir", type=str, help="directory containing who labels")
    parser.add_argument("--cnpi_dir", type=str, help="directory containing country npis")
    parser.add_argument("--ckpt_dir", type=str, help="directory to load checkpoints from")
    parser.add_argument("--save_dir", type=str, help="root directory to save output")

    # experiments args
    parser.add_argument("--mode", type=str, required=True, \
        choices=["doc", "zero_shot", "random_init", "random_init_eta", "random_init_eta_alpha", \
            "from_scratch", "finetune"], help="running mode")
    parser.add_argument("--num_seeds", type=int, default=1, help="number of runs with different seeds")
    parser.add_argument("--save_ckpt", action="store_true", help="save best model checkpoints")

    # training args
    parser.add_argument("--lr", type=float, required=True, help="learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="batch size")
    parser.add_argument("--epochs", type=int, required=True, help="epochs")
    parser.add_argument("--weight_decay", type=float, required=True, help="weight decay")

    # other args
    parser.add_argument("--quiet", action="store_true", help="not showing progress bars")

    return parser.parse_args()

def prepare_data(*args):
    out = []
    for arg in args:
        if not isinstance(arg, torch.Tensor):
            out.append(torch.from_numpy(arg).to(device))
        else:
            out.append(arg.to(device))
    return out

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

def get_label_supports(labels):
    label_cnts = labels.sum(0).tolist()
    return np.array(label_cnts) / np.sum(label_cnts)

@torch.no_grad()
def eval_epoch(test_loader, test_data, test_labels, model, class_weight=None):
    all_preds = []
    test_loss = 0
    model.eval()
    for batch_idx, idxs in enumerate(test_loader):
        pred = model(test_data[idxs])
        if class_weight is None:
            test_loss += F.binary_cross_entropy_with_logits(pred, test_labels[idxs])
        else:
            test_loss += torch.dot(class_weight, \
                F.binary_cross_entropy_with_logits(pred, test_labels[idxs], reduction="none").mean(dim=0))
        all_preds.append(pred)
    test_loss = test_loss / (batch_idx + 1)

    auprc_breakdown = compute_auprc_breakdown(test_labels, torch.cat(all_preds, dim=0))  
    label_cnts_normed = get_label_supports(test_labels)

    return all_preds, test_loss, np.average(auprc_breakdown, weights=label_cnts_normed), np.mean(auprc_breakdown)

def run(train_data, train_labels, test_data, test_labels, configs):
    train_loader = torch.utils.data.DataLoader(
        range(train_data.shape[0]),
        batch_size=configs["batch_size"],
        num_workers=2,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        range(test_data.shape[0]),
        batch_size=configs["batch_size"],
        num_workers=2,
        shuffle=False
    )

    if configs["class_weight"]:
        class_weight = (train_labels.sum() / train_labels.sum(dim=0)) \
            / (train_labels.sum() / train_labels.sum(dim=0)).sum()
    else:
        class_weight = None

    seeds = list(range(2021, 2021 + configs["num_seeds"]))
    
    all_train_weighted_auprcs = []
    all_train_macro_auprcs = []
    all_test_weighted_auprcs = []
    all_test_macro_auprcs = []

    for seed in tqdm(seeds, disable=configs["quiet"]):
        torch.manual_seed(seed)
        train_weighted_auprcs = []
        train_macro_auprcs = []
        test_weighted_auprcs = []
        test_macro_auprcs = []

        if configs["ckpt_dir"]:
            model = torch.load(os.path.join(configs["ckpt_dir"], f"model_{seed}.pt"), map_location=device)
        else:
            model = torch.nn.Linear(
                train_data.shape[-1],
                train_labels.shape[-1],
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])

        best_test_loss = None
        for epoch in range(configs["epochs"]):
            model.zero_grad()
            optimizer.zero_grad()

            # train
            all_preds = []
            all_labels = []
            train_loss = 0
            model.train()
            for batch_idx, idxs in enumerate(train_loader):
                pred = model(train_data[idxs])
                if not configs["class_weight"]:
                    train_loss += F.binary_cross_entropy_with_logits(pred, train_labels[idxs])
                else:
                    train_loss += torch.dot(class_weight,
                                F.binary_cross_entropy_with_logits(pred, \
                                    train_labels[idxs], reduction="none").mean(dim=0))
                all_preds.append(pred.detach())
                all_labels.append(train_labels[idxs])
            train_loss = train_loss / (batch_idx + 1)
            train_loss.backward()
            optimizer.step()

            auprc_breakdown = compute_auprc_breakdown(torch.cat(all_labels, dim=0), torch.cat(all_preds, dim=0))  
            label_cnts_normed = get_label_supports(torch.cat(all_labels, dim=0))

            train_weighted_auprcs.append(np.average(auprc_breakdown, weights=label_cnts_normed))
            train_macro_auprcs.append(np.mean(auprc_breakdown))

            # eval
            test_preds, test_loss, weighted_auprc, macro_auprc = \
                eval_epoch(test_loader, test_data, test_labels, model, class_weight=class_weight)

            if configs["save_ckpt"] and (best_test_loss is None or test_loss < best_test_loss):
                best_test_loss = test_loss
                torch.save(model, os.path.join(configs["save_dir"], f"model_{seed}.pt"))
                torch.save(torch.cat(test_preds, dim=0).detach().cpu(), \
                    os.path.join(save_dir, f"test_preds_{seed}.pt"))

            test_weighted_auprcs.append(weighted_auprc)
            test_macro_auprcs.append(macro_auprc)

        del model
        del optimizer
                
        all_train_weighted_auprcs.append(np.array(train_weighted_auprcs))
        all_train_macro_auprcs.append(np.array(train_macro_auprcs))
        all_test_weighted_auprcs.append(np.array(test_weighted_auprcs))
        all_test_macro_auprcs.append(np.array(test_macro_auprcs))

    return all_train_weighted_auprcs, all_train_macro_auprcs, all_test_weighted_auprcs, all_test_macro_auprcs

if __name__ == "__main__":
    args = parse_args()
    configs = vars(args)
    if not args.quiet:
        print(args)

    # validity check
    if args.mode == "doc":
        for arg in ["who_label_dir", "theta_dir"]:
            if not configs[arg]:
                raise Exception(f"{arg} is needed for document npi prediction")
    else:
        for arg in ["cnpi_dir"]:
            if not configs[arg]:
                raise Exception(f"{arg} is needed for country npi prediction")
    if args.mode in ["zero_shot", "finetune"]:
        for arg in ["ckpt_dir"]:
            if not configs[arg]:
                raise Exception(f"{arg} is needed for zero-shot or fine-tuning")

    exp_time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {exp_time_stamp}")

    # create save directory
    if args.save_ckpt:
        save_dir = os.path.join(args.save_dir, args.mode, exp_time_stamp)
        if os.path.exists(save_dir):
            raise Exception(f"saving directory already exist: {save_dir}")
        os.makedirs(save_dir)
        configs["save_dir"] = save_dir
        print("Saving directory created")

    # load data
    alpha = np.load(os.path.join(args.eta_dir, "alpha.npy"))

    if args.mode == "doc":
        train_ratio = 0.8
        theta = np.load(os.path.join(args.theta_dir, "theta.npy"))
        labels = np.load(os.path.join(args.who_label_dir, "labels.npy"))
        
        rng = np.random.default_rng(seed=2021)
        indices = np.arange(theta.shape[0])
        rng.shuffle(indices)
        
        train_indices = indices[: int(len(indices) * train_ratio)]
        test_indices = indices[int(len(indices) * train_ratio):]
        
        train_theta, test_theta = theta[train_indices], theta[test_indices]
        train_labels, test_labels = labels[train_indices], labels[test_indices]

        train_data, test_data, train_labels, test_labels, alpha = \
            prepare_data(train_theta, test_theta, train_labels, test_labels, alpha)
    else:
        with open(os.path.join(args.cnpi_dir, "cnpis_all.pkl"), "rb") as file:
            cnpis = pickle.load(file)
        with open(os.path.join(args.cnpi_dir, "cnpi_mask.pkl"), "rb") as file:
            cnpi_mask = pickle.load(file)
        cnpis = torch.from_numpy(cnpis)
        cnpi_mask = torch.from_numpy(cnpi_mask).type('torch.LongTensor')
        cnpis = cnpis.reshape(-1, cnpis.shape[-1])
        cnpi_mask = cnpi_mask.reshape(cnpis.shape[0])
        train_cnpis = torch.stack(list(compress(cnpis, cnpi_mask)))
        test_cnpis = torch.stack(list(compress(cnpis, 1 - cnpi_mask)))

        eta = np.load(os.path.join(args.eta_dir, "eta.npy"))
        eta = torch.from_numpy(eta).reshape(-1, eta.shape[-1])
        eta = torch.softmax(eta, dim=-1)

        train_eta = torch.stack(list(compress(eta, cnpi_mask)))
        test_eta = torch.stack(list(compress(eta, 1 - cnpi_mask)))

        train_data, test_data, train_labels, test_labels, alpha = \
            prepare_data(train_eta, test_eta, train_cnpis, test_cnpis, alpha)

    train_data, test_data = torch.matmul(train_data, alpha), torch.matmul(test_data, alpha)

    # run
    if args.save_ckpt:
        with open(os.path.join(save_dir, "configs.json"), "w") as file:
            json.dump(configs, file)
    if args.mode in ["doc", "from_scratch", "finetune"]:
        # training needed
        configs = {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "num_seeds": args.num_seeds,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "save_ckpt": args.save_ckpt,
            "quiet": args.quiet,
            "class_weight": args.mode == "doc",
            "ckpt_dir": args.ckpt_dir if args.mode == "finetune" else None,
            "save_dir": save_dir if args.save_ckpt else None,
        }
        
        all_train_weighted_auprcs, all_train_macro_auprcs, all_test_weighted_auprcs, all_test_macro_auprcs = \
            run(train_data, train_labels, test_data, test_labels, configs)

        mean_max_all_test_weighted_auprcs = np.max(all_test_weighted_auprcs, axis=1).mean()
        std_max_all_test_weighted_auprcs = np.max(all_test_weighted_auprcs, axis=1).std()
        mean_max_all_test_macro_auprcs = np.max(all_test_macro_auprcs, axis=1).mean()
        std_max_all_test_macro_auprcs = np.max(all_test_macro_auprcs, axis=1).std()

        print("=" * 50)
        print(f"Average AUPRCs")
        print(f"Weighted: {mean_max_all_test_weighted_auprcs:.3f} ({std_max_all_test_weighted_auprcs:.3f})")
        print(f"Macro: {mean_max_all_test_macro_auprcs:.3f} ({std_max_all_test_macro_auprcs:.3f})")
        print("=" * 50)

        if args.save_ckpt:
            # save the raw auprcs
            np.save(os.path.join(save_dir, "all_train_weighted_auprcs.npy"), np.vstack(all_train_weighted_auprcs))
            np.save(os.path.join(save_dir, "all_test_weighted_auprcs.npy"), np.vstack(all_test_weighted_auprcs))
            np.save(os.path.join(save_dir, "all_train_macro_auprcs.npy"), np.vstack(all_train_macro_auprcs))
            np.save(os.path.join(save_dir, "all_test_macro_auprcs.npy"), np.vstack(all_test_macro_auprcs))

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

    else:
        # no training needed
        test_loader = torch.utils.data.DataLoader(
            range(test_data.shape[0]),
            batch_size=configs["batch_size"],
            num_workers=2,
            shuffle=False
        )

        all_test_weighted_auprcs = []
        all_test_macro_auprcs = []

        for seed in tqdm(range(2021, 2021 + args.num_seeds), total=args.num_seeds, disable=args.quiet):
            torch.manual_seed(seed)
            if args.mode == "zero_shot":
                model = torch.load(os.path.join(configs["ckpt_dir"], f"model_{seed}.pt"), map_location=device)
            else:
                model = torch.nn.Linear(
                    train_data.shape[-1],
                    train_labels.shape[-1],
                ).to(device)

            if args.mode in ["random_init_eta", "random_init_eta_alpha"]:
                test_eta = torch.softmax(torch.randn_like(test_eta), dim=-1)
                if args.mode == "random_init_eta_alpha":
                    torch.randn_like(alpha)
                test_data, alpha = prepare_data(test_eta, alpha)
                test_data = torch.matmul(test_data, alpha)

            test_preds, _, weighted_auprc, macro_auprc = \
                eval_epoch(test_loader, test_data, test_labels, model, class_weight=None)

            if args.save_ckpt:
                torch.save(torch.cat(test_preds, dim=0).detach().cpu(), \
                    os.path.join(save_dir, f"test_preds_{seed}.pt"))
            
            all_test_weighted_auprcs.append(weighted_auprc)
            all_test_macro_auprcs.append(macro_auprc)

        print("=" * 50)
        print(f"Average AUPRCs")
        print(f"Weighted: {np.mean(all_test_weighted_auprcs):.3f} ({np.std(all_test_weighted_auprcs):.3f})")
        print(f"Macro: {np.mean(all_test_macro_auprcs):.3f} ({np.std(all_test_macro_auprcs):.3f})")
        print("=" * 50)

        if args.save_ckpt:
            results = {
                "weighted": {
                    "test": f"{np.mean(all_test_weighted_auprcs):.3f} ({np.std(all_test_weighted_auprcs):.3f})"
                },
                "macro": {
                    "test": f"{np.mean(all_test_macro_auprcs):.3f} ({np.std(all_test_macro_auprcs):.3f})"
                }
            }
            with open(os.path.join(save_dir, "results.json"), "w") as file:
                json.dump(results, file)