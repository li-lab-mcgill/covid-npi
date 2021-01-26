import os, argparse, time, json
import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from mixmedia import MixMedia

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Infer theta given BOW and eta")
    parser.add_argument("--data_dir", type=str, required=True, help="directory containing eta and bow")
    parser.add_argument("--save_dir", type=str, required=True, help="root directory to save output")
    parser.add_argument("--model_dir", type=str, required=True, help="directory containing the trained mixmedia model")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    return parser.parse_args()

@torch.no_grad()
def get_batch_theta(batch_eta, batch_bow, model):
    model.eval()
    batch_input = torch.cat([batch_bow, batch_eta], dim=1)
    batch_mu_theta = model.mu_q_theta(model.q_theta(batch_input))
    return F.softmax(batch_mu_theta, dim=-1)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    config = vars(args)

    exp_time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    print(f"Experiment time stamp: {exp_time_stamp}")

    # load etas
    eta = np.load(os.path.join(args.data_dir, "merge_etas.npy"))

    # load bow
    bow = sparse.load_npz(os.path.join(args.data_dir, "bow.npz")).toarray()
    print("Data loaded")

    # load model
    with open(os.path.join(args.model_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    model = model.to(device)
    print("Model loaded")

    # infer theta
    theta = []
    eta = torch.from_numpy(eta).type(torch.float).to(device)
    bow = torch.from_numpy(bow).type(torch.float).to(device)

    data_loader = torch.utils.data.DataLoader(
        range(len(eta)),
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False
    )

    for batch_idxs in tqdm(data_loader, total=len(data_loader)):
        batch_eta, batch_bow = eta[batch_idxs], bow[batch_idxs]

        theta.append(get_batch_theta(batch_eta, batch_bow, model))
        
    theta = torch.cat(theta, dim=0)

    # save to disk
    if not os.path.exists(os.path.join(args.save_dir, exp_time_stamp)):
        os.makedirs(os.path.join(args.save_dir, exp_time_stamp))

    with open(os.path.join(os.path.join(args.save_dir, exp_time_stamp), 'config.json'), 'w') as file:
        json.dump(config, file)
    
    theta = theta.detach().cpu().numpy()
    np.save(os.path.join(os.path.join(args.save_dir, exp_time_stamp), f"theta.npy"), theta)