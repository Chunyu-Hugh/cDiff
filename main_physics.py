import torch
import torch.optim as optim
import numpy as np
from models.neural_sampler import NormalizingFlowPosteriorSampler, DiffusionPosteriorSampler
#from evaluation.SBC import sample_sbc_calstats, evaluate_sbc
#from evaluation.TARP import get_ecp_area_difference
#from utils import *
import pandas as pd
import time
import argparse
from scipy.integrate import quad
from scipy import interpolate
from torch.utils.data import TensorDataset, DataLoader
import os

# Physical parameter ranges (training)
pmin, pmax = 0.0, 1.0
amin, amax = -1.0, 0.0
bmin, bmax = 0.0, 1.0
qmin, qmax = 0.0, 1.0
cmin, cmax = 0.0, 1.0
dmin, dmax = 0.0, 1.0

get_u = lambda x, a, b, p: p * x ** a * (1 - x) ** b
get_d = lambda x, a, b, q: q * x ** a * (1 - x) ** b

def get_sigma1(x, p):
    u = get_u(x, p[1], p[2], p[0])
    d = get_d(x, p[4], p[5], p[3])
    return 4 * u + d

def get_sigma2(x, p):
    u = get_u(x, p[1], p[2], p[0])
    d = get_d(x, p[4], p[5], p[3])
    return 4 * d + u

def gen_events(sigma, nevents, xmin=0.1, xmax=1.0):
    norm = quad(sigma, xmin, xmax)[0]
    pdf = lambda x: sigma(x) / norm
    get_cdf = lambda x: quad(pdf, x, xmax)[0]
    xs = np.linspace(xmin, xmax, 200)
    invcdf = interpolate.interp1d([get_cdf(_) for _ in xs], xs, bounds_error=False, fill_value=(xmin, xmax))
    u = np.random.uniform(0, 1, nevents)
    events = invcdf(u)
    return events, norm, pdf

def sample_theta():
    p = np.random.uniform(pmin, pmax)
    a = np.random.uniform(amin, amax)
    b = np.random.uniform(bmin, bmax)
    q = np.random.uniform(qmin, qmax)
    c = np.random.uniform(cmin, cmax)
    d = np.random.uniform(dmin, dmax)
    return {"p": p, "a": a, "b": b, "q": q, "c": c, "d": d}

def sample_physics_data(parms_dict, sample_size):
    theta = np.array([
        parms_dict["p"], parms_dict["a"], parms_dict["b"],
        parms_dict["q"], parms_dict["c"], parms_dict["d"]
    ])
    s1_events, _, _ = gen_events(lambda x: get_sigma1(x, theta), nevents=sample_size)
    s2_events, _, _ = gen_events(lambda x: get_sigma2(x, theta), nevents=sample_size)
    return np.stack([s1_events, s2_events], axis=1)

def generate_raw_events_dataset(N_samples=10000, events_per_sample=200, seed=42):
    np.random.seed(seed)
    all_theta = []
    all_events = []
    for i in range(N_samples):
        parms = sample_theta()
        events = sample_physics_data(parms, events_per_sample)  # shape = [events_per_sample, 2]
        all_theta.append([parms[k] for k in ['p', 'a', 'b', 'q', 'c', 'd']])
        all_events.append(events)  # shape = [events_per_sample, 2]
        if (i+1) % 500 == 0:
            print(f"Generated {i+1}/{N_samples} samples.")
    theta_arr = np.array(all_theta, dtype=np.float32)
    events_arr = np.array(all_events, dtype=np.float32) # [N_samples, events_per_sample, 2]
    dataset = TensorDataset(torch.tensor(theta_arr), torch.tensor(events_arr))
    return dataset

def trainer(data_loader, model, optimizer, scheduler, epochs, device, lr_decay, eval_interval, save_path):
    loss_record = []
    training_time_record = []
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = []
        temport_num = 0
        for batch in data_loader:
            theta, y = batch
            y = y.to(device)
            theta = theta.to(device)

            optimizer.zero_grad()
            loss = model.loss(x=theta, y=y).mean()
            epoch_loss.append(float(loss))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            temport_num += 1
            print(f"Batch {temport_num} of {len(data_loader)}")
        if lr_decay:
            scheduler.step()
        print(
            f"Epoch: {epoch + 1}/{epochs},",
            f"Loss: {np.mean(epoch_loss):.2f},",
            f"LR: {scheduler.get_last_lr()[0]:.4f}"
        )
        loss_record.append(np.mean(epoch_loss))
        training_time_record.append(time.time() - start_time)
        # 可加保存模型
        # if epoch % eval_interval == 0: save_model(model, save_path, epoch)
    epochs_ = list(range(1, len(loss_record)+1))
    df_loss = pd.DataFrame({
        'epochs': epochs_,
        'loss': loss_record,
        'training_time': training_time_record
    })
    return model, df_loss

def main(args):
    # Dataset parameters
    train_size = args.train_size
    test_size  = args.test_size
    batch_size = args.batch_size
    events_per_sample = args.events_per_sample
    seed = args.seed

    # Model parameters
    hidden_dim_summary_net = 32
    n_summaries = 256
    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    alpha = args.alpha

    # Optimizer parameters
    epochs = args.epochs
    lr = args.lr
    lr_decay = args.lr_decay

    eval_interval = args.eval_interval

    # 1. 数据集生成
    print("Generating training set...")
    train_dataset = generate_raw_events_dataset(N_samples=train_size, events_per_sample=events_per_sample, seed=seed)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Generating test set...")
    test_dataset = generate_raw_events_dataset(N_samples=test_size, events_per_sample=events_per_sample, seed=seed+2024)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. 模型定义
    y_dim = events_per_sample * 2
    x_dim = 6
    model_type = args.model
    if model_type == "NormalizingFlow":
        model = NormalizingFlowPosteriorSampler(
            y_dim=y_dim, x_dim=x_dim, n_summaries=n_summaries,
            hidden_dim_decoder=hidden_dim_summary_net, n_flows_decoder=32, alpha=alpha, device=DEVICE
        ).to(DEVICE)
    elif model_type == "Diffusion":
        # （自己根据模型需要定 y_dim）
        num_hidden_layer = args.num_hidden_layer
        model = DiffusionPosteriorSampler(
            y_dim=y_dim, x_dim=x_dim, n_summaries=n_summaries, num_hidden_layer=num_hidden_layer,
            device=DEVICE, sigma_data=0.5 # 可自行指定
        )
    else:
        raise NotImplementedError

    # 3. 训练
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    print("Start training...")
    model, df_loss = trainer(
        train_loader, model, optimizer, optimizer_sched, epochs, DEVICE,
        lr_decay=lr_decay, eval_interval=eval_interval, save_path=args.save_path
    )
    # 保存loss曲线
    df_loss.to_csv(os.path.join(args.save_path, "train_loss.csv"), index=False)
    print("Training finished, loss history saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physics raw event NPE training")

    # Dataset parameters
    parser.add_argument('--train_size', type=int, default=8000, help="Number of training samples")
    parser.add_argument('--test_size', type=int, default=2000, help="Number of test samples")
    parser.add_argument('--events_per_sample', type=int, default=200, help="Events per spectrum")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)

    # Model parameters
    parser.add_argument('--model', type=str, default="Diffusion", help="NormalizingFlow or Diffusion")
    parser.add_argument('--alpha', type=float, default=0.1, help="Lipschitz param for NF")
    parser.add_argument('--num_hidden_layer',type=int, default=4, help="Number of hidden layers for diffusion model")

    # Optimizer/training parameters
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_path', type=str, default="result")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    main(args)