import torch
import torch.nn.functional as F

def get_alpha_schedule(T: int):
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

def train_step(model, optimizer, x0, norm, param_true, T, device):
    model.train()
    optimizer.zero_grad()
    beta, alpha, alpha_bar = get_alpha_schedule(T)
    # Ensure schedule tensors are on the same device as indices/tensors used
    beta = beta.to(device)
    alpha = alpha.to(device)
    alpha_bar = alpha_bar.to(device)

    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=device)
    ab_t = alpha_bar[t].view(-1, 1).to(device)

    noise = torch.randn_like(param_true)
    x_t = torch.sqrt(ab_t) * param_true + torch.sqrt(1 - ab_t) * noise

    noise_pred = model(x0, norm, x_t, t)
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    return loss.item()


