import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_parameter_comparison(truth_params, generated_params, epoch=0, is_normalized=False, save_path="figures/"):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(20, 15))
    if isinstance(generated_params, torch.Tensor):
        generated_params = generated_params.cpu().numpy()
    if isinstance(truth_params, torch.Tensor):
        truth_params = truth_params.cpu().numpy()

    param_names = ['p', 'a', 'b', 'q', 'c', 'd']
    highlight_ranges = {
        'p': (0, 1),
        'a': (-1, 0),
        'b': (0, 1),
        'q': (0, 1),
        'c': (0, 1),
        'd': (0, 1)
    }

    for i in range(generated_params.shape[1]):
        plt.subplot(generated_params.shape[1], 1, i + 1)
        pname = param_names[i]
        if is_normalized:
            param_range = (0, 1)
        else:
            if i % 3 == 0:
                param_range = (0, 3.0)
            elif i % 3 == 1:
                param_range = (-1, 1)
            else:
                param_range = (0, 5.0)
        plt.hist(generated_params[:, i], bins=50, range=param_range, histtype='step', color='red', label='Generated')
        print(f"Param {pname}: std={np.std(generated_params[:, i]):.4f}, mean={np.mean(generated_params[:, i]):.4f}")
        truth_val = truth_params[0, i] if truth_params.ndim > 1 else truth_params[i]
        plt.axvline(x=truth_val, color='blue', label='Truth')
        if pname in highlight_ranges:
            xmin, xmax = highlight_ranges[pname]
            plt.axvspan(xmin, xmax, color='yellow', alpha=0.3, label='Target Range' if i == 0 else None)
        plt.title(f'Parameter {pname}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.suptitle(f'Parameter Comparison (Epoch {epoch})', fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"{save_path}/param_comparison_epoch_{epoch}.png", dpi=300)
    plt.close()


