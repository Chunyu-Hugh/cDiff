import os
import csv
from typing import Optional


class Records:
    def __init__(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.csv_path = os.path.join(ckpt_dir, 'metrics_history.csv')
        self.events_path = os.path.join(ckpt_dir, 'metrics_log.txt')
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['step', 'loss', 'lr', 'coverage', 'bias'])

    def log_loss_lr(self, step: int, loss: float, lr: float):
        if step is None:
            return
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([step, f"{loss:.6f}", f"{lr:.8f}", '', ''])

    def log_eval(self, step: int, coverage: float, bias: float):
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([step, '', '', f"{coverage:.6f}", f"{bias:.6f}"])

    def log_event(self, tag: str, step: int, coverage: float, bias: float, filename: Optional[str] = None):
        with open(self.events_path, 'a', encoding='utf-8') as f:
            file_field = f"\tfile={os.path.basename(filename)}" if filename else ''
            f.write(f"{tag}\tstep={step}\tcoverage={coverage:.6f}\tbias={bias:.6f}{file_field}\n")


