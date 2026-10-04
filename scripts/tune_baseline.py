import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath("code"))

from data import prepare_data
from train import DEFAULT_CFG, run_experiment

data = prepare_data("cuda", processed_dir="data/processed")

for lr in [0.01, 0.02, 0.05, 0.1]:
    cfg = {**DEFAULT_CFG, "lr": lr, "epochs": 5}
    res = run_experiment(cfg, data)
    print(f"lr={lr}: val_acc={res['summary']['val_acc']:.4f}, val_macro_f1={res['summary']['val_macro_f1']:.4f}, val_loss={res['summary']['best_val_loss']:.4f}")
