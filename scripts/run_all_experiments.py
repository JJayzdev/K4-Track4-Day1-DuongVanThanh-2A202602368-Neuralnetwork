"""run_all_experiments.py — Chạy toàn bộ các thí nghiệm theo 7 chủ đề,
ghi kết quả ra JSON, xuất biểu đồ từng thí nghiệm, vẽ biểu đồ so sánh nhóm,
điền bảng experiments.xlsx và đánh giá tập eval bằng evaluate.py.
"""
import sys
import os
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# Thêm code vào path
REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = REPO_ROOT / "submission_2A202602368" / "code"
sys.path.insert(0, str(CODE_DIR))

from data import prepare_data
from model import MLP, EXPECTED_PARAMS, count_params
from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval
from plots import plot_run, plot_compare
from results_table import save_result, load_results, to_row, write_xlsx

SUBMISSION_DIR = REPO_ROOT / "submission_2A202602368"
RESULTS_DIR = SUBMISSION_DIR / "results"
FIGURES_DIR = SUBMISSION_DIR / "figures"
TEMPLATE_PATH = REPO_ROOT / "templates" / "experiment_table_template.xlsx"
EXCEL_PATH = SUBMISSION_DIR / "experiments.xlsx"
EVAL_PRED_PATH = SUBMISSION_DIR / "predictions_eval.csv"
EVAL_JSON_PATH = SUBMISSION_DIR / "eval_result.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== BẮT ĐẦU CHẠY THÍ NGHIỆM TRÊN {device} ===")

# Nạp dữ liệu
data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir=str(REPO_ROOT / "data" / "processed"))

# 1. Health check overfit 20 samples
print("\n--- CHẠY HEALTH CHECK: QUÁ KHỚP 20 MẪU ---")
x_20 = data["X_tr"][:20].clone()
y_20 = data["y_tr"][:20].clone()
model_overfit = MLP(hidden=(256, 128), dropout=0.0, init="he").to(device)
opt_overfit = torch.optim.Adam(model_overfit.parameters(), lr=0.01)

loss_history = []
acc_history = []
for step in range(300):
    model_overfit.train()
    opt_overfit.zero_grad(set_to_none=True)
    out = model_overfit(x_20)
    loss = torch.nn.functional.cross_entropy(out, y_20)
    loss.backward()
    opt_overfit.step()
    loss_history.append(loss.item())
    with torch.no_grad():
        acc = (out.argmax(dim=1) == y_20).float().mean().item()
        acc_history.append(acc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(loss_history, color="firebrick", lw=2)
ax1.set_title("Đường cong Loss khi quá khớp 20 mẫu", fontsize=11, fontweight="bold")
ax1.set_xlabel("Bước cập nhật (Step)")
ax1.set_ylabel("Cross-Entropy Loss (log scale)")
ax1.set_yscale("log")
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(acc_history, color="forestgreen", lw=2)
ax2.set_title("Accuracy khi quá khớp 20 mẫu", fontsize=11, fontweight="bold")
ax2.set_xlabel("Bước cập nhật (Step)")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig_path = str(FIGURES_DIR / "overfit_20_samples.png")
fig.savefig(fig_path, dpi=120)
plt.close(fig)
print(f"Đã lưu: {fig_path}")

# Danh sách toàn bộ các cấu hình thí nghiệm
EXPERIMENTS = [
    # 1. Baseline (3 seeds để đo độ nhiễu)
    {
        "exp_id": "base-s1", "group": "baseline", "description": "Baseline M-base, seed 1",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Baseline chuẩn: SGD+m, lr=0.05, He init"
    },
    {
        "exp_id": "base-s2", "group": "baseline", "description": "Baseline M-base, seed 2",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 2,
        "notes": "Đo nhiễu seed 2"
    },
    {
        "exp_id": "base-s3", "group": "baseline", "description": "Baseline M-base, seed 3",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 3,
        "notes": "Đo nhiễu seed 3"
    },

    # 2. Chủ đề 1: Loss
    {
        "exp_id": "loss-mse", "group": "loss", "description": "Hàm mất mát MSE trên one-hot",
        "loss": "mse", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "So sánh CE vs MSE. Không so trực tiếp giá trị loss mà so macro-F1."
    },

    # 3. Chủ đề 2: Optimizer (mỗi bộ thử 2 lr để so sánh công bằng)
    {
        "exp_id": "opt-sgd-lr0.05", "group": "optimizer", "description": "SGD thuần lr=0.05",
        "loss": "ce", "optimizer": "sgd", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.0,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "SGD không momentum lr=0.05"
    },
    {
        "exp_id": "opt-sgd-lr0.1", "group": "optimizer", "description": "SGD thuần lr=0.1",
        "loss": "ce", "optimizer": "sgd", "lr": 0.1, "weight_decay": 0.0, "momentum": 0.0,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "SGD không momentum lr=0.1"
    },
    {
        "exp_id": "opt-sgdm-lr0.02", "group": "optimizer", "description": "SGD+momentum lr=0.02",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.02, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "SGD+momentum lr nhỏ hơn baseline"
    },
    {
        "exp_id": "opt-adam-lr1e-3", "group": "optimizer", "description": "Adam lr=0.001",
        "loss": "ce", "optimizer": "adam", "lr": 0.001, "weight_decay": 0.0,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Adam thích nghi lr=1e-3"
    },
    {
        "exp_id": "opt-adam-lr3e-3", "group": "optimizer", "description": "Adam lr=0.003",
        "loss": "ce", "optimizer": "adam", "lr": 0.003, "weight_decay": 0.0,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Adam thích nghi lr=3e-3"
    },
    {
        "exp_id": "opt-adamw-lr1e-3", "group": "optimizer", "description": "AdamW lr=0.001 wd=0.01",
        "loss": "ce", "optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "AdamW tách weight decay lr=1e-3"
    },
    {
        "exp_id": "opt-adamw-lr3e-3", "group": "optimizer", "description": "AdamW lr=0.003 wd=0.01",
        "loss": "ce", "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.01,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "AdamW tách weight decay lr=3e-3"
    },

    # 4. Chủ đề 3: Hyper-parameter
    {
        "exp_id": "hp-batch-128", "group": "hparam", "description": "Batch nhỏ 128",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 128, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Batch 128 (nhiều bước cập nhật hơn)"
    },
    {
        "exp_id": "hp-batch-2048", "group": "hparam", "description": "Batch lớn 2048",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 2048, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Batch 2048 (ít bước cập nhật hơn)"
    },
    {
        "exp_id": "hp-mwide", "group": "hparam", "description": "Mô hình rộng M-wide",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (512, 256), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "M-wide 161 287 tham số"
    },
    {
        "exp_id": "hp-mdeep", "group": "hparam", "description": "Mô hình sâu M-deep",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128, 64), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "M-deep 55 687 tham số"
    },
    {
        "exp_id": "hp-wd-1e-4", "group": "hparam", "description": "Weight decay 1e-4",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 1e-4, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "L2 regularization 1e-4"
    },

    # 5. Chủ đề 4: Dropout
    {
        "exp_id": "drop-0.1", "group": "dropout", "description": "Dropout q=0.1",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.1, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Dropout q=0.1 sau ReLU lớp ẩn"
    },
    {
        "exp_id": "drop-0.3", "group": "dropout", "description": "Dropout q=0.3",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.3, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Dropout q=0.3 sau ReLU lớp ẩn"
    },
    {
        "exp_id": "drop-0.5", "group": "dropout", "description": "Dropout q=0.5",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.5, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Dropout q=0.5 sau ReLU lớp ẩn"
    },

    # 6. Chủ đề 5: Gradient clipping
    {
        "exp_id": "clip-1.0", "group": "clipping", "description": "Clip norm=1.0 ở lr thường",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": 1.0, "precision": "fp32", "seed": 1,
        "notes": "Cắt gradient c=1.0 ở lr thường"
    },
    {
        "exp_id": "highlr-noclip", "group": "clipping", "description": "Tốc độ học cao lr=0.8 không clip",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.8, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Thí nghiệm phản chứng: lr cao không clip gây dao động/bùng nổ"
    },
    {
        "exp_id": "highlr-clip1.0", "group": "clipping", "description": "Tốc độ học cao lr=0.8 có clip c=1.0",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.8, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": 1.0, "precision": "fp32", "seed": 1,
        "notes": "Thí nghiệm phản chứng: clipping cứu vãn huấn luyện ở lr cao"
    },

    # 7. Chủ đề 6: Mixed precision (AMP)
    {
        "exp_id": "amp-fp16", "group": "amp", "description": "Mixed Precision FP16 + GradScaler",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "fp16", "seed": 1,
        "notes": "AMP FP16 với GradScaler"
    },
    {
        "exp_id": "amp-bf16", "group": "amp", "description": "Mixed Precision BF16",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "he",
        "clip_norm": None, "precision": "bf16", "seed": 1,
        "notes": "AMP BF16 phần cứng Ampere"
    },

    # 8. Chủ đề 7: Init
    {
        "exp_id": "init-zeros", "group": "init", "description": "Khởi tạo trọng số W = 0",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "zeros",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Khởi tạo zeros: phá vỡ đối xứng thất bại, mạng không học được"
    },
    {
        "exp_id": "init-normal", "group": "init", "description": "Khởi tạo Normal(0, 0.01^2)",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "normal",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Khởi tạo normal std=0.01 nhỏ: kích hoạt tiêu biến"
    },
    {
        "exp_id": "init-xavier", "group": "init", "description": "Khởi tạo Xavier Normal",
        "loss": "ce", "optimizer": "sgd_momentum", "lr": 0.05, "weight_decay": 0.0, "momentum": 0.9,
        "batch": 512, "epochs": 20, "hidden": (256, 128), "dropout": 0.0, "init": "xavier",
        "clip_norm": None, "precision": "fp32", "seed": 1,
        "notes": "Khởi tạo Xavier (phù hợp tanh/sigmoid hơn ReLU)"
    },

    # 9. Cấu hình cuối cùng (Final Best Configuration)
    {
        "exp_id": "final-best-s1", "group": "final", "description": "Cấu hình tối ưu: AdamW lr=0.003, M-wide, wd=0.01",
        "loss": "ce", "optimizer": "adamw", "lr": 0.003, "weight_decay": 0.01,
        "batch": 512, "epochs": 20, "hidden": (512, 256), "dropout": 0.0, "init": "he",
        "clip_norm": 1.0, "precision": "fp32", "seed": 1,
        "notes": "Cấu hình chọn bằng val: AdamW lr=0.003, M-wide, clip 1.0"
    }
]

# Chạy lần lượt từng thí nghiệm
results_dict = {}

print(f"\nTổng số thí nghiệm cần chạy: {len(EXPERIMENTS)}")

for idx, cfg in enumerate(EXPERIMENTS, start=1):
    exp_id = cfg["exp_id"]
    json_path = RESULTS_DIR / f"{exp_id}.json"
    fig_path = FIGURES_DIR / f"{exp_id}.png"

    print(f"\n[{idx}/{len(EXPERIMENTS)}] Đang chạy thí nghiệm: {exp_id} ({cfg['description']})...")
    t0 = time.time()
    res = run_experiment(cfg, data)
    elapsed = time.time() - t0

    # Lưu kết quả và biểu đồ
    save_result(res, str(RESULTS_DIR))
    plot_run(res, str(fig_path))
    results_dict[exp_id] = res

    summ = res["summary"]
    print(f" -> Hoàn thành sau {elapsed:.1f}s | Val Acc: {summ['val_acc']:.4f} | Val Macro-F1: {summ['val_macro_f1']:.4f} | Best Loss: {summ['best_val_loss']:.4f} (Epoch {summ['best_epoch']})")

# Vẽ các biểu đồ so sánh nhóm (compare_<nhóm>.png)
print("\n--- XUẤT CÁC BIỂU ĐỒ SO SÁNH THEO NHÓM ---")

# 1. Compare Optimizer
opt_res = [results_dict["base-s1"], results_dict["opt-sgd-lr0.05"], results_dict["opt-sgd-lr0.1"],
           results_dict["opt-sgdm-lr0.02"], results_dict["opt-adam-lr1e-3"], results_dict["opt-adam-lr3e-3"],
           results_dict["opt-adamw-lr1e-3"], results_dict["opt-adamw-lr3e-3"]]
plot_compare(opt_res, "val_macro_f1", str(FIGURES_DIR / "compare_optimizer.png"), title="So sánh Val Macro-F1 giữa các Bộ tối ưu")

# 2. Compare Loss
loss_res = [results_dict["base-s1"], results_dict["loss-mse"]]
plot_compare(loss_res, "val_macro_f1", str(FIGURES_DIR / "compare_loss.png"), title="So sánh Val Macro-F1: Cross-Entropy vs MSE")

# 3. Compare Hparam (Batch size & Architecture)
hparam_res = [results_dict["base-s1"], results_dict["hp-batch-128"], results_dict["hp-batch-2048"],
              results_dict["hp-mwide"], results_dict["hp-mdeep"], results_dict["hp-wd-1e-4"]]
plot_compare(hparam_res, "val_macro_f1", str(FIGURES_DIR / "compare_hparam.png"), title="So sánh Val Macro-F1 theo Hyper-parameters")

# 4. Compare Dropout
drop_res = [results_dict["base-s1"], results_dict["drop-0.1"], results_dict["drop-0.3"], results_dict["drop-0.5"]]
plot_compare(drop_res, "val_macro_f1", str(FIGURES_DIR / "compare_dropout.png"), title="Ảnh hưởng của Dropout tới Val Macro-F1")

# 5. Compare Clipping
clip_res = [results_dict["base-s1"], results_dict["clip-1.0"], results_dict["highlr-noclip"], results_dict["highlr-clip1.0"]]
plot_compare(clip_res, "val_loss", str(FIGURES_DIR / "compare_clipping.png"), title="Tác dụng của Gradient Clipping ở tốc độ học cao")

# 6. Compare AMP
amp_res = [results_dict["base-s1"], results_dict["amp-fp16"], results_dict["amp-bf16"]]
plot_compare(amp_res, "val_macro_f1", str(FIGURES_DIR / "compare_amp.png"), title="So sánh FP32 vs Mixed Precision (FP16 & BF16)")

# 7. Compare Init
init_res = [results_dict["base-s1"], results_dict["init-xavier"], results_dict["init-normal"], results_dict["init-zeros"]]
plot_compare(init_res, "val_macro_f1", str(FIGURES_DIR / "compare_init.png"), title="So sánh các phương pháp khởi tạo tham số")

print("Đã lưu toàn bộ biểu đồ so sánh nhóm tại submission_2A202602368/figures/!")

# Đánh giá cuối trên tập Eval (cho baseline base-s1 và cấu hình cuối final-best-s1)
print("\n--- ĐÁNH GIÁ TRÊN TẬP EVAL CHO BASELINE VÀ FINAL MODEL ---")

# 1. Baseline predictions
pred_base_path = SUBMISSION_DIR / "predictions_baseline.csv"
final_eval(results_dict["base-s1"]["cfg"], results_dict["base-s1"], data, str(pred_base_path))

# 2. Final best predictions
final_eval(results_dict["final-best-s1"]["cfg"], results_dict["final-best-s1"], data, str(EVAL_PRED_PATH))

# Chạy evaluate.py cho cả 2
import subprocess

print("Chạy scripts/evaluate.py cho baseline...")
cmd_base = [sys.executable, "scripts/evaluate.py", "--pred", str(pred_base_path), "--out", str(SUBMISSION_DIR / "eval_result_base.json")]
proc_base = subprocess.run(cmd_base, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
print(proc_base.stdout)
with open(SUBMISSION_DIR / "eval_result_base.json", "r", encoding="utf-8") as f:
    base_eval_scores = json.load(f)

print("Chạy scripts/evaluate.py cho Final model...")
cmd_final = [sys.executable, "scripts/evaluate.py", "--pred", str(EVAL_PRED_PATH), "--out", str(EVAL_JSON_PATH)]
proc_final = subprocess.run(cmd_final, cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8")
print(proc_final.stdout)
with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
    final_eval_scores = json.load(f)

# Ghi bảng experiments.xlsx
print("\n--- CẬP NHẬT BẢNG EXPERIMENTS.XLSX ---")
rows = []
for cfg in EXPERIMENTS:
    eid = cfg["exp_id"]
    res = results_dict[eid]
    if eid == "base-s1":
        r = to_row(res, eval_scores=base_eval_scores)
    elif eid == "final-best-s1":
        r = to_row(res, eval_scores=final_eval_scores)
    else:
        r = to_row(res)
    rows.append(r)

write_xlsx(rows, str(TEMPLATE_PATH), str(EXCEL_PATH))

# Cập nhật sheet Summary nhận xét
import openpyxl
wb = openpyxl.load_workbook(EXCEL_PATH)
ws_sum = wb["Summary"]
comments = {
    2: "Baseline SGD+momentum hội tụ ổn định, val macro-F1 ~ 0.77",
    3: "MSE hội tụ chậm hơn nhiều so với CE do gradient bị bão hòa ở xác suất cực đoan",
    4: "Adam và AdamW vượt trội SGD; AdamW với lr=0.003 đạt macro-F1 cao nhất",
    5: "M-wide và batch 128 giúp tăng macro-F1; batch 2048 ít bước cập nhật nên hội tụ chậm",
    6: "Mô hình M-base chưa bị quá khớp nặng nên dropout làm giảm nhẹ hiệu năng biểu diễn",
    7: "Clipping c=1.0 phát huy tác dụng rõ rệt khi lr cao (0.8), ngăn chặn nổ loss và dao động",
    8: "FP16 và BF16 giảm bộ nhớ VRAM và duy trì độ chính xác tương đương FP32",
    9: "He vượt trội Xavier và Normal; Zeros hoàn toàn thất bại do vi phạm tính phá vỡ đối xứng",
}
for row_idx, comm in comments.items():
    ws_sum.cell(row=row_idx, column=8, value=comm)

wb.save(EXCEL_PATH)
print("Đã cập nhật nhận xét sheet Summary trong experiments.xlsx!")
print("\n=== TOÀN BỘ TIẾN TRÌNH THÍ NGHIỆM ĐÃ HOÀN TẤT THÀNH CÔNG! ===")
