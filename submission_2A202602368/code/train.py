"""train.py — Vòng lặp huấn luyện, đánh giá, dự đoán và ghi kết quả thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
"""
from __future__ import annotations

import copy
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn qua thử nghiệm val (GUIDE gợi ý 0.01..0.1)
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds_list = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds_list.append(logits.argmax(dim=1))
    return torch.cat(preds_list, dim=0)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    n = len(X)
    cm = np.zeros((7, 7), dtype=np.int64)

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)

        if loss_name.lower() == "ce":
            loss_sum = F.cross_entropy(logits, yb, reduction="sum").item()
        elif loss_name.lower() == "mse":
            one_hot = F.one_hot(yb, num_classes=logits.shape[1]).float()
            loss_sum = F.mse_loss(logits, one_hot, reduction="sum").item()
        else:
            raise ValueError(f"Hàm mất mát không hỗ trợ: {loss_name}")

        total_loss += loss_sum
        pred = logits.argmax(dim=1)
        total_correct += (pred == yb).sum().item()

        yb_np = yb.cpu().numpy()
        pred_np = pred.cpu().numpy()
        np.add.at(cm, (yb_np, pred_np), 1)

    avg_loss = float(total_loss / n)
    acc = float(total_correct / n)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name.lower() == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name.lower() == "mse":
        one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
        return F.mse_loss(logits, one_hot)
    else:
        raise ValueError(f"Loss không hỗ trợ: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    seed = cfg.get("seed", 1)
    set_seed(seed)

    device = data["X_tr"].device
    is_cuda = device.type == "cuda"

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=dropout, init=init).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (
        f"Số tham số {count_params(model)} không khớp {EXPECTED_PARAMS[hidden]}"
    )

    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
        momentum=cfg.get("momentum", 0.9)
    )

    precision = cfg.get("precision", "fp32").lower()
    scaler = None
    if precision == "fp16" and is_cuda:
        scaler = torch.amp.GradScaler("cuda")

    # 1. Đo loss bước 0 trên val ở chế độ eval()
    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])["loss"]

    # Tập con cố định của train để đánh giá train_loss ở chế độ eval() (50 000 mẫu cố định)
    n_train_sub = min(50_000, len(data["X_tr"]))
    X_tr_sub = data["X_tr"][:n_train_sub]
    y_tr_sub = data["y_tr"][:n_train_sub]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)

    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "val_acc": [], "val_macro_f1": [], "grad_norm": [],
        "epoch_time_s": []
    }

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    diverged = False

    gen = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, epochs + 1):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size=batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            # Mixed precision context
            if precision == "fp16" and is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            elif precision == "bf16" and is_cuda:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), clip_norm)
                else:
                    scaler.unscale_(optimizer)
                    gn = clip_gradients(model.parameters(), None)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            grad_norms_epoch.append(gn)

        if is_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        if diverged:
            print(f"[{cfg['exp_id']}] CẢNH BÁO: Loss bị NaN/inf tại epoch {epoch}! Dừng sớm.")
            break

        # Đánh giá cuối epoch ở chế độ eval()
        tr_eval = evaluate(model, X_tr_sub, y_tr_sub, loss_name=cfg["loss"])
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])

        mean_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        # Cập nhật best state theo val_loss
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    # Thống kê tổng hợp tại best_epoch
    best_idx = best_epoch - 1 if best_epoch > 0 and best_epoch <= len(history["val_loss"]) else -1
    best_val_acc = history["val_acc"][best_idx] if best_idx >= 0 else 0.0
    best_val_macro_f1 = history["val_macro_f1"][best_idx] if best_idx >= 0 else 0.0

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")
    avg_time_per_epoch = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    peak_mem_mb = 0.0
    if is_cuda:
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(avg_time_per_epoch),
        "peak_mem_MB": float(peak_mem_mb),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    df.to_csv(path, index=False)
    print(f"Đã ghi {len(df)} dòng dự đoán eval vào {path}")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    init = cfg.get("init", "he")

    model = MLP(hidden=hidden, dropout=0.0, init=init).to(device)
    if result.get("best_state") is not None:
        model.load_state_dict(result["best_state"])
    else:
        print("CẢNH BÁO: Không có best_state trong result, dùng trọng số hiện tại!")

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
