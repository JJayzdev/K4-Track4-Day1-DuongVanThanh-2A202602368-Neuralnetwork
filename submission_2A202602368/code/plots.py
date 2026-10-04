"""plots.py — Vẽ đồ thị huấn luyện và so sánh thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    cfg = result["cfg"]
    hist = result["history"]
    summary = result.get("summary", {})
    best_epoch = summary.get("best_epoch", -1)
    epochs = hist["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Subplot 1: Train & Val Loss
    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="royalblue", lw=2)
    ax.plot(epochs, hist["val_loss"], label="Val Loss", color="crimson", lw=2)
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best epoch ({best_epoch})")
    ax.set_title("Hàm mất mát (Loss)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Subplot 2: Val Accuracy & Macro-F1
    ax = axes[1]
    ax.plot(epochs, hist["val_acc"], label="Val Accuracy", color="forestgreen", lw=2)
    ax.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="darkorange", lw=2)
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best epoch ({best_epoch})")
    ax.set_title("Độ chính xác & Macro-F1", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    # Subplot 3: Grad Norm (trước khi clip)
    ax = axes[2]
    ax.plot(epochs, hist["grad_norm"], label="Grad Norm (trước clip)", color="purple", lw=2)
    if best_epoch in epochs:
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.8, label=f"Best epoch ({best_epoch})")
    ax.set_title("Chuẩn Gradient (Grad Norm)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("L2 Norm", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)

    title_str = (
        f"[{cfg.get('exp_id', 'exp')}] {cfg.get('description', '')}\n"
        f"Opt: {cfg.get('optimizer')} | lr: {cfg.get('lr')} | Batch: {cfg.get('batch')} | "
        f"Loss: {cfg.get('loss')} | Init: {cfg.get('init')} | Drop: {cfg.get('dropout')} | "
        f"Clip: {cfg.get('clip_norm')} | Precision: {cfg.get('precision')}"
    )
    fig.suptitle(title_str, fontsize=11, y=1.03)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ 'val_loss', 'val_macro_f1', 'grad_norm') của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in results:
        cfg = r["cfg"]
        hist = r["history"]
        if metric in hist and hist[metric]:
            label = f"{cfg['exp_id']} ({cfg.get('optimizer', '')}, lr={cfg.get('lr', '')})"
            ax.plot(hist["epoch"], hist[metric], marker="o", ms=3, label=label, lw=1.8)

    ax.set_title(title if title else f"So sánh {metric}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
