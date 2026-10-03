"""data.py — Chuẩn bị và xử lý dữ liệu CoverType cho thí nghiệm huấn luyện.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = Path(processed_dir) / "train.npz"
    eval_path = Path(processed_dir) / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file npz tại {processed_dir}. Hãy chạy `python scripts/split_data.py` trước."
        )

    with np.load(train_path) as tr:
        X_train_full = tr["X"].astype(np.float32)
        y_train_full = tr["y"].astype(np.int64)

    with np.load(eval_path) as ev:
        X_eval = ev["X"].astype(np.float32)
        y_eval = ev["y"].astype(np.int64)
        eval_row_id = ev["row_id"].astype(np.int64)

    # Kiểm tra kích thước và kiểu dữ liệu
    assert X_train_full.shape == (464809, 54), f"Lỗi shape train X: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"Lỗi shape train y: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"Lỗi shape eval X: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Lỗi shape eval y: {y_eval.shape}"
    assert eval_row_id.shape == (116203,), f"Lỗi shape eval_row_id: {eval_row_id.shape}"

    assert X_train_full.dtype == np.float32 and X_eval.dtype == np.float32
    assert y_train_full.dtype == np.int64 and y_eval.dtype == np.int64
    assert 0 <= y_train_full.min() and y_train_full.max() <= 6
    assert 0 <= y_eval.min() and y_eval.max() <= 6

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do: Không được tính trên eval hay val để tránh rò rỉ dữ liệu (data leakage).
    """
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    # Tránh chia cho 0 nếu có đặc trưng có phương sai = 0
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ để tránh side effect.
    """
    X_scaled = X.copy()
    safe_std = np.where(std == 0.0, 1.0, std)
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / safe_std
    return X_scaled


def prepare_data(device: str | torch.device, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    dev = torch.device(device)

    # 1. Nạp và tách dữ liệu
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    # 2. Chuẩn hoá 10 đặc trưng số liên tục CHỈ dựa trên thống kê train
    mean, std = fit_standardizer(X_tr)
    X_tr_norm = apply_standardizer(X_tr, mean, std)
    X_val_norm = apply_standardizer(X_val, mean, std)
    X_eval_norm = apply_standardizer(X_eval, mean, std)

    # 3. Đưa lên device dạng PyTorch tensor
    t_X_tr = torch.tensor(X_tr_norm, dtype=torch.float32, device=dev)
    t_y_tr = torch.tensor(y_tr, dtype=torch.int64, device=dev)
    t_X_val = torch.tensor(X_val_norm, dtype=torch.float32, device=dev)
    t_y_val = torch.tensor(y_val, dtype=torch.int64, device=dev)
    t_X_eval = torch.tensor(X_eval_norm, dtype=torch.float32, device=dev)
    t_y_eval = torch.tensor(y_eval, dtype=torch.int64, device=dev)

    # 4. In thông tin kiểm tra
    print(f"=== Thống kê phân chia dữ liệu ===")
    print(f"Train tập con : X={t_X_tr.shape}, y={t_y_tr.shape}")
    print(f"Validation    : X={t_X_val.shape}, y={t_y_val.shape}")
    print(f"Eval          : X={t_X_eval.shape}, y={t_y_eval.shape}")
    print(f"Thiết bị      : {dev}")

    # Kiểm tra mean và std của 10 cột liên tục trên X_tr
    mean_check = t_X_tr[:, :N_NUMERIC].mean(dim=0).cpu().numpy()
    std_check = t_X_tr[:, :N_NUMERIC].std(dim=0).cpu().numpy()
    print(f"Max abs mean của 10 cột số trên train: {np.abs(mean_check).max():.6f} (kỳ vọng ≈ 0)")
    print(f"Max abs (std - 1) của 10 cột số trên train: {np.abs(std_check - 1.0).max():.6f} (kỳ vọng ≈ 0)")

    # Baseline "luôn đoán lớp đa số" trên val
    val_bincount = torch.bincount(t_y_val, minlength=7)
    maj_class = torch.argmax(val_bincount).item()
    maj_acc = (t_y_val == maj_class).float().mean().item()
    print(f"Lớp đa số trên validation: Lớp {maj_class} (chiếm {val_bincount[maj_class].item()}/{len(t_y_val)} mẫu)")
    print(f"Accuracy của chiến lược đoán lớp đa số trên val: {maj_acc:.4f} (mốc sàn ≈ 0.4876)\n")

    return {
        "X_tr": t_X_tr, "y_tr": t_y_tr,
        "X_val": t_X_val, "y_val": t_y_val,
        "X_eval": t_X_eval, "y_eval": t_y_eval,
        "eval_row_id": eval_row_id,
        "mean": mean, "std": std
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Ghi chú xử lý lô cuối: Nếu số mẫu không chia hết cho batch_size, lô cuối cùng sẽ có kích thước nhỏ hơn
    batch_size (N % batch_size) nhưng vẫn được đưa vào huấn luyện để không bỏ sót mẫu dữ liệu nào.
    """
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
