"""results_table.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    p = Path(results_dir)
    p.mkdir(parents=True, exist_ok=True)
    exp_id = result["cfg"]["exp_id"]
    out_file = p / f"{exp_id}.json"

    data_to_save = {
        "cfg": result["cfg"],
        "history": result["history"],
        "summary": result["summary"]
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    return str(out_file)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    p = Path(results_dir)
    if not p.exists():
        return []
    results = []
    for f in sorted(p.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            results.append(data)
    return results


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg = result["cfg"]
    summary = result.get("summary", {})
    exp_id = cfg["exp_id"]

    hidden_val = cfg.get("hidden", (256, 128))
    if isinstance(hidden_val, (list, tuple)):
        hidden_str = "-".join(str(h) for h in hidden_val)
    else:
        hidden_str = str(hidden_val)

    clip_val = cfg.get("clip_norm", None)
    clip_str = "none" if clip_val is None else str(clip_val)

    opt_val = cfg.get("optimizer", "sgd_momentum")
    opt_map = {
        "sgd": "SGD",
        "sgd_momentum": "SGD+momentum",
        "adam": "Adam",
        "adamw": "AdamW"
    }
    opt_str = opt_map.get(opt_val.lower(), opt_val)

    loss_val = cfg.get("loss", "ce").upper()

    eval_acc = None
    eval_macro_f1 = None
    if eval_scores is not None:
        eval_acc = eval_scores.get("accuracy")
        eval_macro_f1 = eval_scores.get("macro_f1")

    row = {
        "exp_id": exp_id,
        "group": cfg.get("group", "other"),
        "description": cfg.get("description", ""),
        "loss": loss_val,
        "optimizer": opt_str,
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "batch": cfg.get("batch", 512),
        "epochs": cfg.get("epochs", 20),
        "hidden": hidden_str,
        "dropout": cfg.get("dropout", 0.0),
        "clip_norm": clip_str,
        "precision": cfg.get("precision", "fp32"),
        "init": cfg.get("init", "he"),
        "seed": cfg.get("seed", 1),
        "step0_loss": summary.get("step0_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "val_acc": summary.get("val_acc"),
        "val_macro_f1": summary.get("val_macro_f1"),
        "time_per_epoch_s": summary.get("time_per_epoch_s"),
        "peak_mem_MB": summary.get("peak_mem_MB"),
        "diverged": summary.get("diverged", False),
        "eval_acc": eval_acc,
        "eval_macro_f1": eval_macro_f1,
        "figure_file": f"figures/{exp_id}.png",
        "notes": notes or cfg.get("notes", "")
    }
    return row


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path."""
    import openpyxl

    wb = openpyxl.load_workbook(template_path)
    ws = wb["Experiments"]

    header_cols = {}
    for col_idx in range(1, 35):
        val = ws.cell(row=1, column=col_idx).value
        if val is not None:
            header_cols[val] = col_idx

    formula_cols = {"step0_gap_vs_lnC", "gap_val_minus_train", "delta_val_f1_vs_base", "beyond_noise"}

    for r_idx, row_data in enumerate(rows, start=2):
        for k, v in row_data.items():
            if k in header_cols and k not in formula_cols:
                c_idx = header_cols[k]
                ws.cell(row=r_idx, column=c_idx, value=v)

    # Đảm bảo thư mục lưu tồn tại
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    print(f"Đã lưu bảng thí nghiệm tại: {out_path}")

