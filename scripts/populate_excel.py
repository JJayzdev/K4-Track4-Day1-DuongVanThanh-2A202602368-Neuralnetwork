import sys
import json
from pathlib import Path
import openpyxl

sys.stdout.reconfigure(encoding="utf-8")
REPO_ROOT = Path(".").resolve()
sys.path.insert(0, str(REPO_ROOT / "submission_2A202602368" / "code"))

from results_table import to_row, write_xlsx

SUBMISSION_DIR = REPO_ROOT / "submission_2A202602368"
RESULTS_DIR = SUBMISSION_DIR / "results"
TEMPLATE_PATH = REPO_ROOT / "templates" / "experiment_table_template.xlsx"
EXCEL_PATH = SUBMISSION_DIR / "experiments.xlsx"

with open(SUBMISSION_DIR / "eval_result_base.json", "r", encoding="utf-8") as f:
    base_eval_scores = json.load(f)

with open(SUBMISSION_DIR / "eval_result.json", "r", encoding="utf-8") as f:
    final_eval_scores = json.load(f)

# Đọc toàn bộ results
results_dict = {}
for p in RESULTS_DIR.glob("*.json"):
    with open(p, "r", encoding="utf-8") as f:
        res = json.load(f)
        results_dict[res["cfg"]["exp_id"]] = res

# Danh sách thứ tự các thí nghiệm
ORDERED_EXP_IDS = [
    # 1. Baseline
    "base-s1", "base-s2", "base-s3",
    # 2. Loss
    "loss-mse",
    # 3. Optimizer
    "opt-sgd-lr0.05", "opt-sgd-lr0.1", "opt-sgdm-lr0.02",
    "opt-adam-lr1e-3", "opt-adam-lr3e-3",
    "opt-adamw-lr1e-3", "opt-adamw-lr3e-3",
    # 4. Hparam
    "hp-batch-128", "hp-batch-2048", "hp-mwide", "hp-mdeep", "hp-wd-1e-4",
    # 5. Dropout
    "drop-0.1", "drop-0.3", "drop-0.5",
    # 6. Clipping
    "clip-1.0", "highlr-noclip", "highlr-clip1.0",
    # 7. AMP
    "amp-fp16", "amp-bf16",
    # 8. Init
    "init-zeros", "init-normal", "init-xavier",
    # 9. Final
    "final-best-s1"
]

rows = []
for eid in ORDERED_EXP_IDS:
    res = results_dict[eid]
    if eid == "base-s1":
        r = to_row(res, eval_scores=base_eval_scores, notes="Baseline chuẩn: SGD+momentum, lr=0.05, He init")
    elif eid == "final-best-s1":
        r = to_row(res, eval_scores=final_eval_scores, notes="Cấu hình tối ưu chọn bằng val: AdamW lr=0.003, M-wide, clip 1.0")
    else:
        r = to_row(res)
    rows.append(r)

write_xlsx(rows, str(TEMPLATE_PATH), str(EXCEL_PATH))

# Cập nhật sheet Summary và Seeds
wb = openpyxl.load_workbook(EXCEL_PATH)
ws_sum = wb["Summary"]
comments = {
    2: "Baseline SGD+momentum hội tụ tốt (val macro-F1 ~ 0.8368 ± 0.0029, 2σ = 0.0057)",
    3: "MSE hội tụ chậm hơn nhiều so với CE do gradient bị bão hòa ở xác suất cực đoan",
    4: "Adam và AdamW vượt trội SGD; Adam lr=3e-3 và AdamW lr=3e-3 đạt macro-F1 cao nhất (~0.87)",
    5: "M-wide và batch 128 giúp tăng macro-F1; batch 2048 ít bước cập nhật nên hội tụ chậm",
    6: "Mô hình M-base chưa bị quá khớp nên dropout làm giảm nhẹ hiệu năng biểu diễn",
    7: "Clipping c=1.0 phát huy tác dụng rõ rệt khi lr cao (0.8), ngăn chặn nổ loss và dao động",
    8: "FP16 và BF16 giảm bộ nhớ VRAM và duy trì độ chính xác tương đương FP32",
    9: "He vượt trội Xavier và Normal; Zeros hoàn toàn thất bại do vi phạm tính phá vỡ đối xứng",
}
for row_idx, comm in comments.items():
    ws_sum.cell(row=row_idx, column=8, value=comm)

wb.save(EXCEL_PATH)
print("Đã tạo thành công experiments.xlsx với đầy đủ dữ liệu và công thức!")
