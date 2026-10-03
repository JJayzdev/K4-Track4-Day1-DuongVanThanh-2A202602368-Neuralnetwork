import json
import sys

paths = [
    'submission_2A202602368/code/lab.ipynb',
    'code/lab.ipynb'
]

setup_source = [
    "# ===== Cấu hình đường dẫn và môi trường =====\n",
    "import os, sys, json, time, subprocess\n",
    "from pathlib import Path\n",
    "import numpy as np, torch\n",
    "\n",
    "# Xử lý đường dẫn linh hoạt (hỗ trợ cả chạy trên Colab và chạy local)\n",
    "if 'google.colab' in sys.modules:\n",
    "    REPO_NAME = 'K4-Track4-Day1-DuongVanThanh-2A202602368-Neuralnetwork'\n",
    "    if not os.path.exists(f'/content/{REPO_NAME}'):\n",
    "        subprocess.run(['git', 'clone', f'https://github.com/JJayzdev/{REPO_NAME}.git'], check=True)\n",
    "    REPO_ROOT = f'/content/{REPO_NAME}'\n",
    "    OUT_DIR = f'{REPO_ROOT}/submission_2A202602368'\n",
    "    os.chdir(f'{OUT_DIR}/code')\n",
    "    if f'{OUT_DIR}/code' not in sys.path:\n",
    "        sys.path.insert(0, f'{OUT_DIR}/code')\n",
    "else:\n",
    "    REPO_ROOT = '../..'     # thư mục gốc repo, tính từ submission_<MSSV>/code/\n",
    "    OUT_DIR   = '..'        # nơi ghi figures/, results/, experiments.xlsx, predictions_eval.csv\n",
    "    if '.' not in sys.path:\n",
    "        sys.path.insert(0, '.')\n",
    "\n",
    "# Thiết bị: Ưu tiên CUDA nếu có GPU, ngược lại CPU (trên Mac có thể mps)\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))\n",
    "print(f'PyTorch version: {torch.__version__}')\n",
    "print(f'Thiết bị đang sử dụng: {device}')\n",
    "if device.type == 'cuda':\n",
    "    print(f'Tên GPU: {torch.cuda.get_device_name(0)}')\n",
    "\n",
    "# Đảm bảo các thư mục đầu ra tồn tại\n",
    "os.makedirs(f'{OUT_DIR}/figures', exist_ok=True)\n",
    "os.makedirs(f'{OUT_DIR}/results', exist_ok=True)\n",
    "\n",
    "from data import prepare_data, iterate_batches\n",
    "from model import MLP, EXPECTED_PARAMS, count_params, init_weights, activation_stats\n",
    "from optimizer import build_optimizer, clip_gradients\n",
    "from train import DEFAULT_CFG, set_seed, evaluate, predict, run_experiment, final_eval\n",
    "from plots import plot_run, plot_compare\n",
    "from results_table import save_result, load_results, to_row, write_xlsx\n"
]

part0_source = [
    "# 1. Chạy split_data.py nếu chưa có train.npz và eval.npz\n",
    "processed_dir = os.path.join(REPO_ROOT, 'data', 'processed')\n",
    "train_npz = os.path.join(processed_dir, 'train.npz')\n",
    "eval_npz = os.path.join(processed_dir, 'eval.npz')\n",
    "\n",
    "if not (os.path.exists(train_npz) and os.path.exists(eval_npz)):\n",
    "    print('Đang chạy scripts/split_data.py để tạo dữ liệu ban đầu...')\n",
    "    subprocess.run([sys.executable, 'scripts/split_data.py'], cwd=REPO_ROOT, check=True)\n",
    "else:\n",
    "    print('Dữ liệu train.npz và eval.npz đã sẵn sàng.')\n",
    "\n",
    "# 2. Nạp dữ liệu, tách validation 20% stratified seed 42, chuẩn hóa và chuyển lên device\n",
    "data = prepare_data(device=device, val_fraction=0.2, seed=42, processed_dir=processed_dir)\n",
    "\n",
    "# 3. Trích xuất các tensor và kiểm tra kích thước / kiểu dữ liệu\n",
    "X_tr, y_tr = data['X_tr'], data['y_tr']\n",
    "X_val, y_val = data['X_val'], data['y_val']\n",
    "X_eval, y_eval = data['X_eval'], data['y_eval']\n",
    "\n",
    "print('--- KIỂM TRA ĐẦU RA PART 0 ---')\n",
    "print(f'Kích thước X_tr   : {X_tr.shape}, kiểu: {X_tr.dtype}')\n",
    "print(f'Kích thước y_tr   : {y_tr.shape}, kiểu: {y_tr.dtype}')\n",
    "print(f'Kích thước X_val  : {X_val.shape}, kiểu: {X_val.dtype}')\n",
    "print(f'Kích thước y_val  : {y_val.shape}, kiểu: {y_val.dtype}')\n",
    "print(f'Kích thước X_eval : {X_eval.shape}, kiểu: {X_eval.dtype}')\n",
    "print(f'Kích thước y_eval : {y_eval.shape}, kiểu: {y_eval.dtype}')\n",
    "print(f'Số lượng eval_row_id: {len(data[\"eval_row_id\"])}')\n",
    "\n",
    "# Kiểm tra shape theo tiêu chuẩn quy định trong GUIDE\n",
    "assert X_tr.shape == (371847, 54), f'Kỳ vọng 371847 mẫu train nhưng nhận {X_tr.shape[0]}'\n",
    "assert X_val.shape == (92962, 54), f'Kỳ vọng 92962 mẫu val nhưng nhận {X_val.shape[0]}'\n",
    "assert X_eval.shape == (116203, 54), f'Kỳ vọng 116203 mẫu eval nhưng nhận {X_eval.shape[0]}'\n",
    "assert X_tr.dtype == torch.float32 and y_tr.dtype == torch.int64\n",
    "assert X_val.dtype == torch.float32 and y_val.dtype == torch.int64\n",
    "assert X_eval.dtype == torch.float32 and y_eval.dtype == torch.int64\n",
    "assert len(data['eval_row_id']) == 116203\n",
    "\n",
    "# 4. Kiểm tra chuẩn hóa 10 cột số liên tục trên X_tr (mean ~ 0, std ~ 1)\n",
    "numeric_means = X_tr[:, :10].mean(dim=0).cpu().numpy()\n",
    "numeric_stds = X_tr[:, :10].std(dim=0).cpu().numpy()\n",
    "print('\\n--- KIỂM TRA THỐNG KÊ 10 CỘT LIÊN TỤC TRÊN X_tr ---')\n",
    "print(f'Mean của 10 cột liên tục:\\n{np.round(numeric_means, 4)}')\n",
    "print(f'Std của 10 cột liên tục:\\n{np.round(numeric_stds, 4)}')\n",
    "assert np.allclose(numeric_means, 0.0, atol=1e-2), 'Mean của 10 cột trên train chưa chuẩn hóa về 0!'\n",
    "assert np.allclose(numeric_stds, 1.0, atol=1e-2), 'Std của 10 cột trên train chưa chuẩn hóa về 1!'\n",
    "print('=> Chuẩn hóa 10 cột số liên tục đạt yêu cầu (mean ≈ 0, std ≈ 1). 44 cột nhị phân giữ nguyên.')\n",
    "\n",
    "# 5. Đo accuracy của chiến lược luôn đoán lớp đa số trên tập validation\n",
    "val_counts = torch.bincount(y_val, minlength=7)\n",
    "maj_class = torch.argmax(val_counts).item()\n",
    "majority_acc = (y_val == maj_class).float().mean().item()\n",
    "print(f'\\n--- MỐC SÀN DỰ ĐOÁN ĐA SỐ (MAJORITY BASELINE) ---')\n",
    "print(f'Lớp đa số trên validation: Lớp {maj_class} ({val_counts[maj_class].item()}/{len(y_val)} mẫu, chiếm {val_counts[maj_class].item()/len(y_val)*100:.2f}%)')\n",
    "print(f'Accuracy đoán lớp đa số trên val: {majority_acc:.4f} (Mốc sàn tối thiểu mô hình phải vượt ≈ 0.4876)')\n"
]

part0_md = [
    "### Nhận xét Part 0 (Tự viết):\n",
    "1. **Kích thước và phân chia tập dữ liệu**:\n",
    "   - Dữ liệu gốc gồm 581 012 mẫu được chia thành `train` (464 809 mẫu) và `eval` (116 203 mẫu) cố định theo `split_metadata.csv`.\n",
    "   - Tách validation 20% phân tầng theo nhãn từ tập train (seed 42) thu được: **371 847 mẫu train** và **92 962 mẫu validation**.\n",
    "   - Mảng `eval_row_id` được bảo toàn nguyên vẹn đúng thứ tự (116 203 mẫu) để đảm bảo sinh file dự đoán `predictions_eval.csv` chính xác ở Part 4.\n",
    "2. **Cơ chế chuẩn hóa và phòng tránh rò rỉ dữ liệu (Data Leakage)**:\n",
    "   - Chỉ chuẩn hóa 10 đặc trưng số liên tục đầu tiên bằng z-score ($z = (x - \\mu)/\\sigma$); 44 đặc trưng nhị phân One-hot phía sau được giữ nguyên.\n",
    "   - Thông số $(\\mu, \\sigma)$ **bắt buộc chỉ được tính trên tập train rút gọn** (sau khi tách validation) và áp dụng cùng thông số này cho validation và eval.\n",
    "   - **Lý do không được dùng validation hay eval để tính mean/std**: Nếu tính mean/std trên toàn bộ tập dữ liệu hoặc trên eval, mô hình sẽ tiếp cận trước phân phối thống kê của tập đánh giá (data leakage), làm sai lệch tính khách quan và khả năng tổng quát hóa thực tế của mô hình.\n",
    "3. **Chiến lược đoán lớp đa số (Majority Baseline)**:\n",
    "   - Do mất cân bằng lớp (lớp 1 chiếm 48.76%), nếu mô hình luôn dự đoán lớp 1 thì đạt Accuracy $\\approx 0.4876$ trên validation, nhưng Macro-F1 chỉ $\\approx 0.094$.\n",
    "   - Mọi mô hình huấn luyện sau này bắt buộc phải vượt xa mốc sàn Accuracy này, và thước đo đánh giá chất lượng cốt lõi của bài toán là **Macro-F1**."
]

for p in paths:
    with open(p, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cell 1 is setup
    nb['cells'][1]['source'] = setup_source
    # Cell 3 is part 0 code
    nb['cells'][3]['source'] = part0_source
    
    # Check if there is already a Part 0 comment cell after cell 3, if not insert it
    if len(nb['cells']) > 4 and nb['cells'][4]['cell_type'] == 'markdown' and 'Nhận xét Part 0' in ''.join(nb['cells'][4]['source']):
        nb['cells'][4]['source'] = part0_md
    else:
        new_cell = {
            'cell_type': 'markdown',
            'metadata': {},
            'source': part0_md
        }
        nb['cells'].insert(4, new_cell)
        
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Updated {p}')
