"""
Script chia dataset thành train/val một cách ngẫu nhiên
- Val: 10,000 dòng
- Train: phần còn lại
- Shuffle toàn bộ dataset trước khi chia

Usage:
    python split_train_val.py
"""

import csv
import random
import time

# ============================================================================
# CẤU HÌNH
# ============================================================================

INPUT_FILE = "ocr_correction_dataset.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
VAL_SIZE = 200000
RANDOM_SEED = 42

print("=" * 80)
print("Chia Dataset thành Train/Val với Shuffle")
print("=" * 80)
print(f"Input: {INPUT_FILE}")
print(f"Output Train: {TRAIN_FILE}")
print(f"Output Val: {VAL_FILE}")
print(f"Val size: {VAL_SIZE:,} dòng")
print(f"Random seed: {RANDOM_SEED}")
print("=" * 80)

# ============================================================================
# BƯỚC 1: ĐẾM TỔNG SỐ DÒNG
# ============================================================================

print("\n[1/4] Đếm tổng số dòng...")
start_time = time.time()

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    # Đọc header
    header = next(f)
    # Đếm dòng data
    total_rows = sum(1 for _ in f)

print(f"  - Header: {header.strip()}")
print(f"  - Tổng số dòng data: {total_rows:,}")
print(f"  - Train sẽ có: {total_rows - VAL_SIZE:,} dòng")
print(f"  - Val sẽ có: {VAL_SIZE:,} dòng")
print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

# ============================================================================
# BƯỚC 2: TẠO RANDOM INDICES CHO VAL
# ============================================================================

print("\n[2/4] Tạo random indices cho val set...")
start_time = time.time()

# Tạo list các index từ 0 đến total_rows-1
all_indices = list(range(total_rows))

# Shuffle với seed cố định
random.seed(RANDOM_SEED)
random.shuffle(all_indices)

# Lấy 10k indices đầu tiên làm val
val_indices = set(all_indices[:VAL_SIZE])

print(f"  - Đã chọn {len(val_indices):,} indices cho val set")
print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

# ============================================================================
# BƯỚC 3: ĐỌC VÀ CHIA DATASET
# ============================================================================

print("\n[3/4] Đọc và chia dataset...")
start_time = time.time()

# Mở files để ghi
train_file = open(TRAIN_FILE, 'w', encoding='utf-8', newline='')
val_file = open(VAL_FILE, 'w', encoding='utf-8', newline='')

train_writer = csv.writer(train_file)
val_writer = csv.writer(val_file)

# Đọc input file
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    
    # Đọc và ghi header cho cả 2 files
    header = next(reader)
    train_writer.writerow(header)
    val_writer.writerow(header)
    
    # Đọc và phân chia các dòng
    train_count = 0
    val_count = 0
    
    for idx, row in enumerate(reader):
        if idx % 100000 == 0:
            print(f"  - Đã xử lý: {idx:,}/{total_rows:,} dòng ({idx/total_rows*100:.1f}%)")
        
        if idx in val_indices:
            val_writer.writerow(row)
            val_count += 1
        else:
            train_writer.writerow(row)
            train_count += 1

# Đóng files
train_file.close()
val_file.close()

print(f"  - Đã ghi {train_count:,} dòng vào train")
print(f"  - Đã ghi {val_count:,} dòng vào val")
print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

# ============================================================================
# BƯỚC 4: VERIFY
# ============================================================================

print("\n[4/4] Verify kết quả...")
start_time = time.time()

# Đếm lại số dòng trong mỗi file
def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Trừ header

train_lines = count_lines(TRAIN_FILE)
val_lines = count_lines(VAL_FILE)

print(f"  - {TRAIN_FILE}: {train_lines:,} dòng")
print(f"  - {VAL_FILE}: {val_lines:,} dòng")
print(f"  - Tổng: {train_lines + val_lines:,} dòng")

# Kiểm tra
if train_lines + val_lines == total_rows:
    print(f"  ✓ Verify thành công!")
else:
    print(f"  ✗ LỖI: Tổng số dòng không khớp!")
    print(f"    Expected: {total_rows:,}")
    print(f"    Got: {train_lines + val_lines:,}")

print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

# ============================================================================
# TỔNG KẾT
# ============================================================================

print("\n" + "=" * 80)
print("HOÀN THÀNH")
print("=" * 80)
print(f"✓ Train file: {TRAIN_FILE} ({train_lines:,} dòng)")
print(f"✓ Val file: {VAL_FILE} ({val_lines:,} dòng)")
print(f"✓ Đã shuffle với seed = {RANDOM_SEED}")
print("=" * 80)
print("\nBây giờ có thể chạy: python train_bartpho_streaming.py")
print("=" * 80)
