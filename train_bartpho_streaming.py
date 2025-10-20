"""
Script huấn luyện BARTpho với STREAMING từ CSV lớn (37GB, 110M dòng)
Load dataset on-the-fly, không cần load toàn bộ vào RAM

Author: AI Assistant
Date: 2025-10-20

Yêu cầu:
pip install transformers datasets torch
"""

import os
import glob
import time
import torch
import multiprocessing
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# ============================================================================
# CẤU HÌNH
# ============================================================================

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
MODEL_NAME = "vinai/bartpho-syllable-base"
OUTPUT_DIR = "./bartpho_vsc"
FINAL_MODEL_DIR = "./bartpho_vsc_model"

# Hyperparameters
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
NUM_EPOCHS = 5
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WARMUP_STEPS = 10000
LOGGING_STEPS = 2000
SAVE_STEPS = 20000  # Lưu mỗi 10k steps
EVAL_STEPS = 2000   # Eval mỗi 5k steps

# Streaming settings
SHUFFLE_BUFFER = 50000      # Buffer để shuffle (tăng nếu RAM đủ)

# DataLoader optimization - TĂNG TỐC LOAD DATA
# Với streaming dataset (1 shard), set NUM_WORKERS = 0 hoặc 1 để tránh warning
NUM_WORKERS = 0  # Streaming dataset chỉ có 1 shard, không cần nhiều workers
PREFETCH_FACTOR = 2         # Số batches mỗi worker prefetch trước
TOKENIZATION_BATCH_SIZE = 1000  # Batch size cho tokenization (lớn hơn = nhanh hơn)

# GPU settings
device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 80)
print("BARTpho Streaming Training - Load Dataset On-The-Fly")
print("=" * 80)
print(f"Device: {device}")
print(f"Train file: {TRAIN_FILE}")
print(f"Val file: {VAL_FILE}")
print(f"Shuffle buffer: {SHUFFLE_BUFFER:,}")
print(f"DataLoader workers: {NUM_WORKERS} (CPU cores: {multiprocessing.cpu_count()})")
print(f"Prefetch factor: {PREFETCH_FACTOR}")
print("=" * 80)

# ============================================================================
# HÀM ĐẾM SỐ DÒNG CSV
# ============================================================================

def count_csv_rows(csv_file):
    """
    Đếm số dòng trong CSV một cách nhanh chóng
    Không load toàn bộ data vào RAM
    """
    try:
        # Cách nhanh nhất: đọc file và đếm dòng
        with open(csv_file, 'r', encoding='utf-8') as f:
            # Bỏ qua header
            next(f)
            row_count = sum(1 for _ in f)

        return row_count

    except Exception as e:
        print(f"  ✗ Lỗi khi đếm {csv_file}: {e}")
        print("  ! Vui lòng kiểm tra file CSV")
        exit(1)

# ============================================================================
# HÀM TÌM CHECKPOINT MỚI NHẤT
# ============================================================================

def find_latest_checkpoint(output_dir):
    """
    Tìm checkpoint mới nhất trong thư mục output
    
    Args:
        output_dir: Đường dẫn đến thư mục chứa checkpoints
    
    Returns:
        str hoặc None: Đường dẫn đến checkpoint mới nhất, hoặc None nếu không có
    """
    if not os.path.exists(output_dir):
        return None
    
    # Tìm tất cả các thư mục checkpoint-*
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if not checkpoints:
        return None
    
    # Sort theo số step (lấy từ tên thư mục)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    latest_checkpoint = checkpoints[-1]
    
    return latest_checkpoint

# ============================================================================
# HÀM TOKENIZATION
# ============================================================================

def preprocess_function(examples, tokenizer, max_input_length, max_target_length):
    """
    Tokenize input_text và target_text on-the-fly
    """
    # Ensure inputs are valid strings (handle None, NaN, non-string types)
    input_texts = examples['input_text']
    target_texts = examples['target_text']
    
    # Convert to list if not already
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    if not isinstance(target_texts, list):
        target_texts = [target_texts]
    
    # Filter and convert to strings, replace invalid with empty string
    valid_inputs = [str(x) if x is not None and x != '' and str(x) != 'nan' else '' for x in input_texts]
    valid_targets = [str(x) if x is not None and x != '' and str(x) != 'nan' else '' for x in target_texts]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        valid_inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False
    )

    # Tokenize targets
    labels = tokenizer(
        valid_targets,
        max_length=max_target_length,
        truncation=True,
        padding=False
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# ============================================================================
# MAIN TRAINING
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # TỐI ƯU CPU CHO TOKENIZATION
    # ========================================================================
    
    # Tăng số threads cho PyTorch operations (tokenization, data processing)
    torch.set_num_threads(multiprocessing.cpu_count())
    print(f"\n⚡ CPU Optimization: Using {multiprocessing.cpu_count()} threads for tokenization")

    # ========================================================================
    # ĐẾM SỐ DÒNG TRAIN VÀ VAL
    # ========================================================================

    print("\n[1/6] Đếm số dòng train và val...")
    start_time = time.time()
    
    train_samples = count_csv_rows(TRAIN_FILE)
    eval_samples = count_csv_rows(VAL_FILE)
    total_rows = train_samples + eval_samples

    print(f"  → Train: {train_samples:,} dòng ({train_samples/total_rows*100:.2f}%)")
    print(f"  → Eval: {eval_samples:,} dòng ({eval_samples/total_rows*100:.2f}%)")
    print(f"  → Total: {total_rows:,} dòng")
    print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

    # ========================================================================
    # LOAD TRAIN VÀ VAL DATASET WITH STREAMING
    # ========================================================================

    print("\n[2/6] Load train và val dataset với streaming mode...")
    start_time = time.time()

    # Load train CSV với streaming
    train_dataset = load_dataset(
        "csv",
        data_files=TRAIN_FILE,
        streaming=True,
        split="train"
    )
    
    # Load val CSV với streaming
    val_dataset = load_dataset(
        "csv",
        data_files=VAL_FILE,
        streaming=True,
        split="train"
    )

    print(f"  ✓ Datasets streaming đã sẵn sàng trong {time.time() - start_time:.2f}s")

    # ========================================================================
    # TẢI TOKENIZER VÀ MODEL
    # ========================================================================

    print("\n[3/6] Tải tokenizer và model...")
    start_time = time.time()

    # Kiểm tra xem có checkpoint cũ không để load model weights
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    
    if latest_checkpoint:
        step_number = int(latest_checkpoint.split("-")[-1])
        print(f"  ⚡ Tìm thấy checkpoint: {latest_checkpoint}")
        print(f"  ⚡ Step: {step_number:,}")
        print(f"  ⚡ Load MODEL WEIGHTS từ checkpoint (train từ đầu, không resume)")
        
        # Load model weights từ checkpoint
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint)
    else:
        print(f"  → Load pretrained model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if device == "cuda":
        model = model.to(device)

    print(f"  - Model parameters: {model.num_parameters():,}")
    print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # PREPROCESS DATASETS
    # ========================================================================

    print("\n[4/6] Preprocess train và val datasets...")
    start_time = time.time()

    # Shuffle train dataset với buffer (val không cần shuffle)
    train_dataset = train_dataset.shuffle(seed=42, buffer_size=SHUFFLE_BUFFER)

    # Tokenize train dataset on-the-fly với batch lớn hơn để tăng tốc
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
        batched=True,
        batch_size=TOKENIZATION_BATCH_SIZE,  # Tokenize nhiều samples cùng lúc
        remove_columns=["input_text", "target_text"]
    )
    
    # Tokenize val dataset on-the-fly (không shuffle)
    eval_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
        batched=True,
        batch_size=TOKENIZATION_BATCH_SIZE,
        remove_columns=["input_text", "target_text"]
    )

    print(f"  - Train dataset: {train_samples:,} samples")
    print(f"  - Eval dataset: {eval_samples:,} samples")
    print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # TRAINING ARGUMENTS
    # ========================================================================

    print("\n[5/6] Thiết lập training arguments...")

    # Tính số steps dựa trên số dòng thực tế
    steps_per_epoch = train_samples // TRAIN_BATCH_SIZE
    max_steps = steps_per_epoch * NUM_EPOCHS
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=max_steps,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",  # ← BẮT BUỘC để chạy eval!
        save_strategy="steps",  # Lưu theo steps
        save_total_limit=3,  # Giữ 3 checkpoint tốt nhất
        metric_for_best_model="eval_loss",  # Dựa vào eval_loss
        greater_is_better=False,  # Loss càng thấp càng tốt
        load_best_model_at_end=True,  # Load model tốt nhất khi kết thúc
        bf16=True,
        # DataLoader optimization - TĂNG TỐC!
        dataloader_num_workers=NUM_WORKERS,  # Dùng nhiều workers để load song song
        dataloader_prefetch_factor=PREFETCH_FACTOR,  # Prefetch nhiều batches
        dataloader_pin_memory=True,  # Pin memory cho GPU transfer nhanh hơn
        remove_unused_columns=False,
        report_to="none",
    )
    
    print(f"  - Total steps: {max_steps:,}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - Eval every: {EVAL_STEPS:,} steps")
    print(f"  - Save every: {SAVE_STEPS:,} steps")
    print(f"  - Keep best: 3 checkpoints (based on eval_loss)")
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    print("  ✓ Training arguments đã được thiết lập")
    
    # ========================================================================
    # TRAINING
    # ========================================================================

    print("\n[6/6] Bắt đầu training...")
    print(f"  - Train từ đầu (step 0) với model weights đã load")
    print(f"  - Ước tính thời gian: ~{max_steps / (LOGGING_STEPS * 10):.1f} giờ")
    print("-" * 80)
    
    training_start_time = time.time()
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train từ đầu (không resume, chỉ dùng model weights đã load)
    train_result = trainer.train()
    
    training_time = time.time() - training_start_time
    print("-" * 80)
    print(f"  ✓ Training hoàn thành trong {training_time / 3600:.2f} giờ")
    
    # ========================================================================
    # LƯU MODEL
    # ========================================================================

    print("\nLưu model...")
    
    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    
    print(f"  ✓ Model đã lưu tại: {FINAL_MODEL_DIR}")
    
    # ========================================================================
    # TỔNG KẾT
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TỔNG KẾT")
    print("=" * 80)
    print(f"✓ Tổng dữ liệu: {total_rows:,} dòng")
    print(f"✓ Train: {train_samples:,} dòng ({train_samples/total_rows*100:.2f}%)")
    print(f"✓ Eval: {eval_samples:,} dòng ({eval_samples/total_rows*100:.2f}%)")
    print(f"✓ Thời gian training: {training_time / 3600:.2f} giờ")
    print(f"✓ Total steps: {max_steps:,}")
    print(f"✓ Model đã lưu: {FINAL_MODEL_DIR}")
    print("=" * 80)
    print("\nĐể test model: python inference_bartpho.py")
    print("=" * 80)

