import os
import glob
import time
import torch
import numpy as np
import multiprocessing
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

DATA_FILE = "n.csv"  # File dữ liệu chính
TRAIN_VAL_SPLIT = 0.95  # Tỉ lệ train:val = 85:15
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
LOGGING_STEPS = 100
SAVE_STEPS = 5000  # Lưu mỗi 10k steps
EVAL_STEPS = 5000   # Eval mỗi 5k steps

# DataLoader optimization - TĂNG TỐC LOAD DATA
NUM_WORKERS = 8  # Số workers để load data song song (tăng tốc training)
PREFETCH_FACTOR = 2  # Số batches mỗi worker prefetch trước
TOKENIZATION_NUM_PROC = 8  # Số processes cho tokenization song song

# GPU settings
device = "cuda" if torch.cuda.is_available() else "cpu"

# Metrics evaluation settings
MAX_EVAL_SAMPLES = 1000  # Giới hạn số samples để eval metrics (tránh quá chậm)


print("=" * 80)
print("BARTpho Training - Vietnamese Spelling Correction")
print("=" * 80)
print(f"Device: {device}")
print(f"Data file: {DATA_FILE}")
print(f"Train/Val split: {TRAIN_VAL_SPLIT*100:.0f}%/{(1-TRAIN_VAL_SPLIT)*100:.0f}%")
print(f"DataLoader workers: {NUM_WORKERS} (CPU cores: {multiprocessing.cpu_count()})")
print(f"Tokenization processes: {TOKENIZATION_NUM_PROC}")
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
    Tokenize error_text (input) và correct_text (target) on-the-fly
    """
    # Map columns: error_text -> input, correct_text -> target
    input_texts = examples['error_text']
    target_texts = examples['correct_text']
    
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
# HÀM TÍNH METRICS CHO SPELLING CORRECTION
# ============================================================================

def compute_metrics(eval_preds, tokenizer):
    """
    Tính các metrics đánh giá spelling correction:
    - BLEU score: đo lường chất lượng translation/generation
    - Exact Match Accuracy: % câu sửa chính xác 100%
    - Character Error Rate (CER): % lỗi ở mức ký tự
    - Word Error Rate (WER): % lỗi ở mức từ
    - Correction Rate: % câu được sửa (khác input)
    """
    predictions, labels = eval_preds
    
    # Decode predictions và labels
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Replace -100 trong predictions và labels (padding tokens)
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode to text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean text (strip whitespace)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    
    # 1. BLEU Score
    try:
        bleu_metric = evaluate.load("sacrebleu")
        bleu_result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        bleu_score = bleu_result["score"]
    except:
        bleu_score = 0.0
    
    # 2. Exact Match Accuracy
    exact_matches = sum(pred == label for pred, label in zip(decoded_preds, decoded_labels))
    exact_match_accuracy = exact_matches / len(decoded_preds) * 100
    
    # 3. Character Error Rate (CER)
    def calculate_cer(pred, ref):
        """Tính CER giữa 2 strings (Levenshtein distance)"""
        if len(ref) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        
        # Dynamic programming - Levenshtein distance
        d = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
        
        for i in range(len(pred) + 1):
            d[i][0] = i
        for j in range(len(ref) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred) + 1):
            for j in range(1, len(ref) + 1):
                cost = 0 if pred[i-1] == ref[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + cost  # substitution
                )
        
        return d[len(pred)][len(ref)] / len(ref)
    
    cer_scores = [calculate_cer(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    avg_cer = sum(cer_scores) / len(cer_scores) * 100
    
    # 4. Word Error Rate (WER)
    def calculate_wer(pred, ref):
        """Tính WER giữa 2 câu (Levenshtein distance trên words)"""
        pred_words = pred.split()
        ref_words = ref.split()
        
        if len(ref_words) == 0:
            return 0.0 if len(pred_words) == 0 else 1.0
        
        # Dynamic programming
        d = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]
        
        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(ref_words) + 1):
                cost = 0 if pred_words[i-1] == ref_words[j-1] else 1
                d[i][j] = min(
                    d[i-1][j] + 1,
                    d[i][j-1] + 1,
                    d[i-1][j-1] + cost
                )
        
        return d[len(pred_words)][len(ref_words)] / len(ref_words)
    
    wer_scores = [calculate_wer(pred, label) for pred, label in zip(decoded_preds, decoded_labels)]
    avg_wer = sum(wer_scores) / len(wer_scores) * 100
    
    # 5. Correction Rate (% predictions khác ground truth - tức model đã sửa)
    corrections_made = sum(pred != label for pred, label in zip(decoded_preds, decoded_labels))
    correction_rate = corrections_made / len(decoded_preds) * 100
    
    return {
        "bleu": round(bleu_score, 2),
        "exact_match": round(exact_match_accuracy, 2),
        "cer": round(avg_cer, 2),
        "wer": round(avg_wer, 2),
        "correction_rate": round(correction_rate, 2),
    }

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
    # ĐẾM SỐ DÒNG DATASET
    # ========================================================================

    print("\n[1/6] Đếm số dòng dataset...")
    start_time = time.time()
    
    total_rows = count_csv_rows(DATA_FILE)
    train_samples = int(total_rows * TRAIN_VAL_SPLIT)
    eval_samples = total_rows - train_samples

    print(f"  → Total: {total_rows:,} dòng")
    print(f"  → Train: {train_samples:,} dòng ({train_samples/total_rows*100:.2f}%)")
    print(f"  → Eval: {eval_samples:,} dòng ({eval_samples/total_rows*100:.2f}%)")
    print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")

    # ========================================================================
    # LOAD DATASET VÀ SPLIT TRAIN/VAL
    # ========================================================================

    print("\n[2/6] Load toàn bộ dataset vào RAM và split train/val...")
    start_time = time.time()

    # Load toàn bộ dataset vào RAM (không streaming)
    full_dataset = load_dataset(
        "csv",
        data_files=DATA_FILE,
        split="train"
    )
    
    print(f"  ✓ Đã load {len(full_dataset):,} dòng vào RAM")
    
    # Split thành train và val
    dataset_dict = full_dataset.train_test_split(
        test_size=1-TRAIN_VAL_SPLIT,
        seed=42,
        shuffle=True
    )
    
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']

    print(f"  ✓ Dataset loaded và split thành công trong {time.time() - start_time:.2f}s")
    print(f"  ✓ Train: {len(train_dataset):,} samples")
    print(f"  ✓ Eval: {len(eval_dataset):,} samples")

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

    print("\n[4/6] Tokenize train và val datasets...")
    start_time = time.time()

    # Tokenize train dataset với multi-processing
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
        batched=True,
        num_proc=TOKENIZATION_NUM_PROC,  # Multi-processing để tokenize nhanh
        remove_columns=["error_text", "correct_text"],
        desc="Tokenizing train dataset"
    )
    
    # Tokenize val dataset
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH),
        batched=True,
        num_proc=TOKENIZATION_NUM_PROC,
        remove_columns=["error_text", "correct_text"],
        desc="Tokenizing eval dataset"
    )

    print(f"  ✓ Train dataset: {len(train_dataset):,} samples (tokenized)")
    print(f"  ✓ Eval dataset: {len(eval_dataset):,} samples (tokenized)")
    print(f"  ✓ Hoàn thành trong {time.time() - start_time:.2f}s")
    
    # ========================================================================
    # TRAINING ARGUMENTS
    # ========================================================================

    print("\n[5/6] Thiết lập training arguments...")

    # Tính số steps dựa trên số dòng thực tế
    steps_per_epoch = len(train_dataset) // TRAIN_BATCH_SIZE
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
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        metric_for_best_model="bleu",  # Dựa vào BLEU score (tốt hơn loss)
        greater_is_better=True,  # BLEU càng cao càng tốt
        load_best_model_at_end=True,
        bf16=True,
        # Learning rate scheduler
        lr_scheduler_type="cosine",  # Cosine annealing schedule
        # Generation config for metrics
        predict_with_generate=True,  # Generate text để tính metrics
        generation_max_length=MAX_TARGET_LENGTH,
        # DataLoader optimization
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )
    
    print(f"  - Total steps: {max_steps:,}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - LR scheduler: cosine annealing")
    print(f"  - Eval every: {EVAL_STEPS:,} steps")
    print(f"  - Save every: {SAVE_STEPS:,} steps")
    print(f"  - Keep best: 3 checkpoints (based on BLEU score)")
    print(f"  - Metrics: BLEU, Exact Match, CER, WER")
    
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
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
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

