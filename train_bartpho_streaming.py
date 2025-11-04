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
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback
)
import evaluate

DATA_FILE = "spelling_errors.csv"  # File dữ liệu chính
TRAIN_VAL_SPLIT = 0.95  # Tỉ lệ train:val = 85:15
MODEL_NAME = "vinai/bartpho-syllable"
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

# Example sentences for inference after validation
EXAMPLE_SENTENCES = [
    "Đây ià cac phưong tiện vi phạm được camera ghi hình.",
    "Phổ biến nhat ià ioi đỗ không đúng nơi quy dịnh.",
    "Tôi đang học tap tiéng viét ơ trưòng đai hoc.",
    "Hom nay troi mua rat to, toi khong the di hoc duoc."
]


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
    Tính các metrics đánh giá spelling correction (tối ưu với evaluate library):
    - BLEU score: đo lường chất lượng translation/generation
    - Exact Match Accuracy: % câu sửa chính xác 100%
    - Character Error Rate (CER): % lỗi ở mức ký tự
    - Word Error Rate (WER): % lỗi ở mức từ
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
    
    # 3. Character Error Rate (CER) - Dùng evaluate library (nhanh hơn)
    try:
        cer_metric = evaluate.load("cer")
        avg_cer = cer_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        ) * 100
    except:
        avg_cer = 0.0
    
    # 4. Word Error Rate (WER) - Dùng evaluate library (nhanh hơn)
    try:
        wer_metric = evaluate.load("wer")
        avg_wer = wer_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels
        ) * 100
    except:
        avg_wer = 0.0
    
    return {
        "bleu": round(bleu_score, 2),
        "exact_match": round(exact_match_accuracy, 2),
        "cer": round(avg_cer, 2),
        "wer": round(avg_wer, 2),
    }

# ============================================================================
# HÀM INFERENCE CHO EXAMPLE SENTENCES
# ============================================================================

def run_inference_examples(model, tokenizer, examples, device, num_beams=15):
    """
    Chạy inference trên các câu example và log kết quả
    """
    model.eval()
    print("\n" + "=" * 80)
    print("INFERENCE EXAMPLES")
    print("=" * 80)
    
    for i, text in enumerate(examples, 1):
        # Tokenize
        inputs = tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Chuyển sang device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        
        # Decode
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Log
        print(f"\n[{i}] Input:  {text}")
        print(f"    Output: {corrected_text}")
    
    print("=" * 80 + "\n")

# ============================================================================
# CALLBACK ĐỂ CHẠY INFERENCE SAU EVAL
# ============================================================================

class InferenceCallback(TrainerCallback):
    """Callback để chạy inference examples sau mỗi lần evaluation"""
    
    def __init__(self, model, tokenizer, examples, device):
        self.model = model
        self.tokenizer = tokenizer
        self.examples = examples
        self.device = device
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Chạy sau khi evaluation hoàn tất"""
        run_inference_examples(self.model, self.tokenizer, self.examples, self.device)

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

    # Kiểm tra xem có checkpoint để resume
    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    
    if latest_checkpoint:
        step_number = int(latest_checkpoint.split("-")[-1])
        print(f"  ⚡ Tìm thấy checkpoint: {latest_checkpoint}")
        print(f"  ⚡ Step: {step_number:,}")
        print(f"  ⚡ Sẽ RESUME training từ checkpoint này")
        
        # Load tokenizer từ checkpoint để đảm bảo consistency
        tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)
        # Load model từ checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(latest_checkpoint)
    else:
        print(f"  → Không tìm thấy checkpoint, load pretrained model: {MODEL_NAME}")
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
    
    # Giới hạn eval dataset để tăng tốc evaluation
    original_eval_size = len(eval_dataset)
    if len(eval_dataset) > MAX_EVAL_SAMPLES:
        eval_dataset = eval_dataset.select(range(MAX_EVAL_SAMPLES))
        print(f"  → Giới hạn eval dataset: {original_eval_size:,} → {MAX_EVAL_SAMPLES:,} samples (để tăng tốc)")

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
    
    # Auto-detect bf16/fp16 support
    use_bf16 = False
    use_fp16 = False
    if device == "cuda":
        # Check if GPU supports bf16 (Ampere+, compute capability >= 8.0)
        if torch.cuda.get_device_capability()[0] >= 8:
            use_bf16 = True
            print("  → Detected Ampere+ GPU, using bf16 precision")
        else:
            use_fp16 = True
            print("  → Using fp16 precision")
    
    # Determine optimizer
    optimizer_name = "adamw_torch"
    if torch.__version__ >= "2.0":
        optimizer_name = "adamw_torch_fused"
        print("  → Using fused AdamW optimizer")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=max_steps,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=64,  # Giảm để tránh OOM khi generate
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,  # Dùng ratio thay vì fixed steps
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",  # FIX BUG: eval_strategy -> evaluation_strategy
        save_strategy="steps",
        save_total_limit=3,
        metric_for_best_model="bleu",
        greater_is_better=True,
        load_best_model_at_end=True,
        # Mixed precision
        bf16=use_bf16,
        fp16=use_fp16,
        # Optimizer
        optim=optimizer_name,
        weight_decay=0.01,
        # Learning rate scheduler
        lr_scheduler_type="cosine",
        # Label smoothing
        label_smoothing_factor=0.1,
        # Generation config for metrics
        predict_with_generate=True,
        generation_max_length=128,  # Giảm để tăng tốc eval
        # DataLoader optimization
        dataloader_num_workers=NUM_WORKERS,
        dataloader_prefetch_factor=PREFETCH_FACTOR,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        group_by_length=True,  # Bucket by length để tăng tốc
        # Eval optimization
        # Other
        remove_unused_columns=False,
        save_safetensors=True,
        seed=42,
        report_to="none",
    )
    
    print(f"  - Total steps: {max_steps:,}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - LR scheduler: cosine annealing with 3% warmup")
    print(f"  - Weight decay: 0.01, Label smoothing: 0.1")
    print(f"  - Eval every: {EVAL_STEPS:,} steps")
    print(f"  - Save every: {SAVE_STEPS:,} steps")
    print(f"  - Keep best: 3 checkpoints (based on BLEU score)")
    print(f"  - Metrics: BLEU, Exact Match, CER, WER")
    print(f"  - Group by length: True (bucketing for speed)")
    
    # Data collator với pad_to_multiple_of=8 để tăng tốc Tensor Cores
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
    
    print("  ✓ Training arguments đã được thiết lập")
    
    # ========================================================================
    # TRAINING
    # ========================================================================

    print("\n[6/6] Bắt đầu training...")
    if latest_checkpoint:
        print(f"  - RESUME training từ checkpoint: {latest_checkpoint}")
        print(f"  - Tiếp tục từ step {step_number:,}")
    else:
        print(f"  - Bắt đầu training từ đầu (step 0)")
    print(f"  - Ước tính thời gian: ~{max_steps / (LOGGING_STEPS * 10):.1f} giờ")
    print("-" * 80)
    
    training_start_time = time.time()
    
    # Tạo callback để chạy inference sau eval
    inference_callback = InferenceCallback(
        model=model,
        tokenizer=tokenizer,
        examples=EXAMPLE_SENTENCES,
        device=device
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        callbacks=[inference_callback],
    )
    
    # Resume từ checkpoint nếu có, nếu không thì train từ đầu
    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint)
    
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

