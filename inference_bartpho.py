"""
Script inference cho mô hình BARTpho đã fine-tune
Sử dụng để sửa lỗi chính tả tiếng Việt

Usage:
    python inference_bartpho.py
    
    hoặc import và sử dụng:
    from inference_bartpho import correct_spelling
    result = correct_spelling("Chay ì nộp phat nguội")
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================================
# CẤU HÌNH
# ============================================================================

MODEL_DIR = "./bartpho_vsc_model"
MAX_LENGTH = 256

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# TẢI MÔ HÌNH
# ============================================================================

print("=" * 80)
print("BARTpho Spelling Correction - Inference")
print("=" * 80)
print(f"Loading model from: {MODEL_DIR}")
print(f"Device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nHãy chắc chắn bạn đã huấn luyện model bằng cách chạy:")
    print("  python train_bartpho_spelling.py")
    exit(1)

print("=" * 80)

# ============================================================================
# HÀM SỬA LỖI CHÍNH TẢ
# ============================================================================

def correct_spelling(text, num_beams=5, max_length=MAX_LENGTH):
    """
    Sửa lỗi chính tả cho văn bản đầu vào
    
    Args:
        text (str): Văn bản có lỗi chính tả
        num_beams (int): Số beams cho beam search (càng cao càng chính xác nhưng chậm hơn)
        max_length (int): Độ dài tối đa của output
    
    Returns:
        str: Văn bản đã được sửa lỗi
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        max_length=max_length,
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
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )
    
    # Decode
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return corrected_text


def correct_batch(texts, batch_size=8, num_beams=5, max_length=MAX_LENGTH):
    """
    Sửa lỗi chính tả cho nhiều văn bản cùng lúc
    
    Args:
        texts (list): List các văn bản có lỗi
        batch_size (int): Kích thước batch
        num_beams (int): Số beams cho beam search
        max_length (int): Độ dài tối đa của output
    
    Returns:
        list: List các văn bản đã được sửa lỗi
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            max_length=max_length,
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
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )
        
        # Decode
        batch_results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(batch_results)
    
    return results


# ============================================================================
# DEMO INTERACTIVE
# ============================================================================

def interactive_demo():
    """
    Chế độ demo tương tác
    """
    print("\n" + "=" * 80)
    print("DEMO TƯƠNG TÁC - Sửa lỗi chính tả tiếng Việt")
    print("=" * 80)
    print("Nhập văn bản có lỗi chính tả (hoặc 'quit' để thoát)")
    print("-" * 80)
    
    while True:
        print("\nInput: ", end="")
        text = input().strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Tạm biệt!")
            break
        
        if not text:
            continue
        
        # Sửa lỗi
        corrected = correct_spelling(text)
        
        print(f"Output: {corrected}")
        print("-" * 80)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Ví dụ sử dụng
    print("\n📝 VÍ DỤ SỬ DỤNG:")
    print("-" * 80)
    
    # Test cases
    test_cases = [
        "Chay ì nộp phat nguội.",
        "Đây ià cac phưong tiện vi phạm được camera ghi hình.",
        "Phổ biến nhat ià ioi đỗ không đúng nơi quy dịnh.",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n{i}. Input:  {text}")
        corrected = correct_spelling(text)
        print(f"   Output: {corrected}")
    
    print("\n" + "=" * 80)
    
    # Chạy demo tương tác
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nTạm biệt!")

