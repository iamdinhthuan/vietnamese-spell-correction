"""
Gradio Interface cho BARTpho Spelling Correction
Load checkpoint mới nhất từ thư mục training

Usage:
    python gradio_inference.py

Gradio UI sẽ mở tại http://localhost:7860
"""

import os
import glob
import re
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ============================================================================
# CẤU HÌNH
# ============================================================================

CHECKPOINT_DIR = "/home/thuan/data/bartpho_vsc/"
MAX_LENGTH = 256
DEFAULT_NUM_BEAMS = 15

# ============================================================================
# HÀM TÌM CHECKPOINT MỚI NHẤT
# ============================================================================

def find_latest_checkpoint(checkpoint_dir):
    """
    Tìm checkpoint mới nhất trong thư mục
    
    Args:
        checkpoint_dir: Đường dẫn đến thư mục chứa checkpoints
    
    Returns:
        str: Đường dẫn đến checkpoint mới nhất
    """
    # Tìm tất cả các thư mục checkpoint-*
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    
    if not checkpoints:
        raise ValueError(f"Không tìm thấy checkpoint nào trong {checkpoint_dir}")
    
    # Sort theo số step (lấy từ tên thư mục)
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    
    latest_checkpoint = checkpoints[-1]
    step_number = int(latest_checkpoint.split("-")[-1])
    
    return latest_checkpoint, step_number

# ============================================================================
# TẢI MÔ HÌNH
# ============================================================================

print("=" * 80)
print("BARTpho Spelling Correction - Gradio Interface")
print("=" * 80)

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Tìm checkpoint mới nhất
print(f"\nTìm checkpoint mới nhất trong: {CHECKPOINT_DIR}")
try:
    checkpoint_path, step_number = find_latest_checkpoint(CHECKPOINT_DIR)
    print(f"✓ Tìm thấy checkpoint: {checkpoint_path}")
    print(f"  Step: {step_number:,}")
except Exception as e:
    print(f"✗ Lỗi: {e}")
    print("\nKiểm tra lại đường dẫn hoặc chạy training trước:")
    print("  python train_bartpho_streaming.py")
    exit(1)

# Load model và tokenizer
print("\nĐang load model và tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully!")
    print(f"  Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"✗ Lỗi khi load model: {e}")
    exit(1)

print("=" * 80)

# ============================================================================
# HÀM SỬA LỖI CHÍNH TẢ
# ============================================================================

def split_sentences(text):
    """
    Cắt văn bản thành các câu dựa trên dấu chấm (.) và dấu chấm hỏi (?)
    Giữ lại dấu câu ở cuối mỗi câu
    
    Args:
        text (str): Văn bản đầu vào
    
    Returns:
        list: Danh sách các câu
    """
    # Tách theo dấu chấm và dấu chấm hỏi, nhưng giữ lại dấu
    # Pattern: split sau dấu . hoặc ? (và khoảng trắng nếu có)
    sentences = re.split(r'([.?])\s*', text)
    
    # Ghép lại dấu câu với câu tương ứng
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if sentences[i].strip():  # Bỏ qua câu rỗng
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # Thêm dấu câu
            result.append(sentence.strip())
    
    # Xử lý câu cuối nếu không có dấu câu
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    
    return result

def correct_single_sentence(sentence, num_beams=DEFAULT_NUM_BEAMS, max_length=MAX_LENGTH):
    """
    Sửa lỗi chính tả cho một câu đơn
    
    Args:
        sentence (str): Câu có lỗi chính tả
        num_beams (int): Số beams cho beam search
        max_length (int): Độ dài tối đa của output
    
    Returns:
        str: Câu đã được sửa lỗi
    """
    if not sentence or not sentence.strip():
        return ""
    
    # Tokenize input
    inputs = tokenizer(
        sentence,
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

def correct_spelling(text, num_beams=DEFAULT_NUM_BEAMS, max_length=MAX_LENGTH):
    """
    Sửa lỗi chính tả cho văn bản đầu vào
    Cắt văn bản thành các câu, sửa từng câu, rồi merge lại
    
    Args:
        text (str): Văn bản có lỗi chính tả
        num_beams (int): Số beams cho beam search
        max_length (int): Độ dài tối đa của output
    
    Returns:
        str: Văn bản đã được sửa lỗi
    """
    if not text or not text.strip():
        return ""
    
    # Cắt văn bản thành các câu
    sentences = split_sentences(text)
    
    # Sửa từng câu
    corrected_sentences = []
    for sentence in sentences:
        corrected = correct_single_sentence(sentence, num_beams, max_length)
        corrected_sentences.append(corrected)
    
    # Merge các câu lại với khoảng trắng
    result = " ".join(corrected_sentences)
    
    return result

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def gradio_correct_spelling(text, num_beams):
    """
    Wrapper function cho Gradio interface
    """
    try:
        result = correct_spelling(text, num_beams=int(num_beams))
        return result
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Tạo Gradio interface
with gr.Blocks(title="BARTpho Spelling Correction") as demo:
    gr.Markdown(f"""
    # 🔤 BARTpho Spelling Correction
    
    Sửa lỗi chính tả tiếng Việt sử dụng BARTpho
    
    **Checkpoint:** `{checkpoint_path}`  
    **Step:** `{step_number:,}`  
    **Device:** `{device.upper()}`
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Văn bản có lỗi chính tả",
                placeholder="Nhập văn bản có lỗi chính tả vào đây...",
                lines=5
            )
            
            with gr.Row():
                num_beams_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=DEFAULT_NUM_BEAMS,
                    step=1,
                    label="Số beams (càng cao càng chính xác nhưng chậm hơn)"
                )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Xóa", variant="secondary")
                submit_btn = gr.Button("✨ Sửa lỗi chính tả", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="✅ Văn bản đã sửa lỗi",
                lines=5
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["Đây ià cac phưong tiện vi phạm được camera ghi hình.", 15],
            ["Phổ biến nhat ià ioi đỗ không đúng nơi quy dịnh.", 15],
            ["Tôi đang học tap tiéng viét ơ trưòng đai hoc.", 15],
            ["Hom nay troi mua rat to, toi khong the di hoc duoc.", 15],
        ],
        inputs=[input_text, num_beams_slider],
        outputs=output_text,
        fn=gradio_correct_spelling,
        cache_examples=False,
    )
    
    # Event handlers
    submit_btn.click(
        fn=gradio_correct_spelling,
        inputs=[input_text, num_beams_slider],
        outputs=output_text
    )
    
    input_text.submit(
        fn=gradio_correct_spelling,
        inputs=[input_text, num_beams_slider],
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=None,
        outputs=[input_text, output_text]
    )

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n🚀 Đang khởi động Gradio interface...")
    print("   Mở trình duyệt tại: http://localhost:7860")
    print("   Nhấn Ctrl+C để dừng server\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Cho phép truy cập từ mạng nội bộ
        server_port=7860,
        share=True,  # Đặt True nếu muốn share link public
        show_error=True
    )
