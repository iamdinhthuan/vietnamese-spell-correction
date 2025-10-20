# BARTpho Vietnamese Spelling Correction

Sửa lỗi chính tả tiếng Việt sử dụng mô hình BARTpho fine-tuned trên large-scale OCR correction dataset.

## 🌟 Tính năng

- ✅ **Large-scale training**: Hỗ trợ streaming dataset với hàng triệu dòng mà không cần load toàn bộ vào RAM
- ✅ **Resume training**: Tự động tiếp tục training từ checkpoint nếu bị gián đoạn
- ✅ **GPU optimization**: Tối ưu cho training và inference trên GPU
- ✅ **Gradio UI**: Giao diện web thân thiện để test model
- ✅ **Batch inference**: Hỗ trợ xử lý nhiều văn bản cùng lúc

## 📋 Yêu cầu

```bash
pip install transformers datasets torch gradio
```

**Phiên bản khuyến nghị:**
- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30.0
- datasets >= 2.0.0

## 📊 Dataset

Dataset yêu cầu định dạng CSV với 2 cột:
- `input_text`: Văn bản có lỗi chính tả
- `target_text`: Văn bản đã được sửa lỗi

**Ví dụ:**
```csv
input_text,target_text
"Chay ì nộp phat nguội.","Chạy ì nốp phạt nguội."
"Đây ià cac phưong tiện vi phạm","Đây là các phương tiện vi phạm"
```

### Chuẩn bị dataset

1. **Đặt file dataset gốc** với tên `ocr_correction_dataset.csv`

2. **Chia dataset thành train/val:**
```bash
python split_train_val.py
```

Script này sẽ:
- Shuffle toàn bộ dataset với seed cố định
- Tách 200,000 dòng cho validation set
- Phần còn lại cho training set
- Output: `train.csv` và `val.csv`

## 🚀 Training

### Training từ đầu

```bash
python train_bartpho_streaming.py
```

### Resume training từ checkpoint

Nếu training bị gián đoạn, chỉ cần chạy lại lệnh trên:
```bash
python train_bartpho_streaming.py
```

Script tự động:
- Tìm checkpoint mới nhất trong `./bartpho_vsc/`
- Resume training từ checkpoint đó
- Tiếp tục với learning rate và optimizer state đã lưu

**Output:**
- Checkpoints: `./bartpho_vsc/checkpoint-*/`
- Final model: `./bartpho_vsc_model/`

### Hyperparameters

Có thể chỉnh sửa trong file `train_bartpho_streaming.py`:

```python
MAX_INPUT_LENGTH = 256        # Độ dài tối đa input
MAX_TARGET_LENGTH = 256       # Độ dài tối đa output
NUM_EPOCHS = 5                # Số epoch
TRAIN_BATCH_SIZE = 64         # Batch size cho training
EVAL_BATCH_SIZE = 64          # Batch size cho evaluation
LEARNING_RATE = 5e-5          # Learning rate
WARMUP_STEPS = 10000          # Warmup steps
SAVE_STEPS = 10000            # Lưu checkpoint mỗi N steps
EVAL_STEPS = 10000            # Eval mỗi N steps
SHUFFLE_BUFFER = 50000        # Buffer size cho shuffle
NUM_WORKERS = 8               # Số workers cho DataLoader
```

## 🎯 Inference

### 1. Command-line inference

```bash
python inference_bartpho.py
```

**Sử dụng trong code:**
```python
from inference_bartpho import correct_spelling, correct_batch

# Sửa 1 câu
text = "Chay ì nộp phat nguội"
corrected = correct_spelling(text, num_beams=5)
print(corrected)  # "Chạy ì nốp phạt nguội"

# Sửa nhiều câu
texts = ["câu 1 có lỗi", "câu 2 có lỗi", "câu 3 có lỗi"]
results = correct_batch(texts, batch_size=8, num_beams=5)
```

### 2. Gradio Web UI

```bash
python gradio_inference.py
```

Mở trình duyệt tại: **http://localhost:7860**

**Tính năng UI:**
- Input/output textbox
- Điều chỉnh `num_beams` (1-10) bằng slider
- 5 ví dụ mẫu để test nhanh
- Tự động load checkpoint mới nhất từ `/home/thuan/data/bartpho_vsc/`

## 📂 Cấu trúc thư mục

```
.
├── train_bartpho_streaming.py    # Script training chính
├── inference_bartpho.py           # CLI inference
├── gradio_inference.py            # Gradio web UI
├── split_train_val.py             # Chia dataset
├── ocr_correction_dataset.csv     # Dataset gốc (không commit)
├── train.csv                      # Training set (không commit)
├── val.csv                        # Validation set (không commit)
├── bartpho_vsc/                   # Checkpoints (không commit)
│   ├── checkpoint-10000/
│   ├── checkpoint-20000/
│   └── ...
└── bartpho_vsc_model/             # Final model (không commit)
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## 🎨 Ví dụ

```python
# Example 1: Sửa lỗi dấu thanh và dấu phụ
input  = "Chay ì nộp phat nguội."
output = "Chạy ì nốp phạt nguội."

# Example 2: Sửa lỗi chữ cái
input  = "Đây ià cac phưong tiện vi phạm được camera ghi hình."
output = "Đây là các phương tiện vi phạm được camera ghi hình."

# Example 3: Sửa lỗi từ
input  = "Phổ biến nhat ià ioi đỗ không đúng nơi quy dịnh."
output = "Phổ biến nhất là lỗi đỗ không đúng nơi quy định."

# Example 4: Sửa lỗi phức tạp
input  = "Tôi đang học tap tiéng viét ơ trưòng đai hoc."
output = "Tôi đang học tập tiếng việt ở trường đại học."
```

## ⚙️ Cấu hình cho môi trường khác

### Thay đổi đường dẫn checkpoint

**Trong `gradio_inference.py`:**
```python
CHECKPOINT_DIR = "/home/thuan/data/bartpho_vsc/"  # Đổi đường dẫn này
```

**Trong `inference_bartpho.py`:**
```python
MODEL_DIR = "./bartpho_vsc_model"  # Đổi đường dẫn này
```

### Training trên nhiều GPU

Sử dụng `torchrun` hoặc `accelerate`:

```bash
# Với torchrun (PyTorch distributed)
torchrun --nproc_per_node=2 train_bartpho_streaming.py

# Với accelerate
accelerate config  # Cấu hình 1 lần
accelerate launch train_bartpho_streaming.py
```

## 📈 Monitoring Training

Trong quá trình training, log sẽ hiển thị:
- Loss value mỗi `LOGGING_STEPS` (mặc định: 2000 steps)
- Evaluation metrics mỗi `EVAL_STEPS` (mặc định: 10000 steps)
- Checkpoint được lưu mỗi `SAVE_STEPS` (mặc định: 10000 steps)

**Ví dụ log:**
```
{'loss': 0.5234, 'learning_rate': 4.5e-05, 'epoch': 0.5, 'step': 10000}
{'eval_loss': 0.4123, 'eval_runtime': 245.3, 'eval_samples_per_second': 815.2}
```

## 🐛 Troubleshooting

### 1. Out of Memory (OOM)

Giảm batch size trong `train_bartpho_streaming.py`:
```python
TRAIN_BATCH_SIZE = 32  # Giảm từ 64
EVAL_BATCH_SIZE = 32
```

### 2. Training chậm

Tăng số workers và prefetch:
```python
NUM_WORKERS = 16               # Tăng workers
PREFETCH_FACTOR = 8            # Tăng prefetch
TOKENIZATION_BATCH_SIZE = 2000 # Tăng batch size tokenization
```

### 3. Checkpoint không load

Kiểm tra đường dẫn:
```bash
ls -la ./bartpho_vsc/checkpoint-*
```

### 4. Gradio không kết nối

Thử port khác:
```python
demo.launch(server_port=7861)  # Đổi port
```

## 📝 Model Information

- **Base model**: [`vinai/bartpho-syllable-base`](https://huggingface.co/vinai/bartpho-syllable-base)
- **Task**: Seq2Seq text correction
- **Architecture**: BART (Bidirectional and Auto-Regressive Transformers)
- **Tokenization**: Syllable-level tokenization cho tiếng Việt
- **Parameters**: ~135M parameters

## 📄 License

MIT License

## 🙏 Acknowledgments

- [VinAI Research](https://github.com/VinAIResearch) - BARTpho model
- [Hugging Face](https://huggingface.co/) - Transformers library

## 📧 Contact

Nếu có vấn đề hoặc câu hỏi, vui lòng tạo issue trên GitHub.

---

**Happy spell checking! 🎉**
