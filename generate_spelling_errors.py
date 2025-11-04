import csv
from openai import OpenAI
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import threading

# Cấu hình API
MODEL_NAME = "gpt-oss-120b"
WORKERS_PER_API = 100  # Số lượng workers cho mỗi API

# Cấu hình cho từng API riêng biệt (cập nhật api_key trước khi chạy)
API_PROFILES = [
    {
        "name": "api_1",
        "base_url": "https://mkp-api.fptcloud.com",
        "api_key": "sk-699KeHi8mgV8lJs6Zzabeg",
    },
    {
        "name": "api_2",
        "base_url": "https://mkp-api.fptcloud.com",
        "api_key": "sk-ReMoT_JddifgWQYRaKavSQ",
    },
    {
        "name": "api_3",
        "base_url": "https://mkp-api.fptcloud.com",
        "api_key": "sk-dYz-py-tTTjYfFo-KIQcZw",
    },
    {
        "name": "api_4",
        "base_url": "https://mkp-api.fptcloud.com",
        "api_key": "sk-R2dcnukugsvc9-GKIQUHNg",
    },
]

# Lock để đảm bảo ghi file an toàn
write_lock = threading.Lock()

# File paths
INPUT_FILE = "input.txt"
OUTPUT_FILE = "spelling_errors.csv"

# Prompt hướng dẫn model tạo lỗi chính tả
SYSTEM_PROMPT = """Tạo lỗi chính tả tiếng Việt tự nhiên.

QUY TẮC:
- GIỮ NGUYÊN dấu câu
- GIỮ NGUYÊN viết tắt
- CHỈ tạo lỗi: ký tự gần nhau (a→s), sai dấu (ò→ó), sai âm (d→gi, n→ng)
- Tạo 1-3 lỗi mỗi câu

Chỉ trả về câu có lỗi."""

def generate_error_text(api_client, correct_text):
    """Gọi API để tạo văn bản có lỗi chính tả"""
    try:
        response = api_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": correct_text
                }
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9
        )

        # Kiểm tra nếu API trả về None (do hết token)
        if response.choices[0].message.content is None:
            return None

        error_text = response.choices[0].message.content.strip()
        return error_text

    except Exception as e:
        return None

def process_single_line(idx, correct_text, api_client):
    """Xử lý một dòng text"""
    error_text = generate_error_text(api_client, correct_text)
    return idx, correct_text, error_text

def process_file():
    """Đọc file input, tạo lỗi và lưu vào CSV với multi-threading"""

    # Đọc file input
    print(f"Đang đọc file {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Tìm thấy {len(lines)} câu")

    # Kiểm tra xem file output đã tồn tại chưa để tiếp tục từ dòng cuối
    processed_count = 0
    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8', newline='') as f:
            processed_count = sum(1 for _ in csv.reader(f)) - 1  # Trừ header
        print(f"File output đã tồn tại với {processed_count} dòng. Tiếp tục từ dòng {processed_count + 1}...")
        mode = 'a'
        write_header = False
    else:
        print("Tạo file output mới...")
        mode = 'w'
        write_header = True

    if not API_PROFILES:
        raise ValueError("Cần cấu hình ít nhất một API trong API_PROFILES.")

    clients = []
    for idx, profile in enumerate(API_PROFILES):
        api_key = profile.get("api_key")
        base_url = profile.get("base_url")
        api_name = profile.get("name") or f"api_{idx + 1}"

        if not api_key or "REPLACE_WITH_API_KEY" in api_key:
            raise ValueError(f"Chưa cấu hình api_key cho {api_name}.")
        if not base_url:
            raise ValueError(f"Chưa cấu hình base_url cho {api_name}.")

        clients.append(
            {
                "name": api_name,
                "client": OpenAI(api_key=api_key, base_url=base_url)
            }
        )

    num_apis = len(clients)
    total_workers = num_apis * WORKERS_PER_API

    # Lấy danh sách câu cần xử lý
    remaining_lines = lines[processed_count:]
    total_lines = len(lines)

    print(f"Sử dụng {num_apis} API song song, mỗi API {WORKERS_PER_API} workers (tổng {total_workers} workers)")
    print(f"Bắt đầu xử lý {len(remaining_lines)} câu còn lại...\n")

    # Mở file CSV để ghi
    csvfile = open(OUTPUT_FILE, mode, encoding='utf-8', newline='')
    fieldnames = ['correct_text', 'error_text']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Ghi header nếu là file mới
    if write_header:
        writer.writeheader()
        csvfile.flush()

    success_count = 0
    fail_count = 0

    try:
        with ExitStack() as stack:
            api_slots = [
                {
                    "name": cfg["name"],
                    "client": cfg["client"],
                    "executor": stack.enter_context(ThreadPoolExecutor(max_workers=WORKERS_PER_API))
                }
                for cfg in clients
            ]

            future_to_meta = {}
            for offset, text in enumerate(remaining_lines):
                idx = processed_count + offset + 1
                slot = api_slots[offset % num_apis]
                future = slot["executor"].submit(process_single_line, idx, text, slot["client"])
                future_to_meta[future] = (idx, text, slot["name"])

            for future in as_completed(future_to_meta):
                idx, correct_text, api_name = future_to_meta[future]

                try:
                    _, _, error_text = future.result()
                except Exception:
                    error_text = None

                if error_text:
                    with write_lock:
                        writer.writerow({
                            'correct_text': correct_text,
                            'error_text': error_text
                        })
                        csvfile.flush()

                    success_count += 1
                    print(f"[{idx}/{total_lines}] ✓ ({api_name}) {correct_text[:60]}...")
                else:
                    fail_count += 1
                    print(f"[{idx}/{total_lines}] ✗ ({api_name}) Không tạo được lỗi: {correct_text[:60]}...")

    finally:
        csvfile.close()

    print(f"\n{'='*60}")
    print(f"Hoàn thành!")
    print(f"  - Tổng số câu: {total_lines}")
    print(f"  - Thành công: {success_count}")
    print(f"  - Thất bại: {fail_count}")
    print(f"  - Kết quả được lưu tại: {OUTPUT_FILE}")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        process_file()
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng. Dữ liệu đã được lưu tới dòng hiện tại.")
    except Exception as e:
        print(f"\n\nLỗi: {e}")
        print("Dữ liệu đã được lưu tới dòng hiện tại.")
