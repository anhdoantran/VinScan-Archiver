import gradio as gr
import easyocr
import cv2
import numpy as np
import google.generativeai as genai
import re
import os

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG (BẢO MẬT)
# ==========================================
# Lấy API Key từ Secrets của Hugging Face
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_ai = genai.GenerativeModel('gemini-2.5-flash')

# Khởi tạo OCR (Tải model Tiếng Việt, Anh, Pháp)
reader = easyocr.Reader(['vi', 'en', 'fr'], gpu=False)

# ==========================================
# 2. CÁC HÀM XỬ LÝ LOGIC
# ==========================================
def kiem_tra_rac(text):
    if not text.strip(): return True
    special_chars = len(re.findall(r'[^a-zA-Z0-9à-ỹÀ-Ỹ\s\-\—\.\,\?\!]', text))
    return (special_chars / len(text)) > 0.4 if len(text) > 0 else True

def xu_ly_anh_va_ocr(anh):
    if anh is None: return None, "⚠️ Vui lòng tải ảnh lên!"
    # Tiền xử lý ảnh làm rõ nét
    gray = cv2.cvtColor(anh, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # Quét chữ
    result = reader.readtext(enhanced, detail=0, paragraph=True, width_ths=0.7, y_ths=0.3)
    return enhanced, "\n\n".join(result)

def hieu_dinh_ai(raw_text):
    if not raw_text or len(raw_text) < 10: return "Chưa có dữ liệu thô để xử lý."
    if kiem_tra_rac(raw_text):
        return "⚠️ CẢNH BÁO: Hình ảnh quá mờ. Máy không nhận diện được chữ chính xác. Vui lòng tự nhập liệu dựa trên ảnh đã lọc."
    try:
        prompt = f"""Bạn là chuyên gia biên tập phục chế tài liệu cổ. 
        Hãy sửa lỗi chính tả bản quét sau, giữ nguyên các dấu gạch đầu dòng (—) và cấu trúc đoạn:
        {raw_text}"""
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi AI: {str(e)}"

def thuc_hien_luu(van_ban, ten_file, che_do, so_trang):
    if not van_ban: return "⚠️ Không có dữ liệu để lưu!", so_trang, None
    # Chuẩn hóa tên file
    file_name = (ten_file.strip().replace(" ", "_") if ten_file else "VinScan_Output") + ".txt"
    mode = "a" if che_do == "Lưu nối tiếp (Thêm trang)" else "w"
    
    with open(file_name, mode, encoding="utf-8") as f:
        f.write(f"\n\n{'='*15} TRANG {int(so_trang)} {'='*15}\n\n")
        f.write(van_ban)
    
    return f"✅ Đã lưu Trang {int(so_trang)} vào {file_name}", so_trang + 1, file_name

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN (UI/UX)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as vinscan:
    
    # THANH MENU BÊN TRÁI (SIDEBAR)
    with gr.Sidebar(label="Menu Điều hướng"):
        gr.Markdown("## 🏛️ VinScan Admin")
        menu_home = gr.Button("🏠 Trang chủ", variant="secondary")
        menu_ocr = gr.Button("🔍 Công cụ Số hóa", variant="secondary")
        menu_guide = gr.Button("📖 Hướng dẫn", variant="secondary")
        gr.Markdown("---")
        gr.Markdown("Tác giả: **Tran Anh Doan**\nSinh viên: **Nam Can Tho University**")

    # --- TRANG 1: TRANG CHỦ ---
    with gr.Column(visible=True) as home_page:
        gr.Markdown("# CHÀO MỪNG ĐẾN VỚI VINSCAN ARCHIVER")
        gr.Markdown("""
        ### Hệ thống Số hóa và Phục chế Văn bản Cổ tích hợp Trí tuệ Nhân tạo
        VinScan Archiver là giải pháp giúp chuyển đổi các trang sách cũ, ố mờ thành văn bản kỹ thuật số sạch sẽ, chính xác.
        
        **Quy trình hoạt động:**
        1. **Xử lý ảnh:** Sử dụng OpenCV để tăng độ tương phản, làm nổi bật nét chữ.
        2. **Nhận diện (OCR):** Công nghệ Deep Learning (EasyOCR) đa ngôn ngữ Việt - Anh - Pháp.
        3. **Tu chỉnh (AI):** Dùng Gemini AI để sửa lỗi chính tả và khôi phục văn phong gốc.
        4. **Quản lý:** Lưu trữ theo số trang và cho phép tải file trực tiếp.
        """)
        btn_start = gr.Button("Bắt đầu ngay 🚀", variant="primary")

    # --- TRANG 2: WORKSPACE SỐ HÓA ---
    with gr.Column(visible=False) as ocr_page:
        gr.Markdown("# 🔍 Không gian làm việc Số hóa")
        with gr.Row():
            # Cột trái: Ảnh
            with gr.Column(scale=1):
                input_img = gr.Image(label="Tải ảnh trang sách", type="numpy")
                btn_ocr = gr.Button("🔍 PHÂN TÍCH & QUÉT CHỮ", variant="primary")
                with gr.Accordion("Xem ảnh qua bộ lọc (OpenCV)", open=False):
                    output_enhanced = gr.Image(label="Ảnh đã làm rõ nét")

            # Cột phải: Văn bản
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("📄 Văn bản Sau tu chỉnh (AI)"):
                        output_ai = gr.Textbox(label="Kết quả cuối cùng (Có thể chỉnh sửa tay)", lines=12, interactive=True)
                        
                        with gr.Group():
                            with gr.Row():
                                file_name_in = gr.Textbox(label="Tên cuốn sách", placeholder="Nam_Phong_1917", scale=2)
                                page_num_in = gr.Number(label="Số trang hiện tại", value=1, scale=1)
                            save_mode = gr.Radio(["Lưu nối tiếp (Thêm trang)", "Tạo file mới (Ghi đè)"], label="Chế độ lưu", value="Lưu nối tiếp (Thêm trang)")
                            btn_save = gr.Button("💾 THỰC HIỆN LƯU & XUẤT FILE", variant="primary")
                        
                        file_download = gr.File(label="📥 Tải file .txt về máy")
                        status_msg = gr.Textbox(label="Trạng thái", lines=1, interactive=False)
                        btn_ai = gr.Button("✨ Nhờ AI sửa lỗi chính tả", variant="secondary")

                    with gr.TabItem("🔍 Bản quét thô (OCR)"):
                        output_raw = gr.Textbox(label="Văn bản máy quét nhìn thấy", lines=18)

    # --- TRANG 3: HƯỚNG DẪN ---
    with gr.Column(visible=False) as guide_page:
        gr.Markdown("# 📖 Hướng dẫn sử dụng chi tiết")
        gr.Markdown("""
        - **Bước 1:** Tải ảnh trang sách cổ lên (ưu tiên ảnh chụp thẳng, đủ sáng).
        - **Bước 2:** Bấm **'Quét chữ'**. Nếu ảnh mờ, hãy mở mục 'Bộ lọc OpenCV' để kiểm tra.
        - **Bước 3:** Bấm **'Nhờ AI sửa lỗi'** để làm sạch văn bản thô.
        - **Bước 4:** Kiểm tra lại nội dung, đặt tên file và bấm **'Lưu'**.
        - **Bước 5:** Bấm vào file xuất hiện trong mục 'Tải file' để lưu về máy tính.
        """)

    # --- LOGIC ĐIỀU HƯỚNG ---
    def go_home(): return {home_page: gr.update(visible=True), ocr_page: gr.update(visible=False), guide_page: gr.update(visible=False)}
    def go_ocr():  return {home_page: gr.update(visible=False), ocr_page: gr.update(visible=True), guide_page: gr.update(visible=False)}
    def go_guide(): return {home_page: gr.update(visible=False), ocr_page: gr.update(visible=False), guide_page: gr.update(visible=True)}

    menu_home.click(fn=go_home, outputs=[home_page, ocr_page, guide_page])
    menu_ocr.click(fn=go_ocr, outputs=[home_page, ocr_page, guide_page])
    btn_start.click(fn=go_ocr, outputs=[home_page, ocr_page, guide_page])
    menu_guide.click(fn=go_guide, outputs=[home_page, ocr_page, guide_page])

    # --- XỬ LÝ SỰ KIỆN DỮ LIỆU ---
    btn_ocr.click(fn=xu_ly_anh_va_ocr, inputs=input_img, outputs=[output_enhanced, output_raw])
    btn_ai.click(fn=hieu_dinh_ai, inputs=output_raw, outputs=output_ai)
    btn_save.click(fn=thuc_hien_luu, inputs=[output_ai, file_name_in, save_mode, page_num_in], outputs=[status_msg, page_num_in, file_download])

vinscan.launch()