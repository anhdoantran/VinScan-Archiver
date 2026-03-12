import gradio as gr
import easyocr
import cv2
import numpy as np
import google.generativeai as genai
import re

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
# THAY THẾ "API_KEY_CUA_BAN" BẰNG MÃ AIza... CỦA BẠN
genai.configure(api_key="AIzaSyB5UeVYuvK6YdDj_ef5JbLdAqSUT6orRBw")
model_ai = genai.GenerativeModel('gemini-2.5-flash')
reader = easyocr.Reader(['vi', 'en', 'fr'], gpu=False)

# ==========================================
# 2. CÁC HÀM XỬ LÝ LÕI
# ==========================================

def kiem_tra_rac(text):
    """Kiểm tra nếu OCR chỉ quét ra rác (ký tự lạ) để cảnh báo AI"""
    if not text.strip(): return True
    special_chars = len(re.findall(r'[^a-zA-Z0-9à-ỹÀ-Ỹ\s\-\—\.\,\?\!]', text))
    return (special_chars / len(text)) > 0.4 if len(text) > 0 else True

def xu_ly_anh_va_ocr(anh):
    if anh is None: return None, "Vui lòng tải ảnh lên!"
    
    # Tiền xử lý ảnh xám + CLAHE (Làm rõ chữ mà không vỡ nét)
    gray = cv2.cvtColor(anh, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # OCR trích xuất văn bản (giữ đoạn văn)
    result = reader.readtext(enhanced, detail=0, paragraph=True, width_ths=0.7, y_ths=0.3)
    raw_text = "\n\n".join(result)
    
    return enhanced, raw_text

def hieu_dinh_ai(raw_text):
    if not raw_text or len(raw_text) < 10: return "Dữ liệu thô quá ít để xử lý."
    
    if kiem_tra_rac(raw_text):
        return "⚠️ CẢNH BÁO: Hình ảnh quá mờ. Máy không nhận diện được chữ. Vui lòng nhìn ảnh bên trái và tự nhập liệu."

    try:
        prompt = f"""Bạn là chuyên gia phục chế văn bản cổ. 
        NHIỆM VỤ: Tu chỉnh lỗi chính tả từ bản quét OCR sau.
        YÊU CẦU BẮT BUỘC:
        1. GIỮ NGUYÊN dấu gạch đầu dòng (—) hội thoại và các dấu câu (?, !, ...).
        2. GIỮ ĐÚNG cấu trúc xuống dòng và ngắt đoạn của bản gốc.
        3. Khôi phục từ sai dấu, sai chữ dựa trên ngữ cảnh tiếng Việt/Pháp.
        4. KHÔNG tự ý bịa thêm nội dung ngoài văn bản gốc.
        
        VĂN BẢN GỐC:
        {raw_text}"""
        
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi kết nối AI: {str(e)}"

def luu_du_lieu(van_ban, ten_file, che_do, so_trang):
    if not van_ban: return "⚠️ Không có dữ liệu để lưu!", so_trang
    
    # Chuẩn hóa tên file
    file_name = (ten_file.strip().replace(" ", "_") if ten_file else "Tai_Lieu_Luu_Tru") + ".txt"
    mode = "a" if che_do == "Lưu nối tiếp (Thêm trang)" else "w"
    
    with open(file_name, mode, encoding="utf-8") as f:
        # Ghi header trang để dễ quản lý
        f.write(f"\n\n{'='*20} TRANG {int(so_trang)} {'='*20}\n\n")
        f.write(van_ban)
        f.write("\n")
        
    res_msg = f"✅ Đã lưu Trang {int(so_trang)} vào '{file_name}'"
    return res_msg, so_trang + 1 # Tự động nhảy số trang lên 1

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN (UI)
# ==========================================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as vinscan:
    gr.Markdown("<h1 style='text-align: center; color: #1e3a8a;'>🏛️ VINSCAN ARCHIVER</h1>")
    gr.Markdown("<p style='text-align: center;'>Hệ thống Số hóa Di sản: OCR Thông minh & Phục chế AI</p>")

    with gr.Row():
        # Cột trái: Hình ảnh
        with gr.Column(scale=1):
            input_img = gr.Image(label="Ảnh tài liệu gốc", type="numpy")
            btn_ocr = gr.Button("🔍 PHÂN TÍCH ẢNH & OCR", variant="primary")
            
            with gr.Accordion("Xem ảnh qua bộ lọc làm rõ chữ", open=False):
                output_enhanced = gr.Image(label="Ảnh Grayscale + CLAHE")

        # Cột phải: Văn bản và Lưu trữ
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("📄 Kết quả Tu chỉnh (AI)"):
                    output_ai = gr.Textbox(label="Văn bản đã phục chế (Có thể sửa tay)", lines=12, interactive=True)
                    
                    # Khu vực quản lý file
                    with gr.Group():
                        with gr.Row():
                            file_name_in = gr.Textbox(label="Tên cuốn sách", placeholder="Ví dụ: Nam_Phong_1917", scale=2)
                            page_num_in = gr.Number(label="Số trang", value=1, scale=1)
                        
                        save_mode = gr.Radio(
                            ["Lưu nối tiếp (Thêm trang)", "Tạo file mới (Ghi đè)"],
                            label="Chế độ lưu trữ",
                            value="Lưu nối tiếp (Thêm trang)"
                        )
                        btn_save = gr.Button("💾 THỰC HIỆN LƯU & NHẢY TRANG", variant="primary")
                    
                    status_msg = gr.Textbox(label="Trạng thái", lines=1, interactive=False)
                    btn_ai = gr.Button("✨ Nhờ AI sửa lỗi & giữ đoạn", variant="secondary")

                with gr.TabItem("🔍 Bản quét thô (OCR)"):
                    output_raw = gr.Textbox(label="Dữ liệu máy quét chưa xử lý", lines=18)

    # Sự kiện
    btn_ocr.click(fn=xu_ly_anh_va_ocr, inputs=input_img, outputs=[output_enhanced, output_raw])
    btn_ai.click(fn=hieu_dinh_ai, inputs=output_raw, outputs=output_ai)
    btn_save.click(
        fn=luu_du_lieu, 
        inputs=[output_ai, file_name_in, save_mode, page_num_in], 
        outputs=[status_msg, page_num_in]
    )

vinscan.launch()