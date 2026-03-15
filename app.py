import gradio as gr
import easyocr
import cv2
import numpy as np
import google.generativeai as genai
import re
import os
import difflib

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG
# ==========================================
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)
model_ai = genai.GenerativeModel('gemini-2.5-flash')
reader = easyocr.Reader(['vi', 'en', 'fr'], gpu=False)

# ==========================================
# 2. CÁC HÀM XỬ LÝ LOGIC & THUẬT TOÁN
# ==========================================
def kiem_tra_rac(text):
    if not text.strip(): return True
    special_chars = len(re.findall(r'[^a-zA-Z0-9à-ỹÀ-Ỹ\s\-\—\.\,\?\!]', text))
    return (special_chars / len(text)) > 0.4 if len(text) > 0 else True

def theo_doi_su_thay_doi(raw_text, ai_text):
    """Thuật toán đối chiếu và bôi màu sự thay đổi của văn bản (Text Diff)"""
    if not raw_text or not ai_text: return ""
    
    words_raw = raw_text.split()
    words_ai = ai_text.split()
    diff = difflib.ndiff(words_raw, words_ai)
    
    html_output = "<div style='font-family: Arial; line-height: 1.6; padding: 15px; background-color: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb;'>"
    for word in diff:
        if word.startswith('- '):
            html_output += f"<span style='color: #ef4444; text-decoration: line-through; margin-right: 4px;'>{word[2:]}</span>"
        elif word.startswith('+ '):
            html_output += f"<span style='color: #10b981; font-weight: bold; background-color: #d1fae5; padding: 0 2px; border-radius: 3px; margin-right: 4px;'>{word[2:]}</span>"
        elif word.startswith('  '):
            html_output += f"<span style='color: #374151; margin-right: 4px;'>{word[2:]}</span>"
            
    html_output += "</div>"
    return html_output

def xu_ly_anh_va_ocr(anh):
    """Hàm xử lý ảnh qua bộ lọc OpenCV và đọc chữ OCR"""
    if anh is None: return None, "⚠️ Vui lòng tải ảnh lên!"
    gray = cv2.cvtColor(anh, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    result = reader.readtext(enhanced, detail=0, paragraph=True, width_ths=0.7, y_ths=0.3)
    return enhanced, "\n\n".join(result)

def hieu_dinh_ai(raw_text):
    """Hàm gọi AI phục chế văn bản"""
    if not raw_text or len(raw_text) < 10: return "Chưa có dữ liệu thô để xử lý."
    if kiem_tra_rac(raw_text):
        return "⚠️ CẢNH BÁO: Hình ảnh quá mờ. Máy không nhận diện được chữ chính xác."
    try:
        prompt = f"""Bạn là chuyên gia biên tập phục chế tài liệu cổ. 
        Hãy sửa lỗi chính tả bản quét sau, giữ nguyên các dấu gạch đầu dòng (—) và cấu trúc đoạn:
        {raw_text}"""
        response = model_ai.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Lỗi AI: {str(e)}"

def xu_ly_ai_va_so_sanh(raw):
    """Hàm trung gian chạy AI và xuất màn hình đối chiếu"""
    ai_fixed = hieu_dinh_ai(raw)
    diff_html = theo_doi_su_thay_doi(raw, ai_fixed)
    return ai_fixed, diff_html

def thuc_hien_luu(van_ban, ten_file, so_trang):
    """Hàm lưu trữ file text cộng dồn vào ổ đĩa"""
    if not van_ban: return "⚠️ Không có dữ liệu để lưu!", so_trang, None
    
    folder_path = "Kho_Du_Lieu"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_name = (ten_file.strip().replace(" ", "_") if ten_file else "VinScan_Output") + ".txt"
    file_path = os.path.join(folder_path, file_name)
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'='*15} TRANG {int(so_trang)} {'='*15}\n\n")
        f.write(van_ban)
    
    return f"✅ Đã lưu tiếp Trang {int(so_trang)} vào {file_name}", so_trang + 1, file_path

def lam_moi_trang():
    """Hàm Reset trả 9 giá trị rỗng cho 9 ô trên giao diện"""
    return None, None, "", "", "", "", 1, None, "🔄 Đã làm mới toàn bộ không gian làm việc!"

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN (UI/UX)
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as vinscan:
    
    with gr.Sidebar(label="Menu Điều hướng"):
        gr.Markdown("## 🏛️ VinScan Admin")
        menu_home = gr.Button("🏠 Trang chủ", variant="secondary")
        menu_ocr = gr.Button("🔍 Công cụ Số hóa", variant="secondary")
        menu_guide = gr.Button("📖 Hướng dẫn", variant="secondary")
        gr.Markdown("---")
        gr.Markdown("Tác giả: **Tran Anh Doan**\nSinh viên: **Nam Can Tho University**")

    with gr.Column(visible=True) as home_page:
        gr.Markdown("# CHÀO MỪNG ĐẾN VỚI VINSCAN ARCHIVER")
        btn_start = gr.Button("Bắt đầu ngay 🚀", variant="primary")

    with gr.Column(visible=False) as ocr_page:
        gr.Markdown("# 🔍 Không gian làm việc Số hóa")
        with gr.Row():
            # CỘT TRÁI
            with gr.Column(scale=1):
                input_img = gr.Image(label="1. Tải ảnh trang sách", type="numpy")
                
                with gr.Row():
                    btn_ocr = gr.Button("🔍 QUÉT CHỮ", variant="primary")
                    btn_reset = gr.Button("🔄 LÀM MỚI (RESET)", variant="stop")
                    
                with gr.Accordion("Xem ảnh qua bộ lọc (OpenCV)", open=False):
                    output_enhanced = gr.Image(label="Ảnh đã làm rõ nét")

            # CỘT PHẢI
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🔍 2. Bản quét thô (OCR)"):
                        output_raw = gr.Textbox(label="Dữ liệu gốc từ máy quét", lines=18)

                    with gr.TabItem("✨ 3. Tu chỉnh AI & Lưu trữ"):
                        btn_ai = gr.Button("✨ BẤM VÀO ĐÂY ĐỂ AI SỬA LỖI TỪ BẢN QUÉT THÔ", variant="primary")
                        output_ai = gr.Textbox(label="Kết quả AI (Có thể chỉnh sửa tay)", lines=8, interactive=True)
                        
                        gr.Markdown("### 🔍 Phân tích Vết sửa của AI (Thuật toán tự code)")
                        output_diff = gr.HTML(label="Bản đối chiếu Lỗi")

                        with gr.Group():
                            with gr.Row():
                                file_name_in = gr.Textbox(label="Tên cuốn sách (Tự động nối trang nếu trùng tên)", placeholder="Ví dụ: Nam_Phong", scale=2)
                                page_num_in = gr.Number(label="Số trang đang lưu", value=1, scale=1)
                            
                            btn_save = gr.Button("💾 LƯU CỘNG DỒN & XUẤT FILE", variant="secondary")
                        
                        file_download = gr.File(label="📥 Tải file .txt về máy")
                        status_msg = gr.Textbox(label="Trạng thái", lines=1, interactive=False)

    with gr.Column(visible=False) as guide_page:
        gr.Markdown("# 📖 Hướng dẫn sử dụng")

    # LOGIC ĐIỀU HƯỚNG CÁC TRANG
    def go_home(): return {home_page: gr.update(visible=True), ocr_page: gr.update(visible=False), guide_page: gr.update(visible=False)}
    def go_ocr():  return {home_page: gr.update(visible=False), ocr_page: gr.update(visible=True), guide_page: gr.update(visible=False)}
    def go_guide(): return {home_page: gr.update(visible=False), ocr_page: gr.update(visible=False), guide_page: gr.update(visible=True)}

    menu_home.click(fn=go_home, outputs=[home_page, ocr_page, guide_page])
    menu_ocr.click(fn=go_ocr, outputs=[home_page, ocr_page, guide_page])
    btn_start.click(fn=go_ocr, outputs=[home_page, ocr_page, guide_page])
    menu_guide.click(fn=go_guide, outputs=[home_page, ocr_page, guide_page])

    # ==========================================
    # 4. KẾT NỐI SỰ KIỆN NÚT BẤM
    # ==========================================
    btn_ocr.click(fn=xu_ly_anh_va_ocr, inputs=input_img, outputs=[output_enhanced, output_raw])
    btn_ai.click(fn=xu_ly_ai_va_so_sanh, inputs=output_raw, outputs=[output_ai, output_diff])
    btn_save.click(fn=thuc_hien_luu, inputs=[output_ai, file_name_in, page_num_in], outputs=[status_msg, page_num_in, file_download])
    
    btn_reset.click(
        fn=lam_moi_trang, 
        inputs=[], 
        outputs=[input_img, output_enhanced, output_raw, output_ai, output_diff, file_name_in, page_num_in, file_download, status_msg]
    )

if __name__ == "__main__":
    vinscan.launch()