import gradio as gr
import requests

# --- Câu hỏi gốc ---
questions_vi = [
    "1. Tôi thường để ý đến những âm thanh nhỏ mà người khác không để ý.",
    "2. Khi đọc truyện, tôi thấy khó hiểu được ý định của các nhân vật.",
    "3. Tôi thấy dễ hiểu “ẩn ý” khi ai đó đang nói chuyện với tôi.",
    "4. Tôi thường tập trung vào bức tranh tổng thể hơn là các chi tiết nhỏ.",
    "5. Tôi biết cách nhận ra khi người nghe tôi nói bắt đầu thấy chán.",
    "6. Tôi thấy dễ dàng làm nhiều việc cùng lúc.",
    "7. Tôi có thể dễ dàng nhận ra người khác đang nghĩ gì hoặc cảm thấy gì chỉ bằng cách nhìn vào khuôn mặt họ.",
    "8. Nếu bị gián đoạn, tôi có thể nhanh chóng quay lại công việc đang làm.",
    "9. Tôi thích thu thập thông tin về các loại chủ đề hoặc vật thể khác nhau.",
    "10. Tôi thấy khó hiểu được ý định của mọi người."
]

# --- Mapping hiển thị ---
gender_choices = ["Nam", "Nữ"]
ethnicity_choices = [
    'Không xác định', 'Châu Á', 'Người da đen', 'Người gốc Tây Ban Nha / Mỹ Latinh',
    'Người Latin', 'Trung Đông', 'Người Thái Bình Dương', 'Nam Á',
    'Thổ Nhĩ Kỳ', 'Người da trắng châu Âu', 'Khác'
]
bool_choices = ["Không", "Có"]
country_choices = [
    'Afghanistan','AmericanSamoa','Angola','Argentina','Armenia','Aruba',
    'Australia','Austria','Azerbaijan','Bahamas','Bangladesh','Belgium',
    'Bolivia','Brazil','Burundi','Canada','China','Cyprus','Czech Republic',
    'Egypt','Ethiopia','France','Germany','Hong Kong','Iceland','India',
    'Iran','Iraq','Ireland','Italy','Japan','Jordan','Kazakhstan','Malaysia',
    'Mexico','Netherlands','New Zealand','Nicaragua','Niger','Oman','Pakistan',
    'Romania','Russia','Saudi Arabia','Serbia','Sierra Leone','South Africa',
    'Spain','Sri Lanka','Sweden','Tonga','Ukraine','United Arab Emirates',
    'United Kingdom','United States','Viet Nam'
]
relation_choices = ["Bản thân","Cha/mẹ","Anh/chị/em","Người thân khác","Khác"]

# --- Hàm gọi API FastAPI ---
def predict_via_api(*args):
    # args = A1-A10 + age + gender + ethnicity + jaundice + autism + country + used_app + relation
    json_data = {
        "A1_Score": args[0], "A2_Score": args[1], "A3_Score": args[2], "A4_Score": args[3], "A5_Score": args[4],
        "A6_Score": args[5], "A7_Score": args[6], "A8_Score": args[7], "A9_Score": args[8], "A10_Score": args[9],
        "age": args[10],
        "gender": args[11],
        "ethnicity": args[12],
        "jaundice": args[13],
        "autism": args[14],
        "country_of_res": args[15],
        "used_app_before": args[16],
        "relation": args[17]
    }
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=json_data)
        return response.json().get("prediction", "Lỗi trả về từ server")
    except Exception as e:
        return f"Lỗi khi gọi API: {e}"

# --- Giao diện Gradio ---
with gr.Blocks(title="Dự đoán Tự Kỷ") as demo:
    gr.Markdown("""
    ## 🧩 Bài kiểm tra Dự đoán Tự Kỷ
    Vui lòng trả lời các câu hỏi (0 = Không đúng, 1 = Đúng) và nhập các thông tin khác.  
    Nhấn **Dự đoán** để xem kết quả.
    """)

    inputs = []

    # 10 câu hỏi AQ10
    for q in questions_vi:
        inputs.append(gr.Radio([0,1], label=q, type="index"))

    # Tuổi
    inputs.append(gr.Number(label="Tuổi"))

    # Giới tính
    inputs.append(gr.Radio(gender_choices, label="Giới tính", type="index"))

    # Dân tộc
    inputs.append(gr.Dropdown(ethnicity_choices, label="Dân tộc", type="index"))

    # Vàng da, Tiền sử tự kỷ, Đã sử dụng app
    for label in ["Vàng da", "Tiền sử tự kỷ", "Đã sử dụng ứng dụng trước đó"]:
        inputs.append(gr.Radio(bool_choices, label=label, type="index"))

    # Quê quán
    inputs.append(gr.Dropdown(country_choices, label="Quê quán", type="index"))

    # Quan hệ với người làm test
    inputs.append(gr.Dropdown(relation_choices, label="Quan hệ với người làm test", type="index"))

    # Button & Output
    predict_btn = gr.Button("🚀 Dự đoán")
    output = gr.Markdown()
    predict_btn.click(fn=predict_via_api, inputs=inputs, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
