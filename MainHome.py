import streamlit as st
import base64
from pathlib import Path

st.set_page_config(
    page_title="Xử lý Ảnh Số - Trang chủ",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🖼️"
)

image_file = 'images/test.jpg'

def get_image_base64(image_path):
    try:
        img_bytes = Path(image_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file ảnh nền tại '{image_path}'")
        return None
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh nền: {e}")
        return None

encoded_image = get_image_base64(image_file)
if encoded_image:
    sidebar_css = f"""
    <style>
    /* CSS cho ảnh nền Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("data:image/jpg;base64,{encoded_image}"); 
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.9;  /* Mờ đi chút để text dễ nhìn hơn */
    }}

    /* Thêm màu nền và hiệu ứng cho các mục trong sidebar */
    a[data-testid="stSidebarNavLink"] {{
        display: block !important;
        padding: 0.8rem 1.2rem !important;
        border-radius: 8px;
        text-decoration: none !important;
        transition: background-color 0.3s ease, transform 0.3s ease;
        color: #FFFFFF !important; 
        font-weight: 600 !important;
        font-size: 1.1em !important;
    }}

    a[data-testid="stSidebarNavLink"]:hover {{
        background-color: rgba(255, 255, 255, 0.2) !important;
        transform: scale(1.05);
    }}

    a[data-testid="stSidebarNavLink"][aria-current="page"] {{
        background-color: rgba(255, 255, 255, 0.3) !important;
    }}

    a[data-testid="stSidebarNavLink"][aria-current="page"] span.st-emotion-cache-6tkfeg {{
        font-weight: 700 !important;
    }}

    /* Cải thiện hover và hiệu ứng cho icon */
    a[data-testid="stSidebarNavLink"] span[data-testid="stIconEmoji"] {{
        vertical-align: middle;
        margin-right: 10px;
    }}

    /* Tùy chỉnh màu chữ các mục khi hover */
    a[data-testid="stSidebarNavLink"]:hover span.st-emotion-cache-6tkfeg {{
        color: #FFD700 !important; /* Vàng khi hover */
    }}

    </style>
    """
    st.markdown(sidebar_css, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

# Cấu hình các trang
pages = [
    st.Page("pages/GioiThieu.py", title="Trang Chủ", icon="🏠"),
    st.Page("pages/1_NhanDienKhuonMat.py", title="Nhận diện Khuôn mặt", icon="🙂"),
    st.Page("pages/2_NhanDienTraiCay.py", title="Nhận diện Trái cây", icon="🍎"),
    st.Page("pages/3_XuLyAnhChuong03.py", title="Xử lý Ảnh Cơ bản (C3)", icon="⚙️"),
    st.Page("pages/4_XuLyAnhChuong04.py", title="Xử lý Miền tần số (C4)", icon="📊"),
    st.Page("pages/5_XuLyAnhChuong05.py", title="Khôi phục Ảnh (C5)", icon="🔧"),
    st.Page("pages/6_XuLyAnhChuong09.py", title="Xử lý Hình thái (C9)", icon="🔬"),
    st.Page("pages/7_Sign_Language.py", title="Ngôn ngữ Kí hiệu (ASL)", icon="🖐️"),
]

# Tạo navigation
pg = st.navigation(pages)
pg.run()
