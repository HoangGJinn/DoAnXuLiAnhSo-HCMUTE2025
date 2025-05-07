import streamlit as st
from PIL import Image


st.title("Chào mừng đến với Ứng dụng Xử lý Ảnh Số 👋")
st.markdown("Khám phá các kỹ thuật xử lý ảnh cơ bản và nâng cao, bao gồm các bộ lọc, biến đổi miền tần số, nhận dạng khuôn mặt và phát hiện đối tượng.")

try:
     banner_image = Image.open('images/maxresdefault.jpg') # Đảm bảo bạn có ảnh này
     st.image(banner_image, use_container_width=True)
except FileNotFoundError:
     st.warning("Không tìm thấy ảnh banner mặc định (images/maxresdefault.jpg).")
st.markdown("---")

st.header("Các tính năng chính")
st.markdown("Ứng dụng cung cấp các module xử lý và nhận dạng ảnh sau:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("⚙️ Xử lý ảnh cơ bản (Chương 3)")
    st.markdown(
        """
        Thực hiện các phép biến đổi cơ bản trên ảnh:
        - Âm bản (Negative)
        - Biến đổi Logarit, Lũy thừa (Power)
        - Giãn tuyến tính (Piecewise Linear)
        - Histogram và Cân bằng Histogram (Grayscale & Color)
        - Các bộ lọc không gian: Box, Gaussian, Median, Sharpen, Gradient.
        - Phân ngưỡng (Thresholding).
        """
    )

    st.subheader("🙂 Nhận dạng Khuôn mặt")
    st.markdown(
        """
        Phát hiện và nhận dạng khuôn mặt trong thời gian thực (webcam),
        từ video có sẵn hoặc trên ảnh tĩnh tải lên.
        - Sử dụng mô hình YuNet (phát hiện) và SFace (trích xuất đặc trưng).
        - Phân loại bằng SVM đã huấn luyện (tối đa 5 khuôn mặt/khung hình).
        - Hiển thị nhãn "Cannot Recognize!" nếu độ tin cậy thấp.
        """
    )

    st.subheader("🔧 Khôi phục ảnh (Chương 5)")
    st.markdown(
        """
        Thực hiện các kỹ thuật khôi phục ảnh bị suy biến:
        - Tạo nhiễu chuyển động (Motion Blur).
        - Lọc ngược (Inverse Filter) để khôi phục ảnh ít nhiễu.
        - Kết hợp lọc Median và lọc ngược cho ảnh nhiều nhiễu.
        - Lọc Wiener (kết hợp Median) để khôi phục ảnh có nhiễu.
        """
    )


with col2:
    st.subheader("📊 Xử lý Miền tần số (Chương 4)")
    st.markdown(
        """
        Áp dụng các kỹ thuật xử lý ảnh trong miền tần số:
        - Hiển thị phổ biên độ (Spectrum).
        - Lọc thông cao (Highpass Butterworth Filter).
        - Tạo và vẽ bộ lọc Notch Reject.
        - Loại bỏ nhiễu Moire.
        """
    )

    st.subheader("🍎 Nhận dạng Trái cây (YOLOv8)")
    st.markdown(
        """
        Phát hiện các loại trái cây cụ thể trong ảnh tải lên:
        - Sử dụng mô hình YOLOv8 tùy chỉnh ('yolov8n_trai_cay.onnx').
        - Nhận diện các lớp: Sầu riêng, Táo, Thanh long (có thể mở rộng).
        - Vẽ bounding box và hiển thị tên lớp cùng độ tin cậy.
        """
    )

    st.subheader("🔬 Xử lý Hình thái Ảnh (Chương 9)")
    st.markdown(
        """
        Áp dụng các phép toán hình thái học cơ bản và nâng cao:
        - Phép co (Erosion) với cấu trúc phần tử lớn.
        - Phép giãn (Dilation) với cấu trúc phần tử nhỏ.
        - Trích xuất đường biên (Boundary Extraction).
        - Lấp đầy lỗ (Hole Filling) trong đối tượng.
        - Tìm và đánh dấu các thành phần liên thông (Connected Components).
        - Ứng dụng: Đếm số lượng hạt gạo trong ảnh.
        """
    )
st.markdown("---")

# Tạo tiêu đề với màu được chọn
st.markdown("<h2 style='color:#ff4b4b;'>🎯 Tính năng nâng cao</h2>", unsafe_allow_html=True)
st.markdown("Ứng dụng phát hiện ngôn ngữ kí hiệu theo (ASL) từ webcam:")
st.markdown(
    """
    - Sử dụng mô hình MediaPipe Hands để phát hiện và theo dõi bàn tay, thư viện OpenCV để xử lý ảnh.
    - Nhận diện các ký hiệu cơ bản như "A", "B", "C", ... "Z".
    - Hiển thị ký hiệu nhận diện được trên webcam trong thời gian thực thành 1 từ và có thể đọc nó lên cho người dùng.
    """
)


st.markdown("---")

st.header("Hướng dẫn sử dụng")
st.markdown(
    """
    1.  **Chọn chức năng:** Sử dụng thanh điều hướng bên trái (sidebar) để chọn tác vụ bạn muốn thực hiện (ví dụ: "Xử lý ảnh Cơ bản", "Nhận dạng Khuôn mặt", "Nhận dạng Trái cây").
    2.  **Cung cấp đầu vào:**
        * Đối với các chức năng xử lý ảnh cơ bản hoặc nhận dạng trái cây, hãy **tải lên hình ảnh** của bạn bằng nút "Tải lên ảnh". Một số chức năng có thể có ảnh mặc định để bạn thử nghiệm.
        * Đối với nhận dạng khuôn mặt, bạn có thể chọn **"Real-Time Camera"** để dùng webcam, **"Process Video File"** để tải video lên hoặc chọn video mặc định, hoặc **"Process Static Image"** để tải ảnh tĩnh lên.
    3.  **Xem kết quả:**
        * Kết quả xử lý hoặc nhận dạng sẽ được hiển thị trên trang.
        * Đối với video/webcam, nhấn nút "Stop" để dừng.
        * Đối với xử lý ảnh/trái cây, nhấn nút "Bắt đầu..." (nếu có) để thực hiện.
    4.  **Khám phá:** Hãy thử nghiệm với các hình ảnh và chức năng khác nhau!
    """
)

st.markdown("---")
st.markdown("Đây là đồ án môn học xử lí ảnh số - HCMUTE - 2025.")
st.markdown("Ứng dụng được phát triển bởi [Nguyễn Hoàng Giáp / 23110096] và [Nguyễn Thành Vinh / 23110172]")
