import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from library.Chapter03 import (
        Negative, Logarit, Power, PiecewiseLinear, CalculateAndShowHistogram,
        HistEqual, HistEqualColor, LocalHist, HistStat, MySmoothBox, Gauss,
        Hubble, MyMedianFilter, Sharp, Gradient
    )
    st.info("Đã tải các thành công các chức năng của chapter 3")
except ImportError:
    st.error("Không thể tải lên chapter3. Hãy đảm bảo chapter 3 có trong folder.")
    st.stop()
except Exception as e:
    st.error(f"Xảy ra lỗi khi tải lên: {e}")
    st.stop()

DEFAULT_IMAGE_DIR = Path("./images/ImageProcessingChapter03")
DEFAULT_IMAGE_MAP = {
    "1. Negative Image.": "negative_default.jpg",
    "2. Lograrit ảnh.": "logarit_default.jpg",
    "3. Lũy thừa ảnh.": "luythua_default.jpg",
    "4. Biến đổi tuyến tính từng phần.": "tuyentinhtungphan_default.jpg",
    "5. Histogram": "histogram_default.jpg",
    "6. Cân bằng Histogram": "canbanghistogram_default.jpg",
    "7. Cân bằng Histogram của ảnh màu.": "canbanghistogramanhmau_default.jpg",
    "8. Local Histogram.": "LocalHistogram_default.png",
    "9. Thống kê Histogram": "ThongKeHistogram_default.png",
    "10. Lọc Box": "LocBox_default.jpg",
    "11. Lọc Gauss": "LocGauss_default.jpg",
    "12. Phân Ngưỡng": "PhanNguong_default.jpg",
    "13. Lọc Median": "Loc_Median_default.jpg",
    "14. Sharpen": "Shapen_default.jpg",
    "15. Gradient": "Gradient_default.jpg"
}

OPERATIONS_C3 = {
    "1. Negative Image.": {"func": Negative, "color_input": False},
    "2. Lograrit ảnh.": {"func": Logarit, "color_input": False},
    "3. Lũy thừa ảnh.": {"func": Power, "color_input": False},
    "4. Biến đổi tuyến tính từng phần.": {"func": PiecewiseLinear, "color_input": False},
    "5. Histogram": {"func": CalculateAndShowHistogram, "color_input": False},
    "6. Cân bằng Histogram": {"func": HistEqual, "color_input": False},
    "7. Cân bằng Histogram của ảnh màu.": {"func": HistEqualColor, "color_input": True}, # Requires color
    "8. Local Histogram.": {"func": LocalHist, "color_input": False},
    "9. Thống kê Histogram": {"func": HistStat, "color_input": False},
    "10. Lọc Box": {"func": MySmoothBox, "color_input": False},
    "11. Lọc Gauss": {"func": Gauss, "color_input": False},
    "12. Phân Ngưỡng": {"func": Hubble, "color_input": False},
    "13. Lọc Median": {"func": MyMedianFilter, "color_input": False},
    "14. Sharpen": {"func": Sharp, "color_input": False},
    "15. Gradient": {"func": Gradient, "color_input": False}
}


st.title("Xử lý ảnh Cơ bản (Chương 3) ⚙️")

selected_operation_name = st.selectbox(
    "Chọn phép xử lý:",
    options=list(OPERATIONS_C3.keys()),
    key="c3_operation_select"
)

operation_details = OPERATIONS_C3[selected_operation_name]
processing_function = operation_details["func"]
requires_color = operation_details["color_input"]
default_image_filename = DEFAULT_IMAGE_MAP.get(selected_operation_name) # Get default image if mapped

img_input = None
uploaded_file = None
use_default = False

uploaded_file = st.file_uploader(
    "Tải lên ảnh",
    type=["bmp", "png", "jpg", "jpeg"],
    key="c3_uploader"
)

if uploaded_file is None and default_image_filename:
    use_default = st.checkbox(f"Sử dụng ảnh mặc định ({default_image_filename})", value=True, key="c3_use_default")

input_image_path_or_buffer = None
input_caption = "Chưa chọn ảnh"

if uploaded_file:
    input_image_path_or_buffer = uploaded_file
    input_caption = f"Ảnh tải lên: {uploaded_file.name}"
    st.info("Đang sử dụng ảnh tải lên.")
elif use_default and default_image_filename:
    default_image_path = DEFAULT_IMAGE_DIR / default_image_filename
    if default_image_path.is_file():
        input_image_path_or_buffer = default_image_path
        input_caption = f"Ảnh mặc định: {default_image_filename}"
        st.info(f"Đang sử dụng ảnh mặc định: {default_image_filename}")
    else:
        st.warning(f"Không tìm thấy ảnh mặc định: {default_image_path}")
else:
     st.warning("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mặc định (nếu có).")

col1, col2 = st.columns(2)

if input_image_path_or_buffer:
    try:
        img_pil = Image.open(input_image_path_or_buffer)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with col1:
            st.subheader("Ảnh gốc")
            st.image(img_pil, caption=input_caption, use_container_width=True)
        if requires_color:
             img_input_for_processing = img_bgr
        else:
             img_input_for_processing = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        st.divider()

        with col2:
            st.subheader("Kết quả xử lý")
            try:
                with st.spinner("Đang xử lý..."):
                    output_result = processing_function(img_input_for_processing)
                    if output_result is not None:
                        if isinstance(output_result, plt.Figure):
                             st.pyplot(output_result)
                        elif isinstance(output_result, np.ndarray):
                             if output_result.ndim == 2:
                                 st.image(output_result, caption=f"Kết quả: {selected_operation_name}", use_container_width=True)
                             elif output_result.ndim == 3 and output_result.shape[2] == 3:
                                 output_rgb = cv2.cvtColor(output_result, cv2.COLOR_BGR2RGB)
                                 st.image(output_rgb, caption=f"Kết quả: {selected_operation_name}", use_container_width=True)
                             else:
                                 st.error("Định dạng ảnh đầu ra không hợp lệ.")
                        else:
                             st.error("Hàm xử lý không trả về hình ảnh hoặc biểu đồ hợp lệ.")
                    else:
                        st.error("Hàm xử lý không trả về kết quả.")

            except Exception as e:
                st.error(f"Lỗi trong quá trình xử lý '{selected_operation_name}': {e}")
                import traceback
                st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Không thể đọc hoặc xử lý ảnh đầu vào: {e}")